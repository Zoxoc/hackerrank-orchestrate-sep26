"""Core financial state reconstruction for Buy or Wait?

Step 1 only: gather a reliable financial picture per request.
No forecasting, no payment decision, no optimization here.

Rules applied (from problem_statement.md / AGENTS.md):
- Reserve pending debits; never count pending credits until settled.
- Count confirmed salary on settlement date; ignore bonuses, commissions,
  refunds, lottery, investment gains until settled.
- Exclude failed / cancelled / unrealized / non_cash from cash flow.
- linked_event_id is informational; each row is judged by its own cash state
  (so a cancelled row is excluded while its settled re-issue counts once).
- Foreign-currency cash events convert with the FX row for the settlement
  date and stated from->to direction (amount * rate).
- Blank amounts are NEVER zero; they link via images.csv related_event_id
  to dataset/media/images/<image_id>.png and are flagged missing.
- Recurrence is flagged only when history supports it (>=2 settled
  occurrences in the same category before request_date).
- Messages/images are untrusted evidence; attached with relevance flags,
  never as overriding instructions.
"""

import csv
import os
from collections import defaultdict
from datetime import date

DATASET_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "..", "dataset")
# When imported as code.financial_state, __file__ is <root>/code/financial_state.py,
# so dataset is <root>/dataset. Normalize both cases:
if not os.path.isdir(DATASET_DIR):
    DATASET_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "dataset")
DATASET_DIR = os.path.normpath(DATASET_DIR)

CASH_STATUSES = ("settled", "pending", "scheduled")
EXCLUDED_STATUSES = ("failed", "cancelled", "unrealized")

# Keywords marking income that must NOT count until settled (even if scheduled).
UNCONFIRMED_CREDIT_KEYWORDS = (
    "bonus", "commission", "refund", "lottery", "investment",
    "valuation", "unrealized", "gain",
)


def parse_date(s):
    s = (s or "").strip()
    if not s:
        return None
    return date.fromisoformat(s[:10])


def parse_amount(s):
    s = (s or "").strip()
    if not s:
        return None
    try:
        return float(s)
    except ValueError:
        return None


def load_csv(name):
    path = os.path.join(DATASET_DIR, name)
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def build_fx_index(fx_rows):
    """(rate_date, from_ccy, to_ccy) -> rate float."""
    index = {}
    for r in fx_rows:
        try:
            index[(r["rate_date"].strip(), r["from_currency"].strip(), r["to_currency"].strip())] = float(
                r["rate"].strip()
            )
        except (ValueError, AttributeError):
            continue
    return index


def convert_to_home(amount, from_ccy, home_ccy, settlement_date, fx_index):
    """Return (converted_amount, rate_used, fx_note). No chaining; direct pair only."""
    if amount is None:
        return None, None, "missing_amount"
    if from_ccy == home_ccy:
        return amount, 1.0, "same_currency"
    key = (settlement_date, from_ccy, home_ccy)
    rate = fx_index.get(key)
    if rate is None:
        return None, None, f"missing_fx:{settlement_date}:{from_ccy}->{home_ccy}"
    return amount * rate, rate, "converted"


def is_unconfirmed_credit(event):
    """True for bonuses/commissions/refunds/lottery/investment gains (not base salary)."""
    text = f"{event.get('event_type','')} {event.get('category','')} {event.get('description','')}".lower()
    if "salary" in text:
        return False  # base salary is the confirmed-income case
    return any(k in text for k in UNCONFIRMED_CREDIT_KEYWORDS)


def classify_event(event, request_date, home_ccy, fx_index):
    """Classify one event row relative to a request. Returns a dict."""
    status = (event.get("status") or "").strip()
    direction = (event.get("direction") or "").strip()
    settle = parse_date(event.get("settlement_date"))
    raw_amount = parse_amount(event.get("amount"))
    ccy = (event.get("currency") or "").strip() or home_ccy

    info = {
        "event_id": event.get("event_id"),
        "event_type": event.get("event_type"),
        "category": event.get("category"),
        "direction": direction,
        "status": status,
        "raw_amount": raw_amount,
        "currency": ccy,
        "event_date": event.get("event_date"),
        "settlement_date": event.get("settlement_date"),
        "flexibility": (event.get("flexibility") or "").strip(),
        "linked_event_id": (event.get("linked_event_id") or "").strip(),
        "amount_missing": raw_amount is None,
        "converted_amount": None,
        "fx_note": None,
        "cash_role": "excluded",
        "reason": "",
    }

    if raw_amount is not None:
        conv, rate, note = convert_to_home(
            raw_amount, ccy, home_ccy,
            (event.get("settlement_date") or "").strip(), fx_index,
        )
        info["converted_amount"] = conv
        info["fx_note"] = note if note != "converted" else f"converted@{rate}"
    else:
        info["fx_note"] = "missing_amount"

    # --- exclusion rules ---
    if status in ("failed", "cancelled"):
        info["reason"] = f"status_{status}_excluded"
        return info
    if status == "unrealized" or direction == "non_cash":
        info["reason"] = "unrealized_or_non_cash_excluded"
        return info
    if status not in CASH_STATUSES:
        info["reason"] = f"unknown_status_{status}_excluded"
        return info
    if raw_amount is None:
        info["reason"] = "blank_amount_needs_image"
        return info
    if info["converted_amount"] is None:
        info["reason"] = info["fx_note"]
        return info

    is_future = settle is not None and settle >= request_date
    is_history = settle is not None and settle < request_date

    if direction == "debit":
        if is_history and status == "settled":
            info["cash_role"] = "history"
            info["reason"] = "settled_before_request_already_in_balance"
        elif status == "pending" and is_future:
            info["cash_role"] = "reserved_pending_debit"
            info["reason"] = "pending_debit_reserved"
        elif status == "scheduled":
            info["cash_role"] = "scheduled_obligation"
            info["reason"] = "scheduled_debit_is_future_outflow"
        elif status == "settled" and is_future:
            info["cash_role"] = "scheduled_obligation"
            info["reason"] = "settled_with_future_settlement_treated_as_outflow"
        else:
            info["cash_role"] = "history"
            info["reason"] = "past_debit_history"
    elif direction == "credit":
        if status == "pending":
            info["reason"] = "pending_credit_not_counted_until_settled"
        elif status == "scheduled" and is_unconfirmed_credit(event):
            info["reason"] = "scheduled_unconfirmed_credit_not_counted"
        elif status == "settled" and is_history:
            info["cash_role"] = "history"
            info["reason"] = "settled_credit_history"
        elif status in ("scheduled", "settled") and is_future:
            # Confirmed salary (or other confirmed credit incl. settled refunds)
            # counts on settlement date. Refunds that are already settled in
            # the future are kept; pending ones above are excluded.
            info["cash_role"] = "confirmed_future_credit"
            info["reason"] = "confirmed_credit_on_settlement_date"
        else:
            info["reason"] = "credit_not_counted"
    else:
        info["reason"] = f"direction_{direction}_excluded"

    return info


def detect_recurring(history_infos):
    """Group settled history by category; flag only with >=2 occurrences."""
    by_cat = defaultdict(list)
    for h in history_infos:
        if h["cash_role"] == "history" and h["converted_amount"] is not None:
            by_cat[h["category"]].append(h)
    recurring = {}
    for cat, items in sorted(by_cat.items()):
        if len(items) >= 2:
            amounts = [i["converted_amount"] for i in items]
            dates = sorted(i["settlement_date"] for i in items)
            recurring[cat] = {
                "count": len(items),
                "avg_amount": round(sum(amounts) / len(amounts), 2),
                "last_3_dates": dates[-3:],
                "event_ids": [i["event_id"] for i in items[-5:]],
            }
    return recurring


def build_request_state(request, profiles_by_user, events_by_user,
                        options_by_request, messages, images, fx_index):
    """Build the financial picture for one request. No forecasting here."""
    user_id = request["user_id"]
    request_id = request["request_id"]
    request_date = parse_date(request["request_date"])
    profile = profiles_by_user.get(user_id, {})
    home_ccy = (profile.get("home_currency") or "").strip()

    raw_events = events_by_user.get(user_id, [])
    classified = [classify_event(e, request_date, home_ccy, fx_index) for e in raw_events]

    reserved = [c for c in classified if c["cash_role"] == "reserved_pending_debit"]
    scheduled_out = [c for c in classified if c["cash_role"] == "scheduled_obligation"]
    future_credits = [c for c in classified if c["cash_role"] == "confirmed_future_credit"]
    history = [c for c in classified if c["cash_role"] == "history"]
    excluded = [c for c in classified if c["cash_role"] == "excluded"]
    missing = [c for c in classified if c["amount_missing"]]

    # Linked pairs for transparency (both directions).
    by_id = {c["event_id"]: c for c in classified}
    linked_pairs = []
    for c in classified:
        lid = c["linked_event_id"]
        if lid and lid in by_id:
            linked_pairs.append({
                "from": c["event_id"],
                "to": lid,
                "from_role": c["cash_role"],
                "to_role": by_id[lid]["cash_role"],
            })

    recurring = detect_recurring(history)

    # Flexible future obligations adjustable later (decision step uses these).
    adjustable = [
        c for c in classified
        if c["cash_role"] in ("reserved_pending_debit", "scheduled_obligation")
        and (c["flexibility"] or "fixed") != "fixed"
    ]

    # Relevant messages: same user; flag request-linked and event-linked.
    event_ids = {c["event_id"] for c in classified}
    rel_messages = []
    for m in messages:
        if m.get("user_id") != user_id:
            continue
        m_req = (m.get("request_id") or "").strip()
        m_ev = (m.get("related_event_id") or "").strip()
        if m_req == request_id or (m_ev and m_ev in event_ids) or not m_req:
            rel_messages.append({
                "message_id": m.get("message_id"),
                "request_linked": m_req == request_id,
                "event_linked": m_ev in event_ids,
                "related_event_id": m_ev or None,
                "source_type": m.get("source_type"),
                "sent_at": m.get("sent_at"),
                "text": m.get("message_text"),
            })
    rel_messages.sort(key=lambda m: (not m["request_linked"], not m["event_linked"], m["sent_at"] or ""))

    # Relevant images: same user+request or linked to user events.
    rel_images = []
    for im in images:
        if im.get("user_id") != user_id:
            continue
        im_req = (im.get("request_id") or "").strip()
        im_ev = (im.get("related_event_id") or "").strip()
        if im_req == request_id or im_ev in event_ids:
            path = os.path.join(DATASET_DIR, "media", "images", f"{im.get('image_id')}.png")
            rel_images.append({
                "image_id": im.get("image_id"),
                "related_event_id": im_ev or None,
                "request_linked": im_req == request_id,
                "path": path,
                "exists": os.path.exists(path),
            })

    # Attach image existence to missing-amount events.
    img_by_event = {i["related_event_id"]: i for i in rel_images if i["related_event_id"]}
    for c in missing:
        hit = img_by_event.get(c["event_id"])
        c["linked_image"] = hit["image_id"] if hit else None
        c["linked_image_exists"] = bool(hit and hit["exists"])

    totals = {
        "reserved_pending_debits": round(sum(c["converted_amount"] for c in reserved), 2),
        "scheduled_obligations": round(sum(c["converted_amount"] for c in scheduled_out), 2),
        "confirmed_future_credits": round(sum(c["converted_amount"] for c in future_credits), 2),
    }

    return {
        "request_id": request_id,
        "user_id": user_id,
        "request_date": request.get("request_date"),
        "requested_amount": parse_amount(request.get("requested_amount")),
        "desired_completion_date": request.get("desired_completion_date"),
        "allows_partial_payment": (request.get("allows_partial_payment") or "").strip().lower() == "true",
        "request_type": request.get("request_type"),
        "profile": {
            "home_currency": home_ccy,
            "current_available_balance": parse_amount(profile.get("current_available_balance")),
            "minimum_balance_to_keep": parse_amount(profile.get("minimum_balance_to_keep")),
            "payment_methods_user_will_consider": [
                p for p in (profile.get("payment_methods_user_will_consider") or "").split("|") if p
            ],
            "max_installment_months": (profile.get("max_installment_months") or "").strip() or None,
            "protect": [p for p in (profile.get("expense_categories_to_protect") or "").split("|") if p],
            "willing_to_reduce": [
                p for p in (profile.get("expense_categories_user_is_willing_to_reduce") or "").split("|") if p
            ],
            "willing_to_stop": [
                p for p in (profile.get("expense_categories_user_is_willing_to_stop") or "").split("|") if p
            ],
        },
        "payment_options": options_by_request.get(request_id, []),
        "counts": {
            "total_user_events": len(raw_events),
            "history": len(history),
            "reserved_pending_debits": len(reserved),
            "scheduled_obligations": len(scheduled_out),
            "confirmed_future_credits": len(future_credits),
            "excluded": len(excluded),
            "blank_amount": len(missing),
        },
        "totals_home_currency": totals,
        "reserved_pending_debits": reserved,
        "scheduled_obligations": scheduled_out,
        "confirmed_future_credits": future_credits,
        "adjustable_future": adjustable,
        "excluded": excluded,
        "blank_amount_events": missing,
        "linked_pairs": linked_pairs,
        "recurring_candidates": recurring,
        "messages": rel_messages,
        "images": rel_images,
    }


def summarize_state(s):
    """One-screen human-readable summary for verification."""
    p = s["profile"]
    lines = [
        f"{s['request_id']} | user={s['user_id']} | date={s['request_date']} "
        f"| want={s['requested_amount']} {p['home_currency']} by {s['desired_completion_date']} "
        f"(partial_allowed={s['allows_partial_payment']})",
        f"  balance={p['current_available_balance']} min_keep={p['minimum_balance_to_keep']} "
        f"considers={p['payment_methods_user_will_consider']} max_inst={p['max_installment_months']}",
        f"  events: total={s['counts']['total_user_events']} history={s['counts']['history']} "
        f"reserved={s['counts']['reserved_pending_debits']} sched_out={s['counts']['scheduled_obligations']} "
        f"future_credits={s['counts']['confirmed_future_credits']} excluded={s['counts']['excluded']} "
        f"blank={s['counts']['blank_amount']}",
        f"  totals({p['home_currency']}): reserved={s['totals_home_currency']['reserved_pending_debits']} "
        f"sched_out={s['totals_home_currency']['scheduled_obligations']} "
        f"future_credits={s['totals_home_currency']['confirmed_future_credits']}",
        f"  options={len(s['payment_options'])} messages={len(s['messages'])} images={len(s['images'])} "
        f"recurring_cats={len(s['recurring_candidates'])} adjustable_future={len(s['adjustable_future'])}",
    ]
    for c in s["reserved_pending_debits"][:3]:
        lines.append(f"    RESERVED {c['event_id']} {c['category']} {c['converted_amount']} on {c['settlement_date']}")
    for c in s["confirmed_future_credits"][:3]:
        lines.append(f"    FUTURE+  {c['event_id']} {c['category']} {c['converted_amount']} on {c['settlement_date']}")
    for c in s["blank_amount_events"][:3]:
        lines.append(f"    BLANK   {c['event_id']} needs_image={c.get('linked_image')} exists={c.get('linked_image_exists')}")
    top_rec = list(s["recurring_candidates"].items())[:4]
    for cat, r in top_rec:
        lines.append(f"    RECUR   {cat} x{r['count']} avg={r['avg_amount']}")
    return "\n".join(lines)
