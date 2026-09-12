"""90-day baseline forecast for Buy or Wait?

Step 2 only: simulate balance day-by-day for 90 days with NO request
payment applied. The later decision step overlays candidate payments
onto this baseline.

Inputs come from code/financial_state.py state dicts.
Rules (per approved plan):
- Include: reserved pending debits, scheduled debits, confirmed future
  salary credits, recurring essentials (history-supported only).
- Exclude: pending/unconfirmed credits, failed/cancelled/unrealized,
  one-time history (already in starting balance), blank/missing-FX amounts.
- Recurring projection: per category, median interval from last seen dates
  (fallback 30 days), average amount, projected forward inside the window.
- Messages/images: listed in assumptions; applied only when they carry an
  explicit date + amount (v1: flagged, not auto-applied -- hook provided).
- Deterministic: sorted ledger, event_id tie-break, 2-decimal rounding.
"""

from datetime import date, timedelta

FORECAST_DAYS = 90


def _parse(d):
    return date.fromisoformat(d[:10]) if d else None


def _median(xs):
    xs = sorted(xs)
    n = len(xs)
    if n == 0:
        return None
    mid = n // 2
    if n % 2 == 1:
        return xs[mid]
    return (xs[mid - 1] + xs[mid]) / 2


def project_recurring(state, start, end):
    """Project recurring DEBIT categories forward. Returns (flows, assumptions)."""
    flows = []  # (date, signed_amount, label, event_ref)
    assumptions = []
    for cat, r in sorted(state.get("recurring_candidates", {}).items()):
        dates = [_parse(d) for d in r.get("all_dates", r.get("last_3_dates", [])) if d]
        dates = sorted(d for d in dates if d)
        if len(dates) >= 2:
            intervals = [(dates[i + 1] - dates[i]).days for i in range(len(dates) - 1)]
            step = int(round(_median(intervals))) if intervals else 30
            step = max(7, min(90, step))
        else:
            step = 30
        avg = float(r.get("avg_amount", 0))
        if avg <= 0:
            continue
        last = max(dates) if dates else start - timedelta(days=step)
        nxt = last + timedelta(days=step)
        # fast-forward to window
        while nxt < start:
            nxt += timedelta(days=step)
        projected = 0
        d = nxt
        while d <= end:
            flows.append((d, -round(avg, 2), f"recurring:{cat}", None))
            projected += 1
            d += timedelta(days=step)
        assumptions.append(
            f"projected {cat}: every ~{step}d at {round(avg, 2)} x{projected} "
            f"(from {r.get('count')} settled occurrences)"
        )
    return flows, assumptions


def _add_months(d, n):
    """Step n calendar months preserving day-of-month (clamped)."""
    month = d.month - 1 + n
    year = d.year + month // 12
    month = month % 12 + 1
    import calendar as _cal
    return date(year, month, min(d.day, _cal.monthrange(year, month)[1]))


SALARY_END_KEYWORDS = (
    "final", "last pay", "termination", "end of contract", "resign",
    "layoff", "laid off", "retrench", "severance",
)


def project_salary(state, start, end):
    """Project base salary monthly. Returns (flows, assumptions).

    Amount: latest scheduled regular salary if present, else latest settled
    regular salary. Anchor: latest known salary date. Already-counted
    scheduled salary dates are skipped to avoid double counting.
    A latest salary marked as final/terminal stops all projection.
    """
    sched = state.get("scheduled_salary", []) or []
    hist = state.get("salary_history", []) or []
    latest = sched[-1] if sched else (hist[-1] if hist else None)
    if latest is None:
        return [], ["no regular salary found: no salary projected"]
    if any(k in (latest.get("description") or "").lower() for k in SALARY_END_KEYWORDS):
        return [], [f"salary ended ({latest.get('description')} on "
                    f"{latest.get('settlement_date')}): no salary projected"]
    amount = latest["converted_amount"]
    known = {_parse(x["settlement_date"]) for x in sched + hist if x.get("settlement_date")}
    known = {d for d in known if d}
    anchor = max(known) if known else start
    # first projection strictly after anchor
    d = _add_months(anchor, 1)
    flows = []
    n = 0
    while d <= end:
        if d >= start and d not in known:
            flows.append((d, +round(amount, 2), "recurring:salary", None))
            n += 1
        d = _add_months(d, 1)
    return flows, [f"projected salary: monthly at {round(amount, 2)} x{n}"]


def collect_message_adjustments(state):
    """Hook for explicit message/image date+amount amendments.

    V1: flags relevant messages in assumptions without auto-applying,
    because free-text parsing is not deterministic enough for cash flow.
    Returns (flows, notes) with flows empty for now.
    """
    notes = []
    for m in state.get("messages", []):
        if m.get("request_linked") or m.get("event_linked"):
            notes.append(
                f"message {m.get('message_id')} noted ({m.get('source_type')}) "
                f"sent {m.get('sent_at')}: applied only if explicit date+amount"
            )
    for im in state.get("images", []):
        notes.append(
            f"image {im.get('image_id')} linked to {im.get('related_event_id')} "
            f"exists={im.get('exists')}: blank amounts stay excluded until OCR"
        )
    return [], notes


def build_baseline_forecast(state, horizon_days=FORECAST_DAYS):
    """Simulate baseline balances. Returns forecast dict."""
    start = _parse(state["request_date"])
    end = start + timedelta(days=horizon_days)
    opening = float(state["profile"]["current_available_balance"])
    minimum = float(state["profile"]["minimum_balance_to_keep"])

    dated = []  # (date, signed, label, ref)

    for c in state.get("reserved_pending_debits", []):
        d = _parse(c["settlement_date"])
        if d and start <= d <= end:
            dated.append((d, -c["converted_amount"], f"reserved:{c['category']}", c["event_id"]))
    for c in state.get("scheduled_obligations", []):
        d = _parse(c["settlement_date"])
        if d and start <= d <= end:
            dated.append((d, -c["converted_amount"], f"scheduled:{c['category']}", c["event_id"]))
    for c in state.get("confirmed_future_credits", []):
        d = _parse(c["settlement_date"])
        if d and start <= d <= end:
            dated.append((d, +c["converted_amount"], f"salary:{c['category']}", c["event_id"]))

    rec_flows, rec_notes = project_recurring(state, start, end)
    dated.extend(rec_flows)
    sal_flows, sal_notes = project_salary(state, start, end)
    dated.extend(sal_flows)
    msg_flows, msg_notes = collect_message_adjustments(state)
    dated.extend(msg_flows)

    dated.sort(key=lambda t: (t[0], t[3] or t[2]))

    # aggregate per day preserving order
    per_day = {}
    for d, amt, label, ref in dated:
        per_day.setdefault(d, []).append((amt, label, ref))

    daily_balances = {}
    flows_used = []
    bal = round(opening, 2)
    min_bal = bal
    first_breach = None
    cur = start
    while cur <= end:
        for amt, label, ref in per_day.get(cur, []):
            bal = round(bal + amt, 2)
            flows_used.append({
                "date": cur.isoformat(), "amount": amt,
                "label": label, "ref": ref,
                "balance_after": bal,
            })
        daily_balances[cur.isoformat()] = bal
        if bal < min_bal:
            min_bal = bal
        if first_breach is None and bal < minimum:
            first_breach = cur.isoformat()
        cur += timedelta(days=1)

    return {
        "request_id": state["request_id"],
        "start": start.isoformat(),
        "end": end.isoformat(),
        "opening_balance": opening,
        "minimum_to_keep": minimum,
        "daily_balances": daily_balances,
        "baseline_min_balance": round(min_bal, 2),
        "baseline_safe": first_breach is None,
        "first_breach_date": first_breach,
        "flows_used": flows_used,
        "assumptions": rec_notes + sal_notes + msg_notes + [
            f"excluded {state['counts']['excluded']} failed/cancelled/unrealized/pending-credit rows",
            f"blank amounts excluded: {state['counts']['blank_amount']}",
        ],
    }


def is_payment_safe(forecast, payment_date, amount):
    """Check if a single full payment on payment_date keeps balance >= minimum.

    Replays baseline flows plus one extra outflow. Deterministic helper for
    the later decision step; no method ranking here.
    """
    start = _parse(forecast["start"])
    end = _parse(forecast["end"])
    minimum = forecast["minimum_to_keep"]
    pay_day = _parse(payment_date)
    if pay_day is None or not (start <= pay_day <= end):
        return False
    # rebuild daily net from flows_used
    from collections import defaultdict
    net = defaultdict(float)
    for f in forecast["flows_used"]:
        net[f["date"]] += f["amount"]
    net[pay_day.isoformat()] -= float(amount)
    bal = forecast["opening_balance"]
    cur = start
    while cur <= end:
        bal = round(bal + net.get(cur.isoformat(), 0.0), 2)
        if bal < minimum:
            return False
        cur += timedelta(days=1)
    return True
