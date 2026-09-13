"""Message actions for Buy or Wait?

Converts unstructured messages into typed, dated amendments the forecast
understands. Rule-based and deterministic (regex, English + Indonesian).

Pipeline position: STATE (collects) -> MESSAGE ACTIONS (this file) ->
FORECAST (applies) -> DECISION.

Amendment types (8): SET_SALARY, MOVE_MONEY_DATE, STOP_SALARY, ADD_BILL,
ADD_CONFIRMED_PAY, SCALE_BILL, DEDUPE_TRANSFER, CONFIRM_SKIP.

Safety: explicit amount+date or no action; messages never override rules;
conflicts prefer explicit amendment > newer same-source message >
settled events > financially safer reading.
"""

import re
from datetime import date

MONEY_RE = re.compile(r"\d{1,3}(?:,\d{2,3})+(?:\.\d{1,2})?|\d+\.\d{1,2}|\d{4,}")
ISO_DATE_RE = re.compile(r"(\d{4})-(\d{2})-(\d{2})")
DMY_DATE_RE = re.compile(r"(\d{1,2})-(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*-(\d{4})",
                         re.IGNORECASE)
TEXTUAL_DATE_RE = re.compile(
    r"(\d{1,2})\s+(January|February|March|April|May|June|July|August|"
    r"September|October|November|December)\s+(\d{4})", re.IGNORECASE)
PERCENT_RE = re.compile(r"(\d+(?:\.\d+)?)\s*%")
CCY_RE = re.compile(r"\b(IDR|INR|ZAR|USD|EUR|Rp|Rs|R|€|\$)\b", re.IGNORECASE)

_MONTHS = {m: i + 1 for i, m in enumerate(
    ["jan", "feb", "mar", "apr", "may", "jun",
     "jul", "aug", "sep", "oct", "nov", "dec"])}

# Subject vocabularies (lowercased substrings).
SALARY_WORDS = ("salary", "salaries", "payroll", "payslip", "payday",
                "upah", "gaji", "penggajian", "paycheck", "wages")
# Timing references that pin a pay change to a real payroll (the date itself
# may then come from the event schedule). A bare level statement with no
# timing reference is never applied as an override.
NEXT_PAYROLL_WORDS = ("next payroll", "next pay", "next salary",
                      "upcoming pay", "upcoming salary", "penggajian berikutnya",
                      "gaji berikutnya", "next payslip", "payday",
                      "siklus penggajian berikut")
# "Resumes" language: the stated pay continues monthly afterwards.
RESUME_HINTS = ("resume", "restart", "mulai kembali", "kembali normal",
                "berlanjut", "continue")
# Prize/claim contexts never stop salary (handled as CONFIRM_SKIP instead).
PRIZE_VETO_WORDS = ("prize", "hadiah", "lottery", "lotere", "claim", "klaim")
BONUS_WORDS = ("bonus", "commission", "komisi", "arrears", "tunggakan",
               "adjustment", "penyesuaian", "one-time", "one time", "satu kali")
PAYOUT_WORDS = ("payout", "invoice", "faktur", "pembayaran faktur", "claim",
                "tagihan")
BILL_WORDS = ("childcare", "rent", "lease", "sewa", "bill", "tagihan",
              "payment", "pembayaran", "deduction", "potongan", "subscription")
REFUND_WORDS = ("refund", "pengembalian", "rebate")
INVEST_WORDS = ("portfolio", "market value", "investment", "investasi",
                "units", "unit", "nav")
PRIZE_WORDS = ("prize", "hadiah", "lottery", "lotere", "reward")

SKIP_HINTS = ("pending", "awaiting", "not yet", "not been", "has not",
              "haven't", "unapproved", "until approved", "until.*complet",
              "can change", "withdrawable", "processing", "no .* sold",
              "no cash", "no units", "no further", "tertunda", "menunggu", "belum",
              "belum disetujui", "belum dikredit", "diproses", "tidak.*dijual")
CHANGE_HINTS = ("increased", "increase", "raised", "rise", "rose", "naik",
                "menjadi", "reduced", "reduction", "reduced to", "cut",
                "temporary", "changed", "berubah", "new amount",
                "updated", "diperbarui", "revised", "disesuaikan")
# Declarations of the regular/base pay level (no change verb needed).
BASE_HINTS = ("gaji pokok", "base salary", "regular salary", "gaji rutin",
              "confirmed salary", "basic salary", "gaji yang dikonfirmasi")
FIRST_PAY_HINTS = ("first salary", "first pay", "gaji pertama", "will be eur",
                   "will be idr", "will be zar", "will be usd", "will be inr",
                   "confirmed credit date", "disetujui", "setujui",
                   "dikonfirmasi")
FIRST_PAY_PAIRS = (("approved", "settlement"), ("confirmed", "credit"),
                   ("settlement", "expected"), ("pembayaran", "disetujui"),
                   ("invoice", "approved"))
MOVE_HINTS = ("replaces", "revised date", "rescheduled", "moved to",
              "now expected", "menggantikan", "revisi", "diubah",
              "tanggal baru", "instead")
STOP_HINTS = (("contract", "ended"), ("contract", "no "), ("no ", "renewal"),
              ("no ", "confirmed"), ("kontrak", "berakhir"),
              ("tidak", "konfirmasi"), ("final", "payroll"),
              ("seasonal", "ended"), ("off-season", "no"))
TRANSFER_HINTS = (("transfer", "between"), ("matching", "debit"),
                  ("same account holder",), ("own accounts",),
                  ("antara", "akun"))
SCALE_HINTS = (("increases", "%"), ("increase", "%"), ("naik", "%"),
               ("raised", "%"), ("rose", "%"))


def _parse_date(s):
    s = (s or "").strip()
    if not s:
        return None
    try:
        return date.fromisoformat(s[:10])
    except ValueError:
        return None


def _extract_iso_dates(text):
    return [f"{y}-{m}-{d}" for y, m, d in ISO_DATE_RE.findall(text or "")]


def _extract_any_dates(text):
    found = _extract_iso_dates(text)
    for d, mon, y in DMY_DATE_RE.findall(text or ""):
        found.append(f"{y}-{_MONTHS[mon.lower()[:3]]:02d}-{int(d):02d}")
    for d, mon, y in TEXTUAL_DATE_RE.findall(text or ""):
        found.append(f"{y}-{_MONTHS[mon.lower()[:3]]:02d}-{int(d):02d}")
    return found


def _clean_for_amounts(text):
    """Remove reference codes, dates, and phone-like tokens so only real
    money figures remain for amount extraction."""
    t = text or ""
    t = re.sub(r"\b(EMP|SER|MER|BAN|FIN|TXN|REF|CASE|ORDER|INV|ACCT|ACC|PNR)[- ]?\d+\b",
               " ", t, flags=re.IGNORECASE)
    t = ISO_DATE_RE.sub(" ", t)
    t = DMY_DATE_RE.sub(" ", t)
    t = TEXTUAL_DATE_RE.sub(" ", t)
    t = re.sub(r"\b\d{3,}-\d{4,}\b", " ", t)  # phone fragments
    t = re.sub(r"\(\d+\)", " ", t)  # parenthesised codes
    return t


def _extract_amounts(text):
    out = []
    for m in MONEY_RE.finditer(_clean_for_amounts(text).replace(" ", "")):
        try:
            out.append(float(m.group(0).replace(",", "")))
        except ValueError:
            continue
    return out


def _nearest_salary_amount(text):
    """Amount figure closest to salary words (avoids nearby bonus figures)."""
    low = _clean_for_amounts(text).lower()
    spans = []
    for m in MONEY_RE.finditer(low.replace(" ", "")):
        try:
            spans.append((m.start(), float(m.group(0).replace(",", ""))))
        except ValueError:
            continue
    if not spans:
        return None
    sal_pos = []
    for w in SALARY_WORDS:
        start = 0
        while True:
            i = low.find(w, start)
            if i < 0:
                break
            sal_pos.append(i)
            start = i + 1
    if not sal_pos:
        return spans[0][1]
    best, best_d = spans[0][1], None
    for pos, amt in spans:
        d = min(abs(pos - s) for s in sal_pos)
        if best_d is None or d < best_d:
            best, best_d = amt, d
    return best


def _has(text, words):
    t = (text or "").lower()
    return any(w in t for w in words)


def _has_pair(text, pairs):
    t = (text or "").lower()
    return any(all(w in t for w in pair) for pair in pairs)


def parse_message(msg, state):
    """Parse one message dict -> list of amendment dicts (usually 0-1)."""
    text = msg.get("text") or ""
    mid = msg.get("message_id")
    amendments = []

    def note(action, applied, note_text, params=None):
        entry = {"action": action, "message_id": mid,
                 "applied": applied, "note": note_text,
                 "params": params or {}, "sent_at": msg.get("sent_at")}
        amendments.append(entry)
        return entry

    # 1. Internal-transfer duplicates (exclusive: nothing else applies).
    if _has_pair(text, TRANSFER_HINTS):
        pair = _find_transfer_pair(state)
        if pair:
            return [note("DEDUPE_TRANSFER", True,
                         f"internal move {pair[0]}<->{pair[1]}: debit leg excluded (net 0)",
                         {"debit_event_id": pair[0], "credit_event_id": pair[1]})]
        return [note("DEDUPE_TRANSFER", False,
                     "transfer described but no matching unsettled pair; "
                     "settled pairs already sit inside today's balance, no change")]

    out = []
    # 2. Bonus/commission/arrears part explicitly pending -> stays excluded.
    if _has(text, BONUS_WORDS + REFUND_WORDS) and _has(text, SKIP_HINTS):
        out.append({"action": "CONFIRM_SKIP", "message_id": mid, "applied": True,
                    "note": "bonus/commission/refund part unconfirmed: stays excluded",
                    "params": {}, "sent_at": msg.get("sent_at")})

    # 3. Regular-salary level: change verbs OR base-pay declarations.
    # Amount chosen = figure nearest salary words (not a nearby bonus figure).
    # Skipped for %-scale messages about rent/lease (handled by SCALE_BILL).
    rent_scale = _has_pair(text, SCALE_HINTS) and _has(text, ("rent", "lease", "sewa"))
    if ((_has(text, CHANGE_HINTS) or _has(text, BASE_HINTS))
            and _has(text, SALARY_WORDS) and not rent_scale):
        amount = _nearest_salary_amount(text)
        dates = _extract_any_dates(text)
        eff, eff_source = None, "explicit"
        if dates:
            eff = dates[-1]
        elif _has(text, NEXT_PAYROLL_WORDS):
            # "Your next salary is X": the date is the series' own next
            # occurrence on/after the request (interpolated, never invented).
            eff, eff_source = _next_payroll_date(state), "next_payroll"
        extra = "; bonus/commission/arrears parts stay excluded" \
            if _has(text, BONUS_WORDS) else ""
        if amount is not None and eff:
            out.append({"action": "SET_SALARY", "message_id": mid, "applied": True,
                        "note": f"regular salary set to {amount} from {eff}{extra}",
                        "params": {"amount": amount, "from_date": eff,
                                   "date_source": eff_source},
                        "sent_at": msg.get("sent_at")})
        elif not out:
            out.append({"action": "SET_SALARY", "message_id": mid, "applied": False,
                        "note": "pay level described but explicit amount/date missing; no change",
                        "params": {}, "sent_at": msg.get("sent_at")})

    # 4. Date moves (only when no new pay level was set above).
    if not any(a["action"] == "SET_SALARY" and a["applied"] for a in out) \
            and _has(text, MOVE_HINTS) \
            and _has(text, SALARY_WORDS + PAYOUT_WORDS + ("payroll", "pay", "payment", "pembayaran")):
        dates = _extract_any_dates(text)
        target = _find_move_target(state)
        if dates and target:
            entry = note("MOVE_MONEY_DATE", True,
                         f"{target} moved to {dates[-1]}",
                         {"event_id": target, "new_date": dates[-1]})
            return out + [entry]
        out.append({"action": "MOVE_MONEY_DATE", "message_id": mid, "applied": False,
                    "note": "date move described but explicit date or target missing; no change",
                    "params": {}, "sent_at": msg.get("sent_at")})

    # 5. First/confirmed pay or approved invoice (adds a dated credit).
    # Skipped when the message is really about an unconfirmed bonus.
    if _has(text, FIRST_PAY_HINTS) or _has_pair(text, FIRST_PAY_PAIRS):
        if _has(text, SALARY_WORDS + PAYOUT_WORDS):
            bonus_only = _has(text, BONUS_WORDS) and not _has(text, BASE_HINTS)
            if not bonus_only:
                amounts = _extract_amounts(text)
                dates = _extract_any_dates(text)
                if amounts and dates:
                    if _duplicate_credit(state, amounts[0], dates[-1]):
                        out.append({"action": "ADD_CONFIRMED_PAY", "message_id": mid,
                                    "applied": False,
                                    "note": f"confirmed {amounts[0]} on {dates[-1]} already covered; no duplicate",
                                    "params": {"amount": amounts[0], "date": dates[-1]},
                                    "sent_at": msg.get("sent_at")})
                    else:
                        out.append({"action": "ADD_CONFIRMED_PAY", "message_id": mid,
                                    "applied": True,
                                    "note": f"confirmed credit {amounts[0]} on {dates[-1]}",
                                    "params": {"amount": amounts[0], "date": dates[-1]},
                                    "sent_at": msg.get("sent_at")})
                elif not out:
                    out.append({"action": "ADD_CONFIRMED_PAY", "message_id": mid,
                                "applied": False,
                                "note": "confirmed pay described but explicit amount/date missing; no change",
                                "params": {}, "sent_at": msg.get("sent_at")})

    # 6. Brand-new recurring bill.
    if _has(text, ("new recurring", "begins", "starts", "mulai", "baru")) \
            and _has(text, BILL_WORDS) \
            and not _has(text, SKIP_HINTS):
        amounts = _extract_amounts(text)
        dates = _extract_any_dates(text)
        start = dates[-1] if dates else _same_month_anchor(state, text)
        if amounts and start:
            out.append({"action": "ADD_BILL", "message_id": mid, "applied": True,
                        "note": f"new recurring outflow {amounts[0]} monthly from {start}",
                        "params": {"amount": amounts[0], "start_date": start},
                        "sent_at": msg.get("sent_at")})
        elif not out:
            out.append({"action": "ADD_BILL", "message_id": mid, "applied": False,
                        "note": "new bill described but amount unclear; not invented, no change",
                        "params": {}, "sent_at": msg.get("sent_at")})

    # 7. Scaled recurring bill (rent +x%).
    if _has_pair(text, SCALE_HINTS) and _has(text, BILL_WORDS):
        pct = PERCENT_RE.search(text or "")
        cat = _which_bill_category(text, state)
        if pct and cat:
            factor = 1.0 + float(pct.group(1)) / 100.0
            out.append({"action": "SCALE_BILL", "message_id": mid, "applied": True,
                        "note": f"{cat} scaled x{round(factor, 4)} from next occurrence",
                        "params": {"category": cat, "factor": round(factor, 4)},
                        "sent_at": msg.get("sent_at")})
        elif not out:
            out.append({"action": "SCALE_BILL", "message_id": mid, "applied": False,
                        "note": "scale described but % or category unclear; no change",
                        "params": {}, "sent_at": msg.get("sent_at")})

    # 8. Hard STOP: income ended AND no amounts stated (pure stop).
    # Prize/claim contexts never stop salary (they close the prize, not the job).
    if _has_pair(text, STOP_HINTS) and not _extract_amounts(text) \
            and not _has(text, PRIZE_VETO_WORDS) and (
            _has(text, SALARY_WORDS) or _has(text, ("contract", "kontrak", "income",
                                                    "penghasilan", "shift", "renewal"))):
        out.append({"action": "STOP_SALARY", "message_id": mid, "applied": True,
                    "note": "income source ended with no renewal: salary projection stopped",
                    "params": {}, "sent_at": msg.get("sent_at")})

    # 9. Explicit confirmations of non-cash (document the correct skip).
    if _has(text, SKIP_HINTS) and _has(
            text, BONUS_WORDS + REFUND_WORDS + INVEST_WORDS + PRIZE_WORDS + PAYOUT_WORDS):
        if not any(a["applied"] for a in out):
            out.append({"action": "CONFIRM_SKIP", "message_id": mid, "applied": True,
                        "note": "unconfirmed/processing credit stays excluded until settled",
                        "params": {}, "sent_at": msg.get("sent_at")})

    if not out:
        return [note("CONFIRM_SKIP", False, "no actionable content; noted only")]
    return out


# ---------- helpers needing state context ----------

def _state_events(state):
    out = []
    for key in ("reserved_pending_debits", "scheduled_obligations",
                "confirmed_future_credits"):
        out.extend(state.get(key, []) or [])
    return out


def _find_transfer_pair(state, window_days=7):
    by_amt = {}
    for c in _state_events(state):
        amt = round(c.get("converted_amount") or 0, 2)
        by_amt.setdefault(amt, []).append(c)
    for amt, items in by_amt.items():
        if amt <= 0:
            continue
        debits = [i for i in items if i.get("direction") == "debit"]
        credits = [i for i in items if i.get("direction") == "credit"]
        if not debits or not credits:
            continue
        d, c_ = debits[0], credits[0]
        dd, cd = _parse_date(d.get("settlement_date")), _parse_date(c_.get("settlement_date"))
        if dd and cd and abs((dd - cd).days) <= window_days:
            return (d["event_id"], c_["event_id"])
    return None


def _find_move_target(state):
    fut = list(state.get("confirmed_future_credits", []) or [])
    if fut:
        fut.sort(key=lambda c: c.get("settlement_date") or "")
        return fut[0]["event_id"]
    sched = list(state.get("scheduled_salary", []) or [])
    if sched:
        return sched[-1]["event_id"]
    return None


def _next_payroll_date(state):
    """Next payroll date: scheduled/expected credits first, else the own
    series' next monthly occurrence (established cadence interpolated,
    only called for messages that reference the next payroll).
    """
    req = _parse_date(state.get("request_date"))
    fut = [c.get("settlement_date") for c in
           (state.get("confirmed_future_credits", []) or []) if c.get("settlement_date")]
    if req:
        fut = [d for d in fut if (_parse_date(d) or req) >= req]
    if fut:
        return sorted(fut)[0]
    sched = [x.get("settlement_date") for x in
             (state.get("scheduled_salary", []) or []) if x.get("settlement_date")]
    if req:
        sched = [d for d in sched if (_parse_date(d) or req) >= req]
    if sched:
        return sorted(sched)[0]
    hist = [x.get("settlement_date") for x in
            (state.get("salary_history", []) or []) if x.get("settlement_date")]
    if hist and req:
        from forecast import _add_months as _add_m
        nxt = _add_m(max(_parse_date(d) for d in hist if _parse_date(d)), 1)
        if nxt >= req:
            return nxt.isoformat()
    return None


def _same_month_anchor(state, text):
    dates = _extract_any_dates(text)
    if dates:
        return dates[-1][:7] + "-01"
    return None


def _duplicate_credit(state, amount, on_date):
    for c in state.get("confirmed_future_credits", []) or []:
        try:
            same_amt = abs(float(c.get("converted_amount") or 0) - amount) < 0.01
        except (TypeError, ValueError):
            same_amt = False
        if same_amt and c.get("settlement_date") == on_date:
            return True
    # Also duplicate when it coincides with the projected monthly salary
    # (same date as anchor + 1 month, same amount): the projection already
    # counts it, so adding it again would double-count one payday.
    hist = [x for x in (state.get("salary_history", []) or [])
            if x.get("settlement_date")]
    if hist:
        try:
            latest = max(hist, key=lambda x: x["settlement_date"])
            from forecast import _add_months as _add_m
            anchor = _parse_date(latest["settlement_date"])
            if anchor and _add_m(anchor, 1).isoformat() == on_date \
                    and abs(float(latest.get("converted_amount") or 0) - amount) < 0.01:
                return True
        except (TypeError, ValueError):
            pass
    return False


def _which_bill_category(text, state):
    t = (text or "").lower()
    cats = list((state.get("recurring_candidates", {}) or {}).keys())
    for hint, cat in (("rent", "rent"), ("sewa", "rent"), ("lease", "rent"),
                      ("childcare", "childcare"), ("subscription", "subscription")):
        if hint in t and (cat in cats or cat == "childcare"):
            return cat
    for cat in cats:
        if cat.lower() in t:
            return cat
    return None


def build_adjustments(state, fx_index=None):
    """Parse all attached messages -> forecast adjustments + notes.

    Returns {"salary_override", "stop_salary", "moved_dates",
    "extra_flows", "scales", "exclude_event_ids", "amendments", "notes"}.
    Latest same-source salary directive wins (messages sorted by sent_at).
    """
    amendments = []
    for m in sorted(state.get("messages", []) or [],
                    key=lambda x: x.get("sent_at") or ""):
        amendments.extend(parse_message(m, state))

    adj = {"salary_override": None, "stop_salary": False, "moved_dates": {},
           "extra_flows": [], "scales": {}, "exclude_event_ids": set(),
           "salary_seeds": [], "amendments": amendments, "notes": []}
    home = (state.get("profile", {}) or {}).get("home_currency", "")
    msg_text = {m.get("message_id"): (m.get("text") or "")
                for m in state.get("messages", []) or []}

    salary_sets = [a for a in amendments
                   if a["action"] == "SET_SALARY" and a["applied"]]
    stops = [a for a in amendments
             if a["action"] == "STOP_SALARY" and a["applied"]]
    if salary_sets and stops:
        # Newest directive wins across salary life-cycle events.
        last_set = max(salary_sets, key=lambda a: a.get("sent_at") or "")
        last_stop = max(stops, key=lambda a: a.get("sent_at") or "")
        keep_set = (last_set.get("sent_at") or "") >= (last_stop.get("sent_at") or "")
        if not keep_set:
            salary_sets = []
        else:
            stops = []
    if salary_sets:
        latest = max(salary_sets, key=lambda a: a.get("sent_at") or "")
        adj["salary_override"] = latest["params"]
    if stops:
        adj["stop_salary"] = True
        adj["salary_override"] = None

    for a in amendments:
        if not a["applied"]:
            adj["notes"].append(f"{a['message_id']}: {a['note']}")
            continue
        p = a["params"]
        if a["action"] == "MOVE_MONEY_DATE":
            adj["moved_dates"][p["event_id"]] = p["new_date"]
            adj["notes"].append(f"{a['message_id']}: {a['note']}")
        elif a["action"] == "ADD_CONFIRMED_PAY":
            adj["notes"].append(f"{a['message_id']}: {a['note']}")
            if _has(msg_text.get(a["message_id"]), RESUME_HINTS):
                # "Resumes": the pay continues monthly (seeded, not one-off).
                adj["salary_seeds"].append(
                    {"amount": p["amount"], "date": p["date"]})
                adj["notes"].append(
                    f"{a['message_id']}: resumed pay continues monthly from {p['date']}")
            else:
                adj["extra_flows"].append(
                    {"date": p["date"], "amount": p["amount"],
                     "label": f"msg:{a['message_id']}"})
        elif a["action"] == "ADD_BILL":
            adj["extra_flows"].append(
                {"date": p["start_date"], "amount": -p["amount"],
                 "label": f"msg:{a['message_id']}", "monthly": True})
            adj["notes"].append(f"{a['message_id']}: {a['note']}")
        elif a["action"] == "SCALE_BILL":
            adj["scales"][p["category"]] = p["factor"]
            adj["notes"].append(f"{a['message_id']}: {a['note']}")
        elif a["action"] == "DEDUPE_TRANSFER":
            adj["exclude_event_ids"].add(p["debit_event_id"])
            adj["notes"].append(f"{a['message_id']}: {a['note']}")
        elif a["action"] in ("SET_SALARY", "STOP_SALARY"):
            adj["notes"].append(f"{a['message_id']}: {a['note']}")
        elif a["action"] == "CONFIRM_SKIP":
            adj["notes"].append(f"{a['message_id']}: {a['note']}")
    return adj
