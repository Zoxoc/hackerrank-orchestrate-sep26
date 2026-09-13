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
    """Project recurring DEBIT categories forward. Returns (flows, assumptions).

    Monthly cadences (median interval 27-32 days) step by calendar month on
    the usual day-of-month (bills are due on a date, not every 30 days);
    other cadences step by median interval days.
    """
    import calendar as _cal
    from collections import Counter as _Counter
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
        monthly = 27 <= step <= 32 and len(dates) >= 2
        if monthly:
            dom = _Counter(d.day for d in dates).most_common(1)[0][0]
            last = max(dates)
            n, d, projected = 1, None, 0
            while True:
                cand = _add_months(last, n)
                cand = date(cand.year, cand.month,
                            min(dom, _cal.monthrange(cand.year, cand.month)[1]))
                if cand > end:
                    break
                if cand >= start:
                    flows.append((cand, -round(avg, 2), f"recurring:{cat}", None))
                    projected += 1
                n += 1
            cadence_note = f"monthly on day {dom}"
        else:
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
            cadence_note = f"every ~{step}d"
        assumptions.append(
            f"projected {cat}: {cadence_note} at {round(avg, 2)} x{projected} "
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


def project_salary(state, start, end, salary_override=None, stop=False,
                   salary_seeds=None):
    """Project confirmed recurring pay. Returns (flows, assumptions).

    Monthly payroll (median interval 24-33 days) is projected on payday;
    professional freelance-style series (6-23 days) at their own cadence.
    Gig/weekly-platform/variable pay is never projected. Payday rule: when
    the latest known pay arrived a full cycle after the previous one the
    cycle shifted (continue from it); a mid-cycle extra never moves payday.
    A latest pay marked as final/terminal stops all projection.
    salary_override {"amount", "from_date"}: projections on/after from_date
    use the new amount. salary_seeds [{"amount", "date"}]: confirmed
    (re)starts counted once on their date, then continued monthly.
    stop=True: no projection at all.
    """
    import calendar as _cal2
    from collections import Counter as _Counter2
    if stop:
        return [], ["salary stopped by message amendment: no salary projected"]
    sched = state.get("scheduled_salary", []) or []
    hist = state.get("salary_history", []) or []
    latest = sched[-1] if sched else (hist[-1] if hist else None)
    seeds = salary_seeds or []
    if latest is None and not seeds:
        return [], ["no regular salary found: no salary projected"]
    if latest is not None and any(
            k in (latest.get("description") or "").lower() for k in SALARY_END_KEYWORDS):
        return [], [f"salary ended ({latest.get('description')} on "
                    f"{latest.get('settlement_date')}): no salary projected"]
    hdates = sorted(_parse(x["settlement_date"]) for x in hist
                    if x.get("settlement_date"))
    hdates = [d for d in hdates if d]
    confirmed = bool(sched or seeds or salary_override)
    band = None
    if len(hdates) >= 2:
        ivs = sorted((hdates[i + 1] - hdates[i]).days for i in range(len(hdates) - 1))
        med = ivs[len(ivs) // 2]
        if 24 <= med <= 33:
            band = "monthly"
        elif 6 <= med <= 23 and len(hdates) >= 3:
            band = "interval"
    if band is None and not confirmed:
        return [], ["pay series not monthly-confirmed: not projected"]
    amount = latest["converted_amount"] if latest is not None else None
    notes = []
    if salary_override:
        from_date = _parse(salary_override.get("from_date"))
        new_amount = float(salary_override.get("amount"))
        notes.append(f"message salary override: {new_amount} from "
                     f"{salary_override.get('from_date')}")
        if amount is None:
            amount = new_amount
    else:
        from_date, new_amount = None, None
    if amount is None:
        amount = seeds[0]["amount"] if seeds else 0
    known = {_parse(x["settlement_date"]) for x in sched + hist if x.get("settlement_date")}
    known = {d for d in known if d}
    seed_dates = {_parse(s["date"]) for s in seeds if s.get("date")}
    seed_dates = {d for d in seed_dates if d}
    if band == "monthly" or not band:
        # Payday: a latest pay that arrived a full cycle late shifted the
        # cycle (continue from it); a mid-cycle extra never moves payday.
        if len(hdates) >= 2 and (hdates[-1] - hdates[-2]).days >= 27:
            payday = hdates[-1].day
            base = hdates[-1]
        elif known:
            payday = _Counter2(d.day for d in known).most_common(1)[0][0]
            base = max(known)
        elif seed_dates:
            payday = _Counter2(d.day for d in seed_dates).most_common(1)[0][0]
            base = min(seed_dates)
        else:
            payday = start.day
            base = start
        step_mode = ("monthly", payday, base)
    else:
        ivs = sorted((hdates[i + 1] - hdates[i]).days for i in range(len(hdates) - 1))
        step_mode = ("interval", ivs[len(ivs) // 2], max(hdates))
    # first projection strictly after the latest known salary
    n, flows, count = 1, [], 0
    if step_mode[0] == "monthly":
        _, payday, base = step_mode
        while True:
            cand = _add_months(base, n)
            cand = date(cand.year, cand.month,
                        min(payday, _cal2.monthrange(cand.year, cand.month)[1]))
            if cand > end:
                break
            if cand >= start and cand not in known and cand not in seed_dates:
                amt = new_amount if (from_date and cand >= from_date) else amount
                flows.append((cand, +round(amt, 2), "recurring:salary", None))
                count += 1
            n += 1
        cadence_note = f"monthly(day {payday})"
    else:
        _, step, base = step_mode
        cand = base + timedelta(days=step)
        while cand < start:
            cand += timedelta(days=step)
        while cand <= end:
            if cand not in known and cand not in seed_dates:
                amt = new_amount if (from_date and cand >= from_date) else amount
                flows.append((cand, +round(amt, 2), "recurring:salary", None))
                count += 1
            cand += timedelta(days=step)
        cadence_note = f"every ~{step}d"
    for s in seeds:
        d = _parse(s.get("date"))
        if d and start <= d <= end:
            flows.append((d, +round(float(s["amount"]), 2), "recurring:salary", None))
            count += 1
    flows.sort(key=lambda t: t[0])
    return flows, [f"projected salary: {cadence_note} at {round(amount, 2)} "
                   f"x{count}"] + notes


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


def build_baseline_forecast(state, horizon_days=FORECAST_DAYS, adjustments=None):
    """Simulate baseline balances. Returns forecast dict.

    adjustments: optional dict from message_actions.build_adjustments
    (moved_dates, exclude_event_ids, extra_flows, scales, salary_override,
    stop_salary, notes). None = pure events baseline.
    """
    adj = adjustments or {}
    moved = adj.get("moved_dates", {}) or {}
    excluded_ids = adj.get("exclude_event_ids", set()) or set()
    scales = adj.get("scales", {}) or {}
    start = _parse(state["request_date"])
    end = start + timedelta(days=horizon_days)
    opening = float(state["profile"]["current_available_balance"])
    minimum = float(state["profile"]["minimum_balance_to_keep"])

    dated = []  # (date, signed, label, ref)

    def _eff_date(c):
        d = _parse(c["settlement_date"])
        new = moved.get(c["event_id"])
        return _parse(new) if new else d

    for key, sign, kind in (("reserved_pending_debits", -1, "reserved"),
                            ("scheduled_obligations", -1, "scheduled"),
                            ("confirmed_future_credits", +1, "salary")):
        for c in state.get(key, []) or []:
            if c["event_id"] in excluded_ids:
                continue
            d = _eff_date(c)
            if d and start <= d <= end:
                amt = c["converted_amount"]
                if kind == "salary" and adj.get("salary_override"):
                    ov = adj["salary_override"]
                    if d >= _parse(ov.get("from_date")):
                        amt = float(ov.get("amount"))
                dated.append((d, sign * amt,
                              f"{kind}:{c['category']}", c["event_id"]))

    rec_flows, rec_notes = project_recurring(state, start, end)
    for cat, factor in scales.items():
        rec_flows = [(d, round(a * factor, 2), lbl, ref)
                     if lbl == f"recurring:{cat}" else (d, a, lbl, ref)
                     for d, a, lbl, ref in rec_flows]
        rec_notes.append(f"scaled {cat} x{factor} by message amendment")
    dated.extend(rec_flows)
    sal_flows, sal_notes = project_salary(
        state, start, end,
        salary_override=adj.get("salary_override"),
        stop=bool(adj.get("stop_salary")),
        salary_seeds=adj.get("salary_seeds"))
    dated.extend(sal_flows)
    for ef in adj.get("extra_flows", []) or []:
        d0 = _parse(ef.get("date"))
        if not d0:
            continue
        if ef.get("monthly"):
            d = d0
            while d <= end:
                if d >= start:
                    dated.append((d, ef["amount"], ef.get("label", "msg"), None))
                d = _add_months(d, 1)
        elif start <= d0 <= end:
            dated.append((d0, ef["amount"], ef.get("label", "msg"), None))
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
        "assumptions": rec_notes + sal_notes + msg_notes + list(adj.get("notes", []) or []) + [
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
