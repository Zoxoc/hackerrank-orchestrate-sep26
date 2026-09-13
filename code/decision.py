"""Payment decision logic for Buy or Wait?

Five questions in order (see approved plan):
1. amount_safe_to_pay  (binary search in cents on request_date)
2. earliest_date_for_full_payment (forward scan, baseline = no changes)
3. plan ranking over eligible candidates (full/partial/installments/wait)
4. spending-change fallback search (stop/reduce adjustable recurring)
5. explanation + deterministic validator (broken rows fall back safely)

Conventions: all money rounded to 2dp; plan amounts print whole when
integral else 2dp ("68432", "620.40"); dates chronological; every plan
completes by desired_completion_date and holds the minimum every day.
"""

from datetime import date, timedelta

FORECAST_DAYS = 90


def _parse(d):
    return date.fromisoformat(d[:10]) if d else None


def fmt_amt(x):
    r = round(float(x) + 0.0, 2)
    if r == int(r):
        return str(int(r))
    return f"{r:.2f}"


def fmt_money(x, ccy):
    r = round(float(x), 2)
    s = f"{r:,.2f}"
    if s.endswith(".00"):
        s = s[:-3]
    return f"{ccy} {s}"


def baseline_nets(forecast):
    nets = {}
    for f in forecast.get("flows_used", []):
        nets[f["date"]] = round(nets.get(f["date"], 0.0) + f["amount"], 2)
    return nets


def simulate(forecast, payments, nets=None):
    """True iff opening + baseline nets + payments keeps minimum every day."""
    start = _parse(forecast["start"])
    end = _parse(forecast["end"])
    minimum = forecast["minimum_to_keep"]
    net = dict(nets) if nets else baseline_nets(forecast)
    for dstr, amt in payments:
        net[dstr] = round(net.get(dstr, 0.0) - float(amt), 2)
    bal = forecast["opening_balance"]
    cur = start
    while cur <= end:
        bal = round(bal + net.get(cur.isoformat(), 0.0), 2)
        if bal < minimum:
            return False
        cur += timedelta(days=1)
    return True


def earliest_full_date(state, forecast, requested):
    start = _parse(state["request_date"])
    end = _parse(forecast["end"])
    cur = start
    while cur <= end:
        if simulate(forecast, [(cur.isoformat(), requested)]):
            return cur.isoformat()
        cur += timedelta(days=1)
    return None


def safe_today_cents(state, forecast, requested):
    hi = int(round(float(requested) * 100))
    lo, best = 0, 0
    while lo <= hi:
        mid = (lo + hi) // 2
        if simulate(forecast, [(state["request_date"], mid / 100.0)]):
            best, lo = mid, mid + 1
        else:
            hi = mid - 1
    return best / 100.0


# ---------- installment options ----------

def _opt_int(v, default=0):
    try:
        return int(float(v))
    except (TypeError, ValueError):
        return default


def installment_schedules(state):
    """Yield dicts for eligible installment options (schedule + totals)."""
    req_date = _parse(state["request_date"])
    deadline = _parse(state["desired_completion_date"])
    considers = state["profile"]["payment_methods_user_will_consider"]
    if "installments" not in considers:
        return
    try:
        max_months = float(state["profile"]["max_installment_months"]) \
            if state["profile"]["max_installment_months"] else None
    except (TypeError, ValueError):
        max_months = None
    for o in sorted(state.get("payment_options", []),
                    key=lambda x: x.get("payment_option_id") or ""):
        if (o.get("payment_method") or "").strip() != "installments":
            continue
        n = _opt_int(o.get("number_of_payments"))
        freq = _opt_int(o.get("payment_frequency_days"))
        first = _parse(o.get("first_payment_date"))
        try:
            per = float(o.get("payment_amount"))
            total = float(o.get("total_payable_amount"))
        except (TypeError, ValueError):
            continue
        if not (n >= 2 and freq > 0 and first and per > 0 and total > 0):
            continue
        if first < req_date:
            continue
        sched = [(first + timedelta(days=i * freq)).isoformat() for i in range(n)]
        if _parse(sched[-1]) > deadline:
            continue
        if max_months is not None:
            span_months = ((n - 1) * freq) / 30.44
            if span_months > max_months + 1e-9:
                continue
        yield {"option_id": o.get("payment_option_id"), "schedule": sched,
               "per": per, "total": total, "n": n}


# ---------- candidate search ----------

def _rank_key(total, start, n_payments, option_id=""):
    return (round(float(total), 2), start or "", n_payments, option_id or "")


def find_best_plan(state, forecast, requested, safe, earliest, nets=None,
                   allow_changes=False):
    """Return best eligible plan dict or None. Pure ranking, no fallback."""
    req_date = state["request_date"]
    deadline = _parse(state["desired_completion_date"])
    considers = state["profile"]["payment_methods_user_will_consider"]
    cands = []

    if "full_payment" in considers and safe >= requested and \
            _parse(req_date) <= deadline:
        if simulate(forecast, [(req_date, requested)], nets):
            cands.append({"method": "full_payment", "status": "affordable_now",
                          "payments": [(req_date, requested)], "total": requested,
                          "key": _rank_key(requested, req_date, 1)})

    if "partial_payment" in considers and state["allows_partial_payment"] \
            and 0 < safe < requested and earliest \
            and _parse(earliest) <= deadline:
        rest = round(requested - safe, 2)
        pays = [(req_date, safe), (earliest, rest)]
        if simulate(forecast, pays, nets):
            cands.append({"method": "partial_payment",
                          "status": "affordable_with_plan", "payments": pays,
                          "total": requested,
                          "key": _rank_key(requested, req_date, 2)})

    for inst in installment_schedules(state):
        pays = [(d, inst["per"]) for d in inst["schedule"]]
        if simulate(forecast, pays, nets):
            cands.append({"method": "installments",
                          "status": "affordable_with_plan", "payments": pays,
                          "total": inst["total"], "option_id": inst["option_id"],
                          "key": _rank_key(inst["total"], inst["schedule"][0],
                                           inst["n"], inst["option_id"])})

    if "full_payment" in considers and earliest and earliest != req_date \
            and _parse(earliest) <= deadline:
        cands.append({"method": "wait", "status": "affordable_later",
                      "payments": [(earliest, requested)], "total": requested,
                      "key": _rank_key(requested, earliest, 1)})

    if not cands:
        return None
    cands.sort(key=lambda c: c["key"])
    best = cands[0]
    if allow_changes:
        best["status"] = "affordable_with_plan"
    return best


# ---------- spending changes ----------

def _change_actions(state):
    """Atomic actions: ("stop", cat) / ("reduce", cat). Deterministic order."""
    adj = state.get("adjustable_recurring", {}) or {}
    actions = []
    for cat in sorted(adj):
        info = adj[cat]
        if info.get("can_stop"):
            actions.append(("stop", cat))
        if info.get("can_reduce") and info.get("minimum") is not None:
            actions.append(("reduce", cat))
    return actions


def _apply_changes_to_nets(forecast, state, change_set):
    """Copy baseline nets with recurring category flows stopped/reduced."""
    nets = baseline_nets(forecast)
    req = _parse(state["request_date"])
    adj = state.get("adjustable_recurring", {}) or {}
    stop_cats = {c for a, c in change_set if a == "stop"}
    red = {c: adj[c]["minimum"] for a, c in change_set if a == "reduce"}
    if not (stop_cats or red):
        return nets
    per_day_cat = {}
    for f in forecast.get("flows_used", []):
        if f["label"].startswith("recurring:"):
            cat = f["label"].split(":", 1)[1]
            if cat in stop_cats or cat in red:
                per_day_cat.setdefault((f["date"], cat), 0.0)
                per_day_cat[(f["date"], cat)] += f["amount"]
    for (dstr, cat), old_sum in per_day_cat.items():
        if _parse(dstr) < req:
            continue
        if cat in stop_cats:
            nets[dstr] = round(nets.get(dstr, 0.0) - old_sum, 2)
        else:
            n_items = sum(1 for f in forecast["flows_used"]
                          if f["date"] == dstr
                          and f["label"] == f"recurring:{cat}")
            new_sum = round(red[cat] * max(n_items, 1), 2)
            nets[dstr] = round(nets.get(dstr, 0.0) - old_sum + new_sum, 2)
    return nets


def _changes_to_strings(change_set, state):
    adj = state.get("adjustable_recurring", {}) or {}
    out = []
    for action, cat in change_set:
        eid = adj[cat]["event_id"]
        if action == "stop":
            out.append(f"stop:{eid}")
        else:
            out.append(f"reduce_to:{eid}:{fmt_amt(adj[cat]['minimum'])}")
    return out


def search_with_changes(state, forecast, requested, safe, earliest):
    """Bounded fallback: singles, pairs, triples of top-saving actions."""
    actions = _change_actions(state)
    if not actions:
        return None, None
    # Rank actions by projected 90-day saving, deterministic tie-break.
    scored = []
    base_nets = baseline_nets(forecast)
    for a in actions:
        trial = _apply_changes_to_nets(forecast, state, [a])
        save = round(sum(base_nets.values()) - sum(trial.values()), 2)
        scored.append((-save, a))
    scored.sort(key=lambda t: (t[0], t[1]))
    ranked = [a for _, a in scored]
    pool, seen = [], set()
    for a in ranked[:8]:
        pool.append(a)
    combos = [[a] for a in pool]
    top6 = pool[:6]
    combos += [list(c) for i in range(len(top6))
               for c in [top6[i], ] for j in range(i + 1, len(top6))
               for c in [(top6[i], top6[j])]]
    top5 = pool[:5]
    for i in range(len(top5)):
        for j in range(i + 1, len(top5)):
            for k in range(j + 1, len(top5)):
                combos.append([top5[i], top5[j], top5[k]])
    # Dedupe combos sharing an event (stop+reduce same event forbidden).
    seen_keys, ordered = set(), []
    for combo in combos:
        eids = []
        for action, cat in combo:
            eids.append(state["adjustable_recurring"][cat]["event_id"])
        if len(set(eids)) != len(eids):
            continue
        key = tuple(sorted((a, c) for a, c in combo))
        if key in seen_keys:
            continue
        seen_keys.add(key)
        ordered.append(combo)
    results = []
    for combo in ordered:
        nets = _apply_changes_to_nets(forecast, state, combo)
        best = find_best_plan(state, forecast, requested, safe, earliest,
                              nets=nets, allow_changes=True)
        if best:
            results.append((len(combo), best["key"], combo, best))
    if not results:
        return None, None
    results.sort(key=lambda t: (t[0], t[1]))
    _, _, combo, best = results[0]
    return best, _changes_to_strings(combo, state)


# ---------- explanations ----------

def _cite(state, forecast):
    """Short evidence cite: first message/OCR/salary assumption, if any."""
    for a in forecast.get("assumptions", []):
        low = a.lower()
        if low.startswith("message ") or "per employer" in low \
                or low.startswith("salary ended") or "projected salary: monthly" in low:
            return a
    return None


def explain(state, forecast, method, status, payments, requested, changes):
    p = state["profile"]
    ccy, minimum = p["home_currency"], p["minimum_balance_to_keep"]
    deadline = state["desired_completion_date"]
    if method == "full_payment" and status == "affordable_now":
        return (f"Pay {fmt_money(requested, ccy)} today. This leaves at least "
                f"{fmt_money(minimum, ccy)} available over the next 90 days.")
    if method == "full_payment":  # via spending changes
        desc = _changes_desc(changes, state)
        return (f"{desc}, then pay {fmt_money(requested, ccy)} today. This leaves "
                f"at least {fmt_money(minimum, ccy)} available.")
    if method == "partial_payment":
        first, (d2, rest) = payments[0], payments[1]
        return (f"Pay {fmt_money(first[1], ccy)} on {first[0]}, then "
                f"{fmt_money(rest, ccy)} on {d2}. This leaves at least "
                f"{fmt_money(minimum, ccy)} available.")
    if method == "installments":
        n = len(payments)
        return (f"Use {n} installments of {fmt_money(payments[0][1], ccy)}, "
                f"starting {payments[0][0]}. This leaves at least "
                f"{fmt_money(minimum, ccy)} available.")
    if method == "wait":
        return (f"Wait until {payments[0][0]}, then pay "
                f"{fmt_money(requested, ccy)} in full. Paying sooner would put "
                f"the {fmt_money(minimum, ccy)} minimum at risk.")
    return (f"Do not make this payment by {deadline}. None of the available "
            f"options keeps the {fmt_money(minimum, ccy)} minimum protected.")


def _changes_desc(changes, state):
    adj = state.get("adjustable_recurring", {}) or {}
    parts = []
    for ch in changes or []:
        if ch.startswith("stop:"):
            eid = ch.split(":")[1]
            desc = next((v["description"] for v in adj.values()
                         if v["event_id"] == eid), "flexible spending")
            parts.append(f"Stop the {desc}")
        else:
            _, eid, amt = ch.split(":")
            desc = next((v["description"] for v in adj.values()
                         if v["event_id"] == eid), "flexible spending")
            parts.append(f"reduce the {desc} to {amt}")
    if not parts:
        return "Adjust spending"
    if len(parts) == 1:
        first = parts[0]
        return first[0].upper() + first[1:] if first.startswith("reduce") else first
    head = ", ".join(parts[:-1])
    tail = parts[-1]
    return head + " and " + tail[0].lower() + tail[1:]


# ---------- main entry + validator ----------

def decide_request(state, forecast):
    requested = float(state["requested_amount"])
    req_date = state["request_date"]
    safe = safe_today_cents(state, forecast, requested)
    earliest = earliest_full_date(state, forecast, requested)
    best = find_best_plan(state, forecast, requested, safe, earliest)
    changes = []
    if best is None:
        best, changes = search_with_changes(state, forecast, requested, safe,
                                            earliest)
    if best is None:
        row = {"request_id": state["request_id"],
               "amount_safe_to_pay": safe,
               "affordability_status": "not_affordable",
               "recommended_payment_method": "not_recommended",
               "payment_plan": "none",
               "earliest_date_for_full_payment": earliest or "",
               "spending_changes_needed": "none",
               "decision_explanation": explain(state, forecast, "not_recommended",
                                               "not_affordable", [], requested, [])}
    else:
        changes = changes or []
        plan = "|".join(f"{d}:{fmt_amt(a)}" for d, a in best["payments"])
        row = {"request_id": state["request_id"],
               "amount_safe_to_pay": safe,
               "affordability_status": best["status"],
               "recommended_payment_method": best["method"],
               "payment_plan": plan,
               "earliest_date_for_full_payment": earliest or "",
               "spending_changes_needed": "|".join(changes) if changes else "none",
               "decision_explanation": explain(state, forecast, best["method"],
                                               best["status"], best["payments"],
                                               requested, changes)}
    problems = validate_row(row, state)
    if problems:
        row = {"request_id": state["request_id"],
               "amount_safe_to_pay": min(safe, requested),
               "affordability_status": "not_affordable",
               "recommended_payment_method": "not_recommended",
               "payment_plan": "none",
               "earliest_date_for_full_payment": "",
               "spending_changes_needed": "none",
               "decision_explanation": explain(state, forecast, "not_recommended",
                                               "not_affordable", [], requested, [])}
    return row


def validate_row(row, state):
    """Return list of problems (empty = valid). Never raises."""
    problems = []
    try:
        requested = float(state["requested_amount"])
        safe = float(row["amount_safe_to_pay"])
        if not (0 <= safe <= requested + 1e-9):
            problems.append("safe_out_of_bounds")
    except (TypeError, ValueError):
        problems.append("safe_not_numeric")
        return problems
    status, method = row["affordability_status"], row["recommended_payment_method"]
    if status not in ("affordable_now", "affordable_with_plan",
                      "affordable_later", "not_affordable"):
        problems.append("bad_status")
    if method not in ("full_payment", "partial_payment", "installments",
                      "wait", "not_recommended"):
        problems.append("bad_method")
    early, req_date = row["earliest_date_for_full_payment"], state["request_date"]
    if status == "affordable_now" and early != req_date:
        problems.append("now_needs_request_date")
    if method == "partial_payment":
        if status != "affordable_with_plan":
            problems.append("partial_needs_plan_status")
        parts = (row["payment_plan"] or "").split("|")
        if len(parts) != 2:
            problems.append("partial_needs_two_payments")
        else:
            try:
                d1, a1 = parts[0].split(":"); d2, a2 = parts[1].split(":")
                if d1 != req_date or d2 != early:
                    problems.append("partial_dates_wrong")
                if abs((float(a1) + float(a2)) - requested) > 0.011:
                    problems.append("partial_sum_wrong")
                if not (0 < float(a1) < requested):
                    problems.append("partial_first_wrong")
            except ValueError:
                problems.append("partial_unparseable")
        if not state["allows_partial_payment"]:
            problems.append("partial_not_allowed")
    if method == "installments":
        if not _matches_option(row["payment_plan"], state):
            problems.append("installments_not_matching_option")
    if method == "wait" and not early:
        problems.append("wait_needs_date")
    if method == "not_recommended" and row["payment_plan"] != "none":
        problems.append("fallback_plan_must_be_none")
    chg = row["spending_changes_needed"]
    if chg != "none":
        items = chg.split("|")
        if len(items) > 3:
            problems.append("too_many_changes")
        seen_events = set()
        adj = state.get("adjustable_recurring", {}) or {}
        valid_ids = {v["event_id"] for v in adj.values()}
        for item in items:
            if item.startswith("stop:"):
                eid = item[5:]
                if eid in seen_events:
                    problems.append("change_event_reused")
                seen_events.add(eid)
                if eid not in valid_ids:
                    problems.append(f"change_not_adjustable:{eid}")
            elif item.startswith("reduce_to:"):
                try:
                    _, eid, amt = item.split(":")
                    float(amt)
                except ValueError:
                    problems.append(f"change_unparseable:{item}")
                    continue
                if eid in seen_events:
                    problems.append("change_event_reused")
                seen_events.add(eid)
                if eid not in valid_ids:
                    problems.append(f"change_not_adjustable:{eid}")
            else:
                problems.append(f"change_bad_shape:{item}")
    return problems


def _matches_option(plan_str, state):
    try:
        pairs = [p.split(":") for p in (plan_str or "").split("|")]
        dates = [d for d, _ in pairs]
        amts = [float(a) for _, a in pairs]
    except ValueError:
        return False
    for o in state.get("payment_options", []):
        if (o.get("payment_method") or "").strip() != "installments":
            continue
        n = _opt_int(o.get("number_of_payments"))
        freq = _opt_int(o.get("payment_frequency_days"))
        first = _parse(o.get("first_payment_date"))
        try:
            per = float(o.get("payment_amount"))
        except (TypeError, ValueError):
            continue
        if not (first and n == len(dates)):
            continue
        expect = [(first + timedelta(days=i * freq)).isoformat() for i in range(n)]
        if dates == expect and all(abs(a - per) < 0.011 for a in amts):
            return True
    return False
