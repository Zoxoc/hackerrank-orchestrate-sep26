"""Automated evaluation of the decision pipeline on sample_requests.csv.

Isolated test harness: reads the 25 public worked examples, runs the
production pipeline (no modifications to it), compares field by field,
and writes a report. Rerun after any engine change:

    python3 code/evaluation/test_samples.py [--report PATH]

Match rules (documented, strict where scoring is strict):
- affordability_status, recommended_payment_method,
  earliest_date_for_full_payment, spending_changes_needed: EXACT match
  (empty earliest == empty earliest).
- amount_safe_to_pay: numeric tolerance |got-want| <= max(1.0, 5% of want).
- payment_plan: EQUIVALENT match -- same dated entries in order, amounts
  within 0.011 ("none" must equal "none"). String equality is sufficient
  but not required.
- decision_explanation: VALIDITY check only (never exact-matched) --
  non-empty, mentions the home currency, and contains at least one figure.
  Reported separately; does not affect row PASS/FAIL.

Row verdict: PASS iff all six decisive fields pass. Exit code 0 iff all
25 rows pass, else 1 (no engine code is touched either way).
"""

import os
import re
import sys
from collections import defaultdict

EVAL_DIR = os.path.dirname(os.path.abspath(__file__))
CODE_DIR = os.path.dirname(EVAL_DIR)
REPO_ROOT = os.path.dirname(CODE_DIR)
sys.path.insert(0, CODE_DIR)
os.chdir(REPO_ROOT)

from financial_state import (load_csv, build_fx_index, build_request_state)
from forecast import build_baseline_forecast
from message_actions import build_adjustments
from decision import decide_request

try:
    from ocr_amounts import resolve_event_amount
    HAVE_OCR = True
except ImportError:
    HAVE_OCR = False

SAFE_REL_TOL = 0.05
SAFE_ABS_FLOOR = 1.0
PLAN_AMT_TOL = 0.011
DEFAULT_REPORT = os.path.join(EVAL_DIR, "sample_test_report.txt")


def check_exact(got, want):
    return (got or "") == (want or ""), "exact"


def check_safe(got, want):
    try:
        g, w = float(got), float(want)
    except (TypeError, ValueError):
        return False, "unparseable"
    tol = max(SAFE_ABS_FLOOR, abs(w) * SAFE_REL_TOL)
    ok = abs(g - w) <= tol + 1e-9
    return ok, f"|{g}-{w}|<={tol:.2f}" if ok else f"|{g}-{w}|>{tol:.2f}"


def _parse_plan(s):
    if (s or "") == "none":
        return []
    entries = []
    for part in s.split("|"):
        d, a = part.split(":")
        entries.append((d.strip(), float(a)))
    return entries


def check_plan(got, want):
    if (got or "") == (want or ""):
        return True, "exact-string"
    try:
        g, w = _parse_plan(got), _parse_plan(want)
    except (ValueError, AttributeError):
        return False, "unparseable"
    if len(g) != len(w):
        return False, f"{len(g)} entries != {len(w)}"
    for (gd, ga), (wd, wa) in zip(g, w):
        if gd != wd:
            return False, f"date {gd} != {wd}"
        if abs(ga - wa) > PLAN_AMT_TOL:
            return False, f"amount {ga} != {wa}"
    return True, "equivalent-schedule"


def check_explanation(row, state):
    text = (row.get("decision_explanation") or "").strip()
    if not text:
        return False, "empty"
    ccy = (state.get("profile", {}) or {}).get("home_currency", "")
    if ccy and ccy not in text:
        return False, "missing-currency"
    if not re.search(r"\d", text):
        return False, "no-figures"
    return True, "grounded"


FIELD_CHECKS = [
    ("affordability_status", check_exact, True),
    ("recommended_payment_method", check_exact, True),
    ("amount_safe_to_pay", check_safe, True),
    ("payment_plan", check_plan, True),
    ("earliest_date_for_full_payment", check_exact, True),
    ("spending_changes_needed", check_exact, True),
]


def evaluate():
    samples = load_csv("sample_requests.csv")
    expected = {r["request_id"]: r for r in samples}
    profs = {r["user_id"]: r for r in load_csv("financial_profiles.csv")}
    ev_by = defaultdict(list)
    for e in load_csv("financial_events.csv"):
        ev_by[e["user_id"]].append(e)
    opts = defaultdict(list)
    for o in load_csv("request_payment_options.csv"):
        opts[o["request_id"]].append(o)
    msgs, imgs = load_csv("messages.csv"), load_csv("images.csv")
    fx = build_fx_index(load_csv("exchange_rates.csv"))
    resolver = resolve_event_amount if HAVE_OCR else None

    lines = []
    field_ok = defaultdict(int)
    expl_ok = 0
    passed_rows = 0
    for req in samples:
        rid = req["request_id"]
        state = build_request_state(req, profs, ev_by, opts, msgs, imgs, fx,
                                    ocr_resolver=resolver)
        forecast = build_baseline_forecast(
            state, adjustments=build_adjustments(state))
        row = decide_request(state, forecast)
        want = expected[rid]
        fails = []
        for field, check, _ in FIELD_CHECKS:
            if field == "amount_safe_to_pay":
                ok, why = check(row[field], want[field])
            elif field == "payment_plan":
                ok, why = check(row[field], want[field])
            else:
                ok, why = check(row[field], want[field])
            if ok:
                field_ok[field] += 1
            else:
                fails.append(f"{field} [{why}]")
        ok_e, why_e = check_explanation(row, state)
        expl_ok += ok_e
        verdict = "PASS" if not fails else "FAIL"
        passed_rows += (verdict == "PASS")
        reason = "all fields match" if not fails else "; ".join(fails)
        lines.append(f"{rid} | {verdict} | reason: {reason}")
        for field, _, _ in FIELD_CHECKS:
            lines.append(f"    {field}: expected={want[field]!r} actual={row[field]!r}")
        lines.append(f"    decision_explanation: valid={ok_e} ({why_e})")
        lines.append(f"    actual explanation: {row['decision_explanation'][:160]}")
    n = len(samples)
    summary = [f"Overall: {passed_rows}/{n} ({passed_rows / n:.0%})"]
    for field, _, _ in FIELD_CHECKS:
        summary.append(f"  {field}: {field_ok[field]}/{n} "
                       f"({field_ok[field] / n:.0%})")
    summary.append(f"  decision_explanation valid: {expl_ok}/{n} "
                   f"({expl_ok / n:.0%}) (validity only, not exact match)")
    return lines, summary, passed_rows, n


def main():
    report = (sys.argv[sys.argv.index("--report") + 1]
              if "--report" in sys.argv and
              len(sys.argv) > sys.argv.index("--report") + 1
              else DEFAULT_REPORT)
    detail, summary, passed_rows, n = evaluate()
    text = "\n".join(detail + ["", "SUMMARY"] + summary) + "\n"
    print(text)
    with open(report, "w", encoding="utf-8") as f:
        f.write(text)
    print(f"Report saved to {report}")
    sys.exit(0 if passed_rows == n else 1)


if __name__ == "__main__":
    main()
