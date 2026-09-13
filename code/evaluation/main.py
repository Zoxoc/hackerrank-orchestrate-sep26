"""Local scorer: pipeline decisions vs public sample_requests.py (reference only).

Usage:
    python3 code/evaluation/main.py
"""

import os
import sys
from collections import defaultdict

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
os.chdir(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))

from financial_state import (load_csv, build_fx_index, build_request_state)
from forecast import build_baseline_forecast
from message_actions import build_adjustments
from decision import decide_request
from ocr_amounts import resolve_event_amount


def main():
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

    keys = ["affordability_status", "recommended_payment_method",
            "payment_plan", "earliest_date_for_full_payment",
            "spending_changes_needed"]
    tally = {k: 0 for k in keys}
    for r in samples:
        s = build_request_state(r, profs, ev_by, opts, msgs, imgs, fx,
                                ocr_resolver=resolve_event_amount)
        row = decide_request(s, build_baseline_forecast(
            s, adjustments=build_adjustments(s)))
        e = expected[r["request_id"]]
        line = []
        for k in keys:
            got, want = (row[k] or ""), (e[k] or "")
            mark = "=" if got == want else "!"
            if got == want:
                tally[k] += 1
            line.append(f"{k.split('_')[0]}:{mark}")
        print(f"{r['request_id']}: {' '.join(line)}")
    print(f"matches /{len(samples)}:", tally)


if __name__ == "__main__":
    main()
