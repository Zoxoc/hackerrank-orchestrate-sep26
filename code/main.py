"""Buy or Wait? - full pipeline entry point.

Reads dataset/, decides every request, writes output.csv at repo root.

Usage:
    python3 code/main.py [--warmup-ocr] [--limit N] [--out PATH]

--warmup-ocr  OCR every image up front (same results as lazy default).
--limit N     process only the first N requests (debugging).
--out PATH    output path (default: <repo root>/output.csv).
"""

import argparse
import csv
import os
import sys
import time
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from financial_state import (load_csv, build_fx_index, build_request_state,
                             summarize_state)
from forecast import build_baseline_forecast
from message_actions import build_adjustments
from decision import decide_request, validate_row

try:
    from ocr_amounts import resolve_event_amount, warmup as ocr_warmup
    HAVE_OCR = True
except ImportError:
    HAVE_OCR = False
    resolve_event_amount = None

COLUMNS = ["request_id", "amount_safe_to_pay", "affordability_status",
           "recommended_payment_method", "payment_plan",
           "earliest_date_for_full_payment", "spending_changes_needed",
           "decision_explanation"]

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def run(requests, profiles_by_user, events_by_user, options_by_request,
        messages, images, fx_index, use_ocr=True, verbose_every=50):
    rows = []
    stats = defaultdict(int)
    t0 = time.time()
    for i, req in enumerate(requests):
        resolver = resolve_event_amount if (use_ocr and HAVE_OCR) else None
        state = build_request_state(req, profiles_by_user, events_by_user,
                                    options_by_request, messages, images,
                                    fx_index, ocr_resolver=resolver)
        adj = build_adjustments(state)
        forecast = build_baseline_forecast(state, adjustments=adj)
        row = decide_request(state, forecast)
        for p in validate_row(row, state):
            stats[f"invalid:{p}"] += 1
        stats[row["affordability_status"]] += 1
        stats[row["recommended_payment_method"]] += 1
        rows.append(row)
        if verbose_every and (i + 1) % verbose_every == 0:
            print(f"  {i + 1}/{len(requests)}...", flush=True)
    stats["seconds"] = round(time.time() - t0, 1)
    return rows, dict(stats)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--warmup-ocr", action="store_true")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--out", default=os.path.join(REPO_ROOT, "output.csv"))
    ap.add_argument("--no-ocr", action="store_true")
    args = ap.parse_args()

    if args.warmup_ocr and HAVE_OCR:
        from ocr_amounts import _dataset_currencies
        img_dir = os.path.join(REPO_ROOT, "dataset", "media", "images")
        meta = {b: {"currency": c or "INR"}
                for b, c in _dataset_currencies(img_dir).items()}
        res = ocr_warmup(img_dir, meta)
        ok = sum(1 for r in res.values() if r["status"] == "ok")
        print(f"OCR warmup: {ok}/{len(res)} ok")

    os.chdir(REPO_ROOT)
    requests = load_csv("requests.csv")
    if args.limit:
        requests = requests[:args.limit]
    profiles_by_user = {r["user_id"]: r for r in load_csv("financial_profiles.csv")}
    events_by_user = defaultdict(list)
    for e in load_csv("financial_events.csv"):
        events_by_user[e["user_id"]].append(e)
    options_by_request = defaultdict(list)
    for o in load_csv("request_payment_options.csv"):
        options_by_request[o["request_id"]].append(o)
    messages = load_csv("messages.csv")
    images = load_csv("images.csv")
    fx_index = build_fx_index(load_csv("exchange_rates.csv"))

    print(f"Deciding {len(requests)} requests...")
    rows, stats = run(requests, profiles_by_user, events_by_user,
                      options_by_request, messages, images, fx_index,
                      use_ocr=not args.no_ocr)
    with open(args.out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=COLUMNS)
        w.writeheader()
        for row in rows:
            w.writerow({k: row[k] for k in COLUMNS})
    print(f"Wrote {args.out} ({len(rows)} rows)")
    print("Stats:", stats)


if __name__ == "__main__":
    main()
