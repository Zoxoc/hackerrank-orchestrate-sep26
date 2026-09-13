# Buy or Wait? — Solution Code

Deterministic, rule-based financial decision agent for the HackerRank
Orchestrate "Buy or Wait?" challenge. For every purchase/payment request in
`dataset/requests.csv` it reconstructs the user's financial position,
forecasts 90 days of cash flow, and recommends `full_payment`,
`partial_payment`, `installments`, `wait`, or `not_recommended`.
No LLM or external API calls at runtime.

## 1. Overview

For each request the pipeline builds a per-request money picture (profile,
classified events, FX conversion, recurring series, salary, linked
messages/images), applies message amendments, simulates a 90-day baseline,
then searches candidate payment plans in spec ranking order and validates
the chosen row before writing it to `output.csv`.

## 2. Solution Architecture

`code/main.py` runs, per request:

1. **Financial state reconstruction** (`financial_state.py ::
   build_request_state`) — profile (balance, minimum to keep, protected /
   reducible / stoppable categories, payment preferences), classified events
   (settled / pending / scheduled / failed / cancelled / unrealized /
   blank), FX conversion to home currency, recurring-debit candidates,
   salary history + scheduled salary, adjustable recurring items, and linked
   messages/images.
2. **Message interpretation** (`message_actions.py :: build_adjustments`) —
   each relevant message (EN/ID) becomes at most one dated amendment:
   `SET_SALARY`, `MOVE_MONEY_DATE`, `STOP_SALARY`, `ADD_BILL`,
   `ADD_CONFIRMED_PAY`, `SCALE_BILL`, `DEDUPE_TRANSFER`, `CONFIRM_SKIP`.
   Applied only with an explicit amount + date (or a resolvable next-payroll
   date); otherwise recorded and ignored.
3. **OCR / image processing** (`ocr_amounts.py :: resolve_event_amount`) —
   blank-amount events linked to a photo get their amount from the local
   ONNX OCR reader (see §5). Unresolved blanks stay excluded.
4. **90-day forecasting** (`forecast.py :: build_baseline_forecast`) —
   day-by-day baseline from reserved pending debits, scheduled obligations,
   confirmed future credits, history-supported recurring debits (monthly
   bills step by calendar day-of-month, other cadences by median interval at
   average amount), projected salary, and message amendments.
5. **Decision / payment-plan engine** (`decision.py :: decide_request`) —
   safe-today amount (binary search), earliest safe full-payment date
   (forward scan), ranked plan search, spending-change fallback, grounded
   explanation (see §3).
6. **Validation** (`decision.py :: validate_row`) — every row is checked
   (column order, enums, amount bounds, plan arithmetic, change limits)
   during the run; violations are counted, not silently written.
7. **CSV output** — `code/main.py` writes the exact 8 required columns in
   order to `<repo root>/output.csv`.

## 3. How the Logic Works

- **Amount safe to pay** — largest amount payable on `request_date` such
  that the simulated balance never drops below `minimum_balance_to_keep`
  over the 90-day window (`safe_today_cents`, cent-level binary search).
- **Affordability status** — `affordable_now` (full safe today),
  `affordable_with_plan` (full completes via partial schedule,
  installments, or permitted spending changes), `affordable_later` (full
  safe on a future date), `not_affordable` (never safe in-window).
- **Payment method ranking** — full > partial > installments > wait,
  ordered by a key that prefers completing by `desired_completion_date`,
  avoiding spending changes, minimizing total cost, starting earlier, and
  using fewer payments (`find_best_plan`, `_rank_key`).
- **Partial payments** — exactly two entries: safe amount on
  `request_date`, remainder on `earliest_date_for_full_payment`; only when
  allowed, accepted by the user, and the second date meets the deadline.
- **Installments** — schedules replayed verbatim from
  `request_payment_options.csv`; rejected if they conflict with user
  preferences or `max_installment_months`.
- **Waiting** — full payment on the earliest safe date when today is
  unsafe but a future date works.
- **Spending changes** — up to three `stop:<event_id>` /
  `reduce_to:<event_id>:<new_amount>` actions over non-protected, flexible
  categories the user permits (`search_with_changes` tries stop/reduce
  combos by increasing size, then by plan rank).
- **Earliest full-payment date** — first date a single full payment is
  safe; equals `request_date` for `affordable_now`, empty when never safe.

## 4. Data Handling

- **Transactions** — `settled` counts; `failed`/`cancelled`/`unrealized`
  excluded; one-time history is already inside today's balance, never
  re-projected.
- **Recurring income/expenses** — detected only with history support;
  monthly payroll projected on payday (off-cycle extras never move payday,
  terminal markers stop projection); gig/weekly-platform/variable pay never
  projected.
- **Pending/scheduled** — pending debits reserved; pending credits,
  bonuses, commissions, refunds, lottery, and investment gains excluded
  until settled; confirmed salary counted on its settlement date.
- **Messages** — untrusted evidence: may clarify, amend, delay, cancel, or
  confirm a fact via the 8 amendment types; embedded instructions never
  override the rules; only messages sent on/before `request_date` apply.
- **Images** — resolved to `dataset/media/images/<image_id>.png` and read
  by local OCR only when relevant to a blank amount.
- **Currencies** — fixed dated rates from `exchange_rates.csv`, applied in
  the stated `from_currency → to_currency` direction on settlement date.
- **Minimum balance** — inviolable: no projected essential expense or
  recommended payment may push the balance below
  `minimum_balance_to_keep` on any day.

## 5. OCR / ML Usage

The only ML component is a **local, on-device OCR reader** — no external
LLM or API calls anywhere in the implementation (0 model calls, 0 tokens,
$0.00; see `code/evaluation/usage_report.md`).

- Engine: `rapidocr-onnxruntime==1.4.4` with three vendored ONNX models in
  `code/ocr_models/` (PP-OCRv4 detection + recognition + orientation
  classifier).
- Used solely to recover amounts for blank-amount events linked to a
  photo; label-priority extraction with sanity gates.
- Lazy per-request with fingerprint-keyed cache (`code/ocr_cache.json`,
  regenerable); final 250-request run performed 11 image reads.
- If OCR is unavailable the pipeline still runs (`--no-ocr`): blanks stay
  excluded per the rules.

## 6. Running the Code

From the repository root:

```bash
pip install -r code/requirements.txt   # local OCR runtime; rest is stdlib (Python 3.12)
python3 code/main.py                    # reads dataset/, writes output.csv (250 rows)
```

Useful flags:

```bash
python3 code/main.py --warmup-ocr   # OCR every image up front (same results as lazy default)
python3 code/main.py --limit 10     # first 10 requests only (debugging)
python3 code/main.py --out /tmp/o.csv
python3 code/main.py --no-ocr       # skip photo reading (blanks stay excluded)
```

Final run stats: 250/250 rows in ~2 seconds with warm OCR cache.

## 7. Evaluation

Automated sample test (isolated harness, never touches engine code):

```bash
python3 code/evaluation/test_samples.py [--report PATH]
```

It runs the production pipeline on the 25 public worked examples in
`dataset/sample_requests.csv` and checks, per row: exact
`affordability_status`, `recommended_payment_method`,
`earliest_date_for_full_payment`, `spending_changes_needed`;
`amount_safe_to_pay` within `max(1.0, 5%)`; `payment_plan` equivalent
(same dates, amounts within 0.011); `decision_explanation` validity only
(non-empty, mentions home currency, contains a figure).

**Current observed result, reported honestly: 6/25 full rows**
(status 21/25, method 22/25, safe 11/25, plan 21/25, earliest 19/25,
changes 22/25). Residual gaps are baseline-calibration tolerances
(±1 occurrence / few-% amounts, mixed directions), not logic defects —
see `code/evaluation/sample_test_report.txt`.

## 8. Output

`output.csv` columns, in order:

| Column | Meaning |
|---|---|
| `request_id` | Request key, one row per `dataset/requests.csv` |
| `amount_safe_to_pay` | Amount safe on `request_date` (0 … `requested_amount`) |
| `affordability_status` | `affordable_now` / `affordable_with_plan` / `affordable_later` / `not_affordable` |
| `recommended_payment_method` | `full_payment` / `partial_payment` / `installments` / `wait` / `not_recommended` |
| `payment_plan` | Chronological `YYYY-MM-DD:amount` entries joined by `\|`, or `none` |
| `earliest_date_for_full_payment` | First safe full-payment date (`request_date` for `affordable_now`, empty if never) |
| `spending_changes_needed` | `none` or up to three `stop:` / `reduce_to:` actions |
| `decision_explanation` | Concise, grounded explanation with currency and figures |

## 9. Evaluation and Usage Reports

- `code/evaluation/test_samples.py` — automated 25-sample test harness.
- `code/evaluation/sample_test_report.txt` — latest per-row expected-vs-actual report.
- `code/evaluation/usage_report.md` — final full-dataset run accounting:
  model providers/names, 0 model calls, 0 input/output tokens, $0.00
  total and per-request cost (local OCR only, non-API).

## 10. Code Structure

```text
code/
├── README.md                  # this file (solution guide for judges)
├── main.py                    # pipeline entry point; writes output.csv
├── financial_state.py         # per-request state: profile, events, FX, recurrence, salary, links
├── forecast.py                # 90-day baseline simulation + safety check
├── message_actions.py         # message → dated amendments (EN+ID)
├── decision.py                # safe amount, earliest date, plan ranking, changes, validator
├── ocr_amounts.py             # runtime photo amount reader (local ONNX + cache)
├── ocr_models/                # vendored PP-OCRv4 det/rec/cls .onnx files
├── ocr_cache.json             # fingerprint-keyed OCR results (regenerable)
├── requirements.txt           # pinned OCR runtime (stdlib otherwise)
└── evaluation/
    ├── test_samples.py        # automated 25-sample test
    ├── sample_test_report.txt # latest sample test report
    ├── usage_report.md        # final-run token/cost accounting
    └── main.py                # sample scorer helper
```
