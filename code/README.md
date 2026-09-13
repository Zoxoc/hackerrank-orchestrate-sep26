# Buy or Wait? — solution code

Deterministic, rule-based financial decision agent. No LLM calls at runtime.

## Setup

```bash
pip install -r code/requirements.txt
```

This installs the local OCR engine (`rapidocr-onnxruntime` + vendored ONNX
models in `code/ocr_models/`). Everything else is Python standard library.

## Run

```bash
python3 code/main.py
```

Reads `dataset/`, writes `output.csv` at the repository root (one row per
`dataset/requests.csv`, exact required columns and order).

Useful flags:

```bash
python3 code/main.py --warmup-ocr   # OCR every image up front (same results as lazy default)
python3 code/main.py --limit 10     # first 10 requests only (debugging)
python3 code/main.py --out /tmp/o.csv
python3 code/main.py --no-ocr       # skip photo reading (blanks stay excluded)
```

## Score on the public samples

```bash
python3 code/evaluation/main.py
```

Compares the pipeline's decisions on `dataset/sample_requests.csv` against the
worked examples (format/decision-style reference only).

## Modules

| File | Role |
|---|---|
| `code/main.py` | pipeline entry point |
| `code/financial_state.py` | per-request money picture (profile, classified events, FX, recurrence, salary series, messages/images) |
| `code/forecast.py` | 90-day baseline simulation + payment-safety check |
| `code/message_actions.py` | message → dated amendments (EN+ID), applied by the forecast |
| `code/ocr_amounts.py` | runtime photo amount reader (vendored models + cache) |
| `code/decision.py` | safe amount, earliest date, plan ranking, spending changes, validator |
| `code/ocr_models/` | vendored OCR models (also shipped inside the pinned pip wheel) |
| `code/ocr_cache.json` | fingerprint-keyed OCR results (regenerable via `--warmup-ocr`) |
