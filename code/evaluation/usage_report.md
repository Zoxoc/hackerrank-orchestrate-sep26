# Token Usage and Cost Report — Final Full-Dataset Run

Run: `python3 code/main.py`  
Result: 250/250 requests processed and written to `output.csv`.

## Engine

The solution is fully rule-based and deterministic. It makes zero
calls to language models or paid APIs.

Financial-state reconstruction, forecasting, message interpretation,
payment-plan evaluation, decision making, and validation are performed
locally in Python.

The only ML component is a local ONNX OCR reader used for bill-photo
amount extraction. It runs locally on CPU and does not consume API
tokens.

## Model Providers and Names

| Component | Provider | Model | Calls |
|---|---|---|---:|
| Financial state, forecast, message actions, decision engine | None (local) | N/A | 0 |
| Photo OCR | None (local) | PP-OCRv4 detection + recognition + orientation classifier | 11 image reads |

The OCR model is vendored under `code/ocr_models/` and uses
`rapidocr-onnxruntime==1.4.4`. OCR results are cached by file hash.

## LLM Token Accounting

| Metric | Value |
|---|---:|
| Model providers used | None |
| Model calls | 0 |
| Input tokens | 0 |
| Output tokens | 0 |
| Total tokens | 0 |
| Average tokens per request | 0 |
| Estimated total cost | $0.00 |
| Estimated cost per request | $0.00 |

## Local Compute

- Final 250-request run: ~1 second with warm OCR cache.
- 11 requests required image OCR.
- OCR results are cached and reused on subsequent runs.
- Re-running the final dataset produces byte-identical decisions.
- No API keys, credentials, or sensitive configuration are included.