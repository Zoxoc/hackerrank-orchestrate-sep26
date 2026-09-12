"""Runtime OCR amount reader for Buy or Wait?

Reads a bill/receipt/payslip photo with a vendored ONNX OCR engine
(see code/ocr_models/) and extracts the payable amount. Works on unseen
future images: no pre-saved answers, deterministic for identical bytes.

Usage (lazy per-request, the default):
    from ocr_amounts import resolve_event_amount
    res = resolve_event_amount("dataset/media/images/image_05.png", event_currency="INR")
    # -> {"status": "ok", "amount": 704.05, ...} or {"status": "unclear", ...}

Batch warm-up (optional, same results as lazy):
    python3 code/ocr_amounts.py --warmup
"""

import hashlib
import json
import os
import re
import sys

CODE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(CODE_DIR, "ocr_models")
CACHE_PATH = os.path.join(CODE_DIR, "ocr_cache.json")

VENDORED_MODELS = {
    "det_model_path": os.path.join(MODEL_DIR, "ch_PP-OCRv4_det_infer.onnx"),
    "rec_model_path": os.path.join(MODEL_DIR, "ch_PP-OCRv4_rec_infer.onnx"),
    "cls_model_path": os.path.join(MODEL_DIR, "ch_ppocr_mobile_v2.0_cls_infer.onnx"),
}

MIN_MEAN_CONFIDENCE = 0.5

# A candidate must look like money: thousands separators, decimals,
# or at least 4 digits (bare 2-digit fragments like "43" never qualify).
MONEY_RE = re.compile(r"\d{1,3}(?:,\d{2,3})+(?:\.\d{1,2})?|\d+\.\d{1,2}|\d{4,}")
NUM_RE = re.compile(r"\d[\d,]*\.?\d*")

# (label, keywords, take_first_number)
# Ordered by priority. "total"-family takes the LAST number in the row
# (rightmost column); payable/balance rows take the FIRST.
LABEL_RULES = [
    ("balance_due", ("balancedue", "balance"), False),
    ("amount_payable", ("payable", "amountpayable"), True),
    ("grand_total", ("grandtotal",), False),
    ("net_amount", ("netamount", "netpay"), False),
    ("item_bill", ("itembill", "billamount", "invoiceamount"), True),
    ("total_received", ("totalamountreceived", "totalreceived"), False),
    ("total", ("total",), False),
    ("amount_due", ("amountdue", "amounttill"), True),
]

# Rows that must never supply the amount (payments made, running balances).
POISON_ROWS = ("cashpaid", "changepaid", "previousbalance", "paymentsmade",
               "amountpaid", "cashpaid:")

RECEIVED_KEYWORDS = ("amountreceived", "received", "paid")


def _sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def _load_cache():
    try:
        with open(CACHE_PATH, encoding="utf-8") as f:
            return json.load(f)
    except (OSError, ValueError):
        return {}


def _save_cache(cache):
    tmp = CACHE_PATH + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(cache, f, indent=1, sort_keys=True)
    os.replace(tmp, CACHE_PATH)


_ENGINE = None


def get_engine():
    """Singleton OCR engine; prefers vendored models, else package defaults."""
    global _ENGINE
    if _ENGINE is not None:
        return _ENGINE
    from rapidocr_onnxruntime import RapidOCR
    if all(os.path.exists(p) for p in VENDORED_MODELS.values()):
        _ENGINE = RapidOCR(**VENDORED_MODELS)
    else:
        _ENGINE = RapidOCR()  # package-bundled models
    return _ENGINE


def read_lines(image_path):
    """OCR -> [(row_text, mean_conf)] grouped into visual rows, reading order."""
    engine = get_engine()
    result, _elapsed = engine(image_path)
    if not result:
        return []
    items = []
    for box, text, conf in result:
        xs = [p[0] for p in box]
        ys = [p[1] for p in box]
        items.append({"text": text or "", "conf": float(conf or 0),
                      "cx": sum(xs) / len(xs), "cy": sum(ys) / len(ys),
                      "h": max(ys) - min(ys)})
    items.sort(key=lambda i: i["cy"])
    rows = []
    for it in items:
        placed = False
        for row in rows:
            if abs(row["cy"] - it["cy"]) <= max(12.0, row["h"] * 0.6):
                row["items"].append(it)
                row["cy"] = sum(i["cy"] for i in row["items"]) / len(row["items"])
                placed = True
                break
        if not placed:
            rows.append({"cy": it["cy"], "h": it["h"], "items": [it]})
    out = []
    for row in rows:
        parts = sorted(row["items"], key=lambda i: i["cx"])
        text = " ".join(p["text"] for p in parts).strip()
        conf = sum(p["conf"] for p in parts) / len(parts)
        if text:
            out.append((text, conf))
    return out


def _norm(text):
    return re.sub(r"[^a-z0-9]", "", text.lower())


def _money_numbers(text):
    return [m.group(0) for m in MONEY_RE.finditer(text.replace(" ", ""))]


def _parse_number(token):
    try:
        return float(token.replace(",", ""))
    except ValueError:
        return None


def _currency_contradiction_free(full_text, event_currency):
    t = full_text.lower()
    cur = (event_currency or "").upper()
    if cur == "USD":
        return True  # "$" is the expected sign; codes rarely contradict
    if "$" in t and cur != "USD":
        # "$" sometimes means "Rs." in noisy scans; only reject if a
        # non-USD code is explicit.
        if re.search(r"\b(usd|dollar)", t):
            return False
    for code, ccy in (("idr", "IDR"), ("rp", "IDR"), ("usd", "USD"),
                      ("eur", "EUR"), ("zar", "ZAR"), ("inr", "INR")):
        if re.search(r"\b" + code + r"\b", t) and ccy != cur:
            return False
    return True


def extract_amount(lines, event_currency):
    """Pick the payable amount from OCR rows. Returns (amount, meta|None)."""
    if not lines:
        return None, None
    full_text = " ".join(t for t, _ in lines)
    if not _currency_contradiction_free(full_text, event_currency):
        return None, {"reason": "currency_contradiction"}
    norm_rows = [(_norm(t), t, c) for t, c in lines]
    doc_norm = " ".join(n for n, _, _ in norm_rows)
    has_received = any(k in doc_norm for k in RECEIVED_KEYWORDS)

    for label, keywords, take_first in LABEL_RULES:
        if label == "balance_due" and not has_received:
            continue
        for norm, raw, conf in norm_rows:
            if any(p in norm for p in POISON_ROWS):
                continue
            if not any(k in norm for k in keywords):
                continue
            if label == "total" and "subtotal" in norm and "grandtotal" not in norm:
                # A bare "subtotal" row is not the payable total; keep looking
                # for a real total row first (fallback below accepts it).
                continue
            nums = _money_numbers(raw)
            if not nums:
                continue
            token = nums[0] if take_first else nums[-1]
            amount = _parse_number(token)
            if amount is None or amount <= 0 or amount >= 1e12:
                continue
            return amount, {"label": label, "row": raw,
                            "confidence": round(conf, 4)}
    # Fallback: a subtotal row is better than nothing.
    for norm, raw, conf in norm_rows:
        if "subtotal" in norm and "grandtotal" not in norm:
            nums = _money_numbers(raw)
            if nums:
                amount = _parse_number(nums[-1])
                if amount and 0 < amount < 1e12:
                    return amount, {"label": "subtotal_fallback",
                                    "row": raw,
                                    "confidence": round(conf, 4)}
    return None, {"reason": "no_labeled_total_found"}


def resolve_event_amount(image_path, event_currency="INR", event_category="",
                         use_cache=True):
    """OCR one event photo. Never raises; unclear results are explicit."""
    if not image_path or not os.path.exists(image_path or ""):
        return {"status": "unclear", "reason": "image_missing",
                "path": image_path}
    try:
        key = _sha256(image_path) + "|" + (event_currency or "").upper()
    except OSError:
        return {"status": "unclear", "reason": "image_unreadable",
                "path": image_path}
    if use_cache:
        hit = _load_cache().get(key)
        if hit:
            out = dict(hit)
            out["cached"] = True
            return out
    try:
        lines = read_lines(image_path)
    except Exception as exc:  # engine failure -> explicit unclear
        return {"status": "unclear", "reason": f"ocr_engine_error: {exc}",
                "path": image_path}
    if not lines:
        return {"status": "unclear", "reason": "no_text_detected",
                "path": image_path}
    mean_conf = sum(c for _, c in lines) / len(lines)
    if mean_conf < MIN_MEAN_CONFIDENCE:
        return {"status": "unclear", "reason": "low_confidence",
                "confidence": round(mean_conf, 4), "path": image_path}
    amount, meta = extract_amount(lines, event_currency)
    if amount is None:
        out = {"status": "unclear",
               "reason": (meta or {}).get("reason", "extract_failed"),
               "confidence": round(mean_conf, 4), "path": image_path}
    else:
        out = {"status": "ok", "amount": amount,
               "currency": (event_currency or "").upper(),
               "label": (meta or {}).get("label"),
               "confidence": (meta or {}).get("confidence", round(mean_conf, 4)),
               "engine": "rapidocr-onnxruntime",
               "path": image_path}
    if use_cache:
        cache = _load_cache()
        cache[key] = dict(out)
        try:
            _save_cache(cache)
        except OSError:
            pass
    out["cached"] = False
    return out


def warmup(images_dir, event_meta=None):
    """OCR every PNG in a directory (batch mode). Same results as lazy."""
    import glob as _glob
    results = {}
    for path in sorted(_glob.glob(os.path.join(images_dir, "*.png"))):
        meta = (event_meta or {}).get(os.path.basename(path), {})
        results[path] = resolve_event_amount(
            path, event_currency=meta.get("currency", "INR"))
    return results


def _dataset_currencies(img_dir):
    """Map image basename -> event currency using dataset CSVs (best effort)."""
    mapping = {}
    try:
        repo = os.path.dirname(CODE_DIR)
        ds = os.path.join(repo, "dataset")
        import csv as _csv
        ev_ccy = {}
        with open(os.path.join(ds, "financial_events.csv"),
                  newline="", encoding="utf-8") as f:
            for row in _csv.DictReader(f):
                ev_ccy[row["event_id"]] = (row.get("currency") or "").strip()
        with open(os.path.join(ds, "images.csv"),
                  newline="", encoding="utf-8") as f:
            for row in _csv.DictReader(f):
                mapping[row["image_id"] + ".png"] = \
                    ev_ccy.get(row.get("related_event_id"), "")
    except OSError:
        pass
    return mapping


if __name__ == "__main__":
    if "--warmup" in sys.argv:
        img_dir = sys.argv[sys.argv.index("--warmup") + 1] \
            if len(sys.argv) > sys.argv.index("--warmup") + 1 else \
            os.path.join("dataset", "media", "images")
        currencies = _dataset_currencies(img_dir)
        event_meta = {base: {"currency": ccy or "INR"}
                      for base, ccy in currencies.items()}
        res = warmup(img_dir, event_meta)
        ok = sum(1 for r in res.values() if r["status"] == "ok")
        print(f"OCR warmup: {ok}/{len(res)} ok -> {CACHE_PATH}")
        for path, r in sorted(res.items()):
            print(f"  {os.path.basename(path)}: {r}")
