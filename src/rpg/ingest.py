"""Bring outside receipts into the pipeline.

Three entry points, all of which converge on the same canonical record and then
go through the *same* contract gate, feature builder and model as generated
data. That convergence is the point: an uploaded receipt is not on a special
path, so whatever the console shows about it is what the pipeline would do in
a batch.

  1. `read_upload`  -- CSV / JSON / JSONL, in either of the two shapes real
     exports actually arrive in (one row per receipt, or one row per line item).
  2. `build_receipt` -- a single record assembled from form input.
  3. `ocr_receipt`   -- a photo, read with Tesseract and parsed.

The OCR path is the weakest link and is labelled as such in the UI. Tesseract
on a creased thermal receipt under kitchen lighting is a coin flip; that is a
real property of the problem, not something to hide behind a confidence number.
"""

from __future__ import annotations

import io
import json
import re
from typing import Any

import pandas as pd

CANONICAL_COLUMNS = [
    "receipt_id", "user_id", "store_id", "submitted_at", "currency",
    "payment_method", "scanner_version", "image_quality", "items",
    "subtotal", "tax", "total",
]

DEFAULTS = {
    "user_id": "U000000",
    "store_id": "ST001",
    "currency": "USD",
    "payment_method": "card",
    "scanner_version": "v4.0",
    "image_quality": 0.90,
}

TAX_RATE = 0.0825


class IngestError(ValueError):
    """Raised with a message intended to be shown directly to the user."""


# ---------------------------------------------------------------- files ---
def _coerce_items(value: Any) -> list[dict]:
    if isinstance(value, list):
        return [dict(i) for i in value]
    if isinstance(value, str) and value.strip():
        try:
            parsed = json.loads(value)
        except json.JSONDecodeError as exc:
            raise IngestError(
                "The `items` column must contain JSON, for example "
                '[{"sku":"SKU1","name":"Milk","qty":1,"unit_price":4.29}]'
            ) from exc
        if not isinstance(parsed, list):
            raise IngestError("`items` must be a JSON list of line items.")
        return [dict(i) for i in parsed]
    return []


def _normalise(df: pd.DataFrame) -> pd.DataFrame:
    missing = [c for c in ("receipt_id", "total") if c not in df.columns]
    if missing:
        raise IngestError(
            f"Missing required column(s): {', '.join(missing)}. "
            f"A file needs at least receipt_id and total."
        )

    out = df.copy()
    for col, default in DEFAULTS.items():
        if col not in out.columns:
            out[col] = default

    if "items" in out.columns:
        out["items"] = out["items"].map(_coerce_items)
    else:
        out["items"] = [[] for _ in range(len(out))]

    if "submitted_at" not in out.columns:
        out["submitted_at"] = pd.Timestamp.now("UTC")
    out["submitted_at"] = pd.to_datetime(out["submitted_at"], utc=True, errors="coerce")
    if out["submitted_at"].isna().any():
        raise IngestError("Some `submitted_at` values could not be read as timestamps.")

    for col in ("total", "subtotal", "tax", "image_quality"):
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce")

    # Fill the OCR-nullable money columns from the line items where we can.
    computed = out["items"].map(
        lambda items: round(sum(float(i.get("qty", 0)) * float(i.get("unit_price", 0))
                                for i in items), 2)
    )
    if "subtotal" not in out.columns:
        out["subtotal"] = computed
    if "tax" not in out.columns:
        out["tax"] = (out["subtotal"] * TAX_RATE).round(2)

    for c in CANONICAL_COLUMNS:
        if c not in out.columns:
            out[c] = None
    return out[CANONICAL_COLUMNS].sort_values("submitted_at").reset_index(drop=True)


def _pivot_line_items(df: pd.DataFrame) -> pd.DataFrame:
    """Fold a one-row-per-line-item export into receipt grain."""
    required = {"receipt_id", "qty", "unit_price"}
    if not required.issubset(df.columns):
        raise IngestError(
            "A line-item file needs receipt_id, qty and unit_price columns."
        )
    grouped = []
    for rid, g in df.groupby("receipt_id", sort=False):
        items = [
            {
                "sku": str(r.get("sku", "UNKNOWN")),
                "name": str(r.get("item_name", r.get("name", ""))),
                "qty": int(float(r["qty"])),
                "unit_price": float(r["unit_price"]),
            }
            for _, r in g.iterrows()
        ]
        sub = round(sum(i["qty"] * i["unit_price"] for i in items), 2)
        first = g.iloc[0]
        rec = {
            "receipt_id": rid,
            "items": items,
            "subtotal": sub,
            "tax": round(sub * TAX_RATE, 2),
            "total": float(first["total"]) if "total" in g.columns and pd.notna(first.get("total"))
            else round(sub * (1 + TAX_RATE), 2),
        }
        for col in ("user_id", "store_id", "submitted_at", "currency",
                    "payment_method", "scanner_version", "image_quality"):
            if col in g.columns:
                rec[col] = first[col]
        grouped.append(rec)
    return pd.DataFrame(grouped)


def read_upload(data: bytes, filename: str) -> pd.DataFrame:
    """Parse an uploaded CSV / JSON / JSONL file into canonical receipt rows."""
    name = filename.lower()
    if len(data) == 0:
        raise IngestError("That file is empty.")

    try:
        if name.endswith(".csv"):
            df = pd.read_csv(io.BytesIO(data))
        elif name.endswith(".jsonl") or name.endswith(".ndjson"):
            df = pd.read_json(io.BytesIO(data), lines=True)
        elif name.endswith(".json"):
            payload = json.loads(data.decode("utf-8"))
            if isinstance(payload, dict):
                payload = payload.get("receipts", [payload])
            df = pd.DataFrame(payload)
        else:
            raise IngestError("Upload a .csv, .json or .jsonl file.")
    except IngestError:
        raise
    except Exception as exc:
        raise IngestError(f"Could not read that file: {exc}") from exc

    if df.empty:
        raise IngestError("That file parsed, but contains no rows.")

    # Line-item shape: many rows share a receipt_id and there is no items column.
    if "items" not in df.columns and {"qty", "unit_price"}.issubset(df.columns):
        df = _pivot_line_items(df)

    return _normalise(df)


# ----------------------------------------------------------------- form ---
def build_receipt(
    items: list[dict],
    total: float,
    *,
    receipt_id: str = "UPLOAD-0001",
    store_id: str = "ST001",
    user_id: str = "U000000",
    submitted_at: Any = None,
    payment_method: str = "card",
    scanner_version: str = "v4.0",
    image_quality: float = 0.90,
    subtotal: float | None = None,
    tax: float | None = None,
) -> pd.DataFrame:
    """One canonical receipt from form values."""
    if not items:
        raise IngestError("Add at least one line item.")
    computed_sub = round(sum(float(i["qty"]) * float(i["unit_price"]) for i in items), 2)
    rec = {
        "receipt_id": receipt_id,
        "user_id": user_id,
        "store_id": store_id,
        "submitted_at": pd.to_datetime(submitted_at or pd.Timestamp.now("UTC"), utc=True),
        "currency": "USD",
        "payment_method": payment_method,
        "scanner_version": scanner_version,
        "image_quality": float(image_quality),
        "items": items,
        "subtotal": computed_sub if subtotal is None else float(subtotal),
        "tax": round(computed_sub * TAX_RATE, 2) if tax is None else float(tax),
        "total": float(total),
    }
    return pd.DataFrame([rec])[CANONICAL_COLUMNS]


# ------------------------------------------------------------------ OCR ---
# Two shapes of receipt line, tried in order:
#   1. "2  Milk           4.29"   -> leading quantity (my renderer, many stores)
#   2. "BREAD 007225003712 F 2.88 N" -> Walmart-style: name, product code,
#      optional tax flag, price, optional trailing tax letter. No leading qty.
# The price is always the last money-looking token on the line; everything
# before the first long digit run or the price is the name.
LINE_QTY_RE = re.compile(
    r"^(?P<qty>\d{1,3})\s*[xX@]?\s+(?P<name>[A-Za-z][A-Za-z0-9 .'\-/]{2,30}?)\s+"
    r"\$?(?P<price>\d{1,5}[.,]\d{2})\s*$"
)
LINE_NAMED_RE = re.compile(
    r"^(?P<name>[A-Za-z0-9][A-Za-z0-9 .'&\-/]{1,28}?[A-Za-z])\s+"  # product name
    r"(?:\d{6,}\s+)?"                                        # optional SKU/UPC
    r"(?:[A-Z]\s+)?"                                          # optional tax flag (F/T/N)
    r"\$?(?P<price>\d{1,5}[.,]\d{2})"                        # price
    r"(?:\s+[A-Z0-9])?\s*$"                                  # optional trailing code
)
# Lines that are clearly not items, even if they match the shape loosely.
NOT_ITEM_RE = re.compile(
    r"\b(total|subtotal|tax|change|tend|debit|credit|balance|account|ref|"
    r"network|approv|purchase|items sold|store|manager|phone|savings|"
    r"cash|visa|master|payment|declin)\b",
    re.I,
)
# Summary lines lead with a keyword that Tesseract mangles on thermal print:
# TAX -> TAK / 1AX / TAK, TOTAL -> T0TAL / TDTAL, SUBTOTAL -> SUBT0TAL. A line
# STARTING with one of these near-spellings is a summary line, not a product,
# even when the exact word did not survive OCR.
SUMMARY_MISREAD_RE = re.compile(
    r"^(t[a0o][xk]|[t1][o0]tal|subt[o0]tal|[a-z]?tax|chan[gq]e|tend)\b",
    re.I,
)
TOTAL_RE = re.compile(r"\b(total|amount due|balance)\b.*?\$?(\d{1,6}[.,]\d{2})", re.I)
SUBTOTAL_RE = re.compile(r"\bsub\s*total\b.*?\$?(\d{1,6}[.,]\d{2})", re.I)
# Match the tax *amount*, not a rate: ignore any number trailed by % (e.g.
# "TAX 1 7.000 %  3.26" -> 3.26, not 7.000).
TAX_RE = re.compile(r"\btax\b\s*\d*\s*(?:[\d.]+\s*%\s*)?\$?(\d{1,6}[.,]\d{2})(?!\s*%)", re.I)


def _money(s: str) -> float:
    return float(s.replace(",", "."))


def _parse_item_line(line: str) -> dict | None:
    """Return an item dict for a receipt line, or None if it is not an item.

    Handles both the leading-quantity layout and the Walmart-style
    name/SKU/price layout. Excludes summary lines (total, tax, tender, etc.)
    up front so a stray "TOTAL 46.30" is never mistaken for a product.
    """
    if NOT_ITEM_RE.search(line) or SUMMARY_MISREAD_RE.match(line):
        return None
    if m := LINE_QTY_RE.match(line):
        name, qty, price = m.group("name"), int(m.group("qty")), m.group("price")
    elif m := LINE_NAMED_RE.match(line):
        name, qty, price = m.group("name"), 1, m.group("price")
    else:
        return None
    name = name.strip()
    if len(name) < 2:  # reject noise like a lone letter
        return None
    return {
        "sku": "OCR-" + re.sub(r"[^A-Z0-9]", "", name.upper())[:8],
        "name": name,
        "qty": qty,
        "unit_price": _money(price),
    }


def parse_receipt_text(text: str) -> dict[str, Any]:
    """Pull line items and totals out of OCR text.

    A line-oriented parser rather than a layout model: receipts are printed as
    `qty name price`, and that regularity is most of what there is to exploit
    without a trained document model. Lines that do not match are returned in
    `unparsed` instead of being dropped, so the user can see what was missed.
    """
    items: list[dict] = []
    unparsed: list[str] = []
    total = subtotal = tax = None

    for raw in text.splitlines():
        line = raw.strip()
        if not line:
            continue
        if (m := TOTAL_RE.search(line)) and total is None and "sub" not in line.lower():
            total = _money(m.group(2))
            continue
        if (m := SUBTOTAL_RE.search(line)) and subtotal is None:
            subtotal = _money(m.group(1))
            continue
        if (m := TAX_RE.search(line)) and tax is None:
            tax = _money(m.group(1))
            continue
        item = _parse_item_line(line)
        if item is not None:
            items.append(item)
        else:
            unparsed.append(line)

    computed = round(sum(i["qty"] * i["unit_price"] for i in items), 2)
    return {
        "items": items,
        "subtotal": subtotal,
        "tax": tax,
        "total": total if total is not None else round(computed * (1 + TAX_RATE), 2),
        "total_was_read": total is not None,
        "computed_subtotal": computed,
        "unparsed": unparsed,
    }


def ocr_text(image_bytes: bytes) -> str:
    """Run Tesseract. Raises IngestError with a usable message if unavailable."""
    try:
        import pytesseract
        from PIL import Image
    except ImportError as exc:
        raise IngestError(
            "OCR needs pytesseract and Pillow: pip install -r requirements-ocr.txt"
        ) from exc

    try:
        img = Image.open(io.BytesIO(image_bytes))
    except Exception as exc:
        raise IngestError("That file could not be opened as an image.") from exc

    # Grayscale plus a hard threshold: thermal receipts are high-contrast text
    # on light stock, and Tesseract does markedly better on a clean binary image
    # than on a photo with shadows across it.
    img = img.convert("L")
    img = img.point(lambda p: 255 if p > 160 else 0)
    try:
        # psm 6: assume a single uniform block of text, which is what a receipt is.
        return pytesseract.image_to_string(img, config="--psm 6")
    except Exception as exc:
        raise IngestError(
            "Tesseract is not installed on this machine. On Streamlit Cloud add a "
            "packages.txt containing `tesseract-ocr`."
        ) from exc


def ocr_receipt(image_bytes: bytes, **meta) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Photo -> canonical receipt row, plus the parse detail for display."""
    text = ocr_text(image_bytes)
    parsed = parse_receipt_text(text)
    if not parsed["items"]:
        raise IngestError(
            "No line items could be read from that image. Receipts photographed "
            "flat, in even light, with the whole tape in frame work best."
        )
    df = build_receipt(
        parsed["items"],
        total=parsed["total"],
        subtotal=parsed["subtotal"],
        tax=parsed["tax"],
        receipt_id=meta.pop("receipt_id", "OCR-0001"),
        **meta,
    )
    parsed["raw_text"] = text
    return df, parsed


# -------------------------------------------------------------- render ----
def render_receipt_image(record: dict[str, Any], width: int = 420):
    """Draw a receipt as an image.

    Exists so the OCR path can be demonstrated and, more usefully, *tested*:
    render a receipt whose contents are known, read it back, and assert the
    parse matches. Without this the OCR code would be untestable in CI.
    """
    from PIL import Image, ImageDraw, ImageFont

    def font(sz):
        for p in (
            "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf",
            "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        ):
            try:
                return ImageFont.truetype(p, sz)
            except OSError:
                continue
        return ImageFont.load_default()

    items = record["items"]
    height = 190 + 26 * len(items)
    img = Image.new("L", (width, height), 246)
    d = ImageDraw.Draw(img)
    f, fb = font(15), font(18)

    y = 18
    d.text((width // 2 - 52, y), "MARKET 24", font=fb, fill=20)
    y += 30
    d.text((16, y), f"STORE {record.get('store_id', 'ST001')}", font=f, fill=40)
    y += 22
    d.line([(14, y), (width - 14, y)], fill=120)
    y += 14

    for it in items:
        left = f"{it['qty']} {it['name'][:22]}"
        right = f"{it['unit_price']:.2f}"
        d.text((16, y), left, font=f, fill=20)
        d.text((width - 16 - d.textlength(right, font=f), y), right, font=f, fill=20)
        y += 26

    d.line([(14, y), (width - 14, y)], fill=120)
    y += 14
    for label, val in (
        ("SUBTOTAL", record.get("subtotal")),
        ("TAX", record.get("tax")),
        ("TOTAL", record.get("total")),
    ):
        if val is None:
            continue
        txt = f"{val:.2f}"
        d.text((16, y), label, font=f, fill=20)
        d.text((width - 16 - d.textlength(txt, font=f), y), txt, font=f, fill=20)
        y += 24
    return img
