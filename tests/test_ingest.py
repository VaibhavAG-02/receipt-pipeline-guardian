"""Tests for bringing outside receipts in: files, forms, photos."""

from __future__ import annotations

import io
import json

import pandas as pd
import pytest

from rpg.ingest import (
    CANONICAL_COLUMNS,
    IngestError,
    build_receipt,
    ocr_text,
    parse_receipt_text,
    read_upload,
    render_receipt_image,
)

ITEMS = [
    {"sku": "SKU1001", "name": "Whole Milk", "qty": 2, "unit_price": 4.29},
    {"sku": "SKU1010", "name": "Pasta 16oz", "qty": 3, "unit_price": 1.99},
]


# ---------------------------------------------------------------- files ---
def test_reads_receipt_grain_json():
    payload = [{
        "receipt_id": "R1", "store_id": "ST001", "total": 15.75,
        "submitted_at": "2026-06-20T10:00:00Z", "items": ITEMS,
    }]
    df = read_upload(json.dumps(payload).encode(), "x.json")
    assert list(df.columns) == CANONICAL_COLUMNS
    assert len(df) == 1
    assert df.loc[0, "items"][0]["sku"] == "SKU1001"


def test_reads_csv_with_items_as_json_string():
    csv = (
        "receipt_id,store_id,total,submitted_at,items\n"
        f'R1,ST001,15.75,2026-06-20T10:00:00Z,"{json.dumps(ITEMS).replace(chr(34), chr(34) * 2)}"\n'
    )
    df = read_upload(csv.encode(), "x.csv")
    assert len(df) == 1
    assert len(df.loc[0, "items"]) == 2


def test_folds_line_item_shaped_files_to_receipt_grain():
    """Exports often arrive one row per line, not one row per receipt."""
    csv = (
        "receipt_id,sku,item_name,qty,unit_price,store_id\n"
        "R1,SKU1,Milk,2,4.29,ST001\n"
        "R1,SKU2,Pasta,3,1.99,ST001\n"
        "R2,SKU1,Milk,1,4.29,ST002\n"
    )
    df = read_upload(csv.encode(), "x.csv")
    assert len(df) == 2, "two receipts, not three lines"
    r1 = df[df.receipt_id == "R1"].iloc[0]
    assert len(r1["items"]) == 2
    assert r1["subtotal"] == pytest.approx(2 * 4.29 + 3 * 1.99, abs=0.01)


def test_missing_optional_columns_are_defaulted():
    df = read_upload(json.dumps([{"receipt_id": "R1", "total": 9.99}]).encode(), "x.json")
    assert df.loc[0, "currency"] == "USD"
    assert df.loc[0, "payment_method"] == "card"
    assert pd.notna(df.loc[0, "submitted_at"])


def test_errors_are_specific_and_actionable():
    with pytest.raises(IngestError, match="receipt_id"):
        read_upload(json.dumps([{"total": 1.0}]).encode(), "x.json")
    with pytest.raises(IngestError, match="empty"):
        read_upload(b"", "x.csv")
    with pytest.raises(IngestError, match=r"\.csv"):
        read_upload(b"x", "x.txt")
    with pytest.raises(IngestError, match="JSON"):
        read_upload(
            b'receipt_id,total,items\nR1,1.0,"not json"\n', "x.csv"
        )


def test_uploaded_rows_survive_the_contract_gate_unchanged():
    """An uploaded receipt must take the same path as a generated one."""
    from rpg.quality import split_quarantine

    payload = [
        {"receipt_id": "R_OK", "store_id": "ST001", "total": 15.75,
         "submitted_at": "2026-06-20T10:00:00Z", "items": ITEMS},
        {"receipt_id": "R_BAD", "store_id": "ZZ999", "total": 15.75,
         "submitted_at": "2026-06-20T10:00:00Z", "items": ITEMS},
    ]
    df = read_upload(json.dumps(payload).encode(), "x.json")
    clean, quarantined = split_quarantine(df, as_of=pd.Timestamp("2026-06-30T00:00:00Z"))
    assert set(clean["receipt_id"]) == {"R_OK"}
    assert quarantined.iloc[0]["quarantine_reason"] == "unknown_store"


# ----------------------------------------------------------------- form ---
def test_build_receipt_computes_subtotal_and_tax():
    df = build_receipt(ITEMS, total=15.75)
    assert df.loc[0, "subtotal"] == pytest.approx(2 * 4.29 + 3 * 1.99, abs=0.01)
    assert df.loc[0, "tax"] > 0
    assert list(df.columns) == CANONICAL_COLUMNS


def test_build_receipt_rejects_an_empty_basket():
    with pytest.raises(IngestError, match="line item"):
        build_receipt([], total=10.0)


def test_form_receipt_scores_and_responds_to_corruption():
    """The end-to-end promise of the feature: change a field, move the score."""
    from rpg.features import build_features
    from rpg.generate import GenConfig, generate
    from rpg.quality import split_quarantine
    from rpg.train import score, train

    raw = generate(GenConfig(n_receipts=1_500, seed=5))
    clean, _ = split_quarantine(raw, as_of=pd.Timestamp(GenConfig().end))
    train(build_features(clean))

    good = build_receipt(ITEMS, total=15.75)
    bad = build_receipt(
        [ITEMS[0], {**ITEMS[1], "unit_price": 19.90}], total=15.75
    )
    s_good = score(build_features(good))["anomaly_score"].iloc[0]
    s_bad = score(build_features(bad))["anomaly_score"].iloc[0]
    assert s_bad > s_good


# ------------------------------------------------------------------ OCR ---
def _png(record) -> bytes:
    buf = io.BytesIO()
    render_receipt_image(record).save(buf, "PNG")
    return buf.getvalue()


# The OCR tests need the Tesseract *binary*, not just the Python wrapper. Skip
# rather than fail when it is absent, so a contributor without it can still run
# the suite -- CI installs it explicitly so the coverage is not silently lost.
def _tesseract_available() -> bool:
    import shutil
    if shutil.which("tesseract") is None:
        return False
    try:
        import pytesseract  # noqa: F401
    except ImportError:
        return False
    return True


needs_ocr = pytest.mark.skipif(
    not _tesseract_available(), reason="tesseract binary not installed"
)


@needs_ocr
def test_receipt_renderer_produces_a_readable_image():
    img = render_receipt_image({"store_id": "ST001", "items": ITEMS,
                                "subtotal": 14.55, "tax": 1.20, "total": 15.75})
    assert img.width > 200 and img.height > 150


@needs_ocr
def test_ocr_round_trip_recovers_the_line_items():
    """Render a receipt we know, read it back, assert the parse matches.

    This is what makes the OCR path testable at all: without a renderer there
    is no ground truth to compare against in CI.
    """
    record = {"store_id": "ST007", "items": ITEMS + [
        {"sku": "S3", "name": "Cheddar Block", "qty": 1, "unit_price": 4.75}],
        "subtotal": 20.50, "tax": 1.69, "total": 22.19}
    parsed = parse_receipt_text(ocr_text(_png(record)))

    assert len(parsed["items"]) == 3, "all three lines should be recovered"
    qty_price = {(i["qty"], round(i["unit_price"], 2)) for i in parsed["items"]}
    assert (2, 4.29) in qty_price
    assert (3, 1.99) in qty_price
    assert (1, 4.75) in qty_price
    assert parsed["total"] == pytest.approx(22.19, abs=0.01)
    assert parsed["total_was_read"] is True


@needs_ocr
def test_ocr_keeps_lines_it_could_not_parse():
    """Unreadable lines are surfaced, not silently dropped."""
    record = {"store_id": "ST007", "items": ITEMS,
              "subtotal": 14.55, "tax": 1.20, "total": 15.75}
    parsed = parse_receipt_text(ocr_text(_png(record)))
    assert isinstance(parsed["unparsed"], list)
    # The header is not a line item and must land in `unparsed`.
    assert any("MARKET" in line.upper() for line in parsed["unparsed"])


def test_parser_handles_a_misread_tax_label_without_losing_the_receipt():
    """Tesseract routinely reads TAX as TAK or 1AX on thermal print.

    Tax is a nullable OCR field by design -- the marts recompute it from the
    line items -- so a misread label must degrade to null, never to a wrong
    number and never to a crash.
    """
    text = "2 Whole Milk 4.29\n3 Pasta 1.99\nSUBTOTAL 14.55\nTAK 1.20\nTOTAL 15.75"
    parsed = parse_receipt_text(text)
    assert parsed["tax"] is None
    assert parsed["subtotal"] == pytest.approx(14.55)
    assert parsed["total"] == pytest.approx(15.75)
    assert len(parsed["items"]) == 2


def test_parser_falls_back_to_computed_total_when_none_is_printed():
    parsed = parse_receipt_text("2 Whole Milk 4.29\n3 Pasta 1.99")
    assert parsed["total_was_read"] is False
    assert parsed["total"] == pytest.approx(14.55 * 1.0825, abs=0.02)


def test_parser_survives_complete_garbage():
    parsed = parse_receipt_text("~~~ \n \n ????")
    assert parsed["items"] == []
    assert parsed["unparsed"]
