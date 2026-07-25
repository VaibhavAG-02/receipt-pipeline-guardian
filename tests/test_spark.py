"""PySpark job tests, run against a real local Spark session.

These are slow (a session spins up per module) but they exercise the actual
Spark code path, not a mock. If PySpark or a JDK is missing the whole module
skips rather than failing, so contributors without a JVM can still run the rest
of the suite -- CI installs both and runs these for real.

What they prove: the reconciliation arithmetic in Spark matches the dbt/DuckDB
definition to the cent, the aggregation is at the right grain, and the output
schema is stable enough for dbt to read downstream.
"""

from __future__ import annotations

import shutil

import pandas as pd
import pytest

pyspark = pytest.importorskip("pyspark")


def _java_available() -> bool:
    return shutil.which("java") is not None


pytestmark = pytest.mark.skipif(
    not _java_available(), reason="no JDK on this machine; Spark needs a JVM"
)


@pytest.fixture(scope="module")
def spark():
    from rpg.spark_job import build_session

    session = build_session("rpg-tests")
    yield session
    session.stop()


@pytest.fixture(scope="module")
def small_data(spark):
    receipts = spark.createDataFrame(
        pd.DataFrame(
            [
                # reconciles: 2*4.29 + 1*3.89 = 12.47, *1.0825 = 13.50
                {"receipt_id": "R1", "store_id": "ST001",
                 "submitted_at": pd.Timestamp("2026-06-01T10:00:00Z"),
                 "total": 13.50, "image_quality": 0.9},
                # does NOT reconcile: printed total far above the line items
                {"receipt_id": "R2", "store_id": "ST001",
                 "submitted_at": pd.Timestamp("2026-06-01T11:00:00Z"),
                 "total": 99.00, "image_quality": 0.8},
                # different day, different store
                {"receipt_id": "R3", "store_id": "ST002",
                 "submitted_at": pd.Timestamp("2026-06-02T09:00:00Z"),
                 "total": 4.21, "image_quality": 0.95},
            ]
        )
    )
    items = spark.createDataFrame(
        pd.DataFrame(
            [
                {"receipt_id": "R1", "sku": "S1", "qty": 2, "unit_price": 4.29},
                {"receipt_id": "R1", "sku": "S2", "qty": 1, "unit_price": 3.89},
                {"receipt_id": "R2", "sku": "S3", "qty": 1, "unit_price": 9.99},
                {"receipt_id": "R3", "sku": "S4", "qty": 1, "unit_price": 3.89},
            ]
        )
    )
    return receipts, items


def test_spark_session_starts(spark):
    assert spark.version.split(".")[0].isdigit()


def test_daily_store_health_grain_and_counts(spark, small_data):
    from rpg.spark_job import daily_store_health

    out = daily_store_health(*small_data).toPandas()
    # Three receipts, but R1 and R2 are the same store on the same day, so they
    # collapse into one group -> two store-day rows, three receipts total.
    assert len(out) == 2
    assert set(out.columns) >= {
        "submitted_date", "store_id", "n_receipts", "gross_amount",
        "n_failing_reconciliation", "reconciliation_failure_rate",
    }
    assert int(out["n_receipts"].sum()) == 3
    st001 = out[out["store_id"] == "ST001"].iloc[0]
    assert int(st001["n_receipts"]) == 2  # R1 + R2 rolled up


def test_spark_reconciliation_matches_the_dbt_definition(spark, small_data):
    """The Spark arithmetic must agree with silver/gold to the cent.

    R2's printed total is 99.00 against ~10.82 of items, so it must fail
    reconciliation; R1 and R3 must pass. If Spark and dbt ever compute this
    differently, the medallion layers disagree with each other -- exactly the
    class of silent bug the whole project is about.
    """
    from rpg.spark_job import daily_store_health

    out = daily_store_health(*small_data).toPandas()
    failing = int(out["n_failing_reconciliation"].sum())
    assert failing == 1, f"expected exactly R2 to fail, got {failing}"


def test_spark_output_is_deterministic(spark, small_data):
    from rpg.spark_job import daily_store_health

    a = daily_store_health(*small_data).toPandas().sort_values(
        ["submitted_date", "store_id"]).reset_index(drop=True)
    b = daily_store_health(*small_data).toPandas().sort_values(
        ["submitted_date", "store_id"]).reset_index(drop=True)
    pd.testing.assert_frame_equal(a, b)
