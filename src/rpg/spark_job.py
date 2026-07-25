"""PySpark aggregation over the landing zone.

Honest framing: at this demo's volume DuckDB is faster than Spark and the dbt
models already produce these numbers. This job exists because the *shape* of
the transform is what changes at 1TB/day, not the logic -- and it's here so the
Spark path is real code that runs, not a README claim.

Use it when the landing zone outgrows a single machine; otherwise `dbt build`
is the right tool and this is redundant. Run:

    python -m rpg.spark_job --input data/raw --output data/spark_out
"""

from __future__ import annotations

import argparse
from pathlib import Path


def build_session(app_name: str = "receipt-guardian"):
    from pyspark.sql import SparkSession

    return (
        SparkSession.builder.appName(app_name)
        .master("local[*]")
        .config("spark.sql.shuffle.partitions", "8")
        .config("spark.ui.enabled", "false")
        .getOrCreate()
    )


def daily_store_health(receipts_df, items_df):
    """Reconcile reported vs computed totals and roll up per store per day."""
    from pyspark.sql import functions as F

    line_totals = (
        items_df.withColumn("line_amount", F.col("qty") * F.col("unit_price"))
        .groupBy("receipt_id")
        .agg(
            F.count("*").alias("n_items"),
            F.sum("line_amount").alias("computed_subtotal"),
            F.max("qty").alias("max_qty"),
            F.max("unit_price").alias("max_unit_price"),
        )
    )

    joined = receipts_df.join(line_totals, on="receipt_id", how="left").withColumn(
        "computed_total", F.col("computed_subtotal") * F.lit(1.0825)
    )

    reconciled = joined.withColumn(
        "fails_reconciliation",
        F.when(
            F.abs(F.col("total") - F.col("computed_total"))
            / F.greatest(F.col("computed_total"), F.lit(0.01))
            > F.lit(0.02),
            F.lit(1),
        ).otherwise(F.lit(0)),
    )

    return (
        reconciled.withColumn("submitted_date", F.to_date("submitted_at"))
        .groupBy("submitted_date", "store_id")
        .agg(
            F.count("*").alias("n_receipts"),
            F.sum("total").alias("gross_amount"),
            F.avg("image_quality").alias("avg_image_quality"),
            F.sum("fails_reconciliation").alias("n_failing_reconciliation"),
            (
                F.sum("fails_reconciliation") / F.greatest(F.count("*"), F.lit(1))
            ).alias("reconciliation_failure_rate"),
        )
        .orderBy("submitted_date", "store_id")
    )


def run(input_dir: str, output_dir: str | None = None,
        emit_bronze: bool = False) -> int:
    spark = build_session()
    try:
        receipts = spark.read.parquet(str(Path(input_dir) / "receipts.parquet"))
        items = spark.read.parquet(str(Path(input_dir) / "receipt_items.parquet"))
        out = daily_store_health(receipts, items)
        n = out.count()
        if output_dir:
            out.coalesce(1).write.mode("overwrite").parquet(output_dir)
        if emit_bronze:
            # Single parquet file dbt can read via read_parquet(). pandas write
            # avoids Spark's directory-of-parts layout, which dbt-duckdb would
            # have to glob.
            dest = Path(input_dir) / "spark_store_health.parquet"
            out.toPandas().to_parquet(dest, index=False)
        return n
    finally:
        spark.stop()


if __name__ == "__main__":  # pragma: no cover - CLI
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default="data/raw")
    ap.add_argument("--output", default=None,
                    help="optional parquet directory (Spark part-files)")
    ap.add_argument("--emit-bronze", action="store_true",
                    help="write data/raw/spark_store_health.parquet for dbt")
    a = ap.parse_args()
    n = run(a.input, a.output, emit_bronze=a.emit_bronze)
    print(f"Spark aggregated {n} store-day rows"
          + (" -> bronze parquet for dbt" if a.emit_bronze else ""))
