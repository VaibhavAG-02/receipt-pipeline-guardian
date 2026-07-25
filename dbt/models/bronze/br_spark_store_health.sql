-- Bronze layer produced by the PySpark stage (rpg.spark_job --emit-bronze),
-- not by the Python pipeline. It is materialised as a table rather than a view
-- because the source is an external parquet file Spark wrote, and pinning it
-- into DuckDB keeps the dbt DAG self-contained.
--
-- This is the seam between Spark and dbt: Spark does the per-partition
-- aggregation, dbt validates and models it. If the file is absent (Spark was
-- not run) the model is skipped via the enabled flag below.
{{ config(materialized='table', enabled=var('spark_bronze', false)) }}

select
    cast(submitted_date as date)                       as submitted_date,
    store_id,
    cast(n_receipts as bigint)                         as n_receipts,
    cast(gross_amount as double)                       as gross_amount,
    cast(n_failing_reconciliation as bigint)           as n_failing_reconciliation,
    cast(reconciliation_failure_rate as double)        as reconciliation_failure_rate
from read_parquet('{{ var("spark_bronze_path", "../data/raw/spark_store_health.parquet") }}')
