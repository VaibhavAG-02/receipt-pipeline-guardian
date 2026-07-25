-- Cross-engine agreement: the store-day reconciliation counts computed by
-- PySpark must equal those computed by dbt/DuckDB from the same rows. If Spark
-- and dbt ever diverge, the medallion layers silently disagree -- which is the
-- exact failure mode this whole project argues against. Only runs when the
-- Spark bronze layer is present.
{{ config(enabled=var('spark_bronze', false)) }}

with dbt_side as (
    select submitted_date, store_id,
           sum(case when fails_reconciliation then 1 else 0 end) as n_fail
    from {{ ref('fct_receipt') }}
    group by 1, 2
),
spark_side as (
    select submitted_date, store_id, n_failing_reconciliation as n_fail
    from {{ ref('br_spark_store_health') }}
)
select d.submitted_date, d.store_id, d.n_fail as dbt_fail, s.n_fail as spark_fail
from dbt_side d
join spark_side s on s.submitted_date = d.submitted_date and s.store_id = d.store_id
where d.n_fail <> s.n_fail
