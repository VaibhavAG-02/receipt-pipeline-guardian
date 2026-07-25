-- Store health sliced by the dimension attributes in force at the time.
select
    f.submitted_date,
    f.store_id,
    d.region,
    d.store_format,
    d.manager_id,
    count(*)                                              as n_receipts,
    sum(f.reported_total)                                 as gross_amount,
    avg(f.image_quality)                                  as avg_image_quality,
    sum(case when f.fails_reconciliation then 1 else 0 end) as n_failing_reconciliation,
    sum(case when f.fails_reconciliation then 1 else 0 end) * 1.0
        / nullif(count(*), 0)                             as reconciliation_failure_rate,
    sum(f.is_flagged)                                     as n_flagged,
    sum(f.is_flagged) * 1.0 / nullif(count(*), 0)         as flag_rate,
    avg(f.anomaly_score)                                  as avg_anomaly_score
from {{ ref('fct_receipt') }} f
left join {{ ref('dim_store') }} d on d.store_key = f.store_key
group by 1, 2, 3, 4, 5
