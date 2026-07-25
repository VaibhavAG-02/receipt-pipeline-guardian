-- Daily vitals. Three independent signals: schema health (quarantine),
-- arithmetic health (reconciliation), and the model's opinion (flag rate).
-- They move independently, which is what makes a real incident legible.
with r as (
    select
        submitted_date,
        count(*)                                              as n_clean,
        sum(case when fails_reconciliation then 1 else 0 end) as n_failing_reconciliation,
        sum(is_flagged)                                       as n_flagged,
        avg(image_quality)                                    as avg_image_quality
    from {{ ref('fct_receipt') }}
    group by 1
),
q as (
    select cast(submitted_at as date) as submitted_date, count(*) as n_quarantined
    from {{ ref('br_quarantine') }}
    group by 1
)
select
    r.submitted_date,
    r.n_clean,
    coalesce(q.n_quarantined, 0)                              as n_quarantined,
    coalesce(q.n_quarantined, 0) * 1.0
        / nullif(r.n_clean + coalesce(q.n_quarantined, 0), 0) as quarantine_rate,
    r.n_failing_reconciliation,
    r.n_failing_reconciliation * 1.0 / nullif(r.n_clean, 0)   as reconciliation_failure_rate,
    r.n_flagged,
    r.n_flagged * 1.0 / nullif(r.n_clean, 0)                  as flag_rate,
    r.avg_image_quality
from r left join q on q.submitted_date = r.submitted_date
