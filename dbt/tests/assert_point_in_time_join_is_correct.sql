-- The join that SCD2 exists to make correct: every receipt must land inside
-- its resolved store version's validity window. If this returns rows, history
-- is being restated.
select
    f.receipt_id,
    f.submitted_at,
    d.valid_from,
    d.valid_to
from {{ ref('fct_receipt') }} f
join {{ ref('dim_store') }} d on d.store_key = f.store_key
where f.submitted_at < d.valid_from
   or f.submitted_at >= d.valid_to
