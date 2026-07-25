-- Receipt-grain fact. Surrogate FKs to the conformed dimensions; measures and
-- model output as columns. Reported and recomputed totals sit side by side so
-- nobody downstream has to trust an OCR total blindly.
with line_rollup as (
    select
        receipt_id,
        count(*)          as n_items,
        sum(qty)          as total_qty,
        max(qty)          as max_qty,
        sum(line_amount)  as computed_subtotal,
        max(unit_price)   as max_unit_price
    from {{ ref('sl_receipt_items') }}
    group by receipt_id
)
select
    r.receipt_id,
    -- Point-in-time join: the store version in force when the receipt landed.
    d.store_key,
    dd.date_key,
    r.user_id,
    r.store_id,
    r.submitted_at,
    r.submitted_date,
    r.payment_method,
    r.scanner_version,
    r.image_quality,
    r.reported_subtotal,
    r.reported_tax,
    r.reported_total,
    coalesce(l.n_items, 0)         as n_items,
    coalesce(l.total_qty, 0)       as total_qty,
    coalesce(l.max_qty, 0)         as max_qty,
    coalesce(l.max_unit_price, 0)  as max_unit_price,
    l.computed_subtotal,
    l.computed_subtotal * 1.0825   as computed_total,
    r.reported_total - (l.computed_subtotal * 1.0825) as total_residual,
    abs(r.reported_total - (l.computed_subtotal * 1.0825))
        / nullif(l.computed_subtotal * 1.0825, 0)     as total_residual_pct,
    case
        when abs(r.reported_total - (l.computed_subtotal * 1.0825))
             / nullif(l.computed_subtotal * 1.0825, 0) > 0.02
        then true else false
    end                            as fails_reconciliation,
    s.anomaly_score,
    s.is_flagged,
    e.top_factors,
    e.factor_1,
    e.factor_1_value,
    r.is_anomaly_label,
    r.anomaly_type
from {{ ref('sl_receipts') }} r
left join line_rollup l         on l.receipt_id = r.receipt_id
left join {{ ref('br_scores') }} s on s.receipt_id = r.receipt_id
left join {{ ref('br_explanations') }} e on e.receipt_id = r.receipt_id
left join {{ ref('dim_date') }} dd on dd.calendar_date = r.submitted_date
left join {{ ref('dim_store') }} d
       on d.store_natural_key = r.store_id
      and r.submitted_at >= d.valid_from
      and r.submitted_at <  d.valid_to
