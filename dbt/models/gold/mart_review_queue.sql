-- What a reviewer opens each morning: highest score first, with the model's
-- actual per-receipt reasoning (SHAP) next to the deterministic evidence.
select
    f.receipt_id,
    f.submitted_at,
    f.store_id,
    d.region,
    d.store_format,
    f.user_id,
    f.anomaly_score,
    f.top_factors                as model_reasoning,
    f.reported_total,
    f.computed_total,
    f.total_residual,
    f.total_residual_pct,
    f.n_items,
    f.max_qty,
    f.max_unit_price,
    f.image_quality,
    f.fails_reconciliation,
    case
        when f.image_quality < 0.45      then 'poor_image'
        when f.max_qty >= 40             then 'implausible_quantity'
        when f.total_residual_pct > 0.02 then 'does_not_reconcile'
        when f.max_unit_price > 100      then 'price_outlier'
        else 'behavioural'
    end as rule_based_hint
from {{ ref('fct_receipt') }} f
left join {{ ref('dim_store') }} d on d.store_key = f.store_key
where f.is_flagged = 1
order by f.anomaly_score desc
