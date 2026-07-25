-- Line-grain fact for basket analysis, conformed to the same dimensions.
select
    i.receipt_id,
    i.line_no,
    k.sku_key,
    f.store_key,
    f.date_key,
    i.qty,
    i.unit_price,
    i.line_amount,
    i.unit_price / nullif(k.median_unit_price, 0) as price_ratio_vs_median
from {{ ref('sl_receipt_items') }} i
join {{ ref('fct_receipt') }} f on f.receipt_id = i.receipt_id
left join {{ ref('dim_sku') }} k on k.sku_natural_key = i.sku
