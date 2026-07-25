-- Type 1 product dimension: SKUs are overwritten in place because item names
-- here are descriptive, not analytically meaningful history.
select
    md5(sku)                    as sku_key,
    sku                         as sku_natural_key,
    max(item_name)              as item_name,
    count(*)                    as times_sold,
    median(unit_price)          as median_unit_price,
    min(unit_price)             as min_unit_price,
    max(unit_price)             as max_unit_price
from {{ ref('sl_receipt_items') }}
group by sku
