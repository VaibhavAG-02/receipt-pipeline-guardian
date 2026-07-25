select
    receipt_id,
    cast(line_no as integer)                          as line_no,
    sku,
    item_name,
    cast(qty as integer)                              as qty,
    cast(unit_price as double)                        as unit_price,
    cast(qty as double) * cast(unit_price as double)  as line_amount
from {{ ref('br_receipt_items') }}
