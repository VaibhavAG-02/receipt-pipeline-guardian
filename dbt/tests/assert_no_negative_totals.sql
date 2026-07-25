select receipt_id, reported_total from {{ ref('fct_receipt') }} where reported_total <= 0
