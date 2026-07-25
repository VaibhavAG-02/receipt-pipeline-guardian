select q.receipt_id
from {{ ref('br_quarantine') }} q
join {{ ref('fct_receipt') }} f on f.receipt_id = q.receipt_id
