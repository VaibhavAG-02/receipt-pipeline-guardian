-- Canary for silent model failure: a model flagging almost nothing or almost
-- everything is broken even when every column test passes.
select count(*) as n, sum(is_flagged) * 1.0 / nullif(count(*), 0) as flag_rate
from {{ ref('fct_receipt') }}
having flag_rate < 0.005 or flag_rate > 0.30
