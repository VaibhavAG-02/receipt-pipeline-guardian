-- Fairness view: model recall per operational slice, with sample sizes.
-- Aggregate performance can improve while a segment collapses; this is the
-- table that makes that visible, and the gate that blocks promotion reads it.
select
    slice,
    value,
    n,
    n_positive,
    recall,
    flag_rate,
    recall_gap_vs_overall,
    reliable
from {{ ref('br_slice_metrics') }}
order by reliable desc, recall_gap_vs_overall desc nulls last
