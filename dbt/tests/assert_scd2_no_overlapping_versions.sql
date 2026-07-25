-- SCD2 correctness #1: two versions of the same store must never be valid at
-- the same instant. An overlap silently duplicates every fact that joins
-- through it, which inflates revenue and is very hard to notice.
select
    a.store_natural_key,
    a.version_number as v1,
    b.version_number as v2
from {{ ref('dim_store') }} a
join {{ ref('dim_store') }} b
  on  a.store_natural_key = b.store_natural_key
  and a.store_key <> b.store_key
  and a.valid_from < b.valid_to
  and b.valid_from < a.valid_to
