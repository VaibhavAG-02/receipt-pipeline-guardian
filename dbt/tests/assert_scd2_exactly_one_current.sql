-- SCD2 correctness #3: exactly one open version per store.
select store_natural_key, sum(case when is_current then 1 else 0 end) as n_current
from {{ ref('dim_store') }}
group by 1
having sum(case when is_current then 1 else 0 end) <> 1
