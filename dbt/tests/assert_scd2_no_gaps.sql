-- SCD2 correctness #2: consecutive versions must be contiguous. A gap means a
-- receipt submitted in that window resolves to no store at all and drops out
-- of the fact table.
with ordered as (
    select
        store_natural_key,
        valid_to,
        lead(valid_from) over (
            partition by store_natural_key order by valid_from
        ) as next_valid_from
    from {{ ref('dim_store') }}
)
select *
from ordered
where next_valid_from is not null
  and next_valid_from <> valid_to
