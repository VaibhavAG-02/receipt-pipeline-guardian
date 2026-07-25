-- Fan-out guard. The Type 2 join is the classic way a fact table silently
-- multiplies; the fact must stay at exactly the silver receipt grain.
select
    (select count(*) from {{ ref('fct_receipt') }}) as fact_rows,
    (select count(*) from {{ ref('sl_receipts') }}) as silver_rows
having fact_rows <> silver_rows
