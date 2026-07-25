-- Conformed date dimension covering the receipt window.
with bounds as (
    select min(submitted_date) as d0, max(submitted_date) as d1
    from {{ ref('sl_receipts') }}
),
days as (
    select unnest(generate_series(
        (select d0 from bounds),
        (select d1 from bounds),
        interval 1 day
    )) as d
)
select
    cast(strftime(d, '%Y%m%d') as integer) as date_key,
    cast(d as date)                        as calendar_date,
    year(d)                                as calendar_year,
    month(d)                               as calendar_month,
    day(d)                                 as day_of_month,
    dayofweek(d)                           as day_of_week,
    dayofweek(d) in (0, 6)                 as is_weekend,
    strftime(d, '%Y-%m')                   as year_month
from days
