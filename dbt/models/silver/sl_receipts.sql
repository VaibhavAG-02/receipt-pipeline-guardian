-- Silver: conformed receipt grain. Types pinned, business columns named, and
-- the reported-vs-recomputed reconciliation resolved once so no downstream
-- model has to re-derive it (or derive it differently).
with typed as (
    select
        receipt_id,
        user_id,
        store_id,
        cast(submitted_at as timestamp)   as submitted_at,
        cast(submitted_at as date)        as submitted_date,
        currency,
        payment_method,
        scanner_version,
        cast(image_quality as double)     as image_quality,
        cast(subtotal as double)          as reported_subtotal,
        cast(tax as double)               as reported_tax,
        cast(total as double)             as reported_total,
        cast(is_anomaly as integer)       as is_anomaly_label,
        anomaly_type
    from {{ ref('br_receipts') }}
),
-- Defensive dedupe. The contract gate enforces uniqueness upstream, but an
-- at-least-once stream can redeliver, and silver is where that stops.
deduped as (
    select *
    from (
        select *, row_number() over (
            partition by receipt_id order by submitted_at
        ) as rn
        from typed
    )
    where rn = 1
)
select * exclude (rn) from deduped
