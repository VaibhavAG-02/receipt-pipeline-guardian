-- A reviewer must never be handed a flagged receipt with no explanation.
select receipt_id from {{ ref('mart_review_queue') }}
where model_reasoning is null or model_reasoning = ''
