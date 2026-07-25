select receipt_id, anomaly_score from {{ ref('fct_receipt') }}
where anomaly_score < 0 or anomaly_score > 1
