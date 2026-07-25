# What would need to change for production

This project is a portfolio demo. It is engineered carefully, but it is not
production ready, and the gap is worth stating precisely rather than gesturing at.

## Blocking gaps

| Gap | Why it blocks | Rough effort |
|---|---|---|
| **No idempotency** | A replay re-inserts everything. Needs merge keys and upsert semantics, not `CREATE OR REPLACE`. | 1–2 weeks |
| **No incremental processing** | Every run rebuilds all history. Needs watermarks and dbt incremental models with `unique_key`. | 2–3 weeks |
| **No orchestration** | `make all` is not a scheduler. Needs Airflow/Dagster with retries, backfill, SLA alerting, dependency-aware reruns. | 2–4 weeks |
| **No secrets management** | Credentials would live in env vars. Needs a vault and rotation. | 1 week |
| **No PII controls** | Receipts carry user identifiers and payment methods. Needs field-level classification, hashing/tokenisation, retention policy, access control, and an audit trail. | 3–6 weeks |
| **No schema evolution** | A new upstream field breaks the contract gate with no migration path. Needs versioned contracts and a registry. | 2 weeks |
| **In-memory rate limiting** | Per-process and lost on restart. Needs a shared store. | days |
| **Single-node storage** | DuckDB is a file: no concurrency control, no HA, no point-in-time recovery. | architecture decision |
| **No observability** | No structured logs, metric export, tracing or SLOs. | 2 weeks |
| **Model never retrains automatically** | Drift is *detected* here; nothing acts on it. Needs a registry, shadow deployment, staged rollout, rollback. | 4–6 weeks |

## Non-blocking but expected

- Data lineage and a catalogue
- Cost attribution per pipeline
- Load and chaos testing
- On-call runbooks per alert
- Backfill correctness tests over historical windows

## What is already production-shaped

Worth separating from the above, because these are design decisions that would survive the transition:

- Deterministic contract gate kept separate from the ML scorer
- Quarantine rather than drop, with reasons retained
- Data quality as a build gate that fails CI
- Backward-only feature windows and chronological splitting
- Type 2 dimension with point-in-time joins and correctness tests for overlap, gaps and fan-out
- Per-receipt SHAP explanations rather than global importance
- Drift and fairness as release gates, with sample-size-aware reliability
