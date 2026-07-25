# Receipt Pipeline Guardian

A receipt-ingestion pipeline that **catches bad data before it reaches the marts**, rather than discovering it three weeks later in a dashboard nobody trusts.

Every component is open source and runs locally. No cloud account, no managed warehouse, no credentials, no spend.

```bash
pip install -r requirements-dev.txt
make all          # generate -> gate -> train -> score -> dbt build + tests
make app          # open the operator console
```

---

## The problem

Receipt-scanning pipelines fail in ways that are individually boring and collectively expensive: OCR misreads a total, a device clock is wrong, a user double-submits, a weight gets read as a count. None of these throw an exception. They land quietly in the warehouse and corrupt every metric computed downstream.

The usual response is a dashboard of data-quality charts that someone is supposed to check. This project takes a different position: **quality is a build gate**. A batch that violates its contract does not partially land, and CI fails.

## Architecture

```
 generator ──┐                                    ┌── SHAP explanations
             ├─> contract gate ─> features ─> XGBoost ─> scores
 Redpanda ───┘        │                              └── drift + fairness gate
  (optional)          └─> quarantine (queryable)                    │
                                                                    v
   BRONZE  typed passthrough, no business logic
      │
   SILVER  conformed, deduped, SCD versions closed out
      │
    GOLD   star schema: fct_receipt, fct_receipt_line
           dim_store (Type 2), dim_date, dim_sku
           + review queue, store health, vitals, fairness marts
      │
   Streamlit console
```

Two layers of defence that are deliberately kept **separate**:

| | Contract gate | Anomaly model |
|---|---|---|
| Catches | schema violations, impossible values | plausible-looking defects |
| Method | deterministic rules | gradient-boosted trees |
| On failure | quarantine + fail the batch | flag for human review |
| Explainable at 3am | yes | not really |

Conflating these is a common mistake. You do not need ML to know a total of `-12.00` is wrong, and a rules engine will never catch a duplicate submission that looks perfectly normal in isolation. The gate keeps working when the model is stale.

## Results

Measured on a chronological hold-out (25% most recent), 20,000 receipts, ~6% defect rate:

| Metric | Value |
|---|---|
| PR-AUC | **0.917** |
| ROC-AUC | 0.960 |
| Precision @ top 1% | 1.000 |
| Precision @ top 5% | 0.996 |
| Recall @ operating threshold | 0.885 |
| Quarantined by the gate | 1.96% of batch |
| dbt models / data checks | 22 / 60 — **81 passing** (with Spark stage) |
| Python tests | **60 passing** |
| Fairness gate | pass (no slice lags overall recall by >0.20) |

**PR-AUC leads, not accuracy.** With a 6% positive rate, a model predicting "clean" every time scores 94% accuracy and is worthless. ROC-AUC is similarly flattering on imbalanced data. Precision@k is the number that maps to the actual constraint: a reviewer can only work through so many receipts a day.

The split is **chronological, not random**. Behavioural features look at a user's prior 24 hours, so a random split would let the model see the future.

### Recall by defect type

Aggregates hide failure. Per type, at the operating threshold:

| Defect | n | Recall |
|---|---|---|
| arithmetic_mismatch | 48 | 1.00 |
| duplicate_submission | 92 | 1.00 |
| impossible_quantity | 54 | 1.00 |
| ocr_dropout | 55 | 1.00 |
| **price_outlier** | **72** | **0.51** |
| timestamp_skew | 2 | 0.00 |

**Price outliers are the real result here.** A misplaced decimal point on one line of a large basket is genuinely hard: the receipt still reconciles internally, the user's behaviour is normal, and the only signal is a unit price that's implausible for that SKU. Half of them get through. That's the honest headline, not the 0.917.

## Known limitations

Stated plainly, because a demo that claims no weaknesses is not worth reading.

- **The data is synthetic.** Labels come from a generator I wrote. Absolute metric values describe the generator more than they describe production receipts. The *relative* difficulty ranking is the transferable part.
- **`tax_is_null` carries 53% of feature importance.** The OCR-dropout defect is *defined* by nulling that field, so the model gets a near-perfect giveaway. On real data this signal would be weak and noisy. This is the weakest part of the evaluation and I'd expect the aggregate numbers to fall substantially on real receipts.
- **`timestamp_skew` barely reaches the model** — the contract gate catches most of it first, leaving 2 test examples. That row of the table is noise, not a measurement.
- **Drift is measured train-vs-holdout, not across real time.** The generator's distribution is stationary, so the PSI machinery demonstrates the mechanism rather than a real detection.
- **Fairness slices are operational, not demographic.** Receipts carry no protected attributes.
- **The Kafka path is not exercised in CI.** It needs a running broker. The file-based path is fully tested; `src/rpg/stream.py` is verified by inspection only.
- **Spark is redundant at this scale.** DuckDB is faster here. `spark_job.py` exists because the transform's shape is what changes at 1TB/day, and I would rather ship code that runs than a README claim.

## The warehouse

**Medallion layering**, with the discipline that makes it worth doing: no business logic in bronze (or you can never replay raw history), conforming and deduplication in silver, star schema in gold.

**`dim_store` is a real Type 2 dimension.** Store attributes change — remodels alter the format, district realignments move the region — and each change arrives effective-dated. Facts join on `submitted_at >= valid_from and < valid_to`, so a March receipt resolves to the store *as it was in March*. Joining on `store_id` alone would restate last quarter's numbers every time a store is remodelled.

Four tests exist purely to keep that honest: no overlapping versions (an overlap silently duplicates every fact joining through it), no gaps between versions, exactly one current version per store, and every fact landing inside its resolved version's window. A fifth asserts the fact table stays at exactly the silver row count — the Type 2 join is the classic way a fact table quietly fans out.

**That caught a real bug.** The point-in-time test failed on the first build with 115 orphaned receipts: device clock skew produces receipts dated *before* the store's initial load, so they matched no version and dropped out of the fact table entirely. The fix is the standard one — open the first version back to the beginning of time. Without the test it would have shipped as a silent 0.6% revenue undercount.

## Explainability

The review queue carries **per-receipt signed SHAP contributions**, not global feature importance. Global importance answers "what matters on average", which is the same answer for every row and useless to a reviewer holding one receipt. TreeSHAP is exact for tree ensembles and fast enough to run across the whole flagged set.

Signs are preserved. "Arithmetic residual pushed this *up*" and "prior user history pushed it *down*" are different facts, and collapsing to absolute values loses the direction a reviewer needs.

Mean |SHAP| is also reported next to the model's gain-based importance, because the two routinely disagree and the disagreement is informative.

## Drift and fairness as release gates

Three signals, separated because they fail independently:

- **Feature drift** — PSI per feature against the training reference, with quantile bins taken from the reference. Conventional bands (0.10 / 0.25) so operators share a vocabulary.
- **Prediction drift** — the score distribution can move while inputs look stable, which means the model is extrapolating. Usually the earlier warning.
- **Slice performance** — recall per operational slice (scanner version, payment method, store format, image-quality band) with sample sizes attached.

The fairness gate **blocks promotion** if any adequately-sized slice lags overall recall by more than 0.20. Slices below the reliability threshold are reported but never block — an 11-row slice at 0.0 recall is noise, and a gate that acts on it will be switched off within a week.

Receipts carry no demographic attributes, so these slices are operational rather than protected. The mechanism is identical; a protected attribute plugs in unchanged.

## Repository layout

```
src/rpg/
  config.py      paths and thresholds
  drift.py       PSI drift, slice metrics, fairness gate
  explain.py     TreeSHAP per-receipt + global attribution
  generate.py    synthetic receipts with labelled defects
  quality.py     contract rules + batch gates  <- the "Guardian"
  features.py    feature engineering, backward-looking windows only
  train.py       XGBoost + honest evaluation
  pipeline.py    end-to-end orchestration
  stream.py      optional Redpanda/Kafka ingest
  spark_job.py   optional PySpark aggregation
dbt/             bronze (8) -> silver (3) -> gold (10), 58 data checks
app/             Streamlit operator console
tests/           41 pytest tests
scripts/         local CI verifier
```

## Deploying it for free

**Streamlit Community Cloud** hosts the console at no cost. Push to GitHub, connect the repo at [share.streamlit.io](https://share.streamlit.io), set the entry point to `app/streamlit_app.py`.

The app builds the warehouse in-process on first load (about a minute) if it isn't there, which is why no data or model binaries are committed. Streamlit Cloud's filesystem is ephemeral, so it self-heals on restart.

**CI is free** on GitHub Actions for public repos. Every push runs lint, the unit tests, the full pipeline, the quality gate, a model-regression floor (`pr_auc >= 0.75`), and `dbt build`.

`make verify-ci` executes every `run:` step of the workflow locally, in order, so you can catch a broken pipeline before pushing. It does not emulate the runner image or the `uses:` actions.

## Optional paths

```bash
make stream-up                                  # Redpanda + console on :8080
PYTHONPATH=src python -m rpg.stream produce --receipts 5000
PYTHONPATH=src python -m rpg.stream consume
make stream-down

make spark                                      # PySpark aggregation (needs a JDK)
```

Redpanda rather than Kafka: single container, no ZooKeeper, Kafka-API compatible, so the same code points at any Kafka cluster later.

## Not production ready

Deliberately stated: no idempotent or incremental loads, no orchestration, no secrets management, no PII controls, single-node storage. [`PRODUCTION.md`](PRODUCTION.md) sizes each gap rather than gesturing at it.

## Licence

MIT.
