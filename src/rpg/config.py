"""Central paths and tunables.

Everything is file-based and embedded so the project runs with no cloud
account, no credentials and no running services. The Kafka/Redpanda path is
opt-in (see docker-compose.yml); the default path is pure local files.
"""

from __future__ import annotations

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA = Path(os.environ.get("RPG_DATA_DIR", ROOT / "data"))

RAW_DIR = DATA / "raw"  # landing zone, one parquet per ingest batch
QUARANTINE_DIR = DATA / "quarantine"  # rows rejected by the contract gate
ARTIFACTS = DATA / "artifacts"  # model + metrics
WAREHOUSE = DATA / "warehouse.duckdb"  # DuckDB file dbt builds into

MODEL_PATH = ARTIFACTS / "model.json"
METRICS_PATH = ARTIFACTS / "metrics.json"
FEATURE_LIST_PATH = ARTIFACTS / "features.json"

# Kafka / Redpanda (only used by the optional streaming path)
BOOTSTRAP_SERVERS = os.environ.get("RPG_BOOTSTRAP", "localhost:19092")
TOPIC = os.environ.get("RPG_TOPIC", "receipts.raw")

# Generator defaults
DEFAULT_SEED = 42
DEFAULT_N_RECEIPTS = 20_000
ANOMALY_RATE = 0.06  # share of receipts carrying an injected defect

# Data-quality gate thresholds. These are the "SLA" the Guardian enforces.
MAX_NULL_RATE = 0.02
MIN_ROWS_PER_BATCH = 100
FRESHNESS_MAX_AGE_DAYS = 3.0
# A batch failing more than this share of contract checks fails the build.
MAX_QUARANTINE_RATE = 0.15


def ensure_dirs() -> None:
    for d in (DATA, RAW_DIR, QUARANTINE_DIR, ARTIFACTS):
        d.mkdir(parents=True, exist_ok=True)
