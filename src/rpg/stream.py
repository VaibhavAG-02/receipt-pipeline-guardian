"""Optional Redpanda/Kafka path.

The default pipeline is file-based so the project runs with zero services. This
module swaps the *first* step for a real event stream; everything downstream is
unchanged, because both paths converge on the same parquet landing zone.

Redpanda is used rather than Kafka because it is a single container with no
ZooKeeper/KRaft setup, Apache-2.0 licensed, and speaks the Kafka protocol -- so
`confluent-kafka` talks to it unmodified and you could point this at any Kafka
cluster later without touching the code.

The client objects are injected rather than constructed inline. That is not
ceremony: it lets the batching, keying, serialisation and offset-commit
ordering be exercised against an in-memory broker double in CI, where no broker
is available. See `tests/test_stream.py`. That does NOT test the wire protocol
-- only a live broker does that.

Run `docker compose up -d` first, then:
    python -m rpg.stream produce --receipts 5000
    python -m rpg.stream consume --max-messages 5000
"""

from __future__ import annotations

import json
import sys
from typing import Any, Protocol

import pandas as pd

from .config import BOOTSTRAP_SERVERS, RAW_DIR, TOPIC, ensure_dirs
from .generate import GenConfig, generate, to_records


class ProducerLike(Protocol):
    """The slice of confluent_kafka.Producer this module actually uses."""

    def produce(self, topic: str, key: bytes, value: bytes, callback: Any) -> None: ...
    def poll(self, timeout: float) -> int: ...
    def flush(self, timeout: float) -> int: ...


class ConsumerLike(Protocol):
    """The slice of confluent_kafka.Consumer this module actually uses."""

    def subscribe(self, topics: list[str]) -> None: ...
    def poll(self, timeout: float) -> Any: ...
    def commit(self, asynchronous: bool = ...) -> Any: ...
    def close(self) -> None: ...


def _default_producer() -> ProducerLike:
    try:
        from confluent_kafka import Producer
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise SystemExit(
            "confluent-kafka is not installed. It is an optional extra:\n"
            "    pip install -r requirements-stream.txt\n"
            "The default file-based pipeline (`python -m rpg.pipeline`) needs none of this."
        ) from exc
    return Producer({"bootstrap.servers": BOOTSTRAP_SERVERS, "linger.ms": 50})


def _default_consumer() -> ConsumerLike:
    try:
        from confluent_kafka import Consumer
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise SystemExit("confluent-kafka is not installed. See requirements-stream.txt.") from exc
    return Consumer(
        {
            "bootstrap.servers": BOOTSTRAP_SERVERS,
            "group.id": "rpg-ingest",
            "auto.offset.reset": "earliest",
            # Offsets are committed only after the batch is durably written, so
            # a crash mid-batch replays rather than silently losing records.
            "enable.auto.commit": False,
        }
    )


def produce(
    n_receipts: int = 5_000,
    seed: int = 7,
    producer: ProducerLike | None = None,
    topic: str = TOPIC,
) -> int:
    """Publish synthetic receipts to the topic. Returns messages acknowledged."""
    producer = producer or _default_producer()
    df = generate(GenConfig(n_receipts=n_receipts, seed=seed))

    delivered = 0

    def _ack(err, _msg):
        nonlocal delivered
        if err is None:
            delivered += 1

    for rec in to_records(df):
        # Keying by receipt_id gives per-receipt ordering and lets a compacted
        # topic collapse re-sends of the same receipt.
        producer.produce(
            topic,
            key=rec["receipt_id"].encode(),
            value=json.dumps(rec).encode(),
            callback=_ack,
        )
        producer.poll(0)

    producer.flush(30)
    return delivered


def consume(
    max_messages: int = 5_000,
    timeout_s: float = 10.0,
    consumer: ConsumerLike | None = None,
    topic: str = TOPIC,
    out_path=None,
) -> int:
    """Drain the topic into the parquet landing zone. Returns rows written."""
    consumer = consumer or _default_consumer()
    ensure_dirs()
    out_path = out_path or (RAW_DIR / "stream_receipts.parquet")
    consumer.subscribe([topic])

    rows: list[dict[str, Any]] = []
    try:
        while len(rows) < max_messages:
            msg = consumer.poll(timeout_s)
            if msg is None:
                break
            if msg.error():
                continue
            rows.append(json.loads(msg.value()))
        if rows:
            df = pd.DataFrame(rows)
            df["submitted_at"] = pd.to_datetime(df["submitted_at"], utc=True)
            df.to_parquet(out_path, index=False)
            # Commit only after the write succeeded. If to_parquet raises, the
            # offsets stay put and the batch replays instead of being lost.
            consumer.commit(asynchronous=False)
    finally:
        consumer.close()
    return len(rows)


if __name__ == "__main__":  # pragma: no cover - CLI
    import argparse

    ap = argparse.ArgumentParser(description="Redpanda/Kafka path")
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("produce")
    p.add_argument("--receipts", type=int, default=5_000)
    p.add_argument("--seed", type=int, default=7)
    c = sub.add_parser("consume")
    c.add_argument("--max-messages", type=int, default=5_000)
    a = ap.parse_args()

    if a.cmd == "produce":
        print(f"delivered {produce(a.receipts, a.seed)} messages to {TOPIC}")
    else:
        n = consume(a.max_messages)
        print(f"wrote {n} rows to {RAW_DIR / 'stream_receipts.parquet'}")
        if n == 0:
            sys.exit("no messages consumed -- is the producer running?")
