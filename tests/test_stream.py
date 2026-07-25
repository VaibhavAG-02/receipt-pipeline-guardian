"""Stream path tests against an in-memory broker double.

**What this proves:** serialisation round-trips, receipts are keyed correctly,
delivery callbacks are honoured, malformed messages are skipped rather than
crashing the consumer, the landing-zone schema matches what the batch path
produces, and -- the one that actually matters operationally -- offsets are
committed *only after* the parquet write succeeds.

**What this does not prove:** anything about the Kafka wire protocol, broker
configuration, partition assignment, rebalancing or network behaviour. Those
need a live broker. `docker compose up -d` covers it locally; CI does not.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field

import pandas as pd
import pytest

from rpg import stream


# --------------------------------------------------------------- doubles ---
@dataclass
class FakeMessage:
    _value: bytes
    _key: bytes | None = None
    _error: object | None = None

    def value(self) -> bytes:
        return self._value

    def key(self) -> bytes | None:
        return self._key

    def error(self):
        return self._error


@dataclass
class FakeProducer:
    """Mimics confluent_kafka.Producer's async-callback contract."""

    fail_every: int = 0  # if n>0, every nth produce reports a delivery error
    messages: list[tuple[str, bytes, bytes]] = field(default_factory=list)
    _pending: list = field(default_factory=list)
    _count: int = 0
    flushed: bool = False

    def produce(self, topic, key, value, callback):
        self._count += 1
        err = "delivery failed" if self.fail_every and self._count % self.fail_every == 0 else None
        if err is None:
            self.messages.append((topic, key, value))
        # Real producers fire the callback later, from poll()/flush().
        self._pending.append((err, None))

    def poll(self, timeout):
        n = len(self._pending)
        return n

    def flush(self, timeout):
        for err, msg in self._pending:
            self._cb(err, msg)
        self._pending.clear()
        self.flushed = True
        return 0

    # produce() closes over the caller's callback; store it on first use
    def _cb(self, err, msg):
        if self._callback:
            self._callback(err, msg)

    _callback: object = None


class RecordingProducer(FakeProducer):
    """Captures the callback so flush() can fire it, as a real producer does."""

    def produce(self, topic, key, value, callback):
        self._callback = callback
        super().produce(topic, key, value, callback)


@dataclass
class FakeConsumer:
    queue: list[FakeMessage]
    subscribed: list[str] = field(default_factory=list)
    commits: int = 0
    closed: bool = False
    commit_log: list[int] = field(default_factory=list)
    _read: int = 0

    def subscribe(self, topics):
        self.subscribed = list(topics)

    def poll(self, timeout):
        if self._read >= len(self.queue):
            return None
        msg = self.queue[self._read]
        self._read += 1
        return msg

    def commit(self, asynchronous=False):
        self.commits += 1
        self.commit_log.append(self._read)

    def close(self):
        self.closed = True


def _expected_rows(n: int, seed: int = 7) -> int:
    """The generator appends duplicate-submission rows *on top of* n_receipts,
    so the emitted count is n plus however many dupes it injected. Derive it
    rather than hard-coding, or the tests break whenever the rate changes."""
    from rpg.generate import GenConfig, generate

    return len(generate(GenConfig(n_receipts=n, seed=seed)))


def _messages(n: int, seed: int = 7) -> list[FakeMessage]:
    from rpg.generate import GenConfig, generate, to_records

    recs = to_records(generate(GenConfig(n_receipts=n, seed=seed)))
    return [
        FakeMessage(json.dumps(r).encode(), r["receipt_id"].encode()) for r in recs
    ]


# -------------------------------------------------------------- producer ---
def test_produce_publishes_every_receipt():
    p = RecordingProducer()
    expected = _expected_rows(200, seed=3)
    n = stream.produce(n_receipts=200, seed=3, producer=p, topic="t")
    assert n == expected
    assert len(p.messages) == expected
    assert p.flushed, "producer must be flushed or messages can be lost on exit"


def test_produce_keys_by_receipt_id():
    """Keying drives partition affinity and log compaction; it must be the id."""
    p = RecordingProducer()
    stream.produce(n_receipts=50, seed=3, producer=p, topic="t")
    for _topic, key, value in p.messages:
        assert key == json.loads(value)["receipt_id"].encode()


def test_produce_payload_is_json_serialisable_and_complete():
    """Timestamps are the usual thing that breaks json.dumps. Catch it here."""
    p = RecordingProducer()
    stream.produce(n_receipts=20, seed=3, producer=p, topic="t")
    rec = json.loads(p.messages[0][2])
    for field_name in ("receipt_id", "user_id", "store_id", "submitted_at",
                       "total", "items", "currency"):
        assert field_name in rec
    assert isinstance(rec["items"], list)
    # ISO-8601 string, parseable back to a timestamp
    assert pd.notna(pd.to_datetime(rec["submitted_at"]))


def test_produce_counts_only_acknowledged_messages():
    """A delivery error must not be counted as delivered."""
    total = _expected_rows(100, seed=3)
    p = RecordingProducer(fail_every=10)
    n = stream.produce(n_receipts=100, seed=3, producer=p, topic="t")
    expected_ok = total - total // 10
    assert n == expected_ok
    assert len(p.messages) == expected_ok


# -------------------------------------------------------------- consumer ---
def test_consume_writes_landing_zone(tmp_path):
    out = tmp_path / "stream_receipts.parquet"
    expected = _expected_rows(150)
    c = FakeConsumer(_messages(150))
    n = stream.consume(max_messages=500, consumer=c, topic="t", out_path=out)
    assert n == expected
    assert out.exists()
    df = pd.read_parquet(out)
    assert len(df) == expected
    assert c.subscribed == ["t"]
    assert c.closed, "consumer must be closed even on the happy path"


def test_consume_output_matches_batch_path_schema(tmp_path):
    """The streaming and batch paths must converge on the same shape."""
    from rpg.generate import GenConfig, generate

    out = tmp_path / "s.parquet"
    stream.consume(max_messages=500, consumer=FakeConsumer(_messages(100)),
                   topic="t", out_path=out)
    streamed = pd.read_parquet(out)
    batch = generate(GenConfig(n_receipts=100, seed=7))
    assert set(batch.columns) == set(streamed.columns)
    assert str(streamed["submitted_at"].dtype).startswith("datetime64")


def test_consume_skips_error_messages(tmp_path):
    expected = _expected_rows(50)
    msgs = _messages(50)
    msgs.insert(10, FakeMessage(b"", None, _error="broker transport failure"))
    out = tmp_path / "s.parquet"
    n = stream.consume(max_messages=500, consumer=FakeConsumer(msgs),
                       topic="t", out_path=out)
    assert n == expected, "error frames must be skipped, not written"


def test_consume_respects_max_messages(tmp_path):
    out = tmp_path / "s.parquet"
    n = stream.consume(max_messages=25, consumer=FakeConsumer(_messages(200)),
                       topic="t", out_path=out)
    assert n == 25


def test_consume_commits_offsets_only_after_a_successful_write(tmp_path):
    """The at-least-once guarantee.

    If the parquet write fails, offsets must NOT be committed -- otherwise the
    batch is acknowledged and permanently lost. Point the output at a directory
    that cannot be written to force the failure.
    """
    c = FakeConsumer(_messages(30))
    bad_path = tmp_path / "does" / "not" / "exist" / "s.parquet"
    with pytest.raises(OSError):
        stream.consume(max_messages=100, consumer=c, topic="t", out_path=bad_path)
    assert c.commits == 0, "offsets committed despite a failed write"
    assert c.closed, "consumer must still be closed on failure"


def test_consume_commits_once_on_success(tmp_path):
    c = FakeConsumer(_messages(30))
    stream.consume(max_messages=100, consumer=c, topic="t",
                   out_path=tmp_path / "s.parquet")
    assert c.commits == 1


def test_empty_topic_writes_nothing_and_does_not_commit(tmp_path):
    out = tmp_path / "s.parquet"
    c = FakeConsumer([])
    n = stream.consume(max_messages=100, consumer=c, topic="t", out_path=out)
    assert n == 0
    assert not out.exists()
    assert c.commits == 0


# ------------------------------------------------------------ round trip ---
def test_produce_consume_round_trip_preserves_receipts(tmp_path):
    expected = _expected_rows(120, seed=11)
    p = RecordingProducer()
    stream.produce(n_receipts=120, seed=11, producer=p, topic="t")
    queue = [FakeMessage(v, k) for _t, k, v in p.messages]

    out = tmp_path / "s.parquet"
    n = stream.consume(max_messages=500, consumer=FakeConsumer(queue),
                       topic="t", out_path=out)
    assert n == expected

    from rpg.generate import GenConfig, generate

    original = generate(GenConfig(n_receipts=120, seed=11))
    landed = pd.read_parquet(out)
    assert set(landed["receipt_id"]) == set(original["receipt_id"])
    # Totals must survive JSON encode/decode without float drift
    merged = original[["receipt_id", "total"]].merge(
        landed[["receipt_id", "total"]], on="receipt_id", suffixes=("_o", "_l")
    )
    assert (merged["total_o"] == merged["total_l"]).all()
