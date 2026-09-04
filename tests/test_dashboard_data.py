from __future__ import annotations

import json

import pandas as pd

from dashboard.data import (
    confidence_histogram,
    decision_counts,
    detections_by_layer_level,
    fetch_layer_latencies,
    filter_recent,
    potential_fp_rate,
    to_sync_url,
    top_trigger_tokens,
)


def _requests_df(rows: list[dict]) -> pd.DataFrame:
    return pd.DataFrame(
        rows,
        columns=["id", "created_at", "decision", "confidence", "canary_hit", "latency_ms"],
    )


def test_to_sync_url_mapping():
    assert to_sync_url("sqlite+aiosqlite:///./waf.db") == "sqlite+pysqlite:///./waf.db"
    assert (
        to_sync_url("postgresql+asyncpg://u:p@h:5432/waf") == "postgresql+psycopg2://u:p@h:5432/waf"
    )
    assert to_sync_url("postgresql://h/db") == "postgresql://h/db"


def test_decision_counts_orders_known_decisions():
    df = _requests_df(
        [
            {
                "id": "1",
                "created_at": "2026-01-01T00:00:00Z",
                "decision": "allow",
                "confidence": 0.0,
                "canary_hit": False,
                "latency_ms": 1,
            },
            {
                "id": "2",
                "created_at": "2026-01-01T00:01:00Z",
                "decision": "block",
                "confidence": 0.9,
                "canary_hit": False,
                "latency_ms": 2,
            },
            {
                "id": "3",
                "created_at": "2026-01-01T00:02:00Z",
                "decision": "block",
                "confidence": 0.95,
                "canary_hit": False,
                "latency_ms": 3,
            },
        ]
    )
    counts = decision_counts(df)
    assert counts["block"] == 2
    assert counts["allow"] == 1
    assert counts["sanitize"] == 0  # reindexed, not dropped


def test_confidence_histogram_bins():
    df = _requests_df(
        [
            {
                "id": "1",
                "created_at": "2026-01-01T00:00:00Z",
                "decision": "allow",
                "confidence": 0.0,
                "canary_hit": False,
                "latency_ms": 1,
            },
            {
                "id": "2",
                "created_at": "2026-01-01T00:01:00Z",
                "decision": "block",
                "confidence": 0.99,
                "canary_hit": False,
                "latency_ms": 2,
            },
        ]
    )
    hist = confidence_histogram(df, bins=10)
    assert hist.sum() == 2
    assert len(hist) >= 1


def test_top_trigger_tokens_parses_json_and_counts():
    dets = pd.DataFrame(
        {
            "request_id": ["1", "2", "3"],
            "trigger_tokens": [
                json.dumps(["ignore", "instructions"]),
                json.dumps(["ignore"]),
                None,
            ],
        }
    )
    top = top_trigger_tokens(dets)
    assert top["ignore"] == 2
    assert top["instructions"] == 1


def test_detections_by_layer_level_pivot():
    dets = pd.DataFrame(
        {
            "request_id": ["1", "2"],
            "layer": ["heuristic", "heuristic"],
            "level": ["block", "log"],
        }
    )
    pivot = detections_by_layer_level(dets)
    assert pivot.loc["heuristic", "block"] == 1
    assert pivot.loc["heuristic", "log"] == 1


def test_potential_fp_rate():
    req = _requests_df(
        [
            {
                "id": "1",
                "created_at": "2026-01-01T00:00:00Z",
                "decision": "allow",
                "confidence": 0.4,
                "canary_hit": False,
                "latency_ms": 1,
            },
            {
                "id": "2",
                "created_at": "2026-01-01T00:01:00Z",
                "decision": "block",
                "confidence": 0.9,
                "canary_hit": False,
                "latency_ms": 2,
            },
        ]
    )
    dets = pd.DataFrame(
        {"request_id": ["1", "2"], "layer": ["heuristic"] * 2, "level": ["log", "block"]}
    )
    fp = potential_fp_rate(req, dets)
    assert fp is not None
    # request 1 has detections but was allowed -> 1/2
    assert fp == 0.5


def test_filter_recent_empty_and_all():
    df = _requests_df(
        [
            {
                "id": "1",
                "created_at": "2026-01-01T00:00:00Z",
                "decision": "allow",
                "confidence": 0.0,
                "canary_hit": False,
                "latency_ms": 1,
            },
        ]
    )
    assert filter_recent(df, None).equals(df)
    assert filter_recent(pd.DataFrame(), 24).empty


def test_fetch_layer_latencies_parses_prometheus_text():
    body = (
        "# HELP waf_pipeline_layer_duration_sum x\n"
        "# TYPE waf_pipeline_layer_duration_sum counter\n"
        'waf_pipeline_layer_duration_sum{layer="heuristic"} 0.5\n'
        'waf_pipeline_layer_duration_count{layer="heuristic"} 10\n'
        'waf_pipeline_layer_duration_sum{layer="classifier"} 0.2\n'
        'waf_pipeline_layer_duration_count{layer="classifier"} 4\n'
    )

    class FakeResp:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def read(self):
            return body.encode()

    import dashboard.data as d

    orig = d.urlopen

    def fake_urlopen(url, timeout):
        return FakeResp()

    d.urlopen = fake_urlopen
    try:
        lat = fetch_layer_latencies("http://fake/metrics")
    finally:
        d.urlopen = orig

    assert lat["heuristic"] == 50.0
    assert lat["classifier"] == 50.0


def test_fetch_layer_latencies_unreachable_returns_empty():
    lat = fetch_layer_latencies("http://127.0.0.1:1/metrics", timeout=0.2)
    assert lat.empty
