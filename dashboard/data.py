from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any
from urllib.request import urlopen

import pandas as pd
from sqlalchemy import create_engine, text

_DECISIONS = ["allow", "log", "sanitize", "exclude", "block"]


def to_sync_url(async_url: str) -> str:
    """Map the app's async SQLAlchemy URLs to sync drivers for the dashboard."""
    if async_url.startswith("sqlite+aiosqlite"):
        return async_url.replace("sqlite+aiosqlite", "sqlite+pysqlite", 1)
    if async_url.startswith("postgresql+asyncpg"):
        return async_url.replace("postgresql+asyncpg", "postgresql+psycopg2", 1)
    return async_url


@dataclass
class DashboardData:
    requests: pd.DataFrame
    detections: pd.DataFrame
    attributions: pd.DataFrame


def load_db_data(url: str) -> DashboardData:
    engine = create_engine(to_sync_url(url))
    try:
        with engine.connect() as conn:
            requests = pd.read_sql(text("SELECT * FROM request_logs"), conn)
            detections = pd.read_sql(text("SELECT * FROM detections"), conn)
            try:
                attributions = pd.read_sql(text("SELECT * FROM attributions"), conn)
            except Exception:
                # Older DB without the 0002 migration — show empty XAI section.
                attributions = pd.DataFrame()
    finally:
        engine.dispose()
    return DashboardData(requests=requests, detections=detections, attributions=attributions)


def _parse_ts(series: pd.Series) -> pd.Series:
    return pd.to_datetime(series, utc=True, errors="coerce", format="mixed")


def filter_recent(df: pd.DataFrame, hours: float | None) -> pd.DataFrame:
    if df.empty or hours is None:
        return df
    ts = _parse_ts(df["created_at"])
    cutoff = pd.Timestamp.now(tz="UTC") - pd.Timedelta(hours=hours)
    return df[ts >= cutoff]


def decision_counts(requests: pd.DataFrame) -> pd.Series:
    if requests.empty:
        return pd.Series(dtype=int)
    counts = requests["decision"].value_counts()
    return counts.reindex(_DECISIONS, fill_value=0)


def confidence_histogram(requests: pd.DataFrame, bins: int = 20) -> pd.Series:
    if requests.empty:
        return pd.Series(dtype=int)
    conf = requests["confidence"].dropna()
    if conf.empty:
        return pd.Series(dtype=int)
    edges = [i / bins for i in range(bins + 1)]
    intervals = pd.cut(conf, bins=edges, include_lowest=True)
    counts = intervals.value_counts().sort_index()
    labels = [f"{interval.left:.2f}–{interval.right:.2f}" for interval in counts.index]
    return pd.Series(counts.to_numpy(), index=labels)


def _load_tokens(cell: Any) -> list[str]:
    if cell is None or (isinstance(cell, float) and pd.isna(cell)):
        return []
    try:
        value = json.loads(cell)
        return [str(v) for v in value] if isinstance(value, list) else []
    except (TypeError, ValueError):
        return []


def top_trigger_tokens(detections: pd.DataFrame, top: int = 15) -> pd.Series:
    if detections.empty:
        return pd.Series(dtype=int)
    counter: dict[str, int] = {}
    for cell in detections["trigger_tokens"]:
        for token in _load_tokens(cell):
            key = token.strip().lower()
            if key:
                counter[key] = counter.get(key, 0) + 1
    if not counter:
        return pd.Series(dtype=int)
    series = pd.Series(counter).sort_values(ascending=False).head(top)
    return series


def detections_by_layer_level(detections: pd.DataFrame) -> pd.DataFrame:
    if detections.empty:
        return pd.DataFrame()
    pivot = detections.pivot_table(
        index="layer", columns="level", values="request_id", aggfunc="count", fill_value=0
    )
    level_order = [lvl for lvl in _DECISIONS if lvl in pivot.columns]
    other = [c for c in pivot.columns if c not in level_order]
    return pivot[level_order + other]


def potential_fp_rate(requests: pd.DataFrame, detections: pd.DataFrame) -> float | None:
    """Share of flagged requests that were allowed — a proxy for false positives
    until user feedback (Phase 10 retrain queue) provides ground truth."""
    if requests.empty or detections.empty:
        return None
    flagged_ids = set(detections["request_id"].dropna())
    req = requests.dropna(subset=["decision"])
    allowed_flagged = req[(req["id"].isin(flagged_ids)) & (req["decision"] == "allow")]
    if req.empty:
        return None
    return len(allowed_flagged) / len(req)


_METRIC_LINE = re.compile(r"^(?P<name>\w+)(?:\{(?P<labels>[^}]*)\})?\s+(?P<value>[0-9.eE+]+)$")


def _label_value(labels: str, key: str) -> str | None:
    for pair in labels.split(","):
        if "=" not in pair:
            continue
        k, _, v = pair.partition("=")
        if k.strip() == key:
            return v.strip().strip('"')
    return None


def fetch_layer_latencies(metrics_url: str, timeout: float = 5.0) -> pd.Series:
    """Mean per-layer pipeline latency (ms) parsed from the /metrics endpoint."""
    sums: dict[str, float] = {}
    counts: dict[str, float] = {}
    try:
        with urlopen(metrics_url, timeout=timeout) as resp:
            body = resp.read().decode("utf-8", errors="replace")
    except Exception:
        return pd.Series(dtype=float)
    for line in body.splitlines():
        match = _METRIC_LINE.match(line.strip())
        if match is None:
            continue
        name = match.group("name")
        labels = match.group("labels") or ""
        layer = _label_value(labels, "layer")
        if layer is None:
            continue
        try:
            value = float(match.group("value"))
        except ValueError:
            continue
        if name == "waf_pipeline_layer_duration_sum":
            sums[layer] = value
        elif name == "waf_pipeline_layer_duration_count":
            counts[layer] = value
    result = {
        layer: (sums.get(layer, 0.0) / counts[layer]) * 1000.0
        for layer, count in counts.items()
        if count > 0
    }
    series = pd.Series(result, name="mean_ms")
    return series.sort_values(ascending=False)


def recent_detections(detections: pd.DataFrame, limit: int = 50) -> pd.DataFrame:
    if detections.empty:
        return pd.DataFrame()
    df = detections.copy()
    df["created_at"] = _parse_ts(df["created_at"])
    df = df.sort_values("created_at", ascending=False).head(limit)
    cols = [
        c for c in ("created_at", "layer", "level", "confidence", "request_id") if c in df.columns
    ]
    return df[cols]


def top_attributed_tokens(attributions: pd.DataFrame, top: int = 15) -> pd.Series:
    """Aggregate XAI token scores across requests (Phase 8.1 UI)."""
    if attributions.empty:
        return pd.Series(dtype=float)
    counter: dict[str, float] = {}
    for cell in attributions["tokens"]:
        for item in _load_token_dicts(cell):
            token = str(item.get("token", "")).strip().lower()
            score = item.get("score")
            if token and isinstance(score, (int, float)):
                counter[token] = counter.get(token, 0.0) + float(score)
    if not counter:
        return pd.Series(dtype=float)
    return pd.Series(counter).sort_values(ascending=False).head(top)


def _load_token_dicts(cell: Any) -> list[dict[str, Any]]:
    if cell is None or (isinstance(cell, float) and pd.isna(cell)):
        return []
    try:
        value = json.loads(cell) if isinstance(cell, str) else cell
        return [v for v in value if isinstance(v, dict)] if isinstance(value, list) else []
    except (TypeError, ValueError):
        return []


def recent_attributions(attributions: pd.DataFrame, limit: int = 50) -> pd.DataFrame:
    if attributions.empty:
        return pd.DataFrame()
    df = attributions.copy()
    df["created_at"] = _parse_ts(df["created_at"])
    df = df.sort_values("created_at", ascending=False).head(limit)
    df["top_tokens"] = df["tokens"].apply(
        lambda cell: ", ".join(str(item.get("token", "")) for item in _load_token_dicts(cell)[:5])
    )
    cols = [
        c
        for c in (
            "created_at",
            "request_id",
            "method",
            "model",
            "score",
            "top_tokens",
        )
        if c in df.columns
    ]
    return df[cols]
