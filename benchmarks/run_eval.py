"""Evaluate WAF configs on the frozen holdout (benchmarks/freeze_eval_set.py).

Three configurations, all sharing provenance -> normalizer -> decision layers:
  heuristic-only  (ENABLE_HEURISTIC=true,  ENABLE_CLASSIFIER=false)
  classifier-only (ENABLE_HEURISTIC=false, ENABLE_CLASSIFIER=true)
  full-pipeline   (both enabled)

Prediction mapping (graduated response): block/exclude -> positive;
sanitize -> positive when confidence >= 0.5; log/allow -> negative.

Metrics per config: detection rate (TPR on attacks), FPR (on benign),
precision, recall, F1, accuracy, decision distribution and in-process
latency (mean/p50/p95/p99 ms). Per-source breakdown shows where each
configuration catches or misses attacks.

Usage:
  python benchmarks/run_eval.py [--configs heuristic,classifier,full]
                                [--holdout data/eval/benchmark_holdout.jsonl]
                                [--out benchmarks/results.json]

Requires the frozen holdout; see freeze_eval_set.py for sources.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import statistics
import time
from pathlib import Path

os.environ.setdefault("HF_HUB_OFFLINE", "1")
# The global get_settings() used by alert_block re-reads .env; pin overrides
# so no eval run POSTs blocks to a leftover local webhook.
os.environ["ALERTS_ENABLED"] = "false"
os.environ["ALERTS_WEBHOOK_URL"] = ""

from src.config.settings import Settings  # noqa: E402
from src.pipeline.base import PipelineContext  # noqa: E402
from src.pipeline.orchestrator import PipelineOrchestrator  # noqa: E402

HOLDOUT_DEFAULT = Path("data/eval/benchmark_holdout.jsonl")
OUT_DEFAULT = Path("benchmarks/results.json")

CONFIGS: dict[str, dict[str, str]] = {
    "heuristic-only": {"ENABLE_HEURISTIC": "true", "ENABLE_CLASSIFIER": "false"},
    "classifier-only": {"ENABLE_HEURISTIC": "false", "ENABLE_CLASSIFIER": "true"},
    "full-pipeline": {"ENABLE_HEURISTIC": "true", "ENABLE_CLASSIFIER": "true"},
}


def load_holdout(path: Path) -> list[dict]:
    rows: list[dict] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def predict(ctx: PipelineContext, sanitize_threshold: float = 0.5) -> tuple[int, str]:
    decision = ctx.decision.value
    if decision in ("block", "exclude"):
        return 1, decision
    if decision == "sanitize":
        return (1, decision) if ctx.confidence >= sanitize_threshold else (0, decision)
    return 0, decision


def pct(values: list[float], p: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    return ordered[min(int(len(ordered) * p / 100), len(ordered) - 1)]


def metrics_for(rows: list[dict], preds: list[int]) -> dict:
    tp = sum(1 for r, p in zip(rows, preds, strict=True) if r["label"] == 1 and p == 1)
    fp = sum(1 for r, p in zip(rows, preds, strict=True) if r["label"] == 0 and p == 1)
    tn = sum(1 for r, p in zip(rows, preds, strict=True) if r["label"] == 0 and p == 0)
    fn = sum(1 for r, p in zip(rows, preds, strict=True) if r["label"] == 1 and p == 0)
    total = max(tp + fp + tn + fn, 1)
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
        "accuracy": round((tp + tn) / total, 4),
    }


async def run_config(name: str, rows: list[dict]) -> dict:
    overrides = {
        "_env_file": None,
        "DATABASE_URL": "sqlite+aiosqlite:///./.eval_tmp.db",
        "REDIS_URL": "redis://localhost:6379/0",
        "LOG_REQUESTS_ENABLED": "false",
        "RETRAIN_ENABLED": "false",
        "XAI_STORE_ENABLED": "false",
        "XAI_ATTENTION_ENABLED": "false",
        **CONFIGS[name],
    }
    orch = PipelineOrchestrator(Settings(**overrides))

    preds: list[int] = []
    decisions: dict[str, int] = {}
    latencies: list[float] = []

    for row in rows:
        ctx = PipelineContext(raw_body={"messages": [{"role": "user", "content": row["text"]}]})
        start = time.perf_counter()
        await orch.run_pre_inference(ctx)
        latencies.append((time.perf_counter() - start) * 1000)

        pred, decision = predict(ctx)
        preds.append(pred)
        decisions[decision] = decisions.get(decision, 0) + 1

    base = metrics_for(rows, preds)
    latency = {
        "mean": round(statistics.mean(latencies), 2),
        "p50": round(pct(latencies, 50), 2),
        "p95": round(pct(latencies, 95), 2),
        "p99": round(pct(latencies, 99), 2),
        "max": round(max(latencies), 2),
    }

    # Per-source breakdown
    per_source: dict[str, dict] = {}
    for source in sorted({r["source"] for r in rows}):
        subset = [(r, p) for r, p in zip(rows, preds, strict=True) if r["source"] == source]
        n = len(subset)
        flagged = sum(p for _, p in subset)
        per_source[source] = {
            "n": n,
            "flagged": flagged,
            "rate": round(flagged / n, 4) if n else 0.0,
        }

    return {
        "config": name,
        "flags": CONFIGS[name],
        "metrics": base,
        "detection_rate": base["recall"],
        "fpr": round(base["fp"] / max(base["fp"] + base["tn"], 1), 4),
        "decisions": decisions,
        "latency_ms": latency,
        "per_source": per_source,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="WAF frozen-holdout evaluation")
    parser.add_argument("--holdout", type=Path, default=HOLDOUT_DEFAULT)
    parser.add_argument("--configs", type=str, default=",".join(CONFIGS))
    parser.add_argument("--out", type=Path, default=OUT_DEFAULT)
    args = parser.parse_args()

    rows = load_holdout(args.holdout)
    n_pos = sum(r["label"] for r in rows)
    print(f"holdout: {len(rows)} rows ({n_pos} attacks / {len(rows) - n_pos} benign)")

    selected = [c.strip() for c in args.configs.split(",") if c.strip()]
    results: dict[str, dict] = {}
    for name in selected:
        print(f"--- {name} ---")
        results[name] = asyncio.run(run_config(name, rows))
        m = results[name]["metrics"]
        print(
            f"  detection={results[name]['detection_rate']:.3f} "
            f"fpr={results[name]['fpr']:.3f} f1={m['f1']:.3f} "
            f"p95={results[name]['latency_ms']['p95']}ms"
        )

    payload = {
        "holdout": {
            "path": str(args.holdout),
            "rows": len(rows),
            "attacks": n_pos,
            "benign": len(rows) - n_pos,
        },
        "prediction_mapping": "block/exclude=1, sanitize(conf>=0.5)=1, log/allow=0",
        "configs": results,
    }
    args.out.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"written: {args.out}")


if __name__ == "__main__":
    main()
