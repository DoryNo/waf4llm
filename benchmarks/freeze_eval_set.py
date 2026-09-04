"""Freeze the reproducible benchmark holdout from public datasets.

Sources (all public, downloaded once into data/eval/raw/, which is gitignored):
  * AdvBench harmful_behaviors.csv  - 520 direct harmful requests (no jailbreak wrapper).
  * JailbreakBench artifacts (vicuna-13b-v1.5) - adversarial jailbreak prompts per
    method (DSN, GCG white-box; JBC manual, PAIR black-box), jailbroken==True only
    => ~334 verified successful attacks.
  * deepset/prompt-injections - in-domain injection corpus (train parquet, 546 rows;
    203 label=1 injections, 343 label=0 benign). NOTE: partially overlaps the
    protectai classifier training distribution -> contamination documented in README.
  * Alpaca-cleaned - benign user instructions (sampled).

Output: data/eval/benchmark_holdout.jsonl, one row per line:
  {"id": ..., "source": ..., "label": 0|1, "text": ...}   label 1 = attack
Deterministic: fixed seed 42 sampling; dedupe on lowercase text (first wins).
"""

from __future__ import annotations

import csv
import hashlib
import json
import random
from pathlib import Path

import pyarrow.parquet as pq

RAW = Path("data/eval/raw")
OUT = Path("data/eval/benchmark_holdout.jsonl")
SEED = 42

ADVBENCH_SAMPLE = 200
ALPACA_SAMPLE = 380
JBB_METHODS = ("dsn", "gcg", "jbc", "pair")


def _clean(text: str) -> str | None:
    text = (text or "").strip()
    if not text or len(text) < 5:
        return None
    return " ".join(text.split())[:4000]


def load_advbench() -> list[dict]:
    rows: list[dict] = []
    with (RAW / "harmful_behaviors.csv").open(encoding="utf-8-sig", newline="") as fh:
        for r in csv.DictReader(fh):
            text = _clean(r.get("goal") or "")
            if text:
                rows.append({"source": "advbench", "label": 1, "text": text})
    return rows


def load_jbb() -> list[dict]:
    """Keep only jailbroken==True artifacts: verified successful attacks."""
    rows: list[dict] = []
    for method in JBB_METHODS:
        data = json.loads((RAW / f"_jbb_{method}.json").read_text(encoding="utf-8"))
        for item in data["jailbreaks"]:
            if not item.get("jailbroken"):
                continue
            text = _clean(item.get("prompt") or "")
            if text:
                rows.append({"source": f"jbb_{method}", "label": 1, "text": text})
    return rows


def load_prompt_injections() -> list[dict]:
    rows: list[dict] = []
    for r in pq.read_table(str(RAW / "prompt_injections.parquet")).to_pylist():
        text = _clean(r["text"])
        if text:
            rows.append({"source": "pi", "label": int(r["label"]), "text": text})
    return rows


def load_alpaca(rng: random.Random, take: int) -> list[dict]:
    table = pq.read_table(str(RAW / "alpaca.parquet")).to_pylist()
    rows: list[dict] = []
    idx = list(range(len(table)))
    rng.shuffle(idx)
    for i in idx:
        r = table[i]
        parts = [r.get("instruction") or "", r.get("input") or ""]
        text = _clean("\n".join(p for p in parts if p.strip()))
        if text:
            rows.append({"source": "alpaca", "label": 0, "text": text})
        if len(rows) >= take:
            break
    return rows


def main() -> None:
    rng = random.Random(SEED)

    jbb = load_jbb()
    pi = load_prompt_injections()

    advbench = load_advbench()
    rng.shuffle(advbench)
    # JBB first: adversarial prompts are the harder class, so they win dedupe
    # against their own plain-text goals inside advbench.
    selected = [*jbb, *pi, *advbench[:ADVBENCH_SAMPLE], *load_alpaca(rng, ALPACA_SAMPLE)]

    seen: set[str] = set()
    rows: list[dict] = []
    for r in selected:
        key = r["text"].lower()
        if key in seen:
            continue
        seen.add(key)
        rows.append(r)

    rng.shuffle(rows)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with OUT.open("w", encoding="utf-8") as fh:
        for i, r in enumerate(rows):
            fh.write(json.dumps({"id": i, **r}, ensure_ascii=False) + "\n")

    payload = OUT.read_bytes()
    by_source: dict[str, dict[str, int]] = {}
    for r in rows:
        s = by_source.setdefault(r["source"], {"pos": 0, "neg": 0})
        s["pos" if r["label"] == 1 else "neg"] += 1

    print(f"rows={len(rows)}")
    for src in sorted(by_source):
        s = by_source[src]
        print(f"  {src:10s} pos={s['pos']:4d} neg={s['neg']:4d}")
    print("sha256:", hashlib.sha256(payload).hexdigest())


if __name__ == "__main__":
    main()
