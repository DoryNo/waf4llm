from __future__ import annotations

import json
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from src.config.settings import Settings, get_settings
from src.observability.logging import get_logger

logger = get_logger("retrain.dataset")

_LABEL_MAP = {
    "benign": 0,
    "safe": 0,
    "clean": 0,
    "0": 0,
    "injection": 1,
    "malicious": 1,
    "attack": 1,
    "1": 1,
}


@dataclass
class Example:
    text: str
    label: int  # 0 = benign, 1 = injection
    source: str = "unknown"

    def to_dict(self) -> dict[str, Any]:
        return {"text": self.text, "label": self.label, "source": self.source}


@dataclass
class DatasetSplit:
    train: list[Example] = field(default_factory=list)
    val: list[Example] = field(default_factory=list)

    def stats(self) -> dict[str, Any]:
        def _counts(examples: list[Example]) -> dict[str, int]:
            pos = sum(1 for e in examples if e.label == 1)
            return {"total": len(examples), "injection": pos, "benign": len(examples) - pos}

        return {"train": _counts(self.train), "val": _counts(self.val)}


def normalize_label(value: Any) -> int | None:
    """Coerce corpus labels (0/1, benign/injection, safe/malicious) to 0/1."""
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, (int, float)) and value in (0, 1):
        return int(value)
    if isinstance(value, str):
        return _LABEL_MAP.get(value.strip().lower())
    return None


def _coerce_example(raw: dict[str, Any], source: str) -> Example | None:
    """Accept the several JSONL shapes used by public prompt-injection corpora."""
    text = raw.get("text") or raw.get("prompt") or raw.get("content") or raw.get("request")
    if not isinstance(text, str) or not text.strip():
        return None
    label = normalize_label(raw.get("label", raw.get("is_injection", raw.get("attack"))))
    if label is None:
        return None
    return Example(text=text.strip()[:4096], label=label, source=source)


def load_corpus_file(path: str | Path) -> list[Example]:
    """Load one JSONL corpus file; malformed lines are skipped."""
    path = Path(path)
    examples: list[Example] = []
    try:
        for line in path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                raw = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(raw, dict):
                example = _coerce_example(raw, source=path.stem)
                if example:
                    examples.append(example)
    except OSError as e:
        logger.warning("corpus file unreadable", path=str(path), error=str(e))
    return examples


def load_corpora(corpus_dir: str | Path) -> list[Example]:
    """Merge all *.jsonl corpora in a directory (JailbreakBench / AdvBench / HackAPrompt dumps)."""
    corpus_dir = Path(corpus_dir)
    examples: list[Example] = []
    if corpus_dir.is_dir():
        for path in sorted(corpus_dir.glob("*.jsonl")):
            loaded = load_corpus_file(path)
            logger.info("corpus loaded", path=str(path), examples=len(loaded))
            examples.extend(loaded)
    return examples


def dedupe(examples: list[Example]) -> list[Example]:
    """First occurrence wins; text is the identity."""
    seen: set[str] = set()
    unique: list[Example] = []
    for example in examples:
        key = example.text.strip().lower()
        if key in seen:
            continue
        seen.add(key)
        unique.append(example)
    return unique


def split_dataset(examples: list[Example], val_split: float = 0.1, seed: int = 42) -> DatasetSplit:
    """Stratified-ish shuffled split; keeps both classes in val when possible."""
    rng = random.Random(seed)
    shuffled = examples[:]
    rng.shuffle(shuffled)
    positives = [e for e in shuffled if e.label == 1]
    negatives = [e for e in shuffled if e.label == 0]
    val_n_pos = round(len(positives) * val_split)
    val_n_neg = round(len(negatives) * val_split)
    val = positives[:val_n_pos] + negatives[:val_n_neg]
    train = positives[val_n_pos:] + negatives[val_n_neg:]
    return DatasetSplit(train=train, val=val)


async def load_queue_examples(settings: Settings | None = None) -> list[Example]:
    """Drain pending items from the retrain queue DB table (10.1 -> 10.2)."""
    from sqlalchemy import select

    from src.db.models import RetrainItem
    from src.db.session import _get_sessionmaker

    cfg = settings or get_settings()
    examples: list[Example] = []
    try:
        maker = _get_sessionmaker()
        async with maker() as session:
            rows = (
                (
                    await session.execute(
                        select(RetrainItem)
                        .where(RetrainItem.status == "pending")
                        .order_by(RetrainItem.created_at)
                        .limit(20000)
                    )
                )
                .scalars()
                .all()
            )
            label_map = {"injection": 1, "benign": 0}
            for row in rows:
                label = label_map.get(row.label)
                if label is None or not row.text.strip():
                    continue
                examples.append(Example(text=row.text, label=label, source=row.source or "queue"))
    except Exception as e:
        logger.warning("queue load failed", error=str(e))
    _ = cfg
    return examples


async def mark_queue_consumed(items: list[Example], settings: Settings | None = None) -> int:
    """Flag queue rows as consumed after a successful dataset build."""
    from sqlalchemy import update

    from src.db.models import RetrainItem
    from src.db.session import _get_sessionmaker

    texts = [e.text for e in items]
    if not texts:
        return 0
    try:
        maker = _get_sessionmaker()
        async with maker() as session:
            result = await session.execute(
                update(RetrainItem)
                .where(RetrainItem.text.in_(texts), RetrainItem.status == "pending")
                .values(status="consumed")
            )
            await session.commit()
            return int(result.rowcount or 0)
    except Exception as e:
        logger.warning("queue consume mark failed", error=str(e))
        return 0


async def build_dataset(
    *,
    output_dir: str | Path | None = None,
    include_queue: bool = True,
    settings: Settings | None = None,
) -> dict[str, Any]:
    """Merge queue + public corpora into train.jsonl / val.jsonl. Returns stats."""
    cfg = settings or get_settings()
    examples: list[Example] = []
    if include_queue:
        queue_examples = await load_queue_examples(cfg)
        logger.info("queue examples", count=len(queue_examples))
        examples.extend(queue_examples)
    corpus_examples = load_corpora(cfg.retrain_corpus_dir)
    logger.info("corpus examples", count=len(corpus_examples))
    examples.extend(corpus_examples)

    examples = dedupe(examples)
    split = split_dataset(examples, val_split=cfg.retrain_val_split, seed=cfg.retrain_seed)

    out_dir = Path(output_dir or cfg.retrain_dataset_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for name, part in (("train.jsonl", split.train), ("val.jsonl", split.val)):
        with (out_dir / name).open("w", encoding="utf-8") as f:
            for example in part:
                f.write(json.dumps(example.to_dict(), ensure_ascii=False) + "\n")

    if include_queue:
        consumed = await mark_queue_consumed(split.train + split.val, cfg)
        logger.info("queue consumed", count=consumed)

    stats = {"corpora_dir": str(cfg.retrain_corpus_dir), **split.stats()}
    logger.info("dataset built", **stats)
    return stats


# ---------------------------------------------------------------------------
# CLI: python -m src.retrain.dataset
# ---------------------------------------------------------------------------


def _cli() -> int:
    import argparse
    import asyncio

    parser = argparse.ArgumentParser(
        prog="python -m src.retrain.dataset",
        description="Merge retrain queue + corpora into train/val JSONL.",
    )
    parser.add_argument("--corpus-dir", default=None, help="override RETRAIN_CORPUS_DIR")
    parser.add_argument("--out-dir", default=None, help="override RETRAIN_DATASET_DIR")
    parser.add_argument("--no-queue", action="store_true", help="skip DB queue")
    args = parser.parse_args()

    settings = get_settings()
    if args.corpus_dir:
        settings.retrain_corpus_dir = args.corpus_dir
    if args.out_dir:
        settings.retrain_dataset_dir = args.out_dir

    stats = asyncio.run(build_dataset(include_queue=not args.no_queue, settings=settings))
    print(json.dumps(stats, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(_cli())
