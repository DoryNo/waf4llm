from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

from src.config.settings import get_settings
from src.observability.logging import get_logger

logger = get_logger("retrain.train")


def _load_split(path: str | Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return rows


def pr_auc(scores: list[float], labels: list[int]) -> float:
    """Average precision (area under precision-recall curve), pure Python."""
    pairs = sorted(zip(scores, labels, strict=True), key=lambda p: -p[0])
    total_pos = sum(labels) or 1
    tp = 0.0
    fp = 0.0
    ap = 0.0
    prev_recall = 0.0
    for _score, label in pairs:
        if label == 1:
            tp += 1
        else:
            fp += 1
        precision = tp / (tp + fp)
        recall = tp / total_pos
        ap += precision * (recall - prev_recall)
        prev_recall = recall
    return ap


def evaluate(
    model: Any, tokenizer: Any, examples: list[dict[str, Any]], **cfg: Any
) -> dict[str, float]:
    """Run inference on val examples; return accuracy/precision/recall/F1/PR-AUC."""
    import torch

    device = cfg.get("device", "cpu")
    model.eval()
    y_true: list[int] = []
    y_pred: list[int] = []
    scores: list[float] = []
    batch_size = int(cfg.get("batch_size", 16))
    max_length = int(cfg.get("max_length", 512))
    with torch.no_grad():
        for i in range(0, len(examples), batch_size):
            batch = examples[i : i + batch_size]
            texts = [r["text"] for r in batch]
            enc = tokenizer(
                texts, padding=True, truncation=True, max_length=max_length, return_tensors="pt"
            ).to(device)
            logits = model(**enc).logits
            probs = torch.softmax(logits, dim=-1)[:, 1]
            scores.extend(float(p) for p in probs.tolist())
            y_pred.extend(int(p) for p in (probs > 0.5).tolist())
            y_true.extend(int(r["label"]) for r in batch)

    tp = sum(1 for t, p in zip(y_true, y_pred, strict=True) if t == 1 and p == 1)
    fp = sum(1 for t, p in zip(y_true, y_pred, strict=True) if t == 0 and p == 1)
    fn = sum(1 for t, p in zip(y_true, y_pred, strict=True) if t == 1 and p == 0)
    tn = sum(1 for t, p in zip(y_true, y_pred, strict=True) if t == 0 and p == 0)
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "accuracy": (tp + tn) / len(y_true) if y_true else 0.0,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "pr_auc": pr_auc(scores, y_true),
    }


def train_model(
    dataset_dir: str | Path,
    output_dir: str | Path,
    *,
    base_model: str | None = None,
    epochs: int = 2,
    lr: float = 2e-5,
    batch_size: int = 16,
    max_length: int = 512,
    device: str = "cpu",
    seed: int = 42,
) -> dict[str, Any]:
    """Fine-tune the prompt-injection classifier. Requires the `ml` extra."""
    import torch
    from torch.utils.data import DataLoader, Dataset
    from transformers import (
        AutoModelForSequenceClassification,
        AutoTokenizer,
        get_linear_schedule_with_warmup,
    )

    torch.manual_seed(seed)

    base = base_model or get_settings().classifier_model
    train_rows = _load_split(Path(dataset_dir) / "train.jsonl")
    val_rows = _load_split(Path(dataset_dir) / "val.jsonl")
    if not train_rows:
        raise ValueError(f"empty train split: {dataset_dir}/train.jsonl")
    logger.info("training", train=len(train_rows), val=len(val_rows), base=base, device=device)

    tokenizer = AutoTokenizer.from_pretrained(base, trust_remote_code=False)
    model = AutoModelForSequenceClassification.from_pretrained(
        base, num_labels=2, trust_remote_code=False, attn_implementation="eager"
    ).to(device)

    class TextDataset(Dataset):
        def __init__(self, rows: list[dict[str, Any]]) -> None:
            self.rows = rows

        def __len__(self) -> int:
            return len(self.rows)

        def __getitem__(self, idx: int) -> tuple[str, int]:
            row = self.rows[idx]
            return row["text"], int(row["label"])

    loader = DataLoader(
        TextDataset(train_rows), batch_size=batch_size, shuffle=True, drop_last=False
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    total_steps = max(1, len(loader) * epochs)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=0, num_training_steps=total_steps
    )

    model.train()
    t0 = time.monotonic()
    step = 0
    for epoch in range(epochs):
        for texts, labels in loader:
            enc = tokenizer(
                list(texts),
                padding=True,
                truncation=True,
                max_length=max_length,
                return_tensors="pt",
            ).to(device)
            out = model(**enc, labels=torch.tensor(list(labels), device=device))
            out.loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            step += 1
            if step % 20 == 0:
                logger.info(
                    "train step",
                    epoch=epoch,
                    step=step,
                    loss=round(float(out.loss), 4),
                    elapsed=round(time.monotonic() - t0, 1),
                )

    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(out_path)
    tokenizer.save_pretrained(out_path)

    metrics: dict[str, Any] = {}
    if val_rows:
        metrics = evaluate(
            model, tokenizer, val_rows, device=device, batch_size=batch_size, max_length=max_length
        )
        logger.info("eval done", **{k: round(v, 4) for k, v in metrics.items()})

    report = {
        "base_model": base,
        "epochs": epochs,
        "lr": lr,
        "batch_size": batch_size,
        "train_size": len(train_rows),
        "val_size": len(val_rows),
        "duration_seconds": round(time.monotonic() - t0, 1),
        "metrics": metrics,
    }
    (out_path / "train_report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return report


def _cli() -> int:
    import argparse

    parser = argparse.ArgumentParser(
        prog="python -m src.retrain.train",
        description="Fine-tune the prompt-injection classifier on a merged dataset.",
    )
    parser.add_argument("--dataset-dir", default="data/datasets")
    parser.add_argument("--out-dir", default=None, help="artifact dir (default models/v<ts>)")
    parser.add_argument("--base-model", default=None)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--register", action="store_true", help="register result in model registry")
    args = parser.parse_args()

    settings = get_settings()
    out_dir = args.out_dir or f"models/v{time.strftime('%Y%m%d-%H%M%S')}"
    report = train_model(
        args.dataset_dir,
        out_dir,
        base_model=args.base_model,
        epochs=args.epochs,
        lr=args.lr,
        batch_size=args.batch_size,
        max_length=args.max_length,
        device=args.device,
        seed=args.seed or settings.retrain_seed,
    )
    if args.register:
        from src.retrain.registry import ModelRegistry

        version = Path(out_dir).name
        ModelRegistry().register(
            version,
            out_dir,
            base_model=report["base_model"],
            metrics=report["metrics"],
            dataset_stats={"train": report["train_size"], "val": report["val_size"]},
        )
        print(
            f"registered: {version} (promote with: python -m src.retrain.registry promote {version})"
        )
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(_cli())
