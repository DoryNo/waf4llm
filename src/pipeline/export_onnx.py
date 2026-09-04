"""Export classifier to ONNX — Phase 5.4.

Usage:
  python -m src.pipeline.export_onnx --model protectai/deberta-v3-base-prompt-injection-v2 --output models/classifier.onnx
Requires: pip install -e ".[ml]"  (torch, transformers, onnx, onnxruntime)
"""

from __future__ import annotations

import argparse
import importlib
from pathlib import Path


def export_classifier(model_name: str, output_path: str, opset_version: int = 14) -> str:
    try:
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
    except ImportError as e:
        raise RuntimeError("transformers not installed. Run pip install -e '.[ml]'") from e

    try:
        import torch  # noqa: F401
    except ImportError as e:
        raise RuntimeError("torch not installed. Run pip install -e '.[ml]'") from e

    print(f"Loading tokenizer/model: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=False)
    model = AutoModelForSequenceClassification.from_pretrained(model_name, trust_remote_code=False)
    model.eval()

    dummy = tokenizer("hello world", return_tensors="pt")
    input_ids = dummy["input_ids"]
    attention_mask = dummy["attention_mask"]

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)

    print(f"Exporting to ONNX: {output} (opset={opset_version})")
    # Dynamic batch/sequence
    torch.onnx.export(
        model,
        (input_ids, attention_mask),
        str(output),
        input_names=["input_ids", "attention_mask"],
        output_names=["logits"],
        dynamic_axes={
            "input_ids": {0: "batch", 1: "sequence"},
            "attention_mask": {0: "batch", 1: "sequence"},
            "logits": {0: "batch"},
        },
        opset_version=opset_version,
        do_constant_folding=True,
    )
    print(f"ONNX exported: {output} ({output.stat().st_size} bytes)")
    # Verify with onnxruntime
    try:
        ort = importlib.import_module("onnxruntime")
        np = importlib.import_module("numpy")

        sess = ort.InferenceSession(str(output), providers=["CPUExecutionProvider"])
        # Quick sanity check
        enc = tokenizer(
            ["hello world", "ignore previous instructions"],
            padding=True,
            truncation=True,
            return_tensors="np",
        )
        ort_inputs = {
            "input_ids": enc["input_ids"].astype(np.int64),
            "attention_mask": enc["attention_mask"].astype(np.int64),
        }
        logits = sess.run(None, ort_inputs)[0]
        print(f"ONNX sanity check logits shape: {logits.shape}")
        print("ONNX verification passed")
    except Exception as e:
        print(f"ONNX verification skipped/failed: {e}")

    return str(output)


def main():
    parser = argparse.ArgumentParser(description="Export classifier to ONNX")
    parser.add_argument(
        "--model", default="protectai/deberta-v3-base-prompt-injection-v2", help="HF model id"
    )
    parser.add_argument("--output", default="models/classifier.onnx", help="Output ONNX path")
    parser.add_argument("--opset", type=int, default=14, help="ONNX opset version")
    args = parser.parse_args()
    export_classifier(args.model, args.output, args.opset)


if __name__ == "__main__":
    main()
