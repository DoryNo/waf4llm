# Frozen benchmark holdout

The evaluation set is **frozen and reproducible**: fixed public sources,
fixed seed (42), deduplicated by lowercase text, shuffled once.
`data/` is gitignored, so the exact artifact is rebuilt from these commands;
`benchmarks/freeze_eval_set.py` is deterministic (same sha256 on every run).

Reproduce:

```bash
mkdir -p data/eval/raw && cd data/eval/raw
curl -L -o harmful_behaviors.csv \
  https://raw.githubusercontent.com/llm-attacks/llm-attacks/main/data/advbench/harmful_behaviors.csv
curl -L -o _jbb_dsn.json \
  https://raw.githubusercontent.com/JailbreakBench/artifacts/main/attack-artifacts/DSN/white_box/vicuna-13b-v1.5.json
curl -L -o _jbb_gcg.json \
  https://raw.githubusercontent.com/JailbreakBench/artifacts/main/attack-artifacts/GCG/white_box/vicuna-13b-v1.5.json
curl -L -o _jbb_jbc.json \
  https://raw.githubusercontent.com/JailbreakBench/artifacts/main/attack-artifacts/JBC/manual/vicuna-13b-v1.5.json
curl -L -o _jbb_pair.json \
  https://raw.githubusercontent.com/JailbreakBench/artifacts/main/attack-artifacts/PAIR/black_box/vicuna-13b-v1.5.json
curl -L -o prompt_injections.parquet \
  https://huggingface.co/api/datasets/deepset/prompt-injections/parquet/default/train/0.parquet
curl -L -o alpaca.parquet \
  https://huggingface.co/api/datasets/yahma/alpaca-cleaned/parquet/default/train/0.parquet
cd ../../..
python benchmarks/freeze_eval_set.py
python benchmarks/run_eval.py
```

## Composition (1460 rows: 737 attacks / 723 benign)

| Source | Label | Rows | Notes |
|---|---|---|---|
| advbench | attack | 200 | sample of 520 harmful behaviors (harmful, **not** injection-style) |
| jbb_dsn | attack | 95 | JailbreakBench DSN, vicuna-13b, jailbroken=True only |
| jbb_gcg | attack | 80 | JailbreakBench GCG, vicuna-13b, jailbroken=True only |
| jbb_jbc | attack | 90 | JailbreakBench JBC manual, vicuna-13b, jailbroken=True only |
| jbb_pair | attack | 69 | JailbreakBench PAIR, vicuna-13b, jailbroken=True only |
| pi | attack | 203 | deepset/prompt-injections, label=1 (in-domain injections) |
| pi | benign | 343 | deepset/prompt-injections, label=0 |
| alpaca | benign | 380 | sample of 51 760 Alpaca-cleaned instructions |

Deterministic sha256 of `data/eval/benchmark_holdout.jsonl`
(seed 42, one line per row: `{"id", "source", "label", "text"}`):

```
9fdef19a8285986014d114984bdd52c439d00fcb2905d2e3057c46b0eeb8a36b
```

## Caveats

* **Contamination (optimistic bias).** The default classifier
  (`protectai/deberta-v3-base-prompt-injection-v2`) is trained on public
  injection corpora that overlap `deepset/prompt-injections`; the `pi` rows are
  in-domain and partially in the classifier's training distribution. AdvBench,
  JailbreakBench and Alpaca rows are out-of-domain for the model.
* **Scope (pessimistic bias).** AdvBench rows are harmful requests without any
  injection wrapper; a pre-inference *injection* WAF legitimately passes many of
  them to the LLM, which is why advbench recall is near zero — that number is
  reported as-is and discussed in the README Limitations.
* **Threshold tuning was done on this holdout.** The perplexity scorer's
  operating point (`HEURISTIC_PERPLEXITY_MIN_CHARS=60`, ppl mapping
  >8000/1500/800) was selected in a single pass by inspecting the holdout
  distribution itself: benign English text (alpaca) sits mostly below ~500 ppl,
  while GCG/DSN adversarial suffixes start above ~800, and short non-English
  snippets (German texts inside the `pi` benign set) produce spuriously high
  values that a min-length guard removes. Treat the `full+ppl` row as an
  optimistic estimate: the thresholds were chosen on the same data they are
  reported against (no held-out threshold set was carved out).

## The perplexity config (`full+ppl`)

`full+ppl` enables the same heuristic + classifier pipeline plus the
distilgpt2-based perplexity scorer (`HEURISTIC_PERPLEXITY_ENABLED=true`).
It targets adversarial-obfuscation attacks (GCG suffixes, DSN artifacts)
whose token sequence is highly improbable under a small LM trained on
natural text — a signal regex rules and the injection classifier both miss.
Natural-language attacks (JBC, PAIR role-play) have normal perplexity and
are not affected by this layer.

Measured impact on the frozen holdout (see README table):

* GCG detection 61.3% → 97.5%, DSN 93.7% → 97.9%; JBC/PAIR unchanged.
* FPR 0.7% → 1.1% (8 of 723 benign rows; alpaca 3/380, pi-benign 5/343).
* Latency on CPU: adds ~300 ms p95 (distilgpt2 forward pass per request).
