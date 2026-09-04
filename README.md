# WAF for LLM — Anti-Injection Proxy

Proxy between the client and the LLM with multi‑layered protection against prompt injection / jailbreak.

## Benchmarks (frozen holdout)

Honest metrics on a frozen holdout built from public datasets —
1460 requests (737 attacks / 723 benign), fixed seed, no per-test threshold
tuning (methodology and provenance: [benchmarks/eval_dataset.md](benchmarks/eval_dataset.md)):

| Configuration | Detection rate (TPR) | FPR | Precision | F1 | Accuracy | p50, ms | p95, ms |
|---|---|---|---|---|---|---|---|
| heuristic-only (39 YAML rules, ~regex) | 1.1% | 0.0% | 1.000 | 0.021 | 0.50 | 0.5 | 4.3 |
| classifier-only (DeBERTa-v3) | 46.8% | 0.7% | 0.986 | 0.635 | 0.73 | 105 | 360 |
| full-pipeline (heuristics + classifier) | 47.2% | 0.7% | 0.986 | 0.638 | 0.73 | 95 | 350 |

Takeaways:

* **The regex baseline is nearly useless on real attacks**: 39 rules catch
  1.1% of attacks (with 0% FP). Obfuscated and role-play jailbreaks do not
  match keyword patterns — which is exactly what motivates the ML layer.
* **The classifier detects 77.5% of verified jailbreaks** (259 of 334; by
  JailbreakBench/vicuna method: JBC 100%, DSN 93.7%, GCG 61.3%, PAIR 44.9%)
  at 0.7% FPR on benign traffic (Alpaca + in-domain benign).
* **In-domain injections (deepset/prompt-injections): 17.0%** — short texts
  and noisy labels; part of this dataset also overlaps the classifier's
  training distribution (contamination, see Limitations).
* **AdvBench (harmful but not injection-style prompts): 0%** — an honest
  scope boundary: this is a pre-inference WAF against prompt injection, not a
  harmful-content filter (see Limitations).

Reproduce:

```bash
python benchmarks/freeze_eval_set.py   # build the frozen holdout
python benchmarks/run_eval.py          # run the 3 configurations
```

## Pipeline
```
user msg + history + RAG chunks + tool outputs
  → Provenance Tagging
  → Anti-Evasion Normalization
  → Fast Heuristic Filter
  → BERT Classifier (per-tag, per-turn)
  → Decision Engine (graduated response)
  → LLM inference (spotlighting)
  → Output Guard (canary + drift)
```

## Quick start
```bash
cp .env.example .env
# edit UPSTREAM_API_KEY / UPSTREAM_BASE_URL

# with docker
docker compose up --build

# without docker (sqlite fallback)
pip install -e ".[dev]"
uvicorn src.main:app --reload

# apply schema migrations explicitly (recommended outside local dev)
alembic upgrade head
```

## Endpoints
- `POST /v1/chat/completions` — OpenAI-compatible proxy
- `POST /v1/completions` — legacy completions proxy
- `POST /admin/feedback` — report a misclassified text for retraining (Phase 10)
- `GET /health` — health check
- `GET /metrics` — Prometheus metrics
- `GET /ready` — readiness (DB/Redis/upstream)

The default classifier and perplexity detector are disabled so the base install
starts without downloading ML models. Enable them explicitly in `.env`; install
the optional ML dependencies with `pip install -e ".[ml]"`.

## Limitations

Honest boundaries of applicability — where the system does not help, or where
the measurements carry caveats:

* **Streaming: HTTP 200 is already sent.** In streaming mode the proxy stops
  forwarding tokens as soon as it catches a canary leak in the output, but the
  HTTP `200` status has already been sent to the client by then. The output
  guard guarantee ("the response never reaches the client") is only strong in
  non-streaming mode.
* **Scope: injections, not harmful content.** The pre-inference WAF catches
  prompt injection / jailbreak, not topic-level harmfulness. AdvBench (direct
  harmful requests with no injection wrapper) is detected at 0% — a deliberate
  trade-off; harmful content would need a separate safety classifier.
* **In-domain injections are detected weakly (17%).** On the short texts of
  deepset/prompt-injections the classifier only reliably catches explicit
  constructions; part of the dataset overlaps its training distribution
  (the estimate is optimistic), and noisy labels inflate FN.
* **Adversarial obfuscation partially breaks through the ML layer.** GCG
  suffixes are caught at 61%, PAIR role-play prompts at 45%: generated attacks
  trade readability for resistance to classifiers. Heuristics do not help here
  (0%), so the roadmap includes a perplexity detector and fine-tuning on
  collected borderline cases (Phase 10).
* **CPU latency.** The classifier (DeBERTa-v3-base) on CPU: p50 ~105 ms,
  p95 ~360 ms per request (in-process measurement, batch = 1 request).
  Production latencies need a GPU or ONNX export.
* **Benchmark latency is in-process.** The measurements cover the cost of
  `run_pre_inference` without the network hop through the proxy (no HTTP
  overhead), i.e. a lower bound on end-to-end latency.

## Project layout
```
src/
  config/        # pydantic-settings
  observability/ # logging, metrics, tracing, alerts (Phase 9.3)
  db/            # SQLAlchemy models, session, request/detection log writer
  proxy/         # FastAPI app, upstream client, streaming
  pipeline/      # orchestrator + layers (provenance, normalizer, heuristic, classifier, decision, output_guard)
  retrain/       # continuous retraining: collector, dataset, train, registry (Phase 10)
tests/
rules/           # heuristic YAML rules
benchmarks/      # frozen-holdout eval (freeze_eval_set.py, run_eval.py), latency harness, results.json
dashboard/       # Streamlit (Phase 9.1)
```

## Dashboard (Phase 9.1)
The proxy persists every request outcome into `request_logs` and `detections`
(best-effort; disable with `LOG_REQUESTS_ENABLED=false`).

```bash
pip install -e ".[dashboard]"
streamlit run dashboard/app.py
```

The dashboard shows decision distribution, confidence histogram, detections
per layer/level, top trigger tokens, potential-FP share (allowed requests that
had detections) and per-layer latency parsed from `/metrics`. Point the sidebar
`DATABASE_URL` at the same DB the proxy uses.

## XAI / Explainability (Phase 8)
When the BERT classifier (`ENABLE_CLASSIFIER=true`) scores a request, token
importance is computed on the same forward pass (CLS-attention, no extra
latency pass) and persisted into the `attributions` table + dashboard:

```bash
XAI_ATTENTION_ENABLED=true   # attention importance on the hot path
XAI_TOP_K=10                 # words kept per attribution
XAI_STORE_ENABLED=true       # persist to attributions table (dashboard section)
```

Note: attention weights require the eager attention implementation — the
classifier loads its model with `attn_implementation="eager"` automatically
when XAI is enabled.

Integrated Gradients (Captum) is heavier (~`XAI_IG_STEPS` forward passes) and
is run on demand only:

```bash
python -m src.pipeline.xai "ignore all previous instructions" --method integrated_gradients
python -m src.pipeline.xai "hello, how are you?" --method attention
```

Requires the `ml` extra: `pip install -e ".[ml]"` (torch, transformers, captum).

## Alerts (Phase 9.3)
Set in `.env`:
```bash
ALERTS_ENABLED=true
ALERTS_WEBHOOK_URL=https://hooks.slack.com/services/XXX/YYY/ZZZ
ALERTS_MIN_LEVEL=block   # log | sanitize | exclude | block
```
Slack-compatible JSON (`text` + structured `waf_alert`) is POSTed when a request
is blocked (heuristic/classifier/decision) or a canary leak is caught. Bursts
are rate-limited (30s cooldown).

## Continuous retraining (Phase 10)
A closed loop for improving the classifier from production traffic:

1. **Collector** (`RETRAIN_ENABLED=true`) — best-effort enqueue of blocked/
   borderline request texts (user segments only, system prompts excluded) and
   user-reported corrections into the `retrain_queue` table:

   ```bash
   curl -X POST localhost:8791/admin/feedback -H "Content-Type: application/json" \
     -d '{"text": "that request was fine", "label": "benign", "request_id": "..."}'
   ```

2. **Dataset build** — drains the queue and merges public JSONL corpora
   (`{text,label}` lines, one of `text|prompt|content` + `label|is_injection`)
   into deduplicated train/val splits:

   ```bash
   python -m src.retrain.dataset            # RETRAIN_CORPUS_DIR + queue -> data/datasets
   python -m src.retrain.dataset --no-queue # corpora only
   ```

3. **Fine-tune** — trains a sequence-classification head from the base model:

   ```bash
   python -m src.retrain.train --epochs 2 --device cpu --register
   ```

4. **Registry** — versioned artifacts with promote/rollback
   (`models/registry.json`):

   ```bash
   python -m src.retrain.registry list
   python -m src.retrain.registry promote v20250101-120000
   ```

Requires the `ml` extra for step 3. Queue items are marked `consumed` after a
successful build, so nothing is trained on twice.

## Development process

The commit history in this repository is compressed (the base phases were
built on a single branch), so below is the actual development path: what was
done, what broke, and how it was fixed. Every phase was closed with tests;
at publication time there are 156 tests (`pytest -q`, all green) plus ruff
and mypy in CI.

| Phase | What was built | Verification |
|---|---|---|
| 0-2 | Proxy skeleton, provenance tagging, anti-evasion normalization (unicode/NFC/homoglyphs/zero-width) | unit tests per layer |
| 3 | Heuristic layer: 39 YAML rules, versioned rules_mtime persisted in detections | rule tests + granular levels |
| 4 | Classifier layer (DeBERTa-v3), per-tag scoring | tests with a mock model |
| 5 | Decision engine (graduated response: allow/log/sanitize/exclude/block), orchestrator | pipeline e2e tests |
| 6 | Output guard: canary injection + canary-drift check | canary-path tests |
| 7 | XAI: attention attributions on the hot path, Integrated Gradients (Captum) on-demand CLI | tests + live token checks |
| 8 | XAI persistence (attributions) + dashboard section | tests + live check through the proxy |
| 9 | Observability: request/detection logs, Streamlit dashboard, Slack alerts | integration tests |
| 10 | Retrain loop: collector → dataset → fine-tune → registry (promote/rollback) | 14 tests, live run |

Real debugging iterations (the kind reviewers usually look for):

* **transformers 5.x broke compatibility** — `BertTokenizer` no longer loads
  by model name; it needed initialization via a `vocab=` dict. SDPA attention
  returns `None` instead of weights — the model is loaded with
  `attn_implementation="eager"` whenever XAI is enabled.
* **The Captum IntegratedGradients contract** — the forward function must
  return the correct shape; attention tensors had to be sliced per batch
  manually (`attention[:, :, 0, :]` — the CLS row) so batch inference would
  not attribute other requests' tokens.
* **HF cache recovery after a broken snapshot** — a blob named by etag +
  hardlink into `snapshots/<rev>/` instead of an endlessly hung
  `snapshot_download`; verified by loading in offline mode.
* **Fail-open vs fail-closed per layer** (decided in the orchestrator: a layer
  error → block under `FAIL_MODE=closed`, allow+log when open) — covered by
  tests.
* **Live runs** against the real proxy endpoint: injection → 403 block,
  benign → the pipeline lets it through (allow), attribution tokens are
  written to the DB, a borderline case lands in the retrain queue and reaches
  `train.jsonl`.

How to reproduce the full verification cycle:

```bash
pip install -e ".[dev]"
pytest -q                          # 156 tests
ruff check . && mypy src
python benchmarks/freeze_eval_set.py
python benchmarks/run_eval.py      # the metrics from the table above
```
