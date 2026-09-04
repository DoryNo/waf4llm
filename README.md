# WAF для LLM — Anti-Injection Proxy

Proxy between the client and the LLM with multi‑layered protection against prompt injection / jailbreak.

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
the optional ML dependencies with `pip install -e ".[ml]"`. In streaming mode,
the proxy can stop forwarding after a canary leak is detected, but HTTP status
has already been sent as `200`, so non-streaming mode gives the strongest output
guard guarantee.

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
benchmarks/      # latency/accuracy harness and reports
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
