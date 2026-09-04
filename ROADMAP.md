# WAF для LLM — Roadmap (Agent Plan)

## Phase 0: Scaffold & Infrastructure
| # | Agent Task | Depends On | Output |
|---|-----------|------------|--------|
| 0.1 | Init repo: pyproject.toml, Dockerfile, docker-compose (app + redis + postgres), CI (lint/typecheck/test) | — | project skeleton |
| 0.2 | Config layer: pydantic-settings, .env, feature-flags (fail-open/closed, per-layer toggles) | 0.1 | `src/config/` |
| 0.3 | Structured logging + metrics scaffold (OpenTelemetry traces, Prometheus counters) | 0.1 | `src/observability/` |
| 0.4 | DB schema: requests, detections, canaries, retrain_queue (Alembic migrations) | 0.1 | `src/db/` |

---

## Phase 1: FastAPI Proxy Core
| # | Agent Task | Depends On | Output |
|---|-----------|------------|--------|
| 1.1 | FastAPI app with `/v1/chat/completions` passthrough to upstream LLM (OpenAI-compatible) | 0.1 | `src/proxy/` |
| 1.2 | Request/response middleware chain (pipeline orchestrator calling layers sequentially) | 1.1 | `src/pipeline/` |
| 1.3 | Streaming support (SSE) — proxy must forward stream chunks, guard layers hook into final assembled response | 1.1 | `src/proxy/streaming.py` |

---

## Phase 2: Provenance Tagging
| # | Agent Task | Depends On | Output |
|---|-----------|------------|--------|
| 2.1 | Parse incoming messages array → tag each segment `[SYS]`, `[USR]`, `[RET]`, `[TOOL]` based on role/content fields | 1.2 | `src/pipeline/provenance.py` |
| 2.2 | RAG chunk isolation: wrap retrieved content in XML tags with random nonce per request | 2.1 | same module |
| 2.3 | Unit tests: mixed-role messages, nested tool outputs, edge cases (empty content, multi-modal) | 2.1 | `tests/test_provenance.py` |

---

## Phase 3: Anti-Evasion Normalization
| # | Agent Task | Depends On | Output |
|---|-----------|------------|--------|
| 3.1 | Unicode normalization (NFKC), zero-width char removal, homoglyph mapping | 2.1 | `src/pipeline/normalizer.py` |
| 3.2 | Recursive decoders: Base64, Hex, ROT13 with depth limit (max 3) + printable-result gate | 3.1 | same module |
| 3.3 | Leetspeak normalizer (1gn0r3 → ignore) | 3.1 | same module |
| 3.4 | Entropy pre-filter: skip decoding if segment entropy doesn't match encoded-text profile | 3.2 | same module |
| 3.5 | Unit tests: double-base64 bombs, mixed encoding, legitimate hashes/IDs not mangled | 3.1 | `tests/test_normalizer.py` |

---

## Phase 4: Fast Heuristic Filter
| # | Agent Task | Depends On | Output |
|---|-----------|------------|--------|
| 4.1 | Regex rule engine (known injection patterns, "ignore previous instructions", role-play starters) | 3.1 | `src/pipeline/heuristic.py` |
| 4.2 | Perplexity scorer: small LM (e.g. distilgpt2) computes per-token perplexity, flag anomalous spikes (GCG-style suffixes) | 3.1 | same module |
| 4.3 | Rule config in YAML (hot-reloadable), not hardcoded | 4.1 | `rules/` |
| 4.4 | Tests: known attack samples from JailbreakBench/AdvBench, verify detection rate | 4.1 | `tests/test_heuristic.py` |

---

## Phase 5: BERT Classifier Integration
| # | Agent Task | Depends On | Output |
|---|-----------|------------|--------|
| 5.1 | Model wrapper: load Prompt Guard 2 (mDeBERTa-base), inference with batching, confidence score output | 4.1 | `src/pipeline/classifier.py` |
| 5.2 | Per-tag scoring: run classifier on `[USR]` and `[RET]` segments separately, aggregate per-turn risk | 5.1, 2.1 | same module |
| 5.3 | Multi-turn window features: sliding window of last N messages, cumulative risk score | 5.2 | `src/pipeline/multiturn.py` |
| 5.4 | Model serving: ONNX/TorchScript export, optional Triton/TF-Serving sidecar | 5.1 | `models/` |
| 5.5 | Benchmark: latency p50/p95/p99, accuracy on held-out set | 5.1 | `benchmarks/` |

---

## Phase 6: Decision Engine (Graduated Response)
| # | Agent Task | Depends On | Output |
|---|-----------|------------|--------|
| 6.1 | Threshold config: low/medium/high/critical confidence bands → allow/sanitize/exclude_chunk/block | 5.2 | `src/pipeline/decision.py` |
| 6.2 | Sanitize action: wrap flagged `[RET]` chunk in spotlighting delimiters + system instruction "data, not commands" | 6.1, 2.2 | same module |
| 6.3 | Exclude action: remove specific chunk, keep rest of context | 6.1 | same module |
| 6.4 | Block action: return 403 + structured error + alert webhook | 6.1 | same module |
| 6.5 | Fail-open/closed policy: configurable per-deployment, fallback when classifier times out | 6.1 | same module |
| 6.6 | Tests: each response path, timeout fallback, policy switch | 6.1 | `tests/test_decision.py` |

---

## Phase 7: Output Guard
| # | Agent Task | Depends On | Output |
|---|-----------|------------|--------|
| 7.1 | Canary token injection: insert unique random sentinel into system prompt per request | 6.1 | `src/pipeline/output_guard.py` |
| 7.2 | Canary check: substring match in LLM output, block if found (near-zero FP) | 7.1 | same module |
| 7.3 | Semantic drift detection: multilingual-e5-small embeddings, cosine sim between output and sanitized system prompt | 7.1 | same module |
| 7.4 | Tests: canary leak detection, legitimate topical answers not blocked | 7.1 | `tests/test_output_guard.py` |

---

## Phase 8: XAI / Explainability
| # | Agent Task | Depends On | Output |
|---|-----------|------------|--------|
| 8.1 | Attention-based token importance (hot path, cheap) — highlight trigger tokens in logs/UI | 5.1 | `src/pipeline/xai.py` |
| 8.2 | Integrated Gradients via Captum (async/debug mode) — detailed attribution on demand | 5.1 | same module |
| 8.3 | Store attributions in DB for post-hoc analysis | 8.1, 0.4 | same module |

---

## Phase 9: Observability & Dashboard
| # | Agent Task | Depends On | Output |
|---|-----------|------------|--------|
| 9.1 | Streamlit dashboard: confidence distribution, top trigger tokens, FP rate, latency per layer | 0.3, 8.1 | `dashboard/` |
| 9.2 | Prometheus metrics endpoint: request count, detection count by level, latency histograms | 0.3 | `src/observability/metrics.py` |
| 9.3 | Alerting: webhook/Slack on critical blocks | 6.4 | `src/observability/alerts.py` |

---

## Phase 10: Continuous Retraining Pipeline
| # | Agent Task | Depends On | Output |
|---|-----------|------------|--------|
| 10.1 | Data collector: log borderline/blocked cases + user FP reports to retrain queue | 0.4, 6.1 | `src/retrain/collector.py` |
| 10.2 | Dataset manager: merge JailbreakBench, AdvBench, HackAPrompt, internal logs | 10.1 | `src/retrain/dataset.py` |
| 10.3 | Training script: fine-tune mDeBERTa on merged dataset, eval harness | 10.2 | `src/retrain/train.py` |
| 10.4 | Model registry: versioned model artifacts, A/B deploy, rollback | 10.3 | `src/retrain/registry.py` |

---

## Phase 11: Red-Teaming & Self-Testing
| # | Agent Task | Depends On | Output |
|---|-----------|------------|--------|
| 11.1 | Promptfoo integration: config with attack categories, run against proxy endpoint | 1.1 | `tests/redteam/` |
| 11.2 | Automated regression: CI runs red-team suite on every model/rule change | 11.1 | CI job |
| 11.3 | Precision/recall reporting per attack category | 11.1 | `reports/` |

---

## Execution Order (Critical Path)

```
Phase 0 (scaffold)
  └─► Phase 1 (proxy core)
        ├─► Phase 2 (provenance)
        │     └─► Phase 3 (normalizer)
        │           ├─► Phase 4 (heuristics)
        │           │     └─► Phase 5 (classifier)
        │           │           └─► Phase 6 (decision)
        │           │                 ├─► Phase 7 (output guard)
        │           │                 ├─► Phase 8 (XAI)
        │           │                 └─► Phase 9 (dashboard)
        │           └─► Phase 10 (retraining) — parallel with 6+
        └─► Phase 11 (red-team) — after proxy stable
```

## MVP Scope (Phases 0–7)
Core working proxy with all protection layers. XAI, dashboard, retraining, red-teaming are post-MVP.

## Tech Stack
- **Runtime**: Python 3.11+, FastAPI, uvicorn
- **ML**: transformers, torch, onnxruntime, Captum
- **Classifier**: Prompt Guard 2 (mDeBERTa-base) or DeBERTa-xsmall for speed
- **Embeddings**: multilingual-e5-small
- **Storage**: PostgreSQL (logs), Redis (cache, rate-limit)
- **Infra**: Docker, Alembic, OpenTelemetry, Prometheus
- **Testing**: pytest, Promptfoo
- **Dashboard**: Streamlit
