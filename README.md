# WAF for LLM — Anti-Injection Proxy

Proxy between the client and the LLM with multi‑layered protection against prompt injection / jailbreak.

## Benchmarks (frozen holdout)

Честные метрики на замороженном holdout из публичных датасетов —
1460 запросов (737 атак / 723 benign), фиксированный seed, без подбора порогов
под тест (методология и провенанс: [benchmarks/eval_dataset.md](benchmarks/eval_dataset.md)):

| Конфигурация | Detection rate (TPR) | FPR | Precision | F1 | Accuracy | p50, ms | p95, ms |
|---|---|---|---|---|---|---|---|
| heuristic-only (правила, ~regex) | 1.1% | 0.0% | 1.000 | 0.021 | 0.50 | 0.5 | 4.3 |
| classifier-only (DeBERTa-v3) | 46.8% | 0.7% | 0.986 | 0.635 | 0.73 | 105 | 360 |
| full-pipeline (эвристики + классификатор) | 47.2% | 0.7% | 0.986 | 0.638 | 0.73 | 95 | 350 |

Выводы из этой таблицы:

* **regex-базовая линия почти бесполезна на реальных атаках**: 39 правил
  ловят 1.1% атак (и 0% FP). Обфусцированные и ролевые джейлбрейки не матчатся
  ключевыми словами — именно это мотивирует ML-слой.
* **Классификатор детектирует 77.5% верифицированных джейлбрейков**
  (259 из 334, по методам JailbreakBench/vicuna: JBC 100%, DSN 93.7%, GCG 61.3%,
  PAIR 44.9%) при FPR 0.7% на benign-трафике (Alpaca + in-domain benign).
* **In-domain инъекции (deepset/prompt-injections): 17.0%** — короткие тексты и
  шум разметки; плюс частичное пересечение с тренировочной выборкой
  (contamination, см. Limitations).
* **AdvBench (вредоносные, но не инъекционные промпты): 0%** — это честная
  граница скоупа pre-inference WAF против prompt injection, а не фильтр
  вредоносного контента (см. Limitations).

Воспроизведение:

```bash
python benchmarks/freeze_eval_set.py   # собрать замороженный holdout
python benchmarks/run_eval.py          # прогнать 3 конфигурации
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

Честные границы применимости — там, где система не помогает или измерена с оговорками:

* **Streaming: HTTP 200 уже отправлен.** В streaming-режиме прокси перестаёт
  пересылать токены, как только ловит canary-утечку в выводе, но HTTP-статус
  `200` к этому моменту уже отправлен клиенту. Гарантия output guard
  («запрос не дойдёт до LLM / ответ не уйдёт клиенту») сильна только в
  non-streaming режиме.
* **Скоуп: инъекции, а не вредоносный контент.** Pre-inference WAF ловит
  prompt injection / jailbreak, а не тематическую вредоносность. AdvBench
  (прямые вредоносные запросы без инъекционной обёртки) детектируется на 0% —
  это осознанный trade-off, для вредоносного контента нужен отдельный
  safety-классификатор.
* **In-domain инъекции детектируются слабо (17%).** На short-текстах
  deepset/prompt-injections классификатор уверенно ловит только явные
  конструкции; часть датасета пересекается с его тренировочной выборкой
  (оценка оптимистична), а шумная разметка завышает FN.
* **Adversarial-обфускация частично пробивает ML-слой.** GCG-суффиксы ловятся
  на 61%, PAIR-ролевые промпты — на 45%: генерируемые атаки разменивают
  читаемость на устойчивость к классификаторам. Эвристики тут не помогают
  (0%), поэтому в roadmap — perplexity-детектор и дообучение на collected
  borderline-кейсах (Phase 10).
* **Латентность на CPU.** Классификатор (DeBERTa-v3-base) на CPU: p50 ~105 мс,
  p95 ~360 мс на один запрос (замеры in-process, батч = 1 запрос). Для
  прод-латентностей нужен GPU или ONNX-экспорт.
* **Latency в benchmarks — in-process.** Замеры измеряют стоимость
  `run_pre_inference` без сетевого хопа прокси (без HTTP-overhead), т.е. это
  нижняя граница сквозной задержки.

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

История коммитов в репозитории сжата (базовые фазы шли одной веткой), поэтому
ниже — реальный путь разработки: что делалось, что ломалось и как чинилось.
Каждая фаза закрывалась тестами; на момент публикации — 156 тестов
(`pytest -q`, все зелёные), плюс ruff и mypy в CI.

| Фаза | Что сделано | Проверка |
|---|---|---|
| 0-2 | Скелет прокси, provenance-тегирование, anti-evasion normalization (unicode/NFC/homoglyphs/zero-width) | юнит-тесты на каждый слой |
| 3 | Heuristic layer: 39 YAML-правил, versioned rules_mtime в detections | тесты на правила + granular levels |
| 4 | Classifier layer (DeBERTa-v3), per-tag scoring | тесты с mock-моделью |
| 5 | Decision engine (graduated response: allow/log/sanitize/exclude/block), orchestrator | e2e-тесты пайплайна |
| 6 | Output guard: canary injection + canary-drift check | тесты canary путей |
| 7 | XAI: attention-атрибуции на горячем пути, Integrated Gradients (Captum) on-demand CLI | тесты + живая проверка токенов |
| 8 | XAI-персистенция (attributions) + dashboard-секция | тесты + live-проверка через прокси |
| 9 | Observability: request/detection logs, Streamlit dashboard, Slack-алерты | интеграционные тесты |
| 10 | Retrain loop: collector → dataset → fine-tune → registry (promote/rollback) | 14 тестов, live-прогон |

Реальные отладочные итерации (то, за чем обычно ходят в code review):

* **transformers 5.x сломал совместимость** — `BertTokenizer` больше не
  грузится по имени модели, потребовалась инициализация через `vocab=` dict;
  SDPA attention возвращает `None` вместо весов — модель грузится с
  `attn_implementation="eager"`, когда включён XAI.
* **Контракт Captum IntegratedGradients** — forward-функция должна возвращать
  корректную форму; пришлось срезать attention-тензоры по батчу вручную
  (`attention[:, :, 0, :]` — CLS-строка), чтобы батч-инференс не отдавал
  атрибуции чужих запросов.
* **Восстановление HF-кеша после битого снапшота** — blob с именем по etag +
  hardlink в `snapshots/<rev>/` вместо бесконечно зависшего
  `snapshot_download`; проверено загрузкой в offline-режиме.
* **Fail-open vs fail-closed** на уровне каждого слоя (решение в
  orchestrator: ошибка слоя → block при `FAIL_MODE=closed`, allow+лог при
  open) — покрыто тестами.
* **Живые прогоны** через реальный прокси-эндпоинт: инъекция → 403 block,
  benign → пайплайн пропускает (allow), attribution-токены пишутся в БД,
  borderline-кейс попадает в retrain-очередь и доезжает до
  `train.jsonl`.

Как воспроизвести весь цикл проверки:

```bash
pip install -e ".[dev]"
pytest -q                          # 156 тестов
ruff check . && mypy src
python benchmarks/freeze_eval_set.py
python benchmarks/run_eval.py      # метрики из таблицы выше
```
