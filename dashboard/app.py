from __future__ import annotations

import os
import sys

import pandas as pd
import streamlit as st

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dashboard import data as d  # noqa: E402
from src.config.settings import get_settings  # noqa: E402

st.set_page_config(page_title="WAF Dashboard", page_icon="🛡️", layout="wide")

settings = get_settings()
default_db = settings.database_url
default_metrics = "http://localhost:8000/metrics"


@st.cache_data(ttl=5)
def cached_load(url: str) -> d.DashboardData:
    return d.load_db_data(url)


@st.cache_data(ttl=5)
def cached_latencies(metrics_url: str) -> pd.Series:
    return d.fetch_layer_latencies(metrics_url)


st.title("🛡️ WAF для LLM — Dashboard")
st.caption("Phase 9.1 — наблюдаемость: решения, confidence, триггеры, FP, латентность слоёв")

with st.sidebar:
    st.header("Источники данных")
    db_url = st.text_input("DATABASE_URL", value=default_db)
    metrics_url = st.text_input("Metrics endpoint", value=default_metrics)
    window_hours = st.selectbox(
        "Период",
        options=[1, 6, 24, 168, None],
        format_func=lambda h: "всё время" if h is None else f"последние {h} ч",
        index=2,
    )
    st.divider()
    st.caption(
        "Данные пишутся прокси в таблицы request_logs / detections. "
        "Латентность слоёв берётся из Prometheus-эндпоинта."
    )

try:
    raw = cached_load(db_url)
except Exception as e:
    st.error(f"Не удалось прочитать БД: {e}")
    st.info("Проверьте DATABASE_URL (sqlite/ postgres). Пример: sqlite+pysqlite:///./waf.db")
    st.stop()

requests = d.filter_recent(raw.requests, window_hours)
detections = d.filter_recent(raw.detections, window_hours)
attributions = d.filter_recent(raw.attributions, window_hours)

if requests.empty:
    st.warning("Записей нет. Сделайте запросы через прокси — они появятся здесь.")
    st.stop()

# --- KPI row ---
c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("Запросов", len(requests))
blocked = int((requests["decision"] == "block").sum())
c2.metric("Заблокировано", blocked, f"{blocked / max(len(requests), 1):.1%}")
c3.metric("Sanitize/Exclude", int(requests["decision"].isin(["sanitize", "exclude"]).sum()))
c4.metric("Canary-хиты", int(requests["canary_hit"].sum()))
lat = requests["latency_ms"].dropna()
c5.metric("Латентность (avg)", f"{lat.mean():.0f} ms" if not lat.empty else "—")

left, right = st.columns(2)

with left:
    st.subheader("Распределение решений")
    st.bar_chart(d.decision_counts(requests))

    st.subheader("Распределение confidence")
    st.bar_chart(d.confidence_histogram(requests))

with right:
    st.subheader("Детекции по слоям и уровням")
    pivot = d.detections_by_layer_level(detections)
    if pivot.empty:
        st.info("Детекций за период нет")
    else:
        st.bar_chart(pivot)

    fp = d.potential_fp_rate(requests, detections)
    st.subheader("Вероятные ложные срабатывания")
    if fp is None:
        st.info("Недостаточно данных")
    else:
        st.metric("Доля разрешённых с детекциями", f"{fp:.1%}")
        st.caption(
            "Прокси-метрика FP: запросы, где детекции были, но решение — allow. "
            "Точная оценка FP появится с retrain-очередью (Phase 10)."
        )

st.subheader("Топ триггер-токенов")
tokens = d.top_trigger_tokens(detections)
if tokens.empty:
    st.info("Триггер-токены не зафиксированы")
else:
    st.bar_chart(tokens)

st.subheader("Латентность по слоям (из Prometheus)")
latencies = cached_latencies(metrics_url)
if latencies.empty:
    st.info(f"Метрики недоступны по {metrics_url}. Запустите прокси и укажите корректный endpoint.")
else:
    st.bar_chart(latencies)
    st.dataframe(latencies.rename("mean_ms").to_frame().round(2))

st.subheader("Последние детекции")
st.dataframe(d.recent_detections(detections), use_container_width=True)

st.subheader("Атрибуции XAI — важность токенов")
st.caption(
    "Phase 8: attention-based важность токенов (hot path) и Integrated Gradients (on-demand)"
)
if attributions.empty:
    st.info(
        "Атрибуций нет. Они появляются, когда классификатор (ENABLE_CLASSIFIER=true) "
        "и XAI_ATTENTION_ENABLED=true — или через `python -m src.pipeline.xai`."
    )
else:
    top_xai = d.top_attributed_tokens(attributions)
    col_x1, col_x2 = st.columns(2)
    with col_x1:
        st.markdown("**Топ токенов по суммарной важности**")
        if top_xai.empty:
            st.info("Токены не зафиксированы")
        else:
            st.bar_chart(top_xai)
    with col_x2:
        st.markdown("**Методы**")
        st.bar_chart(attributions["method"].value_counts())
    st.markdown("**Последние атрибуции**")
    st.dataframe(d.recent_attributions(attributions), use_container_width=True)

if st.button("🔄 Обновить данные"):
    cached_load.clear()
    cached_latencies.clear()
    st.rerun()
