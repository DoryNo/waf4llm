from __future__ import annotations

import os

import pytest

streamlit = pytest.importorskip("streamlit")
from streamlit.testing.v1 import AppTest  # noqa: E402


@pytest.fixture()
def dashboard_env(tmp_path, monkeypatch):
    db_path = tmp_path / "dash.db"
    import sqlite3

    con = sqlite3.connect(db_path)
    con.executescript(
        """
        CREATE TABLE request_logs (
            id TEXT PRIMARY KEY, created_at TEXT, method TEXT DEFAULT 'POST',
            path TEXT, model TEXT, upstream_model TEXT, decision TEXT,
            confidence REAL, canary_token TEXT, canary_hit INTEGER DEFAULT 0,
            latency_ms INTEGER, request_body TEXT, response_body TEXT
        );
        CREATE TABLE detections (
            id TEXT PRIMARY KEY, request_id TEXT, created_at TEXT, layer TEXT,
            level TEXT, confidence REAL, trigger_tokens TEXT, detail TEXT
        );
        INSERT INTO request_logs (id, created_at, path, decision, confidence, canary_hit, latency_ms)
        VALUES
            ('r1', datetime('now'), '/v1/chat/completions', 'allow', 0.0, 0, 5),
            ('r2', datetime('now'), '/v1/chat/completions', 'block', 0.95, 0, 30),
            ('r3', datetime('now'), '/v1/chat/completions', 'allow', 0.42, 0, 8);
        INSERT INTO detections (id, request_id, created_at, layer, level, confidence, trigger_tokens)
        VALUES
            ('d1', 'r2', datetime('now'), 'heuristic', 'block', 0.95, '["ignore", "instructions"]'),
            ('d2', 'r3', datetime('now'), 'normalizer', 'log', 0.42, NULL);
        """
    )
    con.commit()
    con.close()

    monkeypatch.setenv("DATABASE_URL", f"sqlite+aiosqlite:///{db_path.as_posix()}")
    from src.config.settings import reset_settings_cache

    reset_settings_cache()
    yield
    reset_settings_cache()


def test_dashboard_app_renders_without_errors(dashboard_env, monkeypatch):
    app = AppTest.from_file(
        os.path.join(os.path.dirname(__file__), "..", "dashboard", "app.py"),
        default_timeout=30,
    )
    # Point the sidebar at the test DB and metrics at an unreachable endpoint
    app.run()
    assert not app.exception, f"app raised: {app.exception}"

    # The empty-state warning path: fresh AppTest uses default DB url from settings
    # which is the tmp DB — it has rows, so charts render instead of st.stop().
    assert "Dashboard" in app.title[0].value or "WAF" in app.title[0].value
    assert app.metric  # KPI row rendered


def test_dashboard_app_empty_state_stops_cleanly(tmp_path, monkeypatch):
    db_path = tmp_path / "empty.db"
    import sqlite3

    con = sqlite3.connect(db_path)
    con.executescript(
        """
        CREATE TABLE request_logs (id TEXT PRIMARY KEY, created_at TEXT, decision TEXT, confidence REAL, canary_hit INTEGER, latency_ms INTEGER);
        CREATE TABLE detections (id TEXT PRIMARY KEY, request_id TEXT, created_at TEXT, layer TEXT, level TEXT, confidence REAL, trigger_tokens TEXT);
        """
    )
    con.commit()
    con.close()

    monkeypatch.setenv("DATABASE_URL", f"sqlite+aiosqlite:///{db_path.as_posix()}")
    from src.config.settings import reset_settings_cache

    reset_settings_cache()
    try:
        app = AppTest.from_file(
            os.path.join(os.path.dirname(__file__), "..", "dashboard", "app.py"),
            default_timeout=30,
        )
        app.run()
        assert not app.exception
        assert "Записей нет" in " ".join(w.value for w in app.warning)
    finally:
        reset_settings_cache()
