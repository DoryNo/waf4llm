from __future__ import annotations

import pytest

from src.config.settings import reset_settings_cache
from src.db.log_writer import mark_canary_hit, persist_context
from src.db.session import init_db, reset_engine
from src.pipeline.base import DecisionLevel, LayerResult, PipelineContext


@pytest.fixture()
async def db_env(tmp_path):
    db_path = tmp_path / "log_test.db"
    import os

    old = os.environ.get("DATABASE_URL")
    os.environ["DATABASE_URL"] = f"sqlite+aiosqlite:///{db_path.as_posix()}"
    reset_settings_cache()
    reset_engine()
    await init_db()
    yield
    reset_engine()
    if old is None:
        os.environ.pop("DATABASE_URL", None)
    else:
        os.environ["DATABASE_URL"] = old
    reset_settings_cache()


def _ctx() -> PipelineContext:
    ctx = PipelineContext(request_id="req-log-1", route="/v1/chat/completions")
    ctx.raw_body = {"model": "gpt-4o-mini", "messages": [{"role": "user", "content": "hi"}]}
    ctx.add_result(
        LayerResult(
            layer="heuristic",
            passed=False,
            confidence=0.95,
            level=DecisionLevel.block,
            reason="matched rule",
            trigger_tokens=["ignore", "instructions"],
            extra={"rule_ids": ["r1"]},
        )
    )
    ctx.add_result(LayerResult(layer="provenance", passed=True))  # allow -> not stored
    ctx.decision = DecisionLevel.block
    ctx.confidence = 0.95
    ctx.block_reason = "matched rule"
    return ctx


async def test_persist_context_writes_request_and_detections(db_env):
    ctx = _ctx()
    await persist_context(ctx)

    from sqlalchemy import select

    from src.db.models import Detection, RequestLog
    from src.db.session import _get_sessionmaker

    maker = _get_sessionmaker()
    async with maker() as session:
        logs = (await session.execute(select(RequestLog))).scalars().all()
        dets = (await session.execute(select(Detection))).scalars().all()

    assert len(logs) == 1
    log = logs[0]
    assert log.id == "req-log-1"
    assert log.path == "/v1/chat/completions"
    assert log.model == "gpt-4o-mini"
    assert log.decision == "block"
    assert log.confidence == pytest.approx(0.95)
    assert log.canary_hit is False

    assert len(dets) == 1
    det = dets[0]
    assert det.request_id == "req-log-1"
    assert det.layer == "heuristic"
    assert det.level == "block"
    assert det.trigger_tokens is not None and "ignore" in det.trigger_tokens


async def test_persist_context_all_clean_request(db_env):
    ctx = PipelineContext(request_id="req-clean", route="/v1/completions")
    ctx.raw_body = {"prompt": "hello"}
    ctx.add_result(LayerResult(layer="normalizer", passed=True))
    await persist_context(ctx)

    from sqlalchemy import select

    from src.db.models import Detection, RequestLog
    from src.db.session import _get_sessionmaker

    maker = _get_sessionmaker()
    async with maker() as session:
        logs = (await session.execute(select(RequestLog))).scalars().all()
        dets = (await session.execute(select(Detection))).scalars().all()

    assert len(logs) == 1
    assert logs[0].decision == "allow"
    assert logs[0].path == "/v1/completions"
    assert dets == []


async def test_mark_canary_hit_updates_row(db_env):
    ctx = _ctx()
    ctx.decision = DecisionLevel.allow  # allowed pre-inference
    await persist_context(ctx)

    ctx.canary_hit = True
    await mark_canary_hit(ctx)

    from sqlalchemy import select

    from src.db.models import RequestLog
    from src.db.session import _get_sessionmaker

    maker = _get_sessionmaker()
    async with maker() as session:
        log = (await session.execute(select(RequestLog))).scalars().one()

    assert log.canary_hit is True
    assert log.decision == "block"


async def test_persist_context_failure_is_suppressed(db_env, monkeypatch):
    from src.db import session as db_session

    def broken_maker():
        raise RuntimeError("db down")

    monkeypatch.setattr(db_session, "_get_sessionmaker", broken_maker)
    await persist_context(_ctx())  # must not raise
    await mark_canary_hit(_ctx())  # must not raise
