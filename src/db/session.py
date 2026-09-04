from __future__ import annotations

import asyncio
from collections.abc import AsyncGenerator
from pathlib import Path

from alembic.config import Config
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from alembic import command
from src.config.settings import get_settings

_engine = None
_sessionmaker = None


def _get_engine():
    global _engine
    if _engine is None:
        settings = get_settings()
        url = settings.database_url
        # aiosqlite needs check_same_thread=False handling automatically via async engine
        _engine = create_async_engine(url, echo=False, future=True)
    return _engine


def _get_sessionmaker():
    global _sessionmaker
    if _sessionmaker is None:
        _sessionmaker = async_sessionmaker(
            _get_engine(), expire_on_commit=False, class_=AsyncSession
        )
    return _sessionmaker


async def get_session() -> AsyncGenerator[AsyncSession, None]:
    maker = _get_sessionmaker()
    async with maker() as session:
        yield session


async def init_db() -> None:
    """Apply tracked Alembic migrations without blocking the event loop."""
    await asyncio.to_thread(_upgrade_schema)


def _upgrade_schema() -> None:
    project_root = Path(__file__).resolve().parents[2]
    config = Config(str(project_root / "alembic.ini"))
    config.set_main_option("script_location", str(project_root / "alembic"))
    command.upgrade(config, "head")


def reset_engine() -> None:
    global _engine, _sessionmaker
    _engine = None
    _sessionmaker = None
