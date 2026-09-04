import pytest
from fastapi.testclient import TestClient

from src.db.session import reset_engine
from src.pipeline.orchestrator import reset_orchestrator
from src.proxy.app import create_app
from src.proxy.upstream import reset_upstream


@pytest.fixture()
def app():
    # Use sqlite in-memory-ish file for tests
    import os

    os.environ["DATABASE_URL"] = "sqlite+aiosqlite:///./test_waf.db"
    os.environ["REDIS_URL"] = "redis://localhost:6379/0"
    os.environ["UPSTREAM_BASE_URL"] = "https://api.openai.com/v1"
    os.environ["UPSTREAM_API_KEY"] = "test-key"

    # Reset singletons
    from src.config.settings import reset_settings_cache

    reset_settings_cache()
    reset_engine()
    reset_orchestrator()
    reset_upstream()

    application = create_app()
    return application


@pytest.fixture()
def client(app):
    # Use TestClient (sync) for simplicity
    with TestClient(app) as c:
        yield c
