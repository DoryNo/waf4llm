from __future__ import annotations

import time
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, PlainTextResponse
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest

from src.config.settings import get_settings
from src.observability.logging import configure_logging, get_logger
from src.observability.metrics import metrics
from src.observability.tracing import configure_tracing
from src.proxy.routes import admin_router
from src.proxy.routes import router as proxy_router


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = get_settings()
    configure_logging(settings.log_level)
    configure_tracing()
    logger = get_logger("app")
    logger.info(
        "starting",
        env=settings.app_env,
        upstream=settings.upstream_base_url,
        fail_mode=settings.fail_mode.value,
    )

    # Try to init DB (best-effort — sqlite fallback works, postgres may not be up in dev)
    try:
        from src.db.session import init_db

        await init_db()
        logger.info("db initialized")
    except Exception as e:
        logger.warning("db init failed (continuing)", error=str(e))

    # Check Redis (optional)
    try:
        import redis.asyncio as redis

        r = redis.from_url(settings.redis_url, socket_connect_timeout=2)
        await r.ping()
        await getattr(r, "aclose", r.close)()
        logger.info("redis connected")
    except Exception as e:
        logger.warning("redis not available (continuing)", error=str(e))

    yield

    # Shutdown: close upstream client
    try:
        from src.proxy.upstream import get_upstream

        await get_upstream().close()
    except Exception:
        pass


def create_app() -> FastAPI:
    app = FastAPI(
        title="WAF for LLM — Anti-Injection Proxy",
        version="0.1.0",
        lifespan=lifespan,
    )

    # Proxy routes
    app.include_router(proxy_router)
    app.include_router(admin_router)

    @app.get("/health")
    async def health():
        return {"status": "ok", "version": "0.1.0"}

    @app.get("/ready")
    async def ready():
        settings = get_settings()
        checks: dict[str, str] = {}
        # DB check
        try:
            from sqlalchemy import text

            from src.db.session import _get_engine

            engine = _get_engine()
            async with engine.connect() as conn:
                await conn.execute(text("SELECT 1"))
            checks["db"] = "ok"
        except Exception as e:
            checks["db"] = f"fail: {e}"

        # Redis check (optional — empty REDIS_URL disables it, e.g. sqlite-only local dev)
        if settings.redis_url.strip():
            try:
                import redis.asyncio as redis

                r = redis.from_url(settings.redis_url, socket_connect_timeout=2)
                await r.ping()
                await getattr(r, "aclose", r.close)()
                checks["redis"] = "ok"
            except Exception as e:
                checks["redis"] = f"fail: {e}"
        else:
            checks["redis"] = "ok (disabled)"

        # Upstream check (optional — just report)
        checks["upstream"] = settings.upstream_base_url

        all_ok = all(v.startswith("ok") for k, v in checks.items() if k in ("db", "redis"))
        status_code = 200 if all_ok else 503
        return JSONResponse(status_code=status_code, content={"ready": all_ok, "checks": checks})

    @app.get("/metrics")
    async def metrics_endpoint():
        if not get_settings().prometheus_enabled:
            return PlainTextResponse("metrics disabled", status_code=404)
        data = generate_latest()
        return PlainTextResponse(data.decode("utf-8"), media_type=CONTENT_TYPE_LATEST)

    @app.get("/")
    async def root():
        return {
            "name": "WAF for LLM — Anti-Injection Proxy",
            "version": "0.1.0",
            "endpoints": [
                "/v1/chat/completions",
                "/v1/completions",
                "/admin/feedback",
                "/health",
                "/ready",
                "/metrics",
            ],
        }

    # Request timing middleware (lightweight)
    @app.middleware("http")
    async def timing_middleware(request: Request, call_next):
        start = time.monotonic()
        response = await call_next(request)
        elapsed = time.monotonic() - start
        # Don't double-count proxy route which already records metrics
        if request.url.path not in ("/metrics", "/health", "/ready"):
            try:
                # Proxy routes record their final status themselves.
                if request.url.path not in ("/v1/chat/completions", "/v1/completions"):
                    metrics.request_duration.labels(route=request.url.path).observe(elapsed)
            except Exception:
                pass
        response.headers["x-response-time-ms"] = str(int(elapsed * 1000))
        return response

    return app
