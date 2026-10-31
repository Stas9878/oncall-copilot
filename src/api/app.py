"""Сборка FastAPI-приложения."""

import time
import uuid
import structlog
from contextlib import asynccontextmanager
from fastapi import FastAPI, Request, Response
from collections.abc import AsyncIterator, Awaitable, Callable

from src.worker import broker
from src.core.config import get_settings
from src.core.runtime import open_resources
from src.core.logs import configure_logging, log
from src.observability.telemetry import setup_tracing
from src.api.routes import ask, documents, health, incidents
from src.observability.metrics import REQUEST_SECONDS, REQUESTS
from src.search.index import collection_name, ensure_collection


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    settings = get_settings()
    async with open_resources(settings) as resources:
        await ensure_collection(resources.qdrant, collection_name(settings.models))
        await resources.models.warmup()
        # API только ставит задачи в очередь, выполняет их воркер.
        await broker.startup()
        app.state.resources = resources
        log.info("api_started", environment=settings.environment)
        try:
            yield
        finally:
            await broker.shutdown()
    log.info("api_stopped")


async def request_context(request: Request, call_next: Callable[[Request], Awaitable[Response]]) -> Response:
    """Метка запроса в логах и в ответе: по X-Request-ID находят все записи одного запроса."""
    request_id = request.headers.get("x-request-id") or uuid.uuid4().hex
    structlog.contextvars.clear_contextvars()
    structlog.contextvars.bind_contextvars(request_id=request_id)
    started = time.perf_counter()
    response = await call_next(request)
    # Для /v1/ask/stream это время до начала потока.
    elapsed = time.perf_counter() - started
    response.headers["x-request-id"] = request_id
    # Шаблон маршрута: /v1/documents/{document_id} даёт один ряд на все документы.
    route = getattr(request.scope.get("route"), "path", "unmatched")
    REQUESTS.labels(route, request.method, str(response.status_code)).inc()
    REQUEST_SECONDS.labels(route).observe(elapsed)
    log.info(
        "request",
        method=request.method,
        path=request.url.path,
        status=response.status_code,
        duration_ms=round(elapsed * 1000)
    )
    return response


def create_app() -> FastAPI:
    settings = get_settings()
    configure_logging(json=settings.environment != "local")
    setup_tracing(settings)
    # FastAPI сам пишет спан на каждый HTTP-запрос, если настроен OpenTelemetry.
    # В Phoenix нужны трейсы пайплайна ответа, а HTTP-запросы видны в метриках и логах.
    app = FastAPI(title="On-call Copilot", version="0.1.0", lifespan=lifespan, telemetry={"tracing": False})
    app.middleware("http")(request_context)
    app.include_router(health.router)
    app.include_router(documents.router)
    app.include_router(ask.router)
    app.include_router(incidents.router)
    return app
