"""Фоновый воркер индексации на taskiq."""

import uuid
from contextlib import AsyncExitStack
from taskiq_redis import RedisStreamBroker
from taskiq import Context, TaskiqDepends, TaskiqEvents, TaskiqState
from taskiq.middlewares.prometheus_middleware import PrometheusMiddleware
from taskiq.middlewares.smart_retry_middleware import SmartRetryMiddleware

from src.core.config import get_settings
from src.core.logs import configure_logging
from src.core.runtime import open_resources
from src.ingestion.pipeline import ingest_document
from src.observability.telemetry import setup_tracing

settings = get_settings()

# Redis Stream с группой потребителей: задача подтверждается после выполнения.
# Если воркер упал посреди индексации, задачу через idle_timeout получит другой.
broker = RedisStreamBroker(settings.redis.url, queue_name="ingest", idle_timeout=10 * 60 * 1000).with_middlewares(
    # Vision-модель или Qdrant могут быть недоступны какое-то время: задача повторяется
    # через 10 и 20 секунд со случайным разбросом. Индексация идемпотентна, повтор безопасен.
    SmartRetryMiddleware(default_retry_count=3, default_delay=10, use_jitter=True, use_delay_exponent=True)
)


@broker.on_event(TaskiqEvents.WORKER_STARTUP)
async def startup(state: TaskiqState) -> None:
    # Метрики задач на :9000/metrics. Middleware создаётся только в процессе воркера:
    # API тоже импортирует этот модуль, и его метрики попали бы в общий каталог воркера.
    broker.add_middlewares(PrometheusMiddleware(server_port=9000))
    configure_logging(json=settings.environment != "local")
    setup_tracing(settings)
    state.stack = AsyncExitStack()
    state.resources = await state.stack.enter_async_context(open_resources(settings))
    await state.resources.models.warmup(reranker=False)


@broker.on_event(TaskiqEvents.WORKER_SHUTDOWN)
async def shutdown(state: TaskiqState) -> None:
    await state.stack.aclose()


@broker.task(task_name="ingest_document", retry_on_error=True)
async def ingest_document_task(document_id: str, context: Context = TaskiqDepends()) -> None:
    await ingest_document(context.state.resources, uuid.UUID(document_id))
