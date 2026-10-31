"""Фоновый воркер индексации на taskiq."""

import uuid
from contextlib import AsyncExitStack
from taskiq_redis import RedisStreamBroker
from taskiq import Context, TaskiqDepends, TaskiqEvents, TaskiqState

from src.core.config import get_settings
from src.core.logs import configure_logging
from src.core.runtime import open_resources
from src.ingestion.pipeline import ingest_document

settings = get_settings()

# Redis Stream с группой потребителей: задача подтверждается после выполнения.
# Если воркер упал посреди индексации, задачу через idle_timeout получит другой.
broker = RedisStreamBroker(settings.redis.url, queue_name="ingest", idle_timeout=10 * 60 * 1000)


@broker.on_event(TaskiqEvents.WORKER_STARTUP)
async def startup(state: TaskiqState) -> None:
    configure_logging(json=settings.environment != "local")
    state.stack = AsyncExitStack()
    state.resources = await state.stack.enter_async_context(open_resources(settings))
    await state.resources.models.warmup()


@broker.on_event(TaskiqEvents.WORKER_SHUTDOWN)
async def shutdown(state: TaskiqState) -> None:
    await state.stack.aclose()


@broker.task(task_name="ingest_document")
async def ingest_document_task(document_id: str, context: Context = TaskiqDepends()) -> None:
    await ingest_document(context.state.resources, uuid.UUID(document_id))
