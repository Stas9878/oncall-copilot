"""Проверки для оркестратора: жив ли процесс и готов ли он принимать запросы."""

import asyncio
from sqlalchemy import text
from collections.abc import Awaitable
from fastapi import APIRouter, Response, status
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest

from src.api.deps import ResourcesDep
from src.core.runtime import Resources

router = APIRouter(tags=["health"])

CHECK_TIMEOUT_SECONDS = 2.0


async def _check(probe: Awaitable[object]) -> str:
    try:
        await asyncio.wait_for(probe, CHECK_TIMEOUT_SECONDS)
    except Exception as error:  # noqa: BLE001 — в ответ уходит тип ошибки, а не трейс
        return f"error: {type(error).__name__}"
    return "ok"


async def _ping_postgres(resources: Resources) -> None:
    async with resources.engine.connect() as connection:
        await connection.execute(text("SELECT 1"))


@router.get("/health")
async def health() -> dict[str, str]:
    """Liveness: процесс отвечает, зависимости не проверяются."""
    return {"status": "ok"}


@router.get("/ready")
async def ready(resources: ResourcesDep, response: Response) -> dict[str, object]:
    """Readiness: Postgres, Redis и Qdrant доступны, бэкенды моделей на готовность не влияют."""
    core = dict(
        zip(
            ("postgres", "redis", "qdrant"),
            await asyncio.gather(
                _check(_ping_postgres(resources)),
                _check(resources.redis.ping()),
                _check(resources.qdrant.get_collections())
            ),
            strict=True
        )
    )
    llm = dict(
        zip(
            ("local_llm", "cloud_llm"),
            await asyncio.gather(
                _check(resources.local_llm.models.list()),
                _check(resources.cloud_llm.models.list())
            ),
            strict=True
        )
    )
    is_ready = all(state == "ok" for state in core.values())
    if not is_ready:
        response.status_code = status.HTTP_503_SERVICE_UNAVAILABLE
    return {"status": "ok" if is_ready else "unavailable", "dependencies": core | llm}


@router.get("/metrics", include_in_schema=False)
async def metrics() -> Response:
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)
