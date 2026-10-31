"""Общие подключения процесса: Postgres, Redis, Qdrant, клиенты моделей."""

from openai import AsyncOpenAI
from redis.asyncio import Redis
from dataclasses import dataclass
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from qdrant_client import AsyncQdrantClient
from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession, async_sessionmaker

from src.core.config import Settings
from src.db.session import create_engine, create_sessionmaker


@dataclass(frozen=True, slots=True)
class Resources:
    settings: Settings
    engine: AsyncEngine
    sessions: async_sessionmaker[AsyncSession]
    redis: Redis
    qdrant: AsyncQdrantClient
    local_llm: AsyncOpenAI
    cloud_llm: AsyncOpenAI


@asynccontextmanager
async def open_resources(settings: Settings) -> AsyncIterator[Resources]:
    engine = create_engine(settings.postgres)
    redis = Redis.from_url(settings.redis.url, decode_responses=True)
    qdrant = AsyncQdrantClient(url=settings.qdrant.url)
    llm = settings.llm
    # max_retries=0: повтором и переключением на резервный бэкенд управляет наш код,
    # иначе SDK молча повторит запрос к бэкенду, который исчерпал квоту.
    local_llm = AsyncOpenAI(
        base_url=llm.local_base_url,
        api_key=llm.local_api_key.get_secret_value(),
        timeout=llm.timeout_seconds,
        max_retries=0
    )
    cloud_llm = AsyncOpenAI(
        base_url=llm.cloud_base_url,
        api_key=llm.cloud_api_key.get_secret_value(),
        timeout=llm.timeout_seconds,
        max_retries=0
    )
    try:
        yield Resources(
            settings=settings,
            engine=engine,
            sessions=create_sessionmaker(engine),
            redis=redis,
            qdrant=qdrant,
            local_llm=local_llm,
            cloud_llm=cloud_llm
        )
    finally:
        await local_llm.close()
        await cloud_llm.close()
        await qdrant.close()
        await redis.aclose()
        await engine.dispose()
