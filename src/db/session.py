"""Подключение к Postgres: один пул соединений на процесс."""

from collections.abc import AsyncIterator
from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession, async_sessionmaker, create_async_engine

from src.core.config import PostgresSettings


def create_engine(settings: PostgresSettings) -> AsyncEngine:
    return create_async_engine(
        settings.dsn.get_secret_value(),
        pool_size=settings.pool_size,
        # Соединение, которое Postgres закрыл за время простоя, проверяется до выдачи.
        pool_pre_ping=True
    )


def create_sessionmaker(engine: AsyncEngine) -> async_sessionmaker[AsyncSession]:
    # expire_on_commit=False: объекты остаются читаемыми после commit без нового запроса.
    return async_sessionmaker(engine, expire_on_commit=False)


async def session_scope(factory: async_sessionmaker[AsyncSession]) -> AsyncIterator[AsyncSession]:
    """Сессия на один запрос: commit при успехе, rollback при исключении."""
    async with factory() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise
