"""Alembic: адрес базы берётся из настроек сервиса, схема — из моделей."""

import asyncio
from alembic import context
from sqlalchemy import pool
from logging.config import fileConfig
from sqlalchemy.engine import Connection
from sqlalchemy.ext.asyncio import create_async_engine

from src.db.models import Base
from src.core.config import PostgresSettings

config = context.config
if config.config_file_name is not None:
    fileConfig(config.config_file_name)

target_metadata = Base.metadata
DSN = PostgresSettings().dsn.get_secret_value()


def run_migrations_offline() -> None:
    """Печатает SQL миграций без подключения к базе: alembic upgrade head --sql."""
    context.configure(url=DSN, target_metadata=target_metadata, literal_binds=True)
    with context.begin_transaction():
        context.run_migrations()


def do_run_migrations(connection: Connection) -> None:
    context.configure(connection=connection, target_metadata=target_metadata)
    with context.begin_transaction():
        context.run_migrations()


async def run_async_migrations() -> None:
    engine = create_async_engine(DSN, poolclass=pool.NullPool)
    async with engine.connect() as connection:
        await connection.run_sync(do_run_migrations)
    await engine.dispose()


if context.is_offline_mode():
    run_migrations_offline()
else:
    asyncio.run(run_async_migrations())
