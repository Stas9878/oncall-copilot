"""Команды администратора: команды, ключи и переиндексация."""

import uuid
import asyncio
import argparse
from sqlalchemy import select, update
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from sqlalchemy.ext.asyncio import AsyncSession

from src.core.config import get_settings
from src.security.auth import generate_key
from src.db.session import create_engine, create_sessionmaker
from src.db.models import ApiKey, Document, DocumentStatus, Role, Tenant


@asynccontextmanager
async def _transaction() -> AsyncIterator[AsyncSession]:
    """Подключение и транзакция на одну команду: CLI не держит пул между запусками."""
    engine = create_engine(get_settings().postgres)
    try:
        async with create_sessionmaker(engine).begin() as session:
            yield session
    finally:
        await engine.dispose()


async def _tenant_id(session: AsyncSession, slug: str) -> uuid.UUID:
    tenant_id = await session.scalar(select(Tenant.id).where(Tenant.slug == slug))
    if tenant_id is None:
        raise SystemExit(f"Команды {slug} нет")
    return tenant_id


async def create_tenant(slug: str, name: str, link_domains: list[str], budget: int) -> None:
    async with _transaction() as session:
        session.add(Tenant(slug=slug, name=name, link_domains=link_domains, monthly_token_budget=budget))
    print(f"Команда {slug} создана")


async def create_key(slug: str, role: Role, name: str) -> None:
    async with _transaction() as session:
        key, key_hash = generate_key()
        session.add(ApiKey(tenant_id=await _tenant_id(session, slug), name=name, key_hash=key_hash, role=role))
    # Ключ виден только сейчас: в базе остаётся хеш, восстановить ключ нельзя.
    print(key)


async def reindex(slug: str | None) -> None:
    """Ставит документы в очередь на индексацию заново."""
    # Брокер импортируется здесь: остальным командам воркер с его настройками не нужен.
    from src.worker import broker, ingest_document_task

    async with _transaction() as session:
        query = update(Document).values(status=DocumentStatus.QUEUED, index_version=None).returning(Document.id)
        if slug is not None:
            query = query.where(Document.tenant_id == await _tenant_id(session, slug))
        document_ids = list(await session.scalars(query))
    await broker.startup()
    for document_id in document_ids:
        await ingest_document_task.kiq(str(document_id))
    await broker.shutdown()
    print(f"В очереди {len(document_ids)} документов")


def main() -> None:
    parser = argparse.ArgumentParser(prog="src.cli")
    commands = parser.add_subparsers(dest="command", required=True)

    tenant = commands.add_parser("tenant", help="создать команду")
    tenant.add_argument("slug")
    tenant.add_argument("name")
    tenant.add_argument("--link-domain", action="append", default=[], help="домен, на который ответ может ссылаться")
    tenant.add_argument("--budget", type=int, default=2_000_000, help="токенов облачных моделей в месяц")

    key = commands.add_parser("key", help="выпустить ключ API")
    key.add_argument("slug")
    key.add_argument("role", choices=[role.value for role in Role])
    key.add_argument("--name", default="default")

    reindex_parser = commands.add_parser("reindex", help="переиндексировать документы")
    reindex_parser.add_argument("--tenant", help="только документы этой команды")

    args = parser.parse_args()
    if args.command == "tenant":
        asyncio.run(create_tenant(args.slug, args.name, args.link_domain, args.budget))
    elif args.command == "key":
        asyncio.run(create_key(args.slug, Role(args.role), args.name))
    else:
        asyncio.run(reindex(args.tenant))


if __name__ == "__main__":
    main()
