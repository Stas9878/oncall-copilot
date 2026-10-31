"""Команды администратора: команды и ключи."""

import uuid
import asyncio
import argparse
from sqlalchemy import select
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from sqlalchemy.ext.asyncio import AsyncSession

from src.core.config import get_settings
from src.security.auth import generate_key
from src.db.models import ApiKey, Role, Tenant
from src.db.session import create_engine, create_sessionmaker


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

    args = parser.parse_args()
    if args.command == "tenant":
        asyncio.run(create_tenant(args.slug, args.name, args.link_domain, args.budget))
    else:
        asyncio.run(create_key(args.slug, Role(args.role), args.name))


if __name__ == "__main__":
    main()
