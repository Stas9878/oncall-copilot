"""Ключи API и определение команды по ключу."""

import uuid
import hashlib
import secrets
from sqlalchemy import select
from dataclasses import dataclass
from sqlalchemy.ext.asyncio import AsyncSession

from src.db.models import ApiKey, Role, Tenant

KEY_PREFIX = "occ_"


@dataclass(frozen=True, slots=True)
class Principal:
    """Кто делает запрос: команда, её настройки и роль ключа."""

    key_id: uuid.UUID
    tenant_id: uuid.UUID
    tenant_slug: str
    role: Role
    link_domains: tuple[str, ...]
    monthly_token_budget: int

    @property
    def can_write(self) -> bool:
        return self.role is Role.EDITOR


def generate_key() -> tuple[str, str]:
    """Новый ключ и его хеш. Ключ показывают один раз, в базу пишут только хеш."""
    key = KEY_PREFIX + secrets.token_urlsafe(32)
    return key, hash_key(key)


def hash_key(key: str) -> str:
    # Ключ случайный и длинный, поэтому достаточно SHA-256 без соли.
    # Для паролей пользователей так делать нельзя: там нужен bcrypt или argon2.
    return hashlib.sha256(key.encode()).hexdigest()


async def resolve_key(session: AsyncSession, key: str) -> Principal | None:
    """Находит действующий ключ. None, если ключ неизвестен или отозван."""
    if not key.startswith(KEY_PREFIX):
        return None
    row = (
        await session.execute(
            select(ApiKey.id, ApiKey.role, Tenant)
            .join(Tenant, Tenant.id == ApiKey.tenant_id)
            .where(ApiKey.key_hash == hash_key(key), ApiKey.revoked_at.is_(None))
        )
    ).one_or_none()
    if row is None:
        return None
    key_id, role, tenant = row
    return Principal(
        key_id=key_id,
        tenant_id=tenant.id,
        tenant_slug=tenant.slug,
        role=role,
        link_domains=tuple(tenant.link_domains),
        monthly_token_budget=tenant.monthly_token_budget
    )
