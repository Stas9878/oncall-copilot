"""Кэш готовых ответов в Redis с версией базы знаний команды в ключе."""

import json
import uuid
import hashlib
from redis.asyncio import Redis

ANSWER_TTL_SECONDS = 10 * 60


def _version_key(tenant_id: uuid.UUID) -> str:
    return f"kb_version:{tenant_id}"


async def bump_kb_version(redis: Redis, tenant_id: uuid.UUID) -> None:
    await redis.incr(_version_key(tenant_id))


def _normalize(question: str) -> str:
    return " ".join(question.lower().split()).strip(" ?!.")


async def _answer_key(redis: Redis, tenant_id: uuid.UUID, question: str) -> str:
    version = await redis.get(_version_key(tenant_id)) or "0"
    digest = hashlib.sha256(_normalize(question).encode()).hexdigest()[:32]
    return f"answer:{tenant_id}:{version}:{digest}"


async def get_answer(redis: Redis, tenant_id: uuid.UUID, question: str) -> dict | None:
    raw = await redis.get(await _answer_key(redis, tenant_id, question))
    return json.loads(raw) if raw else None


async def put_answer(redis: Redis, tenant_id: uuid.UUID, question: str, payload: dict) -> None:
    key = await _answer_key(redis, tenant_id, question)
    await redis.set(key, json.dumps(payload, ensure_ascii=False), ex=ANSWER_TTL_SECONDS)
