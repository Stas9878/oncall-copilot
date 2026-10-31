"""Ограничение частоты запросов на ключ API."""

import time
import uuid
from redis.asyncio import Redis
from fastapi import HTTPException, status

WINDOW_SECONDS = 60


async def check_rate(redis: Redis, key_id: uuid.UUID, scope: str, limit: int) -> None:
    window = int(time.time() // WINDOW_SECONDS)
    counter = f"rate:{scope}:{key_id}:{window}"
    async with redis.pipeline(transaction=True) as pipe:
        pipe.incr(counter)
        pipe.expire(counter, WINDOW_SECONDS)
        count, _ = await pipe.execute()
    if count > limit:
        retry_after = WINDOW_SECONDS - int(time.time()) % WINDOW_SECONDS
        raise HTTPException(
            status.HTTP_429_TOO_MANY_REQUESTS,
            f"Не больше {limit} запросов в минуту на ключ",
            headers={"Retry-After": str(retry_after)}
        )
