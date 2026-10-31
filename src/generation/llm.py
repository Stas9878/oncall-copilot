"""Выбор бэкенда модели: облако, а при отказе или исчерпанной квоте — локальный vLLM."""

import uuid
import openai
import asyncio
from datetime import date
from dataclasses import dataclass
from sqlalchemy import func, select
from collections.abc import AsyncIterator
from sqlalchemy.dialects.postgresql import insert

from src.core.logs import log
from src.db.models import TokenUsage
from src.core.runtime import Resources
from src.security.auth import Principal
from src.observability.metrics import LLM_FALLBACKS, LLM_TOKENS, TENANT_BUDGET_TOKENS, TENANT_MONTH_TOKENS

FAILURE_LIMIT = 3
FAILURE_WINDOW_SECONDS = 60
COOLDOWN_SECONDS = 30
QUOTA_COOLDOWN_SECONDS = 15 * 60

# Локальная Qwen3 по умолчанию рассуждает. Для коротких служебных ответов
# рассуждение только тратит время на CPU.
NO_THINKING = {"chat_template_kwargs": {"enable_thinking": False}}


class LLMUnavailable(RuntimeError):
    """Не ответил ни облачный, ни локальный бэкенд."""


@dataclass(frozen=True, slots=True)
class Prompt:
    """Сообщения и лимит ответа для одного бэкенда."""

    messages: list[dict]
    max_tokens: int


@dataclass(frozen=True, slots=True)
class Completion:
    text: str
    backend: str  # cloud или local
    model: str
    prompt_tokens: int
    completion_tokens: int
    # Ответ дала резервная модель: качество ниже, а картинка не учтена.
    degraded_reason: str | None = None


class CloudBreaker:
    """Общий для всех реплик выключатель облака."""

    OPEN = "breaker:cloud:open"
    FAILURES = "breaker:cloud:failures"
    TRIPPED = "breaker:cloud:tripped"

    def __init__(self, resources: Resources) -> None:
        self._redis = resources.redis

    async def is_open(self) -> bool:
        return bool(await self._redis.exists(self.OPEN))

    async def success(self) -> None:
        await self._redis.delete(self.FAILURES, self.TRIPPED)

    async def failure(self, *, quota: bool) -> None:
        if quota:
            # Квота не вернётся через 30 секунд: облако выключается надолго.
            await self._open(QUOTA_COOLDOWN_SECONDS, "quota")
            return
        if await self._redis.exists(self.TRIPPED):
            # Пробный запрос после паузы снова упал: выключаем сразу, без подсчёта сбоев.
            await self._open(COOLDOWN_SECONDS, "probe_failed")
            return
        failures = await self._redis.incr(self.FAILURES)
        await self._redis.expire(self.FAILURES, FAILURE_WINDOW_SECONDS)
        if failures >= FAILURE_LIMIT:
            await self._open(COOLDOWN_SECONDS, "failures")

    async def _open(self, seconds: int, reason: str) -> None:
        async with self._redis.pipeline(transaction=True) as pipe:
            pipe.set(self.OPEN, reason, ex=seconds)
            pipe.set(self.TRIPPED, "1", ex=seconds * 4)
            pipe.delete(self.FAILURES)
            await pipe.execute()
        log.warning("cloud_breaker_open", reason=reason, seconds=seconds)


# 429 с коротким Retry-After — всплеск запросов: один раз ждём и повторяем.
# Длинная пауза дороже, чем ответ резервной модели.
MAX_RETRY_AFTER_SECONDS = 5


def _retry_after(error: Exception) -> float | None:
    if not isinstance(error, openai.APIStatusError) or error.status_code != 429:
        return None
    try:
        seconds = float(error.response.headers.get("retry-after", ""))
    except ValueError:
        return None
    return seconds if 0 <= seconds <= MAX_RETRY_AFTER_SECONDS else None


def _is_quota_error(error: openai.APIStatusError) -> bool:
    # 402 — квота или тариф. 429 без Retry-After у Ollama Cloud тоже означает
    # исчерпанный лимит, а 429 с Retry-After — временную перегрузку.
    if error.status_code == 402:
        return True
    return error.status_code == 429 and "retry-after" not in error.response.headers


async def month_tokens(resources: Resources, tenant_id: uuid.UUID) -> int:
    month = date.today().replace(day=1)
    async with resources.sessions() as session:
        total = await session.scalar(
            select(func.coalesce(func.sum(TokenUsage.prompt_tokens + TokenUsage.completion_tokens), 0)).where(
                TokenUsage.tenant_id == tenant_id, TokenUsage.month == month
            )
        )
    return int(total)


async def record_usage(resources: Resources, tenant_id: uuid.UUID, completion: Completion) -> None:
    """Расход облачных токенов команды. Локальная модель квоту не тратит."""
    LLM_TOKENS.labels(completion.backend, completion.model, "prompt").inc(completion.prompt_tokens)
    LLM_TOKENS.labels(completion.backend, completion.model, "completion").inc(completion.completion_tokens)
    if completion.backend != "cloud":
        return
    statement = insert(TokenUsage).values(
        tenant_id=tenant_id,
        month=date.today().replace(day=1),
        model=completion.model,
        prompt_tokens=completion.prompt_tokens,
        completion_tokens=completion.completion_tokens
    )
    statement = statement.on_conflict_do_update(
        index_elements=[TokenUsage.tenant_id, TokenUsage.month, TokenUsage.model],
        set_={
            "prompt_tokens": TokenUsage.prompt_tokens + statement.excluded.prompt_tokens,
            "completion_tokens": TokenUsage.completion_tokens + statement.excluded.completion_tokens
        }
    )
    async with resources.sessions.begin() as session:
        await session.execute(statement)


class LLMRouter:
    def __init__(self, resources: Resources) -> None:
        self._resources = resources
        self._settings = resources.settings.llm
        self._breaker = CloudBreaker(resources)

    async def cloud_blocked(self, principal: Principal) -> str | None:
        """Причина не обращаться к облачной модели или None, если обращаться можно."""
        if await self._breaker.is_open():
            return "облачная модель недоступна"
        used = await month_tokens(self._resources, principal.tenant_id)
        TENANT_MONTH_TOKENS.labels(principal.tenant_slug).set(used)
        TENANT_BUDGET_TOKENS.labels(principal.tenant_slug).set(principal.monthly_token_budget)
        if used >= principal.monthly_token_budget:
            return "исчерпан месячный бюджет токенов команды"
        return None

    def _cloud_kwargs(self, model: str, prompt: Prompt) -> dict:
        kwargs: dict = {"model": model, "messages": prompt.messages, "max_tokens": prompt.max_tokens, "temperature": 0}
        if model == self._settings.cloud_model:
            kwargs["reasoning_effort"] = self._settings.reasoning_effort
        return kwargs

    def _local_kwargs(self, prompt: Prompt) -> dict:
        return {
            "model": self._settings.local_model,
            "messages": prompt.messages,
            "max_tokens": prompt.max_tokens,
            "temperature": 0,
            "extra_body": NO_THINKING
        }

    async def _call_cloud(self, **kwargs):
        """Вызов облака с одним повтором после короткого Retry-After."""
        try:
            return await self._resources.cloud_llm.chat.completions.create(**kwargs)
        except openai.APIStatusError as error:
            if (pause := _retry_after(error)) is None:
                raise
            await asyncio.sleep(pause)
            return await self._resources.cloud_llm.chat.completions.create(**kwargs)

    async def _on_cloud_error(self, error: Exception) -> None:
        quota = isinstance(error, openai.APIStatusError) and _is_quota_error(error)
        if (
            isinstance(error, openai.APIStatusError)
            and not quota
            and error.status_code < 500
            and error.status_code != 429
        ):
            raise error  # 400, 401, 404: ошибка в самом запросе, резервная модель не поможет
        await self._breaker.failure(quota=quota)
        log.warning("cloud_call_failed", error=type(error).__name__, status=getattr(error, "status_code", None))

    async def complete(
        self,
        principal: Principal,
        cloud: Prompt,
        local: Prompt,
        *,
        vision: bool = False,
        response_format: dict | None = None
    ) -> Completion:
        """Ответ облачной модели, а если она недоступна — локальной."""
        extra = {"response_format": response_format} if response_format else {}
        reason = await self.cloud_blocked(principal)
        if reason is None:
            model = self._settings.vision_model if vision else self._settings.cloud_model
            try:
                response = await self._call_cloud(**self._cloud_kwargs(model, cloud), **extra)
            except (openai.APIConnectionError, openai.APIStatusError) as error:
                await self._on_cloud_error(error)
                reason = "облачная модель не ответила"
            else:
                await self._breaker.success()
                completion = _completion(response, "cloud", model)
                await record_usage(self._resources, principal.tenant_id, completion)
                return completion

        LLM_FALLBACKS.labels(reason).inc()
        try:
            response = await self._resources.local_llm.chat.completions.create(**self._local_kwargs(local), **extra)
        except (openai.APIConnectionError, openai.APIStatusError) as error:
            log.error("local_call_failed", error=type(error).__name__, detail=str(error)[:300])
            raise LLMUnavailable(f"{reason}, резервная модель тоже не ответила") from error
        completion = _completion(response, "local", self._settings.local_model, reason)
        await record_usage(self._resources, principal.tenant_id, completion)
        return completion

    async def stream(
        self, principal: Principal, cloud: Prompt, local: Prompt, *, vision: bool = False
    ) -> AsyncIterator[str | Completion]:
        """Поток частей текста, последним элементом — Completion с расходом токенов."""
        options = {"stream": True, "stream_options": {"include_usage": True}}
        reason = await self.cloud_blocked(principal)
        if reason is None:
            model = self._settings.vision_model if vision else self._settings.cloud_model
            try:
                stream = await self._call_cloud(**self._cloud_kwargs(model, cloud), **options)
            except (openai.APIConnectionError, openai.APIStatusError) as error:
                await self._on_cloud_error(error)
                reason = "облачная модель не ответила"
            else:
                try:
                    async for item in _pieces(stream, "cloud", model):
                        if isinstance(item, Completion):
                            await record_usage(self._resources, principal.tenant_id, item)
                        yield item
                except (openai.APIConnectionError, openai.APIStatusError) as error:
                    await self._on_cloud_error(error)
                    raise LLMUnavailable("облачная модель оборвала ответ") from error
                await self._breaker.success()
                return

        LLM_FALLBACKS.labels(reason).inc()
        try:
            stream = await self._resources.local_llm.chat.completions.create(**self._local_kwargs(local), **options)
        except (openai.APIConnectionError, openai.APIStatusError) as error:
            log.error("local_call_failed", error=type(error).__name__, detail=str(error)[:300])
            raise LLMUnavailable(f"{reason}, резервная модель тоже не ответила") from error
        async for item in _pieces(stream, "local", self._settings.local_model, reason):
            if isinstance(item, Completion):
                await record_usage(self._resources, principal.tenant_id, item)
            yield item

    async def local_json(self, messages: list[dict], schema: dict, *, max_tokens: int) -> str:
        """Ответ локальной модели строго по JSON-схеме: vLLM ограничивает генерацию грамматикой."""
        response = await self._resources.local_llm.chat.completions.create(
            response_format={"type": "json_schema", "json_schema": {"name": "result", "schema": schema}},
            **self._local_kwargs(Prompt(messages, max_tokens))
        )
        return response.choices[0].message.content or ""


def _completion(response, backend: str, model: str, degraded_reason: str | None = None) -> Completion:
    return Completion(
        text=response.choices[0].message.content or "",
        backend=backend,
        model=model,
        prompt_tokens=response.usage.prompt_tokens,
        completion_tokens=response.usage.completion_tokens,
        degraded_reason=degraded_reason
    )


async def _pieces(
    stream, backend: str, model: str, degraded_reason: str | None = None
) -> AsyncIterator[str | Completion]:
    prompt_tokens = completion_tokens = 0
    async for chunk in stream:
        if chunk.usage is not None:
            prompt_tokens, completion_tokens = chunk.usage.prompt_tokens, chunk.usage.completion_tokens
        if chunk.choices and chunk.choices[0].delta.content:
            yield chunk.choices[0].delta.content
    yield Completion("", backend, model, prompt_tokens, completion_tokens, degraded_reason)
