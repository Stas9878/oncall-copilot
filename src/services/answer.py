"""Ответ на вопрос: кэш, поиск, промпт, модель и проверка ответа."""

import base64
from collections.abc import AsyncIterator
from dataclasses import asdict, dataclass, field

from src.core.logs import log
from src.generation import cache
from src.core.runtime import Resources
from src.security.auth import Principal
from src.search.query import search_query
from src.search.retrieval import retrieve
from src.generation.output import OutputFilter
from src.generation.llm import Completion, LLMRouter, Prompt
from src.ingestion.captions import SEARCH_PROMPT, describe_image, image_mime
from src.generation.prompt import (
    ANSWER_MAX_TOKENS,
    LOCAL_ANSWER_MAX_TOKENS,
    NO_ANSWER,
    PromptSource,
    build_messages,
    cited_numbers,
    fit_sources,
    local_source_budget
)

HISTORY_TURNS = 6


@dataclass(frozen=True, slots=True)
class Question:
    text: str
    history: list[dict] = field(default_factory=list)
    image: bytes | None = None


@dataclass(frozen=True, slots=True)
class Citation:
    number: int
    filename: str
    heading: str | None
    page_number: int | None


@dataclass(slots=True)
class Answer:
    answer: str
    citations: list[Citation]
    search_query: str
    backend: str | None = None
    model: str | None = None
    # Почему ответ дала резервная модель. None — штатный ответ.
    degraded_reason: str | None = None
    cached: bool = False


@dataclass(slots=True)
class _Prepared:
    sources: list[PromptSource]
    # Резервная модель видит меньше источников, поэтому её цитаты сверяются с её списком.
    local_sources: list[PromptSource]
    cloud: Prompt
    local: Prompt
    output: OutputFilter
    search_query: str


def _citations(sources: list[PromptSource], text: str) -> list[Citation]:
    by_number = {item.number: item.source for item in sources}
    return [
        Citation(n, by_number[n].filename, by_number[n].heading, by_number[n].page_number)
        for n in cited_numbers(text, len(sources))
    ]


async def describe_screen(resources: Resources, router: LLMRouter, principal: Principal, image: bytes) -> str | None:
    """Суть скрина одной строкой для поиска. None, если облако недоступно."""
    if await router.cloud_blocked(principal):
        return None
    try:
        return await describe_image(
            resources.cloud_llm,
            resources.settings.llm.vision_model,
            image,
            image_mime(image),
            prompt=SEARCH_PROMPT,
            max_tokens=80
        )
    except Exception as error:  # noqa: BLE001 — без описания поиск идёт по одному вопросу
        log.warning("describe_failed", error=type(error).__name__)
        return None


async def _prepare(
    resources: Resources, router: LLMRouter, principal: Principal, question: Question
) -> _Prepared | Answer:
    settings = resources.settings.retrieval
    history = question.history[-HISTORY_TURNS:]
    description = await describe_screen(resources, router, principal, question.image) if question.image else None

    query = await search_query(router, question.text, history)
    # Вопрос идёт первым: эмбеддер обрезает длинный текст с конца.
    retrieval_query = f"{query}\n{description}" if description else query
    sources = await retrieve(
        resources, principal, retrieval_query, limit=settings.top_sections, min_score=settings.min_score
    )
    if not sources:
        return Answer(NO_ANSWER, [], retrieval_query)

    texts = [s.text for s in sources]
    fitted = fit_sources(sources, texts, settings.context_tokens)

    image_url = None
    local_question = question.text
    if question.image is not None:
        image_url = f"data:{image_mime(question.image)};base64,{base64.b64encode(question.image).decode()}"
        # Локальная модель картинку не видит: ей достаётся описание, если оно есть.
        if description:
            local_question += f"\n\nОписание приложенного скрина: {description}"
    # Резервной модели — те же источники в том же порядке, но столько, сколько поместится
    # в её окно. Номера совпадают с облачным вариантом, поэтому цитаты не съезжают.
    budget = local_source_budget(resources.settings.llm.local_context_tokens, local_question, history)
    local_fitted = fit_sources(sources, texts, budget)
    return _Prepared(
        sources=fitted,
        local_sources=local_fitted,
        cloud=Prompt(build_messages(question.text, fitted, history, image_url), ANSWER_MAX_TOKENS),
        local=Prompt(build_messages(local_question, local_fitted, history), LOCAL_ANSWER_MAX_TOKENS),
        output=OutputFilter(),
        search_query=retrieval_query
    )


def _cacheable(question: Question) -> bool:
    return not question.history and question.image is None


async def _cached(resources: Resources, principal: Principal, question: Question) -> Answer | None:
    if not _cacheable(question):
        return None
    hit = await cache.get_answer(resources.redis, principal.tenant_id, question.text)
    if hit is None:
        return None
    hit["citations"] = [Citation(**c) for c in hit["citations"]]
    return Answer(**hit | {"cached": True})


async def _remember(resources: Resources, principal: Principal, question: Question, result: Answer) -> None:
    # Ответ резервной модели не кэшируем.
    if _cacheable(question) and result.degraded_reason is None:
        await cache.put_answer(resources.redis, principal.tenant_id, question.text, asdict(result))


async def answer(resources: Resources, principal: Principal, question: Question) -> Answer:
    if hit := await _cached(resources, principal, question):
        return hit

    router = LLMRouter(resources)
    prepared = await _prepare(resources, router, principal, question)
    if isinstance(prepared, Answer):
        return prepared

    completion = await router.complete(principal, prepared.cloud, prepared.local, vision=question.image is not None)
    result = _finish(prepared, completion, prepared.output.apply(completion.text))
    await _remember(resources, principal, question, result)
    return result


def _finish(prepared: _Prepared, completion: Completion, text: str) -> Answer:
    return Answer(
        answer=text,
        citations=_citations(prepared.sources if completion.backend == "cloud" else prepared.local_sources, text),
        search_query=prepared.search_query,
        backend=completion.backend,
        model=completion.model,
        degraded_reason=completion.degraded_reason
    )


async def answer_stream(resources: Resources, principal: Principal, question: Question) -> AsyncIterator[str | Answer]:
    """Части ответа по мере генерации, последним элементом — Answer с цитатами и метаданными."""
    if hit := await _cached(resources, principal, question):
        yield hit.answer
        yield hit
        return

    router = LLMRouter(resources)
    prepared = await _prepare(resources, router, principal, question)
    if isinstance(prepared, Answer):
        yield prepared.answer
        yield prepared
        return

    shown: list[str] = []
    completion: Completion | None = None
    async for item in router.stream(principal, prepared.cloud, prepared.local, vision=question.image is not None):
        if isinstance(item, Completion):
            completion = item
        elif safe := prepared.output.feed(item):
            shown.append(safe)
            yield safe
    if tail := prepared.output.flush():
        shown.append(tail)
        yield tail
    result = _finish(prepared, completion, "".join(shown))
    await _remember(resources, principal, question, result)
    yield result
