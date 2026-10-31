"""Вопросы к базе знаний команды: обычный ответ и поток (SSE)."""

import json
from dataclasses import asdict
from pydantic import BaseModel, Field
from typing import Annotated, Literal
from collections.abc import AsyncIterator
from fastapi import APIRouter, HTTPException, status
from fastapi.sse import EventSourceResponse, ServerSentEvent

from src.core.logs import log
from src.generation.llm import LLMUnavailable
from src.security.uploads import decode_screen
from src.api.deps import PrincipalDep, ResourcesDep
from src.services.answer import Answer, Question, answer, answer_stream

router = APIRouter(prefix="/v1/ask", tags=["ask"])


class Turn(BaseModel):
    role: Literal["user", "assistant"]
    content: Annotated[str, Field(max_length=4000)]


class AskIn(BaseModel):
    question: Annotated[str, Field(min_length=2, max_length=2000)]
    history: Annotated[list[Turn], Field(max_length=12)] = []
    image_base64: Annotated[str | None, Field(description="Скрин PNG или JPEG в base64")] = None


class CitationOut(BaseModel):
    number: int
    filename: str
    heading: str | None
    page_number: int | None


class AnswerOut(BaseModel):
    answer: str
    citations: list[CitationOut]
    search_query: str
    backend: str | None
    model: str | None
    degraded_reason: str | None
    cached: bool


def _question(body: AskIn) -> Question:
    image = decode_screen(body.image_base64) if body.image_base64 else None
    return Question(body.question.strip(), [turn.model_dump() for turn in body.history], image)


def _event(name: str, payload: dict) -> ServerSentEvent:
    # Встроенная сериализация кодирует кириллицу как \u041f. Клиент это разберёт,
    # но в curl и логах такой поток не прочитать.
    return ServerSentEvent(event=name, raw_data=json.dumps(payload, ensure_ascii=False))


def _out(result: Answer) -> AnswerOut:
    return AnswerOut.model_validate(asdict(result))


@router.post("", response_model=AnswerOut)
async def ask(body: AskIn, principal: PrincipalDep, resources: ResourcesDep) -> AnswerOut:
    try:
        return _out(await answer(resources, principal, _question(body)))
    except LLMUnavailable as error:
        raise HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE, str(error)) from error


@router.post("/stream", response_class=EventSourceResponse)
async def ask_stream(body: AskIn, principal: PrincipalDep, resources: ResourcesDep) -> AsyncIterator[ServerSentEvent]:
    """События: token — часть текста, done — цитаты и метаданные, error — сбой."""
    question = _question(body)
    try:
        async for item in answer_stream(resources, principal, question):
            if isinstance(item, Answer):
                yield _event("done", _out(item).model_dump(exclude={"answer"}))
            else:
                yield _event("token", {"text": item})
    except LLMUnavailable as error:
        log.warning("stream_failed", error=str(error))
        yield _event("error", {"detail": str(error)})
