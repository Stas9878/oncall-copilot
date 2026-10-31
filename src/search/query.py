"""Переформулировка уточняющего вопроса в самостоятельный поисковый запрос."""

import re
import json
import asyncio

from src.core.logs import log
from src.generation.llm import LLMRouter

_FOLLOWUP_PREFIX = re.compile(r"^(?:а|и|также|ещё|еще|но)\b[\s,:-]*", re.IGNORECASE)
REWRITE_TIMEOUT_SECONDS = 10
REWRITE_SCHEMA = {
    "type": "object",
    "properties": {"search_query": {"type": "string", "minLength": 10, "maxLength": 200}},
    "required": ["search_query"],
    "additionalProperties": False
}


def needs_rewrite(question: str, history: list[dict]) -> bool:
    """Короткое продолжение диалога без своей темы."""
    text = question.strip()
    return bool(history) and len(text.split()) <= 6 and bool(_FOLLOWUP_PREFIX.match(text))


def rewrite_by_rules(question: str, history: list[dict]) -> str:
    last_user = next((turn["content"].strip() for turn in reversed(history) if turn["role"] == "user"), "")
    tail = _FOLLOWUP_PREFIX.sub("", question.strip()).strip(" ,?")
    if not last_user:
        return question
    return f"{last_user} {tail}".strip()


async def search_query(router: LLMRouter, question: str, history: list[dict]) -> str:
    if not needs_rewrite(question, history):
        return question
    dialog = "\n".join(f"{turn['role']}: {turn['content']}" for turn in history[-4:])
    messages = [
        {
            "role": "user",
            "content": (
                "Сформулируй один самостоятельный поисковый запрос по базе знаний команды: "
                "тема из истории плюс смысл нового вопроса.\n\n"
                f"История:\n{dialog}\n\nНовый вопрос: {question}"
            )
        }
    ]
    try:
        raw = await asyncio.wait_for(
            router.local_json(messages, REWRITE_SCHEMA, max_tokens=96), REWRITE_TIMEOUT_SECONDS
        )
        return json.loads(raw)["search_query"].strip()
    except Exception as error:  # noqa: BLE001 — переформулировка не должна ронять ответ
        log.warning("rewrite_failed", error=type(error).__name__)
        return rewrite_by_rules(question, history)
