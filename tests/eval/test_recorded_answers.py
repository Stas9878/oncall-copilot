"""Проверка в CI: записанные ответы модели соответствуют эталонным кейсам."""

import json
import pytest
from pathlib import Path

from src.core.config import get_settings
from src.generation.prompt import NO_ANSWER
from src.observability.versions import fingerprint

CASES = {case["id"]: case for case in json.loads(Path("eval/cases.json").read_text(encoding="utf-8"))}
RECORDED = json.loads(Path("eval/recorded_answers.json").read_text(encoding="utf-8"))


def _compact(text: str) -> str:
    # Модель пишет «80 %» с неразрывным пробелом: пробелы при сравнении не важны.
    return "".join(text.split()).lower()


def test_recorded_with_current_configuration():
    assert RECORDED["fingerprint"] == fingerprint(get_settings()), (
        "Промпт или модели изменились: перезапишите ответы командой python -m src.evals record"
    )


def test_every_case_recorded():
    assert set(RECORDED["answers"]) == set(CASES)


@pytest.mark.parametrize("case_id", sorted(CASES))
def test_answer_follows_case(case_id: str):
    case, result = CASES[case_id], RECORDED["answers"][case_id]
    text = _compact(result["answer"])
    if case.get("expect_no_answer"):
        assert result["answer"] == NO_ANSWER
    else:
        assert result["citations"], "ответ без ссылок на источники"
        missing = [fact for fact in case.get("must_contain", []) if _compact(fact) not in text]
        assert missing == [], f"нет обязательных фактов: {missing}"
    leaked = [fact for fact in case.get("must_not_contain", []) if _compact(fact) in text]
    assert leaked == []
