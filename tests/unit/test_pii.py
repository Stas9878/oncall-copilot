import json
import uuid
from pathlib import Path

from src.db.models import DocumentKind
from src.search.retrieval import Source
from src.security.pii import PHONE, Pseudonymizer
from src.generation.prompt import build_messages, fit_sources

POLICY = Path("corpus/lumen/oncall_policy_lumen.md").read_text(encoding="utf-8")


def _source(text: str) -> Source:
    return Source(
        uuid.uuid4(),
        uuid.uuid4(),
        "oncall_policy_lumen.md",
        DocumentKind.MARKDOWN,
        "Эскалация",
        None,
        "markdown",
        text,
        5.0
    )


def test_messages_for_cloud_contain_no_phones():
    pseudonymizer = Pseudonymizer()
    source = _source(POLICY)
    fitted = fit_sources([source], [pseudonymizer.mask(source.text)], budget_tokens=4000)
    messages = build_messages(pseudonymizer.mask("Кому звонить? Мой номер +7 999 123-45-67"), fitted, [])
    payload = json.dumps(messages, ensure_ascii=False)
    assert not PHONE.search(payload)
    assert "Соколова" not in payload
    assert "[ТЕЛЕФОН_1]" in payload


def test_labels_restored_in_answer():
    pseudonymizer = Pseudonymizer()
    pseudonymizer.mask(POLICY)
    restored = pseudonymizer.restore("Звоните [ФИО_1]: [ТЕЛЕФОН_1].")
    assert restored == "Звоните Анна Соколова: +7 916 204-55-81."


def test_same_value_gets_same_label():
    pseudonymizer = Pseudonymizer()
    first = pseudonymizer.mask("Телефон +7 916 204-55-81")
    second = pseudonymizer.mask("Ещё раз: +7 916 204-55-81")
    assert first.endswith("[ТЕЛЕФОН_1]") and second.endswith("[ТЕЛЕФОН_1]")


def test_technical_words_are_not_masked():
    text = "Транзакция от пользователя analytics_ro держала pgbouncer"
    assert Pseudonymizer().mask(text) == text
