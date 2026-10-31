import json
import uuid
from pathlib import Path

from src.db.models import DocumentKind
from src.search.retrieval import Source
from src.generation.prompt import PromptSource
from src.services.extraction import Action, alert_facts, grounded_actions

RUNBOOK_ROW = (
    "Алерт: ReplicaLagHigh | Первое действие: Переключить чтение dispatch на основную базу: DISPATCH_READ_PRIMARY=true"
)


def _prompt_source(text: str) -> PromptSource:
    source = Source(uuid.uuid4(), uuid.uuid4(), "runbook_dispatch.pdf", DocumentKind.PDF, None, 2, "table", text, 3.0)
    return PromptSource(1, source, text)


def test_action_without_quote_in_source_is_dropped():
    sources = [_prompt_source(RUNBOOK_ROW)]
    actions = [
        Action(step="Переключить чтение", source=1, quote="переключить чтение dispatch на основную базу"),
        Action(step="Перезапустить кластер", source=1, quote="Перезапустить весь кластер orders-db"),
        Action(step="Ссылка на несуществующий источник", source=7, quote="Переключить чтение dispatch")
    ]
    kept, dropped = grounded_actions(actions, sources)
    assert [a.step for a in kept] == ["Переключить чтение"]
    assert dropped == 2


def test_alert_facts_taken_without_model():
    facts = alert_facts(json.loads(Path("examples/alert_kira_dispatch.json").read_text(encoding="utf-8")))
    assert facts.alertname == "DispatchQueueGrowing"
    assert facts.started_at == "2026-10-12T14:52:00Z"
    assert facts.symptoms[0].startswith("Очередь dispatch.assign растёт")
