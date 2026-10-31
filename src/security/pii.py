"""Псевдонимизация персональных данных перед облачной моделью."""

import re
import anyio
import threading
from functools import lru_cache
from collections.abc import Sequence
from natasha import Doc, NewsEmbedding, NewsNERTagger, Segmenter

PHONE = re.compile(r"(?<!\d)(?:\+7|8)[\s-]?\(?\d{3}\)?[\s-]?\d{3}[\s-]?\d{2}[\s-]?\d{2}(?!\d)")
EMAIL = re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}")
_PLACEHOLDER = re.compile(r"\[(?:ТЕЛЕФОН|EMAIL|ФИО)_\d+\]")

_ner_lock = threading.Lock()


@lru_cache(maxsize=1)
def _ner() -> tuple[Segmenter, NewsNERTagger]:
    return Segmenter(), NewsNERTagger(NewsEmbedding())


class Pseudonymizer:
    """Одна на запрос: одинаковое значение во всех фрагментах получает одну метку."""

    def __init__(self) -> None:
        self._by_value: dict[str, str] = {}
        self._by_label: dict[str, str] = {}
        self._counters: dict[str, int] = {}

    def _label(self, kind: str, value: str) -> str:
        if value not in self._by_value:
            self._counters[kind] = self._counters.get(kind, 0) + 1
            label = f"[{kind}_{self._counters[kind]}]"
            self._by_value[value] = label
            self._by_label[label] = value
        return self._by_value[value]

    def mask(self, text: str) -> str:
        text = PHONE.sub(lambda m: self._label("ТЕЛЕФОН", m.group(0)), text)
        text = EMAIL.sub(lambda m: self._label("EMAIL", m.group(0)), text)
        return self._mask_persons(text)

    async def mask_many(self, texts: Sequence[str]) -> list[str]:
        """Маскирует тексты в отдельном потоке: NER занимает CPU, а цикл событий обслуживает другие запросы."""
        return await anyio.to_thread.run_sync(lambda: [self.mask(text) for text in texts])

    def _mask_persons(self, text: str) -> str:
        segmenter, tagger = _ner()
        doc = Doc(text)
        # Natasha не потокобезопасна, а тексты разных запросов маскируются в потоках параллельно.
        with _ner_lock:
            doc.segment(segmenter)
            doc.tag_ner(tagger)
        parts, cursor = [], 0
        for span in sorted((s for s in doc.spans if s.type == "PER"), key=lambda s: s.start):
            if span.start < cursor or _PLACEHOLDER.fullmatch(span.text):
                continue
            parts += [text[cursor : span.start], self._label("ФИО", span.text)]
            cursor = span.stop
        parts.append(text[cursor:])
        return "".join(parts)

    def restore(self, text: str) -> str:
        """Возвращает настоящие значения на место меток в ответе модели."""
        return _PLACEHOLDER.sub(lambda m: self._by_label.get(m.group(0), m.group(0)), text)

    @property
    def masked_count(self) -> int:
        return len(self._by_value)
