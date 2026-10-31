"""Ответ модели перед отправкой клиенту."""

import re

# gpt-oss иногда ставит ссылку на источник в полноширинных скобках 【1】,
# хотя промпт просит [1]. Приводим к одному виду, иначе цитаты не распознаются.
FULLWIDTH_CITATION = re.compile(r"【(\d+)[^】]*】")
# gpt-oss пишет неразрывный дефис и узкий неразрывный пробел: «orders‑db».
# На вид разницы нет, но скопированное в терминал имя сервиса или флага не сработает.
TYPOGRAPHY = str.maketrans({"\u2010": "-", "\u2011": "-", "\u202f": " ", "\u00a0": " "})


class OutputFilter:
    """Готовит ответ к показу: целиком или по частям потока."""

    def __init__(self) -> None:
        self._pending = ""

    def apply(self, text: str) -> str:
        return FULLWIDTH_CITATION.sub(lambda m: f"[{m.group(1)}]", text.translate(TYPOGRAPHY))

    def feed(self, piece: str) -> str:
        """Принимает часть потока и отдаёт то, что уже безопасно показать."""
        self._pending += piece
        cut = max(self._pending.rfind(" "), self._pending.rfind("\n"))
        if cut < 0:
            return ""
        ready, self._pending = self._pending[: cut + 1], self._pending[cut + 1 :]
        return self.apply(ready)

    def flush(self) -> str:
        ready, self._pending = self._pending, ""
        return self.apply(ready)
