"""Ответ модели перед отправкой клиенту."""

import re

from src.security.pii import Pseudonymizer

LINK = re.compile(r"https?://([^/\s()\[\]<>\"']+)[^\s()\[\]<>\"']*")
# gpt-oss иногда ставит ссылку на источник в полноширинных скобках 【1】,
# хотя промпт просит [1]. Приводим к одному виду, иначе цитаты не распознаются.
FULLWIDTH_CITATION = re.compile(r"【(\d+)[^】]*】")
REMOVED_LINK = "[ссылка удалена]"
# gpt-oss пишет неразрывный дефис и узкий неразрывный пробел: «orders‑db».
# На вид разницы нет, но скопированное в терминал имя сервиса или флага не сработает.
TYPOGRAPHY = str.maketrans({"\u2010": "-", "\u2011": "-", "\u202f": " ", "\u00a0": " "})


def strip_foreign_links(text: str, allowed_domains: tuple[str, ...]) -> str:
    def replace(match: re.Match) -> str:
        host = match.group(1).lower().split(":")[0]
        if any(host == domain or host.endswith("." + domain) for domain in allowed_domains):
            return match.group(0)
        return REMOVED_LINK

    return LINK.sub(replace, text)


class OutputFilter:
    """Готовит ответ к показу: целиком или по частям потока."""

    def __init__(self, pseudonymizer: Pseudonymizer, allowed_domains: tuple[str, ...]) -> None:
        self._pseudonymizer = pseudonymizer
        self._domains = allowed_domains
        self._pending = ""

    def apply(self, text: str) -> str:
        text = FULLWIDTH_CITATION.sub(lambda m: f"[{m.group(1)}]", text.translate(TYPOGRAPHY))
        return strip_foreign_links(self._pseudonymizer.restore(text), self._domains)

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
