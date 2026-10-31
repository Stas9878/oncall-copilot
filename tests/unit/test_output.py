from src.security.pii import Pseudonymizer
from src.generation.output import REMOVED_LINK, OutputFilter, strip_foreign_links

DOMAINS = ("lumenpay.ru",)


def test_foreign_link_removed():
    text = "Подробно: https://evil.example/login и https://status.lumenpay.ru"
    assert strip_foreign_links(text, DOMAINS) == f"Подробно: {REMOVED_LINK} и https://status.lumenpay.ru"


def test_lookalike_domain_removed():
    assert strip_foreign_links("https://lumenpay.ru.evil.example/x", DOMAINS) == REMOVED_LINK


def test_link_split_across_stream_chunks_removed():
    output = OutputFilter(Pseudonymizer(), DOMAINS)
    pieces = ["Откройте https://ev", "il.example/res", "et сейчас [1]"]
    shown = "".join(output.feed(piece) for piece in pieces) + output.flush()
    assert shown == f"Откройте {REMOVED_LINK} сейчас [1]"


def test_fullwidth_citation_normalized():
    output = OutputFilter(Pseudonymizer(), DOMAINS)
    assert output.apply("Снизьте темп【2†L4-L9】.") == "Снизьте темп[2]."


def test_nonbreaking_hyphen_replaced_in_identifiers():
    output = OutputFilter(Pseudonymizer(), DOMAINS)
    assert output.apply("Флаг DISPATCH_READ_PRIMARY=true для orders‑db, 80 %") == (
        "Флаг DISPATCH_READ_PRIMARY=true для orders-db, 80 %"
    )
