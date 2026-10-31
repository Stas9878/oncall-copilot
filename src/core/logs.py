"""Структурные логи: одна JSON-строка на событие."""

import sys
import logging
import structlog


def configure_logging(level: str = "INFO", json: bool = True) -> None:
    renderer = structlog.processors.JSONRenderer() if json else structlog.dev.ConsoleRenderer()
    structlog.configure(
        processors=[
            structlog.contextvars.merge_contextvars,
            structlog.processors.add_log_level,
            structlog.processors.TimeStamper(fmt="iso", utc=True),
            structlog.processors.format_exc_info,
            renderer
        ],
        wrapper_class=structlog.make_filtering_bound_logger(logging.getLevelNamesMapping()[level]),
        logger_factory=structlog.PrintLoggerFactory(sys.stdout),
        cache_logger_on_first_use=True
    )
    # Библиотеки пишут через стандартный logging. Их INFO-шум (каждый HTTP-запрос
    # к Hugging Face, каждый вызов Qdrant) в логах сервиса не нужен.
    logging.basicConfig(level=logging.WARNING, stream=sys.stdout, format="%(name)s %(levelname)s %(message)s")
    for noisy in ("httpx", "httpcore", "sentence_transformers", "huggingface_hub"):
        logging.getLogger(noisy).setLevel(logging.WARNING)


log = structlog.get_logger()
