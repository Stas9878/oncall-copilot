"""Отпечаток версий: всё, от чего зависит ответ на один и тот же вопрос."""

import hashlib

from src.core.config import Settings
from src.generation.prompt import SYSTEM_PROMPT
from src.ingestion.chunking import CHUNKER_VERSION


def fingerprint(settings: Settings) -> dict[str, str]:
    return {
        "cloud_model": settings.llm.cloud_model,
        "local_model": settings.llm.local_model,
        "vision_model": settings.llm.vision_model,
        "embedder": settings.models.text_embedder,
        "reranker": settings.models.reranker,
        "chunker": CHUNKER_VERSION,
        "retrieval": f"top={settings.retrieval.top_sections},min={settings.retrieval.min_score}",
        "prompt_sha": hashlib.sha256(SYSTEM_PROMPT.encode()).hexdigest()[:12]
    }
