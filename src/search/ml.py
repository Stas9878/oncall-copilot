"""Локальные модели на CPU: эмбеддинги текста и реранкер."""

import anyio
import threading
from collections.abc import Callable, Sequence
from sentence_transformers import CrossEncoder, SentenceTransformer

from src.core.config import ModelSettings

TEXT_DIM = 384


class Models:
    def __init__(self, settings: ModelSettings, max_parallel: int = 2) -> None:
        self._settings = settings
        self._limiter = anyio.CapacityLimiter(max_parallel)
        self._cache: dict[str, SentenceTransformer | CrossEncoder] = {}
        self._lock = threading.Lock()

    def _load[T](self, name: str, factory: Callable[[str], T]) -> T:
        # Загрузка при первом обращении под локом: два параллельных запроса
        # не загрузят одну модель дважды.
        with self._lock:
            if name not in self._cache:
                self._cache[name] = factory(name)
            return self._cache[name]

    def _embedder(self) -> SentenceTransformer:
        return self._load(self._settings.text_embedder, lambda n: SentenceTransformer(n, device="cpu"))

    def _reranker(self) -> CrossEncoder:
        return self._load(self._settings.reranker, lambda n: CrossEncoder(n, device="cpu"))

    async def _run[T](self, fn: Callable[[], T]) -> T:
        return await anyio.to_thread.run_sync(fn, limiter=self._limiter)

    async def embed_texts(self, texts: Sequence[str]) -> list[list[float]]:
        """Векторы фрагментов и вопросов. Нормализованы под косинусное расстояние."""
        vectors = await self._run(
            lambda: self._embedder().encode(list(texts), normalize_embeddings=True, show_progress_bar=False)
        )
        return vectors.tolist()

    async def rerank(self, query: str, passages: Sequence[str]) -> list[float]:
        """Оценка пары «вопрос — фрагмент» кросс-энкодером. Чем больше, тем релевантнее."""
        if not passages:
            return []
        scores = await self._run(
            lambda: self._reranker().predict([(query, p) for p in passages], show_progress_bar=False)
        )
        return [float(score) for score in scores]

    async def warmup(self, *, reranker: bool = True) -> None:
        """Загружает модели заранее, чтобы первый запрос не ждал загрузку."""
        await self.embed_texts(["прогрев"])
        if reranker:
            await self.rerank("прогрев", ["прогрев"])
