"""Настройки сервиса из переменных окружения."""

from functools import lru_cache
from pydantic import Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class _Group(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")


class PostgresSettings(_Group, env_prefix="POSTGRES_"):
    dsn: SecretStr = SecretStr("postgresql+asyncpg://copilot:copilot@localhost:5432/copilot")
    pool_size: int = 10


class RedisSettings(_Group, env_prefix="REDIS_"):
    url: str = "redis://localhost:6379/0"


class QdrantSettings(_Group, env_prefix="QDRANT_"):
    url: str = "http://localhost:6333"


class LLMSettings(_Group, env_prefix="LLM_"):
    """Два бэкенда инференса с OpenAI-совместимым API: локальный vLLM и облако через Ollama."""

    local_base_url: str = "http://localhost:8000/v1"
    local_api_key: SecretStr = SecretStr("EMPTY")
    local_model: str = "Qwen/Qwen3-1.7B"
    cloud_base_url: str = "http://localhost:11434/v1"
    cloud_api_key: SecretStr = SecretStr("ollama")
    cloud_model: str = "gpt-oss:120b-cloud"
    vision_model: str = "gemma4:31b-cloud"
    # gpt-oss рассуждает перед ответом. low сокращает рассуждение и расход квоты.
    reasoning_effort: str = "low"
    # Окно контекста локальной модели, как в --max-model-len у vLLM.
    # Резервный промпт собирается под него: меньше источников и короче ответ.
    local_context_tokens: int = 2048
    timeout_seconds: float = 60.0


class ModelSettings(_Group, env_prefix="MODELS_"):
    """Локальные модели для эмбеддингов и реранкинга, работают на CPU."""

    text_embedder: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    reranker: str = "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1"


class RetrievalSettings(_Group, env_prefix="RETRIEVAL_"):
    top_sections: int = 4
    # Порог оценки кросс-энкодера. Подобран на eval/cases.json: разделы с ответом
    # проходят его, у вопросов без ответа в базе ни один фрагмент не проходит.
    # Оценка поиска в CI проверяет оба условия при каждом изменении.
    min_score: float = -2.5
    # Сколько токенов источников помещается в промпт.
    context_tokens: int = 2500


class StorageSettings(_Group, env_prefix="STORAGE_"):
    uploads_dir: str = "data/uploads"
    max_upload_mb: int = 20
    max_pdf_pages: int = 50


class Settings(_Group):
    environment: str = Field(default="local", alias="APP_ENV")
    postgres: PostgresSettings = Field(default_factory=PostgresSettings)
    redis: RedisSettings = Field(default_factory=RedisSettings)
    qdrant: QdrantSettings = Field(default_factory=QdrantSettings)
    llm: LLMSettings = Field(default_factory=LLMSettings)
    models: ModelSettings = Field(default_factory=ModelSettings)
    retrieval: RetrievalSettings = Field(default_factory=RetrievalSettings)
    storage: StorageSettings = Field(default_factory=StorageSettings)


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Читает окружение один раз за процесс."""
    return Settings()
