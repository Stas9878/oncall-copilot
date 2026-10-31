"""Трейсы в Phoenix."""

from phoenix.otel import register
from openinference.instrumentation import OITracer, TracerProvider

from src.core.config import Settings

_provider: TracerProvider | None = None


def setup_tracing(settings: Settings) -> None:
    """Подключает экспорт трейсов. Вызывается один раз при старте процесса."""
    global _provider
    if settings.telemetry.enabled and _provider is None:
        _provider = register(
            project_name=settings.telemetry.project_name,
            endpoint=settings.telemetry.phoenix_endpoint,
            batch=True,
            auto_instrument=True,
            verbose=False
        )


def tracer() -> OITracer:
    # Без setup_tracing спаны создаются, но никуда не отправляются.
    return (_provider or TracerProvider()).get_tracer("oncall-copilot")
