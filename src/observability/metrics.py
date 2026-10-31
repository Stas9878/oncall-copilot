"""Метрики Prometheus: /metrics у API."""

from prometheus_client import Counter, Gauge, Histogram

REQUESTS = Counter("copilot_http_requests_total", "HTTP-запросы", ["route", "method", "status"])
REQUEST_SECONDS = Histogram(
    "copilot_http_request_seconds",
    "Время ответа API",
    ["route"],
    buckets=(0.05, 0.1, 0.25, 0.5, 1, 2, 4, 8, 15, 30, 60)
)
LLM_TOKENS = Counter("copilot_llm_tokens_total", "Токены моделей", ["backend", "model", "kind"])
LLM_FALLBACKS = Counter("copilot_llm_fallbacks_total", "Ответы резервной модели", ["reason"])
ANSWERS = Counter("copilot_answers_total", "Ответы на вопросы", ["outcome"])
TENANT_MONTH_TOKENS = Gauge("copilot_tenant_month_tokens", "Облачные токены команды за месяц", ["tenant"])
TENANT_BUDGET_TOKENS = Gauge("copilot_tenant_budget_tokens", "Месячный бюджет токенов команды", ["tenant"])
