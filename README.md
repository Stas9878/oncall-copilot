# On-call Copilot

Ассистент инженера для разбора инцидентов. Команда загружает ранбуки в PDF, постмортемы в Markdown, скрины дашбордов и схемы. Инженер задаёт вопрос текстом или прикладывает скрин и получает ответ со ссылками на источники. По алерту Alertmanager сервис собирает карточку инцидента: уровень серьёзности по политике команды и первые шаги, каждый с цитатой из ранбука.

Финальный проект курса «LLM для Python-разработчиков: продвинутый уровень».

## Архитектура

| Компонент | Роль |
|---|---|
| `api` (FastAPI) | `/v1/ask`, `/v1/ask/stream`, `/v1/incidents/extract`, `/v1/documents`, `/ready`, `/metrics` |
| `worker` (taskiq) | индексация загруженных файлов: разбор PDF и OCR, описание картинок, эмбеддинги |
| Postgres | команды и ключи, реестр документов, тексты разделов, расход токенов за месяц |
| Redis | очередь индексации, кэш ответов, лимиты запросов, выключатель облачной модели |
| Qdrant | векторы фрагментов с фильтром по команде |
| Облачная модель | провайдер с OpenAI-совместимым API. По умолчанию Ollama: основная модель `gpt-oss:120b-cloud` и vision-модель `gemma4:31b-cloud` |
| vLLM на CPU | `Qwen/Qwen3-1.7B`: переформулировка вопросов и резервная модель |
| Phoenix, Prometheus | трейсы, метрики и правила алертов |

Путь вопроса: кэш → переформулировка уточняющего вопроса → поиск 50 кандидатов в Qdrant → реранкинг кросс-энкодером → разделы из Postgres → псевдонимизация ПДн → облачная или резервная модель → возврат ПДн, фильтр ссылок, проверка цитат.

## Запуск

Нужны Docker и облачная модель с OpenAI-совместимым API. По умолчанию это [Ollama](https://ollama.com) с выполненным `ollama signin`, другой провайдер подключается через `.env`, как показано ниже. Если vLLM работает в том же Docker, выделите Docker около 14 ГБ памяти: vLLM с моделью и KV-кэшем занимает около 9 ГБ.

```bash
cp .env.example .env
# vLLM запускается первым: при старте он резервирует память под KV-кэш.
# Если vLLM уже работает отдельно на порту 8000, эту команду пропустите.
docker compose --profile local-llm up -d vllm
docker compose up -d
```

Создайте команду, выпустите ключи и загрузите демо-документы. Ключ выводится один раз, поэтому сразу сохраните его в переменную:

```bash
docker compose exec api python -m src.cli tenant lumen "Lumen Pay" --link-domain lumenpay.ru
export EDITOR_KEY=$(docker compose exec -T api python -m src.cli key lumen editor --name demo)
export READER_KEY=$(docker compose exec -T api python -m src.cli key lumen reader --name demo)

for f in corpus/lumen/*; do
  curl -s -H "Authorization: Bearer $EDITOR_KEY" -F "file=@$f" localhost:8080/v1/documents
done
```

Примеры запросов:

```bash
# Вопрос
curl -s localhost:8080/v1/ask -H "Authorization: Bearer $READER_KEY" -H 'content-type: application/json' \
  -d '{"question": "Эквайер отвечает 429, что делать?"}'

# Вопрос со скрином, ответ потоком
curl -sN localhost:8080/v1/ask/stream -H "Authorization: Bearer $READER_KEY" -H 'content-type: application/json' \
  -d "{\"question\": \"Что происходит и что делать?\", \"image_base64\": \"$(base64 < examples/lumen_acquirer_live.png | tr -d '\n')\"}"

# Карточка инцидента по алерту
curl -s localhost:8080/v1/incidents/extract -H "Authorization: Bearer $READER_KEY" -H 'content-type: application/json' \
  -d "{\"alert\": $(cat examples/alert_lumen_acquirer.json)}"
```

### Другой провайдер облачной модели

Адрес, ключ и модели задаются переменными окружения. Например, для бесплатной модели OpenRouter добавьте в `.env`:

```bash
LLM_CLOUD_API_KEY=sk-or-v1-...
LLM_CLOUD_MODEL=nvidia/nemotron-3.5-super-vl:free
LLM_VISION_MODEL=nvidia/nemotron-3.5-super-vl:free
LLM_REASONING_EFFORT=none
```

В `docker-compose.yml` адрес облака задан для Ollama на хосте. Чтобы заменить его у API и воркера, создайте рядом `docker-compose.override.yml`:

```yaml
services:
  api:
    environment:
      LLM_CLOUD_BASE_URL: https://openrouter.ai/api/v1
  worker:
    environment:
      LLM_CLOUD_BASE_URL: https://openrouter.ai/api/v1
```

При запуске без Docker адрес задаётся в `.env`: `LLM_CLOUD_BASE_URL=https://openrouter.ai/api/v1`.

### Запуск API и воркера без Docker

```bash
uv sync
docker compose up -d postgres redis qdrant phoenix
uv run alembic upgrade head
uv run uvicorn src.api.app:create_app --factory --port 8080
uv run taskiq worker src.worker:broker
```

## Тесты и оценка

```bash
uv run pytest                           # unit- и интеграционные тесты, проверка записанных ответов, модель угроз
uv run python -m src.evals seed         # создать команды lumen и kira и загрузить их документы без API
uv run python -m src.evals retrieval    # Hit@4 и MRR поиска, без вызова моделей
uv run python -m src.evals record       # записать ответы модели в eval/recorded_answers.json
uv run python scripts/load_test.py --key $READER_KEY --levels 1 4 8
```

CI в `.github/workflows/ci.yml` запускает линтер, миграции, тесты и оценку поиска без облачных моделей. Тест `tests/eval` падает, если промпт, модели или параметры поиска изменились, а ответы не перезаписаны.

Модель угроз описана в `threats.json`. У каждой угрозы уровня critical и high есть тест, а `tests/test_threats.py` проверяет, что он существует.

## Мониторинг

- Phoenix: http://localhost:6006, проект `oncall-copilot`. В корневом спане `ask` есть команда и отпечаток версий: модели, хеш промпта, версия разбиения, параметры поиска.
- Prometheus: http://localhost:9090, правила в `deploy/prometheus/alerts.yml`.
- Метрики API: http://localhost:8080/metrics, воркера: порт 9000.

## Ограничения

Цифры сняты на Mac mini M4 Pro: 12 ядер, бесплатный тариф Ollama Cloud.

- Пропускная способность упирается в облачную модель: 0.6–0.9 вопроса в секунду при 1, 4 и 8 параллельных запросах. Ollama Cloud обрабатывал облачные запросы по очереди.
- p95 при 4 параллельных запросах в трёх прогонах: 5.4, 6.0 и 11.4 секунды. Критерий в 8 секунд выполняется не всегда.
- При 8 параллельных запросах облачная модель отвечает 429, часть ответов даёт резервная модель, p95 20–24 секунды.
- Повторный вопрос берётся из кэша: p95 0.4 секунды при 8 параллельных повторах.
- Резервная модель `Qwen3-1.7B` на CPU отвечает за 9–18 секунд, обязательные факты есть в 8 из 12 эталонных кейсов. Её ответы помечены полем `degraded_reason` и не кэшируются.
- Все сервисы вместе с vLLM занимают около 13 ГБ: vLLM 9.2 ГБ, API 1.7 ГБ, воркер 1.2 ГБ, остальное 0.9 ГБ.
- NER для псевдонимизации иногда ошибается: пропущенное имя уйдёт в облачную модель, а слово в начале предложения («Эквайер», «Алерт») может стать меткой `[ФИО_n]`.
- OCR ошибается на смеси кириллицы и латиницы: «SMS» в скане распознаётся как «$М$».
