# Один образ для API, воркера и миграций: команда задаётся в docker-compose.yml.
FROM python:3.14-slim

# Tesseract с русским языком: OCR сканированных страниц PDF.
RUN apt-get update \
    && apt-get install -y --no-install-recommends tesseract-ocr tesseract-ocr-rus \
    && rm -rf /var/lib/apt/lists/*

COPY --from=ghcr.io/astral-sh/uv:0.9 /uv /bin/uv

WORKDIR /app
ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_NO_DEV=1 \
    PATH="/app/.venv/bin:$PATH" \
    HF_HOME=/models

# Сначала зависимости, потом код: правка кода не пересобирает слой с torch.
COPY pyproject.toml uv.lock ./
RUN --mount=type=cache,target=/root/.cache/uv uv sync --locked --no-install-project

COPY alembic.ini ./
COPY migrations ./migrations
COPY src ./src
RUN --mount=type=cache,target=/root/.cache/uv uv sync --locked

# Процесс без root. Каталоги томов создаются заранее, чтобы том получил владельца app.
RUN useradd --create-home app && mkdir -p /app/data/uploads /models && chown -R app /app/data /models
USER app
