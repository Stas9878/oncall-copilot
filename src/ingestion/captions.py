"""Описание картинки vision-моделью при загрузке."""

import base64
from openai import AsyncOpenAI

CAPTION_PROMPT = (
    "Это картинка из базы знаний инженерной команды: скрин дашборда, схема архитектуры "
    "или фото документа. Опиши её на русском для поиска. Перечисли тип картинки, названия "
    "сервисов и панелей, дату и время, единицы измерения и видимые значения. Отдельно "
    "опиши аномалии: где и во сколько график резко меняется и до каких значений. "
    "Пиши только то, что видно на картинке, без догадок о причинах. Обычный текст без "
    "Markdown и формул, не больше 12 предложений."
)


# Форматы, которые принимает vision-модель, по первым байтам файла.
IMAGE_SIGNATURES = {b"\x89PNG\r\n\x1a\n": "image/png", b"\xff\xd8\xff": "image/jpeg"}


def image_mime(data: bytes) -> str:
    """MIME по сигнатуре файла. Другие форматы vision-модели не отправляем."""
    for signature, mime in IMAGE_SIGNATURES.items():
        if data.startswith(signature):
            return mime
    raise ValueError("Поддерживаются картинки PNG и JPEG")


# Для вопроса со скрином нужен не полный пересказ картинки, а суть инцидента:
# длинная подпись тянет поиск к похожим дашбордам, а не к нужному ранбуку.
SEARCH_PROMPT = (
    "Это скрин, который инженер приложил к вопросу. Одной строкой до 15 слов "
    "назови, что не так: какой сервис или зависимость, какая метрика и что с ней "
    "происходит. Например: «рост ответов 429 от эквайера, падение успешных платежей». "
    "Без вступления и пояснений."
)


async def describe_image(
    client: AsyncOpenAI, model: str, image: bytes, mime: str, prompt: str = CAPTION_PROMPT, max_tokens: int = 700
) -> str:
    data_url = f"data:{mime};base64,{base64.b64encode(image).decode()}"
    response = await client.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {"url": data_url}}
                ]
            }
        ],
        temperature=0,
        max_tokens=max_tokens,
        # Описанию рассуждение не нужно. Рассуждающая модель иначе потратит на него весь max_tokens.
        reasoning_effort="none"
    )
    caption = (response.choices[0].message.content or "").strip()
    if not caption:
        raise RuntimeError(f"{model} вернула пустое описание, finish_reason={response.choices[0].finish_reason}")
    return caption
