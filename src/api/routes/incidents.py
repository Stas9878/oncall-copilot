"""Карточка инцидента по алерту, заметке и скрину."""

from typing import Annotated, Any
from fastapi import APIRouter, HTTPException, status
from pydantic import BaseModel, Field, model_validator

from src.security.uploads import decode_screen
from src.services.answer import describe_screen
from src.api.deps import PrincipalDep, ResourcesDep
from src.generation.llm import LLMRouter, LLMUnavailable
from src.services.extraction import ExtractedCard, extract_card

router = APIRouter(prefix="/v1/incidents", tags=["incidents"])

MAX_ALERT_CHARS = 20_000


class ExtractIn(BaseModel):
    alert: Annotated[dict[str, Any] | None, Field(description="webhook Alertmanager")] = None
    note: Annotated[str | None, Field(max_length=2000)] = None
    image_base64: str | None = None

    @model_validator(mode="after")
    def _has_input(self) -> ExtractIn:
        if not (self.alert or self.note or self.image_base64):
            raise ValueError("Нужен алерт, заметка или скрин")
        if self.alert and len(str(self.alert)) > MAX_ALERT_CHARS:
            raise ValueError("Алерт слишком большой")
        return self


@router.post("/extract", response_model=ExtractedCard)
async def extract(body: ExtractIn, principal: PrincipalDep, resources: ResourcesDep) -> ExtractedCard:
    image = decode_screen(body.image_base64) if body.image_base64 else None
    try:
        summary = await describe_screen(resources, LLMRouter(resources), principal, image) if image else None
        card = await extract_card(resources, principal, body.alert, body.note, summary)
    except LLMUnavailable as error:
        raise HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE, str(error)) from error
    except ValueError as error:
        raise HTTPException(status.HTTP_502_BAD_GATEWAY, str(error)) from error
    if card is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "В базе знаний команды нет ничего об этом инциденте")
    return card
