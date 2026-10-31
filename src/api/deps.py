"""Зависимости FastAPI: ресурсы процесса, сессия Postgres, проверка ключа."""

import structlog
from typing import Annotated
from collections.abc import AsyncIterator
from sqlalchemy.ext.asyncio import AsyncSession
from fastapi import Depends, HTTPException, Request, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from src.core.runtime import Resources
from src.db.session import session_scope
from src.security.auth import Principal, resolve_key

_bearer = HTTPBearer(auto_error=False)


def get_resources(request: Request) -> Resources:
    return request.app.state.resources


ResourcesDep = Annotated[Resources, Depends(get_resources)]


async def get_session(resources: ResourcesDep) -> AsyncIterator[AsyncSession]:
    async for session in session_scope(resources.sessions):
        yield session


SessionDep = Annotated[AsyncSession, Depends(get_session)]


async def get_principal(
    session: SessionDep,
    credentials: Annotated[HTTPAuthorizationCredentials | None, Depends(_bearer)]
) -> Principal:
    principal = await resolve_key(session, credentials.credentials) if credentials else None
    if principal is None:
        raise HTTPException(
            status.HTTP_401_UNAUTHORIZED,
            "Нужен действующий ключ API в заголовке Authorization: Bearer",
            headers={"WWW-Authenticate": "Bearer"}
        )
    # С этого момента команда попадает в каждую строку лога запроса.
    structlog.contextvars.bind_contextvars(tenant=principal.tenant_slug)
    return principal


PrincipalDep = Annotated[Principal, Depends(get_principal)]


async def require_editor(principal: PrincipalDep) -> Principal:
    """Запись в базу знаний: только ключи с ролью editor."""
    if not principal.can_write:
        raise HTTPException(status.HTTP_403_FORBIDDEN, "Загрузка документов доступна только ключу с ролью editor")
    return principal


EditorDep = Annotated[Principal, Depends(require_editor)]
