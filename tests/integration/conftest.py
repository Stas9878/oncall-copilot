"""Окружение для интеграционных тестов: настоящие Postgres, Redis и Qdrant."""

import uuid
import httpx
import pytest
import hashlib
import pytest_asyncio
from pathlib import Path
from dataclasses import dataclass
from sqlalchemy import delete, text
from qdrant_client import models as qm
from collections.abc import AsyncIterator

from src.api.app import create_app
from src.core.config import get_settings
from src.core.storage import FileStorage
from src.ingestion.pipeline import ingest_document
from src.security.auth import Principal, generate_key
from src.core.runtime import Resources, open_resources
from src.search.index import collection_name, ensure_collection
from src.db.models import ApiKey, Document, DocumentKind, Role, Tenant

pytestmark = pytest.mark.integration


@dataclass(frozen=True, slots=True)
class TestTenant:
    principal: Principal
    reader_key: str
    editor_key: str
    document_ids: dict[str, uuid.UUID]


@pytest_asyncio.fixture(scope="session")
async def resources() -> AsyncIterator[Resources]:
    async with open_resources(get_settings()) as opened:
        try:
            async with opened.engine.connect() as connection:
                await connection.execute(text("SELECT 1"))
        except OSError:
            pytest.skip("Postgres недоступен: запустите docker compose up -d postgres redis qdrant")
        await ensure_collection(opened.qdrant, collection_name(opened.settings.models))
        yield opened


async def _create_tenant(resources: Resources, slug: str, files: list[Path]) -> TestTenant:
    reader_key, reader_hash = generate_key()
    editor_key, editor_hash = generate_key()
    tenant = Tenant(slug=slug, name=slug, link_domains=[])
    async with resources.sessions.begin() as session:
        session.add(tenant)
        await session.flush()
        session.add_all(
            [
                ApiKey(tenant_id=tenant.id, name="reader", key_hash=reader_hash, role=Role.READER),
                ApiKey(tenant_id=tenant.id, name="editor", key_hash=editor_hash, role=Role.EDITOR)
            ]
        )
        documents = {}
        for path in files:
            data = path.read_bytes()
            sha256 = hashlib.sha256(data).hexdigest()
            await FileStorage(resources.settings.storage).save(tenant.id, sha256, DocumentKind.MARKDOWN, data)
            document = Document(
                tenant_id=tenant.id, filename=path.name, kind=DocumentKind.MARKDOWN, sha256=sha256, size_bytes=len(data)
            )
            session.add(document)
            await session.flush()
            documents[path.name] = document.id
    for document_id in documents.values():
        await ingest_document(resources, document_id)
    principal = Principal(uuid.uuid4(), tenant.id, slug, Role.READER, (), 2_000_000)
    return TestTenant(principal, reader_key, editor_key, documents)


@pytest_asyncio.fixture(scope="session")
async def tenants(resources: Resources) -> AsyncIterator[dict[str, TestTenant]]:
    suffix = uuid.uuid4().hex[:8]
    created = {
        "lumen": await _create_tenant(resources, f"test-lumen-{suffix}", [Path("corpus/lumen/oncall_policy_lumen.md")]),
        "kira": await _create_tenant(resources, f"test-kira-{suffix}", [Path("corpus/kira/oncall_policy_kira.md")])
    }
    yield created
    for tenant in created.values():
        tenant_id = tenant.principal.tenant_id
        await resources.qdrant.delete(
            collection_name(resources.settings.models),
            points_selector=qm.FilterSelector(
                filter=qm.Filter(must=[qm.FieldCondition(key="tenant_id", match=qm.MatchValue(value=str(tenant_id)))])
            )
        )
        async with resources.sessions.begin() as session:
            await session.execute(delete(Tenant).where(Tenant.id == tenant_id))


@pytest_asyncio.fixture(scope="session")
async def client(resources: Resources) -> AsyncIterator[httpx.AsyncClient]:
    app = create_app()
    # Ресурсы уже открыты фикстурой: lifespan приложения не запускаем.
    app.state.resources = resources
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as http:
        yield http
