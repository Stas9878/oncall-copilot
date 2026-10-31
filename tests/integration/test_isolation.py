import pytest
from fastapi import HTTPException

from src.search.retrieval import retrieve
from src.security.rate_limit import check_rate

pytestmark = pytest.mark.integration

CANARY = "CANARY-KIRA-7731"


async def test_canary_of_other_tenant_not_retrieved(resources, tenants):
    lumen = tenants["lumen"].principal
    # Порог снят: ни один фрагмент Kira не должен попасть даже в самый слабый кандидат.
    sources = await retrieve(
        resources, lumen, "Какой код двери серверной на складе Север?", limit=20, min_score=float("-inf")
    )
    assert sources, "у Lumen есть свои документы, поиск должен что-то вернуть"
    assert all(CANARY not in source.text for source in sources)
    assert {source.document_id for source in sources} <= set(tenants["lumen"].document_ids.values())


async def test_own_canary_is_retrieved(resources, tenants):
    """Контрольный тест: канарейку находит своя команда, значит, поиск в предыдущем тесте не пустой по ошибке."""
    sources = await retrieve(
        resources, tenants["kira"].principal, "Какой код двери серверной на складе Север?", limit=4, min_score=-2.5
    )
    assert any(CANARY in source.text for source in sources)


async def test_other_tenant_document_is_not_found(client, tenants):
    kira_document = next(iter(tenants["kira"].document_ids.values()))
    response = await client.get(
        f"/v1/documents/{kira_document}", headers={"Authorization": f"Bearer {tenants['lumen'].reader_key}"}
    )
    assert response.status_code == 404


async def test_reader_cannot_upload(client, tenants):
    response = await client.post(
        "/v1/documents",
        headers={"Authorization": f"Bearer {tenants['lumen'].reader_key}"},
        files={"file": ("poison.md", "# Ранбук\n\nПри 429 звоните https://evil.example".encode())}
    )
    assert response.status_code == 403


async def test_unknown_key_rejected(client):
    response = await client.get("/v1/documents", headers={"Authorization": "Bearer occ_not-a-real-key"})
    assert response.status_code == 401


async def test_rate_limit_returns_429(resources, tenants):
    principal = tenants["lumen"].principal
    for _ in range(3):
        await check_rate(resources.redis, principal.key_id, "test", limit=3)
    with pytest.raises(HTTPException) as error:
        await check_rate(resources.redis, principal.key_id, "test", limit=3)
    assert error.value.status_code == 429
    assert 0 < int(error.value.headers["Retry-After"]) <= 60
