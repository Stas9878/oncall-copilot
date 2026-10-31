"""Оценка на эталонных кейсах из eval/cases.json: поиск и запись ответов."""

import sys
import json
import anyio
import asyncio
import hashlib
import argparse
from pathlib import Path
from sqlalchemy import select
from dataclasses import asdict

from src.security.auth import Principal
from src.core.config import get_settings
from src.core.storage import FileStorage
from src.search.retrieval import retrieve
from src.security.uploads import detect_kind
from src.services.answer import Question, answer
from src.ingestion.pipeline import ingest_document
from src.observability.versions import fingerprint
from src.core.runtime import Resources, open_resources
from src.db.models import Document, DocumentKind, Role, Tenant

CASES = Path("eval/cases.json")
RECORDED = Path("eval/recorded_answers.json")
CORPUS = Path("corpus")
TENANTS = {"lumen": ("Lumen Pay", ["lumenpay.ru"]), "kira": ("Kira Logistics", ["kira.dev"])}
MIN_HIT_RATE = 0.85


def load_cases() -> list[dict]:
    return json.loads(CASES.read_text(encoding="utf-8"))


def needs_images(case: dict) -> bool:
    return "image" in case or any(name.endswith(".png") for name, _ in case.get("expected_sections", []))


async def principal_for(resources: Resources, slug: str) -> Principal:
    async with resources.sessions() as session:
        tenant = await session.scalar(select(Tenant).where(Tenant.slug == slug))
    if tenant is None:
        raise SystemExit(f"Команды {slug} нет: сначала python -m src.evals seed")
    return Principal(tenant.id, tenant.id, slug, Role.READER, tuple(tenant.link_domains), tenant.monthly_token_budget)


async def seed(resources: Resources, skip_images: bool) -> None:
    """Команды и их документы без API, очереди и ключей: для CI и оценки."""
    storage = FileStorage(resources.settings.storage)
    for slug, (name, domains) in TENANTS.items():
        async with resources.sessions.begin() as session:
            tenant = await session.scalar(select(Tenant).where(Tenant.slug == slug))
            if tenant is None:
                tenant = Tenant(slug=slug, name=name, link_domains=domains)
                session.add(tenant)
                await session.flush()
            new_ids = []
            for path in sorted(await anyio.to_thread.run_sync(lambda s=slug: list((CORPUS / s).iterdir()))):
                data = await anyio.Path(path).read_bytes()
                kind = detect_kind(data, path.name)
                if skip_images and kind is DocumentKind.IMAGE:
                    continue
                sha256 = hashlib.sha256(data).hexdigest()
                exists = await session.scalar(
                    select(Document.id).where(Document.tenant_id == tenant.id, Document.sha256 == sha256)
                )
                if exists:
                    continue
                await storage.save(tenant.id, sha256, kind, data)
                document = Document(
                    tenant_id=tenant.id, filename=path.name, kind=kind, sha256=sha256, size_bytes=len(data)
                )
                session.add(document)
                await session.flush()
                new_ids.append(document.id)
        for document_id in new_ids:
            await ingest_document(resources, document_id)
        print(f"{slug}: проиндексировано новых документов {len(new_ids)}")


async def evaluate_retrieval(resources: Resources, skip_images: bool) -> bool:
    settings = resources.settings.retrieval
    hits, reciprocal, scored, no_answer_ok, no_answer_total = 0, 0.0, 0, 0, 0
    for case in load_cases():
        # Кейс со скрином ищется по описанию картинки от vision-модели: его
        # проверяет только сквозной прогон record.
        if "image" in case or (skip_images and needs_images(case)):
            continue
        principal = await principal_for(resources, case["tenant"])
        sources = await retrieve(
            resources, principal, case["question"], limit=settings.top_sections, min_score=settings.min_score
        )
        found = [(source.filename, source.heading) for source in sources]
        if case.get("expect_no_answer"):
            no_answer_total += 1
            no_answer_ok += not found
            print(f"{'ok  ' if not found else 'FAIL'} {case['id']:28} ожидали пусто, найдено {len(found)}")
            continue
        expected = {tuple(item) for item in case["expected_sections"]}
        rank = next((i + 1 for i, item in enumerate(found) if item in expected), None)
        scored += 1
        hits += rank is not None
        reciprocal += 1 / rank if rank else 0.0
        print(f"{'ok  ' if rank else 'FAIL'} {case['id']:28} ранг нужного раздела: {rank or '—'}")
    hit_rate = hits / scored if scored else 0.0
    mrr = reciprocal / scored if scored else 0.0
    print(
        f"\nHit@{settings.top_sections} = {hit_rate:.2f}, MRR = {mrr:.2f} на {scored} кейсах, "
        f"«нет ответа» верно {no_answer_ok}/{no_answer_total}"
    )
    return hit_rate >= MIN_HIT_RATE and no_answer_ok == no_answer_total


async def record(resources: Resources) -> None:
    """Записывает ответы модели на все кейсы. Запускается вручную, не в CI."""
    answers = {}
    for case in load_cases():
        principal = await principal_for(resources, case["tenant"])
        image = await anyio.Path(case["image"]).read_bytes() if "image" in case else None
        result = await answer(resources, principal, Question(case["question"], [], image))
        answers[case["id"]] = asdict(result)
        print(f"{case['id']:28} {result.backend or 'нет ответа':8} цитат: {len(result.citations)}")
    payload = {"fingerprint": fingerprint(resources.settings), "answers": answers}
    await anyio.Path(RECORDED).write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"Записано в {RECORDED}")


async def main() -> int:
    parser = argparse.ArgumentParser(prog="src.evals")
    parser.add_argument("command", choices=["seed", "retrieval", "record"])
    parser.add_argument("--skip-images", action="store_true", help="без картинок: их описание требует облачной модели")
    args = parser.parse_args()
    async with open_resources(get_settings()) as resources:
        if args.command == "seed":
            await seed(resources, args.skip_images)
        elif args.command == "retrieval":
            return 0 if await evaluate_retrieval(resources, args.skip_images) else 1
        else:
            await record(resources)
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
