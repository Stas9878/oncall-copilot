"""Нагрузочный тест API: параллельные вопросы в потоковый эндпоинт."""

import json
import math
import time
import anyio
import httpx
import asyncio
import argparse
import statistics
from collections import Counter


def p95(values: list[float]) -> float:
    ordered = sorted(values)
    return ordered[math.ceil(0.95 * len(ordered)) - 1]


async def ask(client: httpx.AsyncClient, question: str) -> dict:
    started = time.perf_counter()
    first_piece, outcome, event = None, "error", None
    try:
        async with client.stream("POST", "/v1/ask/stream", json={"question": question}) as response:
            if response.status_code != 200:
                return {"outcome": f"http_{response.status_code}", "latency": time.perf_counter() - started}
            async for line in response.aiter_lines():
                if line.startswith("event:"):
                    event = line.split(":", 1)[1].strip()
                elif line.startswith("data:") and event == "token" and first_piece is None:
                    first_piece = time.perf_counter() - started
                elif line.startswith("data:") and event == "done":
                    done = json.loads(line[5:])
                    outcome = "cached" if done["cached"] else (done["backend"] or "no_answer")
                elif line.startswith("data:") and event == "error":
                    outcome = "stream_error"
    except httpx.HTTPError as error:
        outcome = type(error).__name__
    return {"outcome": outcome, "latency": time.perf_counter() - started, "first_piece": first_piece}


async def run_level(client: httpx.AsyncClient, questions: list[str], concurrency: int, requests: int) -> None:
    gate = asyncio.Semaphore(concurrency)

    async def one(number: int) -> dict:
        async with gate:
            return await ask(client, questions[number % len(questions)])

    started = time.perf_counter()
    results = await asyncio.gather(*(one(n) for n in range(requests)))
    elapsed = time.perf_counter() - started
    latencies = [r["latency"] for r in results]
    firsts = [r["first_piece"] for r in results if r.get("first_piece") is not None]
    outcomes = Counter(r["outcome"] for r in results)
    print(
        f"параллельно={concurrency:2} запросов={requests:3} rps={requests / elapsed:5.2f} "
        f"p50={statistics.median(latencies):5.1f}s p95={p95(latencies):5.1f}s "
        f"ttft p50={statistics.median(firsts) if firsts else float('nan'):4.1f}s "
        f"исходы={dict(outcomes)}"
    )


async def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="http://localhost:8080")
    parser.add_argument("--key", required=True)
    parser.add_argument("--tenant", default="lumen")
    parser.add_argument("--levels", type=int, nargs="+", default=[1, 4, 8])
    parser.add_argument("--requests", type=int, default=16)
    parser.add_argument("--repeat", action="store_true", help="одни и те же вопросы: проверка кэша")
    args = parser.parse_args()

    cases = json.loads(await anyio.Path("eval/cases.json").read_text(encoding="utf-8"))
    base = [c["question"] for c in cases if c["tenant"] == args.tenant and "image" not in c]
    headers = {"Authorization": f"Bearer {args.key}"}
    async with httpx.AsyncClient(base_url=args.url, headers=headers, timeout=120) as client:
        for level in args.levels:
            questions = base if args.repeat else [f"{q} (запрос {level}-{n})" for n, q in enumerate(base * 4)]
            await run_level(client, questions, level, args.requests)


if __name__ == "__main__":
    asyncio.run(main())
