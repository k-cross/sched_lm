import asyncio
import random
import time
from collections import defaultdict
from typing import TYPE_CHECKING

import aiohttp

from bench.prompt import SYSTEM_PROMPT

if TYPE_CHECKING:
    from bench.sim.workload import TurnRequest

# Header the gateway uses to pick a routing strategy. The round-robin HTTPRoute
# matches on `x-llmd-route: round-robin`; anything else falls through to the
# default prefix-affinity route. See infra/llm-d/inference-pool.yaml.
ROUTE_HEADER = "x-llmd-route"

# RFC-0001 §2 router-path directive header, honored by the forked llm-d-inference-sim
# (third_party/). Value grammar: `<int>[; ttl=<Go duration>][; scope=<id>]`. Sending it
# from the client is the escape hatch that proves the directive path end-to-end before any
# EPP work exists to inject it.
KV_CACHE_PRIORITY_HEADER = "x-kv-cache-priority"

USER_PROMPTS = [
    "Write a short poem about the ocean.",
    "Explain quantum computing in simple terms.",
    "Translate 'hello world' to French and Spanish.",
    "What is the capital of Japan?",
    "Write a Python script to reverse a string.",
]


class BenchmarkResult:
    def __init__(self):
        self.ttfts: list[float] = []
        self.e2e_latencies: list[float] = []
        self.successes = 0
        self.errors = 0


async def _stream_chat(
    session: aiohttp.ClientSession,
    url: str,
    route: str,
    messages: list[dict[str, str]],
    result: BenchmarkResult,
    max_tokens: int = 50,
    kv_priority: str | None = None,
):
    """POST a chat-completions request, timing TTFT from the first streamed chunk."""
    payload = {
        "model": "Qwen/Qwen2-0.5B",
        "messages": messages,
        "stream": True,
        "max_tokens": max_tokens,
    }
    headers = {ROUTE_HEADER: route}
    if kv_priority is not None:
        headers[KV_CACHE_PRIORITY_HEADER] = kv_priority

    start_time = time.monotonic()
    ttft = None

    try:
        async with session.post(url, json=payload, headers=headers) as response:
            if response.status != 200:
                result.errors += 1
                return

            async for line in response.content:
                if line and ttft is None:
                    ttft = time.monotonic() - start_time
                    result.ttfts.append(ttft)

            e2e = time.monotonic() - start_time
            result.e2e_latencies.append(e2e)
            result.successes += 1
    except Exception as e:
        print(f"Request failed: {e}")
        result.errors += 1


async def generate_request(
    session: aiohttp.ClientSession,
    url: str,
    route: str,
    result: BenchmarkResult,
    kv_priority: str | None = None,
):
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": random.choice(USER_PROMPTS)},
    ]
    await _stream_chat(session, url, route, messages, result, kv_priority=kv_priority)


async def _bounded_request(
    sem: asyncio.Semaphore,
    session: aiohttp.ClientSession,
    url: str,
    route: str,
    result: BenchmarkResult,
    kv_priority: str | None = None,
):
    try:
        await generate_request(session, url, route, result, kv_priority=kv_priority)
    finally:
        sem.release()


async def run_traffic(
    target_url: str,
    num_requests: int,
    concurrency: int,
    route: str,
    qps: float = 0.0,
    kv_priority: str | None = None,
) -> BenchmarkResult:
    """Drive `num_requests` at the gateway using the given routing strategy.

    Arrivals are paced open-loop to `qps` (when > 0): request i is dispatched at
    roughly start + i/qps regardless of how slow the backend is, which is the
    realistic way to observe queueing under load. `concurrency` is a safety cap
    on in-flight requests so a stalled backend cannot create unbounded tasks.
    When qps <= 0 the generator runs closed-loop, firing as fast as the
    concurrency cap allows.
    """
    result = BenchmarkResult()
    sem = asyncio.Semaphore(concurrency)

    async with aiohttp.ClientSession() as session:
        tasks: list[asyncio.Task[None]] = []
        start = time.monotonic()

        for i in range(num_requests):
            if qps > 0:
                scheduled = start + i / qps
                delay = scheduled - time.monotonic()
                if delay > 0:
                    await asyncio.sleep(delay)

            # Bound in-flight requests; blocks here when concurrency is saturated.
            await sem.acquire()
            task = asyncio.create_task(
                _bounded_request(sem, session, target_url, route, result, kv_priority)
            )
            tasks.append(task)

        if tasks:
            await asyncio.gather(*tasks)

    return result


async def run_session_traffic(
    target_url: str,
    requests: "list[TurnRequest]",
    concurrency: int,
    route: str,
    kv_priority: str | None = None,
) -> BenchmarkResult:
    """Replay tool-calling sessions against the live gateway.

    Turns of a session are sent *sequentially* (so each turn's growing prefix can hit the
    cache the previous turn seeded), while distinct sessions run concurrently up to
    `concurrency`. This is the live counterpart to the offline engine: the same generated
    workload, but routed by the real EPP instead of the modeled cost function.
    """
    result = BenchmarkResult()

    by_session: dict[int, list[TurnRequest]] = defaultdict(list)
    for req in requests:
        by_session[req.session_id].append(req)
    for turns in by_session.values():
        turns.sort(key=lambda r: r.turn_idx)

    sem = asyncio.Semaphore(concurrency)

    async def run_one(turns: "list[TurnRequest]"):
        async with sem, aiohttp.ClientSession() as session:
            for turn in turns:
                await _stream_chat(
                    session, target_url, route, turn.messages, result, kv_priority=kv_priority
                )

    await asyncio.gather(*(run_one(turns) for turns in by_session.values()))
    return result
