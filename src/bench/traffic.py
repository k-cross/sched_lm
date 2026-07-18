import asyncio
import random
import time
from collections import Counter, defaultdict
from typing import TYPE_CHECKING, Any

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

# RFC-0001 §2 client-supplied session identity. Without it the EPP falls back to hashing
# the conversation's opening messages, which collide across sessions that share a canned
# opener -- concurrent sessions then pollute each other's per-tool gap statistics.
SESSION_HEADER = "x-session-id"

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
        # Failures aggregated by reason so a run against a saturated backend prints one
        # tally instead of a line per dropped request. See error_summary().
        self.error_reasons: Counter[str] = Counter()

    def record_error(self, reason: str) -> None:
        self.errors += 1
        self.error_reasons[reason] += 1

    def error_summary(self) -> str:
        """One-line breakdown of failure reasons, most common first."""
        return ", ".join(f"{n}x {reason}" for reason, n in self.error_reasons.most_common())


async def _stream_chat(
    session: aiohttp.ClientSession,
    url: str,
    route: str,
    messages: list[dict[str, Any]],
    result: BenchmarkResult,
    max_tokens: int = 50,
    kv_priority: str | None = None,
    session_id: int | None = None,
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
    if session_id is not None:
        headers[SESSION_HEADER] = str(session_id)

    start_time = time.monotonic()
    ttft = None

    try:
        async with session.post(url, json=payload, headers=headers) as response:
            if response.status != 200:
                # 503 is the EPP shedding load (saturation / "no endpoint candidates")
                # under a qps the backend can't serve -- expected, not a client bug.
                reason = "load shed (503)" if response.status == 503 else f"http {response.status}"
                result.record_error(reason)
                return

            async for line in response.content:
                # The sim reports request errors (e.g. context-length overflow) as a
                # streaming 200 whose first data chunk is an error object -- surface
                # them instead of scoring silent failures as successes.
                if line.startswith(b'data: {"error"'):
                    result.record_error("in-stream error")
                    return
                if line and ttft is None:
                    ttft = time.monotonic() - start_time
                    result.ttfts.append(ttft)

            e2e = time.monotonic() - start_time
            result.e2e_latencies.append(e2e)
            result.successes += 1
    except aiohttp.ClientPayloadError:
        # Truncated/reset response body -- the EPP resets the ext_proc stream when it
        # sheds a request mid-flight under saturation. Same root cause as a 503.
        result.record_error("load shed (stream reset)")
    except aiohttp.ClientError as e:
        result.record_error(f"transport ({type(e).__name__})")


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
    `concurrency`. Within a session, turns are paced by the workload's arrival gaps --
    think time plus tool latency -- so a timing-observing router (the EPP's
    kv-cache-priority plugin) sees the same re-arrival gaps the generator modeled.
    This is the live counterpart to the offline engine: the same generated workload,
    but routed by the real EPP instead of the modeled cost function.
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
            start = time.monotonic()
            base = turns[0].arrival
            for turn in turns:
                delay = (turn.arrival - base) - (time.monotonic() - start)
                if delay > 0:
                    await asyncio.sleep(delay)
                await _stream_chat(
                    session,
                    target_url,
                    route,
                    turn.messages,
                    result,
                    kv_priority=kv_priority,
                    session_id=turn.session_id,
                )

    await asyncio.gather(*(run_one(turns) for turns in by_session.values()))
    return result
