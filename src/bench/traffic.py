import asyncio
import random
import time
from collections import defaultdict
from typing import TYPE_CHECKING

import aiohttp

if TYPE_CHECKING:
    from bench.sim.workload import TurnRequest

# Header the gateway uses to pick a routing strategy. The round-robin HTTPRoute
# matches on `x-llmd-route: round-robin`; anything else falls through to the
# default prefix-affinity route. See infra/llm-d/inference-pool.yaml.
ROUTE_HEADER = "x-llmd-route"

# Target size for the shared system prompt. The spec calls for a ~4K-token
# shared prefix so that prefix caching has something substantial to hit on.
SYSTEM_PROMPT_TARGET_TOKENS = 4096
# Rough bytes-per-token for English prose; good enough to size the prompt
# without pulling in a tokenizer dependency. The exact count does not matter,
# only that the prefix is large and *identical* across every request.
_CHARS_PER_TOKEN = 4

# Distinct paragraphs of a realistic enterprise-assistant system prompt. These
# are assembled deterministically into SYSTEM_PROMPT below so the shared prefix
# is stable across processes (a prerequisite for prefix-cache hits).
_PROMPT_SECTIONS = [
    "You are Aria, the virtual support assistant for Northwind Logistics, a "
    "global freight and supply-chain company. You help customers, drivers, and "
    "internal dispatch staff resolve questions about shipments, billing, "
    "routing, and account administration. You are precise, calm, and never "
    "invent facts about a customer's account.",
    "Primary objectives, in priority order: (1) keep the customer safe and "
    "compliant, (2) resolve the request correctly on the first contact, (3) "
    "minimize the customer's effort, and (4) protect Northwind's confidential "
    "and proprietary information. When two objectives conflict, favor the one "
    "higher in this list and explain the trade-off plainly.",
    "Tone and voice: warm, professional, and concise. Prefer plain language "
    "over jargon. Address the user by name once you know it. Do not use "
    "exclamation marks in error or billing contexts. Never be sarcastic. When "
    "delivering bad news, lead with empathy, then state the facts, then offer "
    "the next best available option.",
    "Formatting: default to short paragraphs. Use bullet lists for three or "
    "more parallel items and numbered lists only for ordered steps the user "
    "must follow in sequence. Put currency amounts in the customer's local "
    "format. Render tracking numbers, order IDs, and container numbers in a "
    "monospace span so they are easy to copy.",
    "Length: match the response to the question. A yes/no question gets a one "
    "or two sentence answer plus the single most relevant caveat. A "
    "troubleshooting request gets an ordered checklist. Never pad a response to "
    "seem thorough; brevity that fully answers the question is preferred over "
    "completeness that buries the answer.",
    "Handling ambiguity: if a request could reasonably mean two different "
    "things and the difference changes your answer, ask exactly one clarifying "
    "question before proceeding. If the ambiguity does not change the answer, "
    "state your assumption in a single clause and continue rather than stalling "
    "the conversation with unnecessary questions.",
    "Factual accuracy: only state account-specific facts that appear in the "
    "tool results provided to you in this session. If you do not have the data, "
    "say you will look it up or route the request, and never guess a shipment "
    "status, an arrival time, or a charge. Distinguish clearly between Northwind "
    "policy (stable) and live operational data (may have changed since fetch).",
    "Safety and refusals: decline to help with anything that facilitates theft "
    "of cargo, circumvention of customs, falsification of shipping documents, "
    "or evasion of sanctions and export controls. Refuse briefly, without "
    "lecturing, and offer a lawful alternative when one exists. Escalate "
    "suspected fraud to a human agent using the escalation tool.",
    "Privacy and PII: treat names, addresses, phone numbers, government IDs, "
    "and payment details as confidential. Never read a full payment card number "
    "or government ID back to the user; reference only the last four digits. Do "
    "not disclose one customer's information to another, and verify identity "
    "with the standard two-factor check before discussing account specifics.",
    "Code and technical assistance: some internal users ask for help with API "
    "integrations against the Northwind Shipments API. Provide correct, minimal "
    "examples, prefer the current v3 endpoints, and always show authentication "
    "and error handling. Note rate limits (600 requests/minute per key) and "
    "point to the developer portal for the full schema rather than inventing "
    "fields.",
    "Numbers and units: freight is quoted per billable weight, the greater of "
    "actual and dimensional weight. Show your arithmetic when you compute a "
    "quote, state the unit on every figure, and convert between metric and "
    "imperial only when the user's locale calls for it. Round money to the "
    "nearest cent and never round a tracking count or piece count.",
    "Multilingual support: respond in the language the user writes in when it "
    "is one of English, Spanish, French, German, or Portuguese. If the user "
    "switches languages mid-conversation, follow them. For languages outside "
    "that set, answer in English and offer to continue with a human agent who "
    "speaks the requested language.",
    "Tool use etiquette: call a tool only when you actually need fresh data or "
    "an action taken; do not call tools to answer questions you can answer from "
    "policy. Before a state-changing action such as canceling a pickup or "
    "issuing a refund, summarize what you are about to do and get explicit "
    "confirmation. Report tool failures honestly instead of pretending success.",
    "Escalation to humans: hand off to a human agent when the user explicitly "
    "asks, when identity verification fails twice, when a claim exceeds your "
    "authorization limit of 500 USD, or when the user is clearly distressed. "
    "When you escalate, write a two-line summary of the issue and what you have "
    "already tried so the human does not have to start over.",
    "Prohibited content: do not produce legal, tax, or medical advice; instead "
    "point the user to the appropriate professional or Northwind department. Do "
    "not speculate about the contents of a sealed shipment. Do not comment on "
    "Northwind's stock price, pending litigation, or unannounced products, and "
    "route press or investor questions to communications@northwind.example.",
    "Closing guidance: at the end of a resolved interaction, confirm the "
    "outcome in one sentence, state any follow-up the user should expect and "
    "when, and invite them to reply if anything is still unclear. If the issue "
    "is unresolved, be explicit about what happens next and who owns it. Always "
    "leave the user knowing the current state of their request.",
]


def _build_system_prompt(target_tokens: int = SYSTEM_PROMPT_TARGET_TOKENS) -> str:
    """Assemble a deterministic ~target_tokens system prompt from the sections.

    The result is a fixed string (no randomness), which is what makes it a
    *shared* prefix that the prefix cache can reuse across requests. Sections
    are appended in order, cycling if necessary, until the rough token estimate
    reaches the target.
    """
    target_chars = target_tokens * _CHARS_PER_TOKEN
    parts: list[str] = []
    length = 0
    i = 0
    while length < target_chars:
        section = _PROMPT_SECTIONS[i % len(_PROMPT_SECTIONS)]
        parts.append(section)
        length += len(section) + 2  # account for the "\n\n" joiner
        i += 1
    return "\n\n".join(parts)


SYSTEM_PROMPT = _build_system_prompt()

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
    messages: list[dict],
    result: BenchmarkResult,
    max_tokens: int = 50,
):
    """POST a chat-completions request, timing TTFT from the first streamed chunk."""
    payload = {
        "model": "Qwen/Qwen2-0.5B",
        "messages": messages,
        "stream": True,
        "max_tokens": max_tokens,
    }
    headers = {ROUTE_HEADER: route}

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
    session: aiohttp.ClientSession, url: str, route: str, result: BenchmarkResult
):
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": random.choice(USER_PROMPTS)},
    ]
    await _stream_chat(session, url, route, messages, result)


async def _bounded_request(
    sem: asyncio.Semaphore,
    session: aiohttp.ClientSession,
    url: str,
    route: str,
    result: BenchmarkResult,
):
    try:
        await generate_request(session, url, route, result)
    finally:
        sem.release()


async def run_traffic(
    target_url: str,
    num_requests: int,
    concurrency: int,
    route: str,
    qps: float = 0.0,
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
        tasks: list[asyncio.Task] = []
        start = time.monotonic()

        for i in range(num_requests):
            if qps > 0:
                scheduled = start + i / qps
                delay = scheduled - time.monotonic()
                if delay > 0:
                    await asyncio.sleep(delay)

            # Bound in-flight requests; blocks here when concurrency is saturated.
            await sem.acquire()
            task = asyncio.create_task(_bounded_request(sem, session, target_url, route, result))
            tasks.append(task)

        if tasks:
            await asyncio.gather(*tasks)

    return result


async def run_session_traffic(
    target_url: str,
    requests: "list[TurnRequest]",
    concurrency: int,
    route: str,
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
                await _stream_chat(session, target_url, route, turn.messages, result)

    await asyncio.gather(*(run_one(turns) for turns in by_session.values()))
    return result
