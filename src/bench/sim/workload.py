"""Tool-calling session workload: the request *shape* the router has to cope with.

A ``Session`` is a multi-turn conversation. Turn 0 is the shared ~4K system prompt plus a
user message; every later turn appends the previous assistant reply, a tool call, a tool
result, and the next user message. So turn *k*'s token sequence has turn *k-1*'s as a
strict prefix -- the growing prefix is a direct function of the tool-call trace.

Two knobs shape how much prefix is actually reusable:

  * **tool-result size** -- larger results inject more unique tokens per turn, pushing the
    divergence block later and lowering the cross-turn hit rate.
  * **tool-call determinism** -- with some probability a tool returns a *canned* result
    shared across sessions, which turns that turn's tokens into a prefix shared between
    *different* sessions, not just across turns of one session.

Arrival times are baked in at generation time (session Poisson arrivals + per-turn
think-time) so every routing policy sees the *same* arrival stream -- a fair comparison.
The same records drive both the offline engine (via ``block_hashes``) and the live replay
sender (via ``messages``).
"""

import random
from dataclasses import dataclass, field

from bench.sim.blocks import DEFAULT_BLOCK_SIZE, hash_blocks
from bench.traffic import SYSTEM_PROMPT

# A stable word -> token-id vocabulary shared across the whole run. Identical text always
# maps to identical ids, so a shared prefix (system prompt, canned tool result) yields
# identical block hashes -- which is the whole point.
_VOCAB: dict[str, int] = {}


def _tokenize(text: str) -> list[int]:
    ids: list[int] = []
    for word in text.split():
        tid = _VOCAB.get(word)
        if tid is None:
            tid = len(_VOCAB)
            _VOCAB[word] = tid
        ids.append(tid)
    return ids


# Pre-tokenize the shared system prompt once; it is the common prefix of every session.
_SYSTEM_TOKENS = _tokenize(SYSTEM_PROMPT)

_USER_PROMPTS = [
    "where is my shipment right now",
    "why was I charged twice this month",
    "reschedule the pickup for tomorrow",
    "which carrier handles the Berlin route",
    "generate an api key for the sandbox",
]

# Canned (deterministic) tool results, shared across sessions when a tool call is
# deterministic -- e.g. a policy lookup that returns the same document every time.
_CANNED_TOOL_RESULTS = [
    "policy document freight billing terms net thirty standard tariff schedule",
    "carrier directory north south routes service levels transit windows",
    "sandbox environment base url rate limits authentication scopes",
]


@dataclass
class TurnRequest:
    session_id: int
    turn_idx: int
    arrival: float
    tokens: list[int]
    block_hashes: list[int]
    gen_tokens: int
    messages: list[dict] = field(default_factory=list)
    # Ground-truth workload class ("tool" | "rag" | "oneshot"). Used by the engine for
    # per-class metrics only -- policies never see it; they classify from observables.
    kind: str = "tool"


def _unique_words(tag: str, n: int) -> str:
    return " ".join(f"{tag}_w{j}" for j in range(n))


def _arrival_gap(rng: random.Random, rate: float, burst_cv: float) -> float:
    """One inter-arrival gap with mean ``1/rate`` and coefficient of variation ``burst_cv``."""
    if burst_cv == 1.0:
        # Poisson arrivals: exponential gaps (kept as expovariate so existing seeds
        # reproduce the same stream as before burst_cv existed).
        return rng.expovariate(rate)
    # Gamma with shape 1/cv^2 has mean 1/rate and coefficient of variation burst_cv.
    shape = 1.0 / (burst_cv * burst_cv)
    return rng.gammavariate(shape, 1.0 / (rate * shape))


def generate_sessions(
    num_sessions: int,
    turns: int,
    qps: float,
    *,
    tool_result_tokens: int = 64,
    think_time: float = 2.0,
    burst_cv: float = 1.0,
    deterministic_tool_prob: float = 0.3,
    gen_tokens_mean: int = 48,
    block_size: int = DEFAULT_BLOCK_SIZE,
    nominal_turn_seconds: float = 0.5,
    seed: int = 0,
) -> list[TurnRequest]:
    """Generate a policy-independent stream of tool-calling turns, sorted by arrival.

    ``qps`` is the aggregate target turn arrival rate; sessions are launched at
    ``qps / turns`` so the flattened turn stream averages ``qps``. ``burst_cv`` is the
    coefficient of variation of session inter-arrival gaps: 1.0 (default) is a Poisson
    process, >1 draws gamma gaps with the same mean — bursts of near-simultaneous sessions
    separated by lulls, which is what stresses the wait-vs-recompute decision. Within a
    session, turn *k* arrives ``think_time + nominal_turn_seconds`` after turn *k-1* (a
    fixed estimate, independent of which node ends up serving it, to keep the stream fair).
    """
    rng = random.Random(seed)
    session_rate = max(qps / max(turns, 1), 1e-9)

    requests: list[TurnRequest] = []
    session_start = 0.0
    for sid in range(num_sessions):
        session_start += _arrival_gap(rng, session_rate, burst_cv)
        user0 = rng.choice(_USER_PROMPTS)

        # Token sequence and message list both grow turn by turn from a shared base.
        tokens = list(_SYSTEM_TOKENS) + _tokenize(user0)
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user0},
        ]

        arrival = session_start
        for turn_idx in range(turns):
            if turn_idx > 0:
                # Append the previous assistant reply + a tool call + tool result + a new
                # user message: this is what makes turn k extend turn k-1's prefix.
                assistant = _unique_words(f"s{sid}_t{turn_idx}_a", gen_tokens_mean)
                if rng.random() < deterministic_tool_prob:
                    tool_result = rng.choice(_CANNED_TOOL_RESULTS)
                else:
                    tool_result = _unique_words(f"s{sid}_t{turn_idx}_r", tool_result_tokens)
                next_user = _unique_words(f"s{sid}_t{turn_idx}_u", 6)

                segment = f"{assistant} {tool_result} {next_user}"
                tokens = tokens + _tokenize(segment)
                messages = messages + [
                    {"role": "assistant", "content": assistant},
                    {"role": "tool", "content": tool_result},
                    {"role": "user", "content": next_user},
                ]
                arrival += think_time + nominal_turn_seconds

            gen_tokens = max(1, int(rng.gauss(gen_tokens_mean, gen_tokens_mean * 0.2)))
            requests.append(
                TurnRequest(
                    session_id=sid,
                    turn_idx=turn_idx,
                    arrival=arrival,
                    tokens=list(tokens),
                    block_hashes=hash_blocks(tokens, block_size),
                    gen_tokens=gen_tokens,
                    messages=[dict(m) for m in messages],
                )
            )

    requests.sort(key=lambda r: (r.arrival, r.session_id, r.turn_idx))
    return requests


# Session-id offsets keeping the three class streams disjoint (tool sids start at 0).
_RAG_SID_BASE = 1_000_000
_ONESHOT_SID_BASE = 2_000_000


def generate_mixed(
    num_sessions: int,
    turns: int,
    qps: float,
    *,
    mix: dict[str, float] | None = None,
    rag_docs: int = 16,
    rag_doc_tokens: int = 1024,
    rag_zipf: float = 1.1,
    rag_question_tokens: int = 16,
    oneshot_tokens: int = 32,
    tool_result_tokens: int = 64,
    think_time: float = 2.0,
    burst_cv: float = 1.0,
    deterministic_tool_prob: float = 0.3,
    gen_tokens_mean: int = 48,
    block_size: int = DEFAULT_BLOCK_SIZE,
    nominal_turn_seconds: float = 0.5,
    seed: int = 0,
) -> list[TurnRequest]:
    """A mixed-class request stream: tool-calling sessions, RAG queries, and one-shots.

    ``mix`` maps class name to its fraction of the total request budget
    (``num_sessions * turns``); classes have very different prefix-cache economics:

      * ``tool``    -- the multi-turn sessions from :func:`generate_sessions`, unchanged
        (``mix={"tool": 1.0}`` reproduces its stream byte-for-byte).
      * ``rag``     -- single-turn queries: shared system prompt + a document sampled from
        a pool of ``rag_docs`` with Zipf-like popularity (weight ``1/rank^rag_zipf``) + a
        unique question. Hot documents become prefixes shared across *users*.
      * ``oneshot`` -- single-turn, ~``oneshot_tokens`` unique tokens, no shared prefix:
        the zero-reuse class, pure cache pollution.

    Each class gets its own seeded RNG and its own Poisson/gamma arrival stream at
    ``qps * fraction`` (same ``burst_cv`` semantics as sessions); streams are merged and
    sorted so every policy sees one fair arrival order.
    """
    mix = dict(mix) if mix else {"tool": 1.0}
    unknown = set(mix) - {"tool", "rag", "oneshot"}
    if unknown:
        raise ValueError(f"unknown workload classes in mix: {sorted(unknown)}")
    total_budget = num_sessions * turns

    requests: list[TurnRequest] = []

    tool_frac = mix.get("tool", 0.0)
    if tool_frac > 0:
        requests += generate_sessions(
            round(num_sessions * tool_frac),
            turns,
            qps * tool_frac,
            tool_result_tokens=tool_result_tokens,
            think_time=think_time,
            burst_cv=burst_cv,
            deterministic_tool_prob=deterministic_tool_prob,
            gen_tokens_mean=gen_tokens_mean,
            block_size=block_size,
            nominal_turn_seconds=nominal_turn_seconds,
            seed=seed,
        )

    rag_frac = mix.get("rag", 0.0)
    if rag_frac > 0:
        rng = random.Random(f"{seed}:rag")
        rate = max(qps * rag_frac, 1e-9)
        doc_texts = [_unique_words(f"rag_d{d}", rag_doc_tokens) for d in range(rag_docs)]
        weights = [1.0 / (d + 1) ** rag_zipf for d in range(rag_docs)]
        arrival = 0.0
        for i in range(round(total_budget * rag_frac)):
            arrival += _arrival_gap(rng, rate, burst_cv)
            doc = doc_texts[rng.choices(range(rag_docs), weights=weights)[0]]
            user = f"{doc} {_unique_words(f'rag_q{i}', rag_question_tokens)}"
            tokens = list(_SYSTEM_TOKENS) + _tokenize(user)
            requests.append(
                TurnRequest(
                    session_id=_RAG_SID_BASE + i,
                    turn_idx=0,
                    arrival=arrival,
                    tokens=tokens,
                    block_hashes=hash_blocks(tokens, block_size),
                    gen_tokens=max(1, int(rng.gauss(gen_tokens_mean, gen_tokens_mean * 0.2))),
                    messages=[
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": user},
                    ],
                    kind="rag",
                )
            )

    oneshot_frac = mix.get("oneshot", 0.0)
    if oneshot_frac > 0:
        rng = random.Random(f"{seed}:oneshot")
        rate = max(qps * oneshot_frac, 1e-9)
        arrival = 0.0
        for i in range(round(total_budget * oneshot_frac)):
            arrival += _arrival_gap(rng, rate, burst_cv)
            text = _unique_words(f"one_{i}", oneshot_tokens)
            tokens = _tokenize(text)
            requests.append(
                TurnRequest(
                    session_id=_ONESHOT_SID_BASE + i,
                    turn_idx=0,
                    arrival=arrival,
                    tokens=tokens,
                    block_hashes=hash_blocks(tokens, block_size),
                    gen_tokens=max(1, int(rng.gauss(gen_tokens_mean, gen_tokens_mean * 0.2))),
                    messages=[{"role": "user", "content": text}],
                    kind="oneshot",
                )
            )

    requests.sort(key=lambda r: (r.arrival, r.session_id, r.turn_idx))
    return requests
