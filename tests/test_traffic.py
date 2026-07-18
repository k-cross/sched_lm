import asyncio
import time

import pytest

from bench.prompt import (
    _CHARS_PER_TOKEN,
    SYSTEM_PROMPT,
    SYSTEM_PROMPT_TARGET_TOKENS,
    _build_system_prompt,
)
from bench.traffic import (
    ROUTE_HEADER,
    BenchmarkResult,
    run_session_traffic,
    run_traffic,
)


def test_system_prompt_is_deterministic_and_shared():
    # The shared prefix must be identical across builds, otherwise prefix
    # caching never hits.
    assert _build_system_prompt() == _build_system_prompt()
    assert SYSTEM_PROMPT == _build_system_prompt()


def test_system_prompt_reaches_target_size():
    est_tokens = len(SYSTEM_PROMPT) // _CHARS_PER_TOKEN
    # Should be at least the target and within a section of overshoot.
    assert est_tokens >= SYSTEM_PROMPT_TARGET_TOKENS
    assert est_tokens < SYSTEM_PROMPT_TARGET_TOKENS * 1.25


def test_benchmark_result_aggregates_error_reasons():
    result = BenchmarkResult()
    result.record_error("load shed (503)")
    result.record_error("load shed (503)")
    result.record_error("in-stream error")

    assert result.errors == 3
    assert result.error_reasons["load shed (503)"] == 2
    # most_common orders by count, so the summary leads with the dominant reason.
    assert result.error_summary() == "2x load shed (503), 1x in-stream error"


# A realistic sim stream: content chunks (whose "usage":null must not confuse the usage
# parse), the final usage-bearing chunk, and the [DONE] sentinel.
_DEFAULT_STREAM = [
    b'data: {"choices":[{"delta":{"content":"chunk"}}],"usage":null}\n',
    b'data: {"choices":[],"usage":{"prompt_tokens":100,"completion_tokens":10,'
    b'"prompt_tokens_details":{"cached_tokens":96}}}\n',
    b"data: [DONE]\n",
]


class _FakeResponse:
    def __init__(self, recorder, lines=None):
        self._recorder = recorder
        self._lines = _DEFAULT_STREAM if lines is None else lines
        self.status = 200
        self.content = self  # async-iterated below

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return False

    def __aiter__(self):
        async def _gen():
            for line in self._lines:
                yield line

        return _gen()


class _FakeSession:
    def __init__(self, lines=None):
        self.calls = []
        self._lines = lines

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return False

    def post(self, url, json=None, headers=None):
        self.calls.append({"url": url, "headers": headers, "json": json})
        return _FakeResponse(self, self._lines)


@pytest.mark.parametrize("route", ["round-robin", "prefix-affinity", "class-aware-reliability"])
def test_run_traffic_injects_route_header(monkeypatch, route):
    session = _FakeSession()
    monkeypatch.setattr("bench.traffic.aiohttp.ClientSession", lambda: session)

    result = asyncio.run(
        run_traffic("http://gw/v1/chat/completions", 3, concurrency=2, route=route, qps=0)
    )

    assert result.successes == 3
    assert result.errors == 0
    assert len(session.calls) == 3
    assert all(c["headers"][ROUTE_HEADER] == route for c in session.calls)


def _turn(session_id, turn_idx, arrival=0.0):
    from bench.sim.workload import TurnRequest

    return TurnRequest(
        session_id=session_id,
        turn_idx=turn_idx,
        arrival=arrival,
        tokens=[1, 2, 3],
        block_hashes=[1],
        gen_tokens=8,
        messages=[{"role": "user", "content": f"s{session_id} t{turn_idx}"}],
    )


def test_run_session_traffic_records_turn_metrics(monkeypatch):
    session = _FakeSession()
    monkeypatch.setattr("bench.traffic.aiohttp.ClientSession", lambda: session)

    turns = [_turn(0, 0), _turn(0, 1, arrival=0.01), _turn(1, 0)]
    result = asyncio.run(
        run_session_traffic("http://gw/v1/chat/completions", turns, 4, "class-aware-reliability")
    )

    assert result.successes == 3
    # Every request asks for the final usage chunk.
    assert all(c["json"]["stream_options"] == {"include_usage": True} for c in session.calls)
    # One TurnMetric per successful turn, carrying the usage-chunk token counts.
    assert {(t.session_id, t.turn_idx) for t in result.turns} == {(0, 0), (0, 1), (1, 0)}
    assert all(
        (t.prompt_tokens, t.completion_tokens, t.cached_tokens) == (100, 10, 96)
        for t in result.turns
    )
    assert all(t.ttft > 0 and t.e2e >= t.ttft for t in result.turns)
    # Per-session wall time and the run's wall clock are recorded.
    assert set(result.session_times) == {0, 1}
    assert all(v > 0 for v in result.session_times.values())
    assert result.wall_seconds > 0


def test_in_stream_error_records_no_turn_metric(monkeypatch):
    session = _FakeSession(lines=[b'data: {"error":{"message":"context length"}}\n'])
    monkeypatch.setattr("bench.traffic.aiohttp.ClientSession", lambda: session)

    result = asyncio.run(
        run_session_traffic("http://gw/v1/chat/completions", [_turn(0, 0)], 1, "prefix-affinity")
    )

    assert result.errors == 1
    assert result.error_reasons["in-stream error"] == 1
    assert result.turns == []


def test_run_traffic_reports_wall_seconds(monkeypatch):
    session = _FakeSession()
    monkeypatch.setattr("bench.traffic.aiohttp.ClientSession", lambda: session)

    result = asyncio.run(
        run_traffic("http://gw/v1/chat/completions", 2, concurrency=2, route="round-robin", qps=0)
    )

    assert result.successes == 2
    assert result.wall_seconds > 0
    # Single-shot runs carry no per-turn or per-session data.
    assert result.turns == []
    assert result.session_times == {}


def test_run_traffic_paces_to_qps(monkeypatch):
    session = _FakeSession()
    monkeypatch.setattr("bench.traffic.aiohttp.ClientSession", lambda: session)

    qps = 20.0
    n = 5
    start = time.monotonic()
    asyncio.run(
        run_traffic(
            "http://gw/v1/chat/completions", n, concurrency=n, route="prefix-affinity", qps=qps
        )
    )
    elapsed = time.monotonic() - start

    # Open-loop pacing dispatches request i at ~i/qps, so n requests take at
    # least (n-1)/qps seconds regardless of how fast the backend responds.
    assert elapsed >= (n - 1) / qps
