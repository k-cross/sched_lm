import asyncio
import time

import pytest

from bench.prompt import (
    _CHARS_PER_TOKEN,
    SYSTEM_PROMPT,
    SYSTEM_PROMPT_TARGET_TOKENS,
    _build_system_prompt,
)
from bench.traffic import ROUTE_HEADER, run_traffic


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


class _FakeResponse:
    def __init__(self, recorder):
        self._recorder = recorder
        self.status = 200
        self.content = self  # async-iterated below

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return False

    def __aiter__(self):
        async def _gen():
            yield b"data: chunk\n"

        return _gen()


class _FakeSession:
    def __init__(self):
        self.calls = []

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return False

    def post(self, url, json=None, headers=None):
        self.calls.append({"url": url, "headers": headers})
        return _FakeResponse(self)


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
