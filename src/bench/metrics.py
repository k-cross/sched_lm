import aiohttp
import numpy as np


class MetricsClient:
    def __init__(self, prometheus_url: str):
        self.prometheus_url = prometheus_url

    async def query(self, query_str: str):
        url = f"{self.prometheus_url}/api/v1/query"
        async with aiohttp.ClientSession() as session:
            async with session.get(url, params={"query": query_str}) as response:
                if response.status == 200:
                    data = await response.json()
                    return data.get("data", {}).get("result", [])
                return []

    async def _scalar(self, query_str: str) -> float:
        """Run an instant query expected to return a single scalar value."""
        results = await self.query(query_str)
        if results:
            value = results[0].get("value", [0, "0"])[1]
            try:
                return float(value)
            except (ValueError, IndexError):
                return 0.0
        return 0.0

    async def get_prefix_cache_hit_rate(self) -> float:
        # Sum of hits / Sum of queries over the last 5 minutes
        query = (
            "sum(rate(vllm:prefix_cache_hits_total[5m])) / "
            "sum(rate(vllm:prefix_cache_queries_total[5m]))"
        )
        return await self._scalar(query)

    async def get_prefix_cache_counters(self) -> tuple[float, float]:
        """Return the cumulative (hits, queries) counters summed over all sims.

        Used to compute a per-route hit rate from the delta across a single
        traffic run, which is more accurate than a rate() window that blends
        traffic from multiple routes.
        """
        hits = await self._scalar("sum(vllm:prefix_cache_hits_total)")
        queries = await self._scalar("sum(vllm:prefix_cache_queries_total)")
        return hits, queries

    async def get_avg_prefill_time(self) -> float | None:
        """Average prefill time (seconds) from vLLM's histogram, or None.

        Real prefill *GPU* time is only meaningful on a GPU-backed vLLM; the
        CPU simulator does not run prefill kernels. We surface vLLM's
        request_prefill_time histogram if the backend happens to emit it, and
        return None otherwise so the report can show it as not-applicable.
        """
        total = await self._scalar("sum(vllm:request_prefill_time_seconds_sum)")
        count = await self._scalar("sum(vllm:request_prefill_time_seconds_count)")
        if count > 0:
            return total / count
        return None

    # ------------------------------------------------------------------
    # Priority / retention metrics (RFC-0001 §4)
    # ------------------------------------------------------------------

    async def get_pinned_usage(self) -> float:
        """Fraction of cache occupied by marked-and-unexpired blocks (0–1).

        Averages the per-sim gauge rather than summing it: each pod reports its own
        occupancy fraction, so ``sum`` would scale with replica count and exceed 1.0.
        """
        return await self._scalar("avg(vllm:kv_cache_pinned_usage_perc)")

    async def get_pinned_evictions(self) -> float:
        """Cumulative pinned-eviction counter summed over all sims.

        Callers diff across runs to get the per-route delta, like
        :meth:`get_prefix_cache_counters`.
        """
        return await self._scalar("sum(vllm:kv_cache_pinned_evictions_total)")

    async def get_evictions(self, range_seconds: int) -> float:
        """Total blocks evicted to make space (any band) over the last ``range_seconds``.

        The direct cache-contention / eviction-pressure signal (RFC-0001 §4) -- unlike
        :meth:`get_pinned_evictions`, which counts only the marked-and-unexpired subset.
        Uses ``increase()`` rather than a before/after snapshot delta: under the cache
        pressure that produces evictions, memory-limited sim pods can OOM-restart mid-run
        and reset the counter, which would make a raw ``sum`` delta go negative;
        ``increase()`` corrects for per-series counter resets.
        """
        return await self._scalar(f"sum(increase(vllm:kv_cache_evictions_total[{range_seconds}s]))")

    async def _peak_gauge(self, metric: str, range_seconds: int) -> float | None:
        """Peak of a per-pod gauge over ``range_seconds`` (max-over-time, then max across
        pods), or None when the window holds no samples.

        Instant reads after the settle wait can miss transient pressure (pinned leases
        decay in seconds; cache usage swings with in-flight requests), so the report reads
        the window peak. None (vs 0.0) distinguishes a broken scrape / absent metric from a
        genuine idle cache, so the report can render "n/a" instead of a misleading 0%.
        """
        results = await self.query(f"max(max_over_time({metric}[{range_seconds}s]))")
        if not results:
            return None
        try:
            return float(results[0].get("value", [0, "0"])[1])
        except (ValueError, IndexError):
            return None

    async def get_peak_pinned_usage(self, range_seconds: int) -> float | None:
        """Peak fraction of cache held by marked-and-unexpired blocks over the window (0–1)."""
        return await self._peak_gauge("vllm:kv_cache_pinned_usage_perc", range_seconds)

    async def get_peak_cache_usage(self, range_seconds: int) -> float | None:
        """Peak overall KV-cache utilization over the window (0–1) -- how full the cache
        got, the contention level the eviction counter is the churn side of."""
        return await self._peak_gauge("vllm:kv_cache_usage_perc", range_seconds)

    async def get_priority_blocks(self) -> dict[str, float]:
        """Per-priority-band resident block counts summed over all sims."""
        evict_first = await self._scalar(
            'sum(vllm:kv_cache_priority_blocks{priority="evict_first"})'
        )
        high = await self._scalar('sum(vllm:kv_cache_priority_blocks{priority="high"})')
        pinned = await self._scalar('sum(vllm:kv_cache_priority_blocks{priority="pinned"})')
        return {"evict_first": evict_first, "high": high, "pinned": pinned}


def compute_percentiles(
    data: list[float], percentiles: list[int] | None = None
) -> dict[int, float]:
    if percentiles is None:
        percentiles = [50, 90, 99]
    if not data:
        return dict.fromkeys(percentiles, 0.0)
    return {p: np.percentile(data, p) for p in percentiles}
