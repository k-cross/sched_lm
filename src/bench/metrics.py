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
        if results and len(results) > 0:
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


def compute_percentiles(data: list[float], percentiles: list[int] | None = None) -> dict:
    if percentiles is None:
        percentiles = [50, 90, 99]
    if not data:
        return dict.fromkeys(percentiles, 0.0)
    return {p: np.percentile(data, p) for p in percentiles}
