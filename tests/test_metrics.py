import asyncio
from unittest.mock import AsyncMock, patch

import pytest

from bench.metrics import MetricsClient


@pytest.fixture
def mock_metrics_client():
    client = MetricsClient("http://fake-prometheus")
    with patch.object(client, "_scalar", new_callable=AsyncMock) as mock_scalar:
        yield client, mock_scalar


def test_get_pinned_usage(mock_metrics_client):
    client, mock_scalar = mock_metrics_client
    mock_scalar.return_value = 0.45

    usage = asyncio.run(client.get_pinned_usage())

    assert usage == 0.45
    # Averaged across sims, not summed -- a summed fraction would exceed 1.0 with >1 replica.
    mock_scalar.assert_called_once_with("avg(vllm:kv_cache_pinned_usage_perc)")


def test_get_pinned_evictions(mock_metrics_client):
    client, mock_scalar = mock_metrics_client
    mock_scalar.return_value = 120.0

    evictions = asyncio.run(client.get_pinned_evictions())

    assert evictions == 120.0
    mock_scalar.assert_called_once_with("sum(vllm:kv_cache_pinned_evictions_total)")


def test_get_priority_blocks(mock_metrics_client):
    client, mock_scalar = mock_metrics_client

    # Return different values for the three calls
    mock_scalar.side_effect = [10.0, 50.0, 100.0]

    blocks = asyncio.run(client.get_priority_blocks())

    assert blocks == {"evict_first": 10.0, "high": 50.0, "pinned": 100.0}
    assert mock_scalar.call_count == 3
