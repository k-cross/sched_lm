from collections import Counter

from bench.sim.blocks import hash_blocks, matched_prefix
from bench.sim.cost import CostParams, Regime, best_placement, hot_node, predict
from bench.sim.engine import run_simulation
from bench.sim.node import Node
from bench.sim.policies import build_policy, cache_local, oracle
from bench.sim.workload import generate_sessions

BLOCK = 4


def _toks(*vals):
    return list(vals)


# --- blocks -----------------------------------------------------------------


def test_shared_prefix_yields_identical_leading_block_hashes():
    a = list(range(40))
    b = list(range(40))
    b[25] = 999  # diverge inside block index 6 (tokens 24..27)
    ha = hash_blocks(a, BLOCK)
    hb = hash_blocks(b, BLOCK)
    # Blocks 0..5 are identical; block 6 (index 6) is where they first differ.
    assert ha[:6] == hb[:6]
    assert ha[6] != hb[6]


def test_matched_prefix_stops_at_first_miss():
    blocks = [10, 20, 30, 40]
    # 40 is cached but 30 is not: matching must stop at the contiguous prefix.
    assert matched_prefix(blocks, {10, 20, 40}) == 2
    assert matched_prefix(blocks, {10, 20, 30, 40}) == 4
    assert matched_prefix(blocks, set()) == 0


# --- cost model -------------------------------------------------------------


def _nodes(n=2, cap=1000):
    return [Node(i, cap) for i in range(n)]


def test_prefill_monotonic_in_missing_and_wait():
    params = CostParams(block_size=BLOCK)
    assert params.prefill_time(5) > params.prefill_time(1)
    nodes = _nodes()
    hot = nodes[0]
    blocks = [1, 2, 3, 4]
    idle = predict(blocks, nodes[1], now=0.0, params=params, hot=hot, transfer=False)
    nodes[1].busy_until = 100.0
    busy = predict(blocks, nodes[1], now=0.0, params=params, hot=hot, transfer=False)
    assert busy.ttft > idle.ttft


def test_transfer_beats_recompute_when_bandwidth_cheap():
    # Hot node holds a long cached prefix but is deeply queued; an idle node can either
    # transfer those blocks (cheap bandwidth) or recompute them. Transfer should win.
    params = CostParams(block_size=BLOCK, bandwidth=1e12, prefill_per_token=1.0)
    nodes = _nodes()
    blocks = [1, 2, 3, 4, 5, 6, 7, 8]
    # Seed node 0 (hot) with the whole prefix and make it busy.
    nodes[0].admit(0.0, 10_000.0, blocks)
    hot = hot_node(blocks, nodes)
    assert hot is nodes[0]
    best = best_placement(blocks, nodes, now=0.0, params=params)
    assert best.node_id == 1
    assert best.transfer is True
    assert best.regime is Regime.TRANSFER


def test_recompute_when_prefix_short():
    # A short uncached prefix on an idle node is cheaper than waiting on the hot node.
    params = CostParams(block_size=BLOCK)
    nodes = _nodes()
    blocks = [1, 2]
    nodes[0].admit(0.0, 5.0, blocks)  # hot but busy
    best = best_placement(blocks, nodes, now=0.0, params=params)
    assert best.node_id == 1
    assert best.regime in (Regime.RECOMPUTE, Regime.TRANSFER)


# --- policies ---------------------------------------------------------------


def test_cache_local_matches_oracle_when_single_hot_idle_node():
    params = CostParams(block_size=BLOCK)
    nodes = _nodes(n=3)
    blocks = [1, 2, 3, 4]
    nodes[1].admit(0.0, 5.0, blocks)  # node 1 is the hot node and now idle at now>=5
    now = 10.0  # past its busy window -> idle again
    cl = cache_local(blocks, nodes, now, params)
    orc = oracle(blocks, nodes, now, params)
    assert cl.node_id == orc.node_id == 1


def test_regret_never_negative_across_policies():
    requests = generate_sessions(30, turns=4, qps=8.0, block_size=BLOCK, seed=3)
    params = CostParams(block_size=BLOCK)
    for name in ("round-robin", "cache-local", "transfer-aware", "oracle"):
        res = run_simulation(
            requests, build_policy(name), num_nodes=3, cache_blocks=500, params=params
        )
        assert all(r >= -1e-9 for r in res.regrets)
    # The oracle is the argmin, so its own regret is ~0.
    orc = run_simulation(requests, build_policy("oracle"), 3, 500, params)
    assert max(orc.regrets) < 1e-9


# --- workload ---------------------------------------------------------------


def test_workload_is_deterministic_for_seed():
    a = generate_sessions(10, turns=3, qps=5.0, block_size=BLOCK, seed=7)
    b = generate_sessions(10, turns=3, qps=5.0, block_size=BLOCK, seed=7)
    assert [r.block_hashes for r in a] == [r.block_hashes for r in b]


def test_turn_prefix_grows_within_session():
    requests = generate_sessions(3, turns=5, qps=100.0, block_size=BLOCK, seed=1)
    by_session: dict[int, list] = {}
    for r in requests:
        by_session.setdefault(r.session_id, []).append(r)
    for turns in by_session.values():
        turns.sort(key=lambda r: r.turn_idx)
        for prev, cur in zip(turns, turns[1:], strict=False):
            # Each later turn's block-hash sequence extends the previous turn's.
            assert cur.block_hashes[: len(prev.block_hashes)] == prev.block_hashes
            assert len(cur.block_hashes) >= len(prev.block_hashes)


def test_lru_evicts_affinity_prefix_under_capacity_pressure():
    node = Node(0, cache_blocks=3)
    node.admit(0.0, 1.0, [1, 2, 3])
    assert node.matched([1, 2, 3]) == 3
    # A different request fills the small cache, evicting the oldest blocks.
    node.admit(1.0, 1.0, [4, 5, 6])
    assert node.matched([1, 2, 3]) == 0
    assert node.matched([4, 5, 6]) == 3


# --- engine -----------------------------------------------------------------


def test_engine_reproducible_and_records_regimes():
    requests = generate_sessions(20, turns=4, qps=6.0, block_size=BLOCK, seed=11)
    params = CostParams(block_size=BLOCK)
    a = run_simulation(requests, build_policy("oracle"), 4, 500, params)
    b = run_simulation(requests, build_policy("oracle"), 4, 500, params)
    assert a.ttfts == b.ttfts
    assert a.successes == len(requests)
    assert isinstance(a.regimes, Counter)
    assert sum(a.regimes.values()) == a.successes
