import statistics
from collections import Counter

from bench.sim.blocks import hash_blocks, matched_prefix
from bench.sim.cost import CostParams, Regime, best_placement, hot_node, predict
from bench.sim.engine import _is_coupled, request_view, run_simulation
from bench.sim.node import BeladyNode, Node
from bench.sim.policies import (
    POLICY_NAMES,
    ClassAwareReliability,
    RequestView,
    ToolGapIndex,
    build_policy,
    cache_local,
    class_aware,
    classify,
    oracle,
    uses_belady,
)
from bench.sim.workload import Tool, TurnRequest, generate_mixed, generate_sessions

BLOCK = 4


def _view(blocks, *, num_messages=2, has_tool_messages=False, last_message_tokens=6):
    """A RequestView shaped like a session opener unless overridden."""
    return RequestView(
        blocks=blocks,
        prompt_tokens=len(blocks) * BLOCK,
        num_messages=num_messages,
        has_tool_messages=has_tool_messages,
        last_message_tokens=last_message_tokens,
    )


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
    cl = cache_local(_view(blocks), nodes, now, params)
    orc = oracle(_view(blocks), nodes, now, params)
    assert cl.node_id == orc.node_id == 1


def test_regret_never_negative_across_policies():
    requests = generate_sessions(30, turns=4, qps=8.0, block_size=BLOCK, seed=3)
    params = CostParams(block_size=BLOCK)
    for name in POLICY_NAMES:
        res = run_simulation(
            requests, build_policy(name), num_nodes=3, cache_blocks=500, params=params
        )
        assert all(r >= -1e-9 for r in res.regrets)
    # The oracle is the argmin, so its own regret is ~0.
    orc = run_simulation(requests, build_policy("oracle"), 3, 500, params)
    assert max(orc.regrets) < 1e-9


def test_weighted_precise_prefers_hot_idle_node():
    # With one node holding the whole prefix and everyone idle, the prefix scorer decides:
    # the weighted policy must agree with the oracle's node choice.
    params = CostParams(block_size=BLOCK)
    nodes = _nodes(n=3)
    blocks = [1, 2, 3, 4]
    nodes[1].admit(0.0, 5.0, blocks)
    now = 10.0  # past node 1's busy window -> all idle
    p = build_policy("weighted-precise")(_view(blocks), nodes, now, params)
    assert p.node_id == 1
    assert p.transfer is False


def test_weighted_saturation_filter_spills_off_the_hot_node():
    # The hot node holds the whole prefix but is saturated far beyond the filter depth;
    # GIE's filter must drop it so the request spills to an idle (cold) node.
    params = CostParams(block_size=BLOCK)
    nodes = _nodes(n=3)
    blocks = [1, 2, 3, 4]
    nodes[1].admit(0.0, 500.0, blocks)  # hot and deeply queued
    p = build_policy("weighted-precise")(_view(blocks), nodes, 0.0, params)
    assert p.node_id != 1


def test_weighted_policies_never_transfer():
    # Production llm-d has no KV-transfer path; the replicated pipeline must not either.
    requests = generate_sessions(20, turns=3, qps=8.0, block_size=BLOCK, seed=5)
    params = CostParams(block_size=BLOCK)
    for name in ("weighted-precise", "weighted-approx"):
        res = run_simulation(requests, build_policy(name), 3, 500, params)
        assert res.regimes.get(Regime.TRANSFER, 0) == 0


def test_weighted_approx_is_blind_to_evictions():
    params = CostParams(block_size=BLOCK)
    nodes = [Node(i, cache_blocks=3) for i in range(3)]
    blocks = [1, 2, 3]
    approx = build_policy("weighted-approx")
    precise = build_policy("weighted-precise")

    # First routing decision (all nodes cold and idle) ties -> node 0, and the approx
    # router records "node 0 holds blocks" in its index.
    first = approx(_view(blocks), nodes, 0.0, params)
    assert first.node_id == 0
    nodes[0].admit(0.0, 0.1, blocks)

    # Node 0 evicts the prefix under capacity pressure; node 2 becomes the real hot node.
    nodes[0].admit(0.2, 0.1, [7, 8, 9])
    nodes[2].admit(0.4, 0.1, blocks)
    assert nodes[0].matched(blocks) == 0

    now = 100.0  # everyone idle again, so only the prefix scorer differs
    assert approx(_view(blocks), nodes, now, params).node_id == 0  # stale router-side belief
    assert precise(_view(blocks), nodes, now, params).node_id == 2  # true cache state


# --- workload ---------------------------------------------------------------


def test_workload_is_deterministic_for_seed():
    a = generate_sessions(10, turns=3, qps=5.0, block_size=BLOCK, seed=7)
    b = generate_sessions(10, turns=3, qps=5.0, block_size=BLOCK, seed=7)
    assert [r.block_hashes for r in a] == [r.block_hashes for r in b]


def test_burst_cv_one_reproduces_the_poisson_stream():
    # burst_cv=1.0 must be byte-identical to the pre-knob Poisson path so existing
    # seeded experiments stay reproducible.
    a = generate_sessions(10, turns=3, qps=5.0, block_size=BLOCK, seed=7)
    b = generate_sessions(10, turns=3, qps=5.0, block_size=BLOCK, seed=7, burst_cv=1.0)
    assert [(r.arrival, r.block_hashes) for r in a] == [(r.arrival, r.block_hashes) for r in b]


def test_burstier_arrivals_have_higher_gap_variability():
    def gap_cv(burst_cv: float) -> float:
        reqs = generate_sessions(
            200, turns=1, qps=20.0, block_size=BLOCK, seed=9, burst_cv=burst_cv
        )
        starts = sorted(r.arrival for r in reqs)
        gaps = [b - a for a, b in zip(starts, starts[1:], strict=False)]
        return statistics.pstdev(gaps) / statistics.mean(gaps)

    assert gap_cv(4.0) > gap_cv(1.0)


def test_turn_prefix_grows_within_session():
    requests = generate_sessions(3, turns=5, qps=100.0, block_size=BLOCK, seed=1)
    by_session: dict[int, list[TurnRequest]] = {}
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


def test_coupled_true_when_placement_flips_as_queues_drain():
    # Node 0 holds the whole prefix but is deeply queued; node 1 is idle and cold.
    # Under load the oracle avoids node 0 (transfer/recompute on node 1), but with queues
    # drained node 0 (full cache, now idle) wins -- the decision depended on other load.
    params = CostParams(block_size=BLOCK)
    nodes = _nodes(n=2)
    blocks = [1, 2, 3, 4, 5, 6, 7, 8]
    nodes[0].admit(0.0, 10_000.0, blocks)  # hot but busy far into the future
    loaded = best_placement(blocks, nodes, now=0.0, params=params)
    assert loaded.node_id == 1  # sanity: load pushed us off the hot node
    assert _is_coupled(blocks, nodes, 0.0, params, loaded) is True


def test_coupled_false_and_restores_busy_until():
    # Node 0 is the hot, idle node and wins regardless of load; node 1 is busy but never
    # in contention, so draining queues changes nothing -- not coupled. The busy_until of
    # the loaded node must be left untouched after the probe.
    params = CostParams(block_size=BLOCK)
    nodes = _nodes(n=2)
    blocks = [1, 2, 3, 4, 5, 6, 7, 8]
    nodes[0].admit(0.0, 1.0, blocks)  # hot; idle again by now=100
    nodes[1].busy_until = 200.0  # genuinely loaded, but irrelevant to the decision
    now = 100.0
    loaded = best_placement(blocks, nodes, now, params)
    assert loaded.node_id == 0
    assert _is_coupled(blocks, nodes, now, params, loaded) is False
    assert nodes[1].busy_until == 200.0  # probe restored the mutated queue state


def test_engine_reproducible_and_records_regimes():
    requests = generate_sessions(20, turns=4, qps=6.0, block_size=BLOCK, seed=11)
    params = CostParams(block_size=BLOCK)
    a = run_simulation(requests, build_policy("oracle"), 4, 500, params)
    b = run_simulation(requests, build_policy("oracle"), 4, 500, params)
    assert a.ttfts == b.ttfts
    assert a.successes == len(requests)
    assert isinstance(a.regimes, Counter)
    assert sum(a.regimes.values()) == a.successes


# --- mixed workload + class-aware policy -------------------------------------

MIX = {"tool": 0.5, "rag": 0.3, "oneshot": 0.2}


def test_mixed_tool_only_matches_generate_sessions():
    # mix={"tool": 1.0} must reproduce the plain session stream byte-for-byte.
    a = generate_sessions(10, turns=3, qps=5.0, block_size=BLOCK, seed=7)
    b = generate_mixed(10, 3, 5.0, mix={"tool": 1.0}, block_size=BLOCK, seed=7)
    assert [(r.arrival, r.block_hashes, r.kind) for r in a] == [
        (r.arrival, r.block_hashes, r.kind) for r in b
    ]


def test_mixed_workload_deterministic_and_proportioned():
    a = generate_mixed(40, 4, 10.0, mix=MIX, block_size=BLOCK, seed=3)
    b = generate_mixed(40, 4, 10.0, mix=MIX, block_size=BLOCK, seed=3)
    assert [(r.arrival, r.block_hashes, r.kind) for r in a] == [
        (r.arrival, r.block_hashes, r.kind) for r in b
    ]
    counts = Counter(r.kind for r in a)
    # Budget = 40 * 4 = 160 requests: 20 tool sessions x 4 turns, 48 rag, 32 oneshot.
    assert counts == {"tool": 80, "rag": 48, "oneshot": 32}


def test_rag_queries_share_doc_prefix_and_diverge_across_docs():
    reqs = generate_mixed(
        20,
        2,
        10.0,
        mix={"rag": 1.0},
        rag_docs=2,
        rag_doc_tokens=64,
        rag_zipf=0.0,  # flat popularity so both docs appear
        block_size=BLOCK,
        seed=5,
    )

    def lcp(a, b) -> int:
        n = 0
        for x, y in zip(a.block_hashes, b.block_hashes, strict=False):
            if x != y:
                break
            n += 1
        return n

    lcps = {lcp(a, b) for a, b in zip(reqs, reqs[1:], strict=False)}
    # Same-doc pairs share system + doc blocks; cross-doc pairs share only the system
    # prompt, so the two lcp levels must differ by about the doc length in blocks.
    assert max(lcps) - min(lcps) >= 64 // BLOCK - 2
    assert min(lcps) > 0  # system prompt always shared


def test_classifier_matches_ground_truth_on_generated_stream():
    reqs = generate_mixed(30, 3, 10.0, mix=MIX, rag_doc_tokens=256, block_size=BLOCK, seed=9)
    assert all(classify(request_view(r)) == r.kind for r in reqs)


def test_class_aware_oneshot_never_chases_cache():
    # Node 0 holds the one-shot's blocks but is busy; node 1 is idle and cold. A one-shot
    # has no future reuse, so class-aware must take the idle node, not the hot queue.
    params = CostParams(block_size=BLOCK)
    nodes = _nodes(n=2)
    blocks = [1, 2, 3, 4]
    nodes[0].admit(0.0, 50.0, blocks)
    view = _view(blocks, num_messages=1)
    p = class_aware(view, nodes, 0.0, params)
    assert p.node_id == 1


def test_class_aware_rag_affinity_and_saturation_spill():
    params = CostParams(block_size=BLOCK)
    blocks = [1, 2, 3, 4]
    rag = _view(blocks, num_messages=2, last_message_tokens=300)

    # Hot doc node idle -> affinity wins.
    nodes = _nodes(n=2)
    nodes[1].admit(0.0, 1.0, blocks)
    assert class_aware(rag, nodes, 100.0, params).node_id == 1

    # Hot doc node saturated far past the depth limit -> spill off it.
    nodes = _nodes(n=2)
    nodes[1].admit(0.0, 500.0, blocks)
    assert class_aware(rag, nodes, 0.0, params).node_id != 1


def test_per_kind_stats_sum_to_aggregates():
    requests = generate_mixed(20, 3, 8.0, mix=MIX, rag_doc_tokens=256, block_size=BLOCK, seed=13)
    params = CostParams(block_size=BLOCK)
    res = run_simulation(requests, build_policy("class-aware"), 3, 500, params)
    assert set(res.by_kind) == {"tool", "rag", "oneshot"}
    assert sum(k.count for k in res.by_kind.values()) == res.successes
    assert sum(len(k.regrets) for k in res.by_kind.values()) == len(res.regrets)
    assert sum(k.reused_blocks for k in res.by_kind.values()) == res.reused_blocks
    assert sum(k.total_blocks for k in res.by_kind.values()) == res.total_blocks


# --- tool-reliability workload -----------------------------------------------


def test_tool_reliability_off_reproduces_the_plain_stream():
    # The new modeling must be strictly opt-in: default args are byte-for-byte identical.
    a = generate_sessions(12, turns=4, qps=6.0, block_size=BLOCK, seed=4)
    b = generate_sessions(12, turns=4, qps=6.0, block_size=BLOCK, seed=4, tool_reliability=False)
    assert [(r.arrival, r.block_hashes, r.tool_name) for r in a] == [
        (r.arrival, r.block_hashes, r.tool_name) for r in b
    ]
    assert all(r.tool_name is None for r in a)  # no tool signal without the flag


def test_tool_reliability_is_deterministic_and_labels_turns():
    reqs = generate_sessions(8, turns=4, qps=6.0, block_size=BLOCK, seed=4, tool_reliability=True)
    again = generate_sessions(8, turns=4, qps=6.0, block_size=BLOCK, seed=4, tool_reliability=True)
    assert [(r.arrival, r.block_hashes) for r in reqs] == [
        (r.arrival, r.block_hashes) for r in again
    ]
    for r in reqs:
        # Turn 0 has no prior tool call; every later turn names the session's tool.
        assert (r.tool_name is None) == (r.turn_idx == 0)
        assert r.conversation_id == r.session_id


def test_failed_calls_stay_short_and_reliable_calls_are_long():
    # A tool that always fails: every inter-turn gap is the short retry gap.
    flaky = [Tool("flaky", fail_prob=1.0, success_latency=99.0)]
    reqs = generate_sessions(
        1, turns=5, qps=100.0, block_size=BLOCK, seed=1, tool_reliability=True, tools=flaky
    )
    reqs.sort(key=lambda r: r.turn_idx)
    gaps = [b.arrival - a.arrival for a, b in zip(reqs, reqs[1:], strict=False)]
    assert all(abs(g - 0.2) < 1e-9 for g in gaps)  # _DEFAULT_RETRY_GAP


def test_reliable_tool_is_lower_variance_than_coin_flip():
    def gap_std(tool: Tool) -> float:
        reqs = generate_sessions(
            60, turns=4, qps=100.0, block_size=BLOCK, seed=2, tool_reliability=True, tools=[tool]
        )
        by_sess: dict[int, list[TurnRequest]] = {}
        for r in reqs:
            by_sess.setdefault(r.session_id, []).append(r)
        gaps = []
        for turns in by_sess.values():
            turns.sort(key=lambda r: r.turn_idx)
            gaps += [b.arrival - a.arrival for a, b in zip(turns, turns[1:], strict=False)]
        return statistics.pstdev(gaps)

    steady = Tool("steady", fail_prob=0.0, success_latency=1.0)
    coinflip = Tool("coinflip", fail_prob=0.5, success_latency=10.0)
    assert gap_std(steady) < gap_std(coinflip)


# --- ToolGapIndex ------------------------------------------------------------


def test_gap_index_learns_mean_and_confidence():
    idx = ToolGapIndex(alpha=0.5)
    for _ in range(20):
        idx.record("t", 1.0)  # perfectly consistent
    mean, var = idx.predict("t")
    assert abs(mean - 1.0) < 1e-6
    assert var < 1e-6
    assert idx.confidence("t") > 0.99

    for g in (0.1, 5.0, 0.1, 5.0, 0.1, 5.0):  # bimodal -> high variance
        idx.record("bimodal", g)
    assert idx.confidence("bimodal") < idx.confidence("t")


def test_gap_index_capacity_falls_back_to_prior():
    idx = ToolGapIndex(capacity=1, priors={"known": (0.5, 0.01)}, default_prior=(9.0, 81.0))
    idx.record("known", 4.0)  # learned value present...
    idx.record("other", 4.0)  # ...but capacity=1 evicts "known"
    # Evicted tool falls back to its static prior, not the learned 4.0.
    assert idx.predict("known") == (0.5, 0.01)
    # Never-seen tool with no prior -> global default.
    assert idx.predict("mystery") == (9.0, 81.0)


# --- node cache-retention ----------------------------------------------------


def test_retain_until_protects_prefix_from_eviction():
    node = Node(0, cache_blocks=3)
    node.admit(0.0, 1.0, [1, 2, 3], retain_until=100.0)  # pin these
    # New blocks arrive unprotected; the protected ones must survive, evicting the newcomers.
    node.admit(1.0, 1.0, [4, 5, 6])
    assert node.matched([1, 2, 3]) == 3
    assert node.matched([4, 5, 6]) == 0


def test_expired_retention_evicts_like_lru():
    node = Node(0, cache_blocks=3)
    node.admit(0.0, 1.0, [1, 2, 3], retain_until=5.0)
    # By now=10 the pin has expired, so pressure evicts the old blocks as usual.
    node.admit(10.0, 1.0, [4, 5, 6])
    assert node.matched([1, 2, 3]) == 0
    assert node.matched([4, 5, 6]) == 3


def test_unprotected_readmit_clears_stale_pin():
    # Re-admitting a pinned block without protection must drop the old pin, so the default
    # retain_until=0.0 path reproduces pure LRU (block 1 is no longer shielded).
    node = Node(0, cache_blocks=2)
    node.admit(0.0, 1.0, [1], retain_until=100.0)
    node.admit(1.0, 1.0, [1])  # re-admit unprotected -> pin cleared
    # Fill the cache so block 1 must compete on pure recency; it is now the LRU, so it goes.
    node.admit(2.0, 1.0, [2, 3])
    assert node.matched([1]) == 0
    assert node.matched([2, 3]) == 2


def test_all_pinned_evicts_soonest_expiring():
    # When every resident block is still protected, eviction should sacrifice the pin that
    # expires soonest, not the strict-LRU block (which here is pinned far longer).
    node = Node(0, cache_blocks=2)
    node.admit(0.0, 1.0, [1], retain_until=100.0)  # LRU-oldest, but pinned longest
    node.admit(1.0, 1.0, [2], retain_until=10.0)  # expires soonest
    node.admit(2.0, 1.0, [3], retain_until=100.0)  # overflow at now=2, all still pinned
    assert node.matched([2]) == 0  # soonest-expiring pin sacrificed
    assert node.matched([1]) == 1  # longer-lived pin kept despite being LRU-oldest
    assert node.matched([3]) == 1


# --- class-aware-reliability policy ------------------------------------------


def _tool_view(blocks, *, conversation_id, last_tool_name):
    return RequestView(
        blocks=blocks,
        prompt_tokens=len(blocks) * BLOCK,
        num_messages=4,
        has_tool_messages=True,  # -> classify() == "tool"
        last_message_tokens=6,
        conversation_id=conversation_id,
        last_tool_name=last_tool_name,
    )


def test_reliability_policy_in_registry():
    assert "class-aware-reliability" in POLICY_NAMES
    assert isinstance(build_policy("class-aware-reliability"), ClassAwareReliability)


def test_reliability_policy_retains_for_confident_fast_tool():
    params = CostParams(block_size=BLOCK)
    nodes = _nodes(n=2)
    policy = ClassAwareReliability()
    blocks = [1, 2, 3, 4]
    # Drive one conversation returning every 0.3s after calling "fast": the policy learns a
    # short, low-variance gap and should soft-pin the prefix on the last turn.
    last = None
    for k in range(6):
        last = policy(
            _tool_view(blocks, conversation_id=7, last_tool_name="fast"), nodes, k * 0.3, params
        )
    assert last is not None and last.retain_until > 6 * 0.3  # keep-warm window into the future


def test_reliability_policy_no_retention_for_unknown_tool():
    params = CostParams(block_size=BLOCK)
    nodes = _nodes(n=2)
    policy = ClassAwareReliability()
    # An unseen tool falls back to the low-confidence default prior -> no retention gamble.
    p = policy(
        _tool_view([1, 2, 3, 4], conversation_id=1, last_tool_name="brand_new"), nodes, 0.0, params
    )
    assert p.retain_until == 0.0


def test_reliability_policy_beats_class_aware_on_tool_hit_rate():
    # Under cache pressure, keeping fast-returning sessions' prefixes warm should lift the
    # tool-class cross-turn hit rate versus plain class-aware on the same stream.
    requests = generate_mixed(
        60, 5, 40.0, mix={"tool": 1.0}, block_size=BLOCK, tool_reliability=True, seed=21
    )
    params = CostParams(block_size=BLOCK)
    base = run_simulation(requests, build_policy("class-aware"), 3, 60, params)
    rel = run_simulation(requests, build_policy("class-aware-reliability"), 3, 60, params)
    assert rel.by_kind["tool"].hit_rate >= base.by_kind["tool"].hit_rate
    # And the workload exposed a per-tool gap distribution to learn from.
    assert rel.gap_by_tool


# --- Belady's MIN eviction / oracle-belady ------------------------------------


def test_belady_retains_soon_needed_block_that_lru_evicts():
    # Cache capacity=3. Admit blocks [1,2,3], then [4,5,6]. LRU evicts 1,2,3 entirely.
    # With Belady, if block 1 is needed again at request index 2 but blocks 4,5,6 are
    # never needed again, Belady keeps 1 and evicts one of {4,5,6} instead.
    future_uses: dict[int, list[int]] = {
        1: [0, 2],  # block 1 used at request 0 and again at request 2
        2: [0],  # block 2 used only at request 0
        3: [0],  # block 3 used only at request 0
        4: [1],  # block 4 used only at request 1
        5: [1],  # block 5 used only at request 1
        6: [1],  # block 6 used only at request 1
    }
    node = BeladyNode(0, cache_blocks=3, future_uses=future_uses)
    node.admit(0.0, 1.0, [1, 2, 3], at_index=0)
    assert node.matched([1, 2, 3]) == 3

    node.admit(1.0, 1.0, [4, 5, 6], at_index=1)
    # Belady should have kept block 1 (needed at request 2) and evicted 2 and 3 (never
    # needed again) plus one of {4,5,6} that was just admitted but has no future use.
    # After admitting [4,5,6], we have 6 blocks total vs capacity 3 -> 3 evictions.
    # Eviction order: blocks 2,3 (next use inf) then the furthest-future of {1,4,5,6}.
    # Block 1 has next use at index 2; blocks 4,5,6 have next use inf (their only use was
    # the current request 1, and bisect_right(1, 1) -> past the end). So 4,5,6 get evicted
    # after 2,3, but we only need 3 evictions total: evict 2, 3, and one of {4,5,6} that
    # has inf next use.
    # Result: block 1 survives.
    assert 1 in node.cache


def test_belady_evicts_like_lru_when_no_future_knowledge():
    # When every block's future uses list is empty (or absent), Belady degrades to
    # always-evict-first-candidate (all have inf next use), which still frees space.
    node = BeladyNode(0, cache_blocks=3, future_uses={})
    node.admit(0.0, 1.0, [1, 2, 3], at_index=0)
    node.admit(1.0, 1.0, [4, 5, 6], at_index=1)
    # All old blocks have inf next use, so all three are evicted.
    assert node.matched([1, 2, 3]) == 0
    assert node.matched([4, 5, 6]) == 3


def test_oracle_belady_in_registry():
    assert "oracle-belady" in POLICY_NAMES
    # oracle-belady reuses the oracle routing function; the eviction difference is carried
    # by uses_belady(), the single source of truth the CLI/report consult.
    assert build_policy("oracle-belady") is oracle
    assert uses_belady("oracle-belady") and not uses_belady("oracle")


def test_oracle_belady_regret_is_zero():
    # Like the LRU oracle, oracle-belady routes optimally on current state so per-request
    # regret vs. itself must be ~0. (Belady only changes eviction, not routing.)
    requests = generate_sessions(30, turns=4, qps=8.0, block_size=BLOCK, seed=3)
    params = CostParams(block_size=BLOCK)
    res = run_simulation(requests, build_policy("oracle-belady"), 3, 500, params, belady=True)
    assert max(res.regrets) < 1e-9


def test_oracle_belady_hit_rate_ge_oracle_lru():
    # Empirical (not construction-guaranteed): with a global, routing-agnostic future-use
    # map and a contiguous-prefix hit metric, Belady's MIN is only an approximation, so this
    # >= holds for this fixed seed rather than for all workloads. See docs/policies.md.
    requests = generate_mixed(
        60, 5, 40.0, mix={"tool": 1.0}, block_size=BLOCK, tool_reliability=True, seed=21
    )
    params = CostParams(block_size=BLOCK)
    lru = run_simulation(requests, build_policy("oracle"), 3, 60, params)
    bel = run_simulation(requests, build_policy("oracle-belady"), 3, 60, params, belady=True)
    assert bel.hit_rate >= lru.hit_rate - 1e-9
