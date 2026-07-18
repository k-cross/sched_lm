from bench.program_metrics import ProgramMetrics, compute_program_metrics
from bench.traffic import BenchmarkResult, TurnMetric


def _turn(sid, idx, ttft=0.1, prompt=100, cached=0):
    return TurnMetric(
        session_id=sid,
        turn_idx=idx,
        ttft=ttft,
        e2e=ttft + 0.5,
        prompt_tokens=prompt,
        completion_tokens=10,
        cached_tokens=cached,
    )


def _result(turns, successes=None, wall=10.0, session_times=None):
    result = BenchmarkResult()
    result.turns = list(turns)
    result.successes = len(turns) if successes is None else successes
    result.wall_seconds = wall
    result.session_times = session_times or {}
    return result


def test_empty_result_yields_inapplicable_metrics():
    metrics = compute_program_metrics(BenchmarkResult())
    assert metrics == ProgramMetrics(
        throughput_rps=0.0,
        session_p50=None,
        session_p90=None,
        zero_recompute_rate=None,
        ttft_by_turn={},
        coverage_by_turn={},
    )


def test_zero_recompute_block_rounding():
    # Previous prompt 100 tokens -> block-aligned threshold floor(100/16)*16 = 96:
    # cached 96 is a zero-recompute return, cached 80 is not.
    hit = _result([_turn(0, 0, prompt=100), _turn(0, 1, prompt=200, cached=96)])
    miss = _result([_turn(1, 0, prompt=100), _turn(1, 1, prompt=200, cached=80)])

    assert compute_program_metrics(hit).zero_recompute_rate == 1.0
    assert compute_program_metrics(miss).zero_recompute_rate == 0.0


def test_zero_recompute_skips_turn_zero_gaps_and_missing_usage():
    turns = [
        # Session 0: turn 0 never counts as a "returning" turn; (0,1) does and hits.
        _turn(0, 0, prompt=100, cached=100),
        _turn(0, 1, prompt=200, cached=96),
        # Session 1: turn 1 errored out, so (0, 2) is not a consecutive pair.
        _turn(1, 0, prompt=100),
        _turn(1, 2, prompt=300, cached=300),
        # Session 2: previous turn carried no usage payload -- unjudgeable.
        _turn(2, 0, prompt=0),
        _turn(2, 1, prompt=200, cached=200),
    ]
    assert compute_program_metrics(_result(turns)).zero_recompute_rate == 1.0


def test_ttft_and_coverage_keyed_by_turn_position():
    turns = [
        _turn(0, 0, ttft=0.4, prompt=100, cached=0),
        _turn(1, 0, ttft=0.6, prompt=100, cached=0),
        _turn(0, 1, ttft=0.2, prompt=200, cached=100),
        _turn(1, 1, ttft=0.1, prompt=200, cached=200),
    ]
    metrics = compute_program_metrics(_result(turns))

    assert metrics.ttft_by_turn == {0: 0.5, 1: 0.15000000000000002}
    assert metrics.coverage_by_turn == {0: 0.0, 1: 0.75}


def test_throughput_and_session_percentiles():
    times = {sid: float(sid + 1) for sid in range(10)}  # 1..10 s
    metrics = compute_program_metrics(_result([], successes=20, wall=4.0, session_times=times))

    assert metrics.throughput_rps == 5.0
    assert metrics.session_p50 == 5.5
    assert metrics.session_p90 == 9.1
