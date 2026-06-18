from __future__ import annotations

from dataclasses import replace

import numpy as np

from qbp_sim.cli import _apply_virtual_swap_policy_args, _build_parser
from qbp_sim.config import SimulationInputConfig
from qbp_sim.core.replay import replay_event_stream
import qbp_sim.core.producer as producer_module
from qbp_sim.events import QBPEvent
from qbp_sim.simulator import GillespieQBPConfig, GillespieQBPSimulator, VirtualSwapPolicy
from qbp_sim.trace import EventTraceReader, EventTraceWriter
from tests.support import _assert_state_invariants, _swap_index


def _empty_config(n_nodes: int, *, max_min: bool = True) -> GillespieQBPConfig:
    return GillespieQBPConfig(
        generation_rates=np.zeros((n_nodes, n_nodes), dtype=np.float64),
        demand_rates=np.zeros((n_nodes, n_nodes), dtype=np.float64),
        swap_rates=np.zeros(n_nodes, dtype=np.float64),
        service_rates=np.zeros((n_nodes, n_nodes), dtype=np.float64),
        virtual_swap_policy=VirtualSwapPolicy(mode="max_min" if max_min else "bp"),
    )


def test_max_min_policy_selects_minimum_preferred_output_queue() -> None:
    config = _empty_config(4)
    config.swap_rates[0] = 1.0
    initial_q = np.zeros((4, 4), dtype=np.int64)
    initial_q[0, 1] = initial_q[1, 0] = 5
    initial_q[0, 2] = initial_q[2, 0] = 5
    initial_q[0, 3] = initial_q[3, 0] = 5
    initial_q[1, 2] = initial_q[2, 1] = 2
    initial_q[1, 3] = initial_q[3, 1] = 1
    initial_q[2, 3] = initial_q[3, 2] = 10

    sim = GillespieQBPSimulator(config, seed=101, initial_q=initial_q)
    expected_idx = _swap_index(sim, 0, 1, 3)

    assert sim.producer.max_min_swap_best_idx[0] == expected_idx
    assert sim.producer.max_min_swap_best_output[0] == 1

    event = sim.produce_next_event()
    assert event is not None
    assert event.event_type == "max_min_swap"
    assert event.swap_idx == expected_idx

    sim.apply_event(event)

    assert sim.state.q[0, 1] == 4
    assert sim.state.q[0, 3] == 4
    assert sim.state.q[1, 3] == 2
    assert sim.state.h_mu.sum() == 0
    assert sim.virtual_swap_requests == 0
    assert sim.swaps_completed == 1
    _assert_state_invariants(sim)


def test_max_min_policy_has_no_swap_rate_without_preferred_candidate() -> None:
    config = _empty_config(3)
    config.swap_rates[0] = 1.0
    initial_q = np.zeros((3, 3), dtype=np.int64)
    initial_q[0, 1] = initial_q[1, 0] = 2
    initial_q[0, 2] = initial_q[2, 0] = 2
    initial_q[1, 2] = initial_q[2, 1] = 1

    sim = GillespieQBPSimulator(config, seed=102, initial_q=initial_q)

    assert sim.producer.max_min_swap_best_idx[0] == -1
    assert sim.producer.active_max_min_swap_total == 0.0
    assert sim.produce_next_event() is None


def test_max_min_service_request_admits_demand_without_alpha_scarcity() -> None:
    config = _empty_config(2)
    config.service_rates[0, 1] = config.service_rates[1, 0] = 1.0
    initial_d = np.zeros((2, 2), dtype=np.int64)
    initial_d[0, 1] = initial_d[1, 0] = 1

    sim = GillespieQBPSimulator(config, seed=103, initial_d=initial_d)
    event = sim.produce_next_event()
    assert event is not None
    assert event.event_type == "service_request"

    sim.apply_event(event)

    assert sim.state.d[0, 1] == 0
    assert sim.state.h_r[0, 1] == 1
    assert sim.state.alpha[0, 1] == 0
    assert sim.virtual_service_requests == 1
    _assert_state_invariants(sim)


def test_max_min_instant_service_after_service_request_fulfills_existing_inventory(tmp_path) -> None:
    config = replace(_empty_config(2), instant_service_fulfillment=True)
    config.service_rates[0, 1] = config.service_rates[1, 0] = 1.0
    initial_d = np.zeros((2, 2), dtype=np.int64)
    initial_q = np.zeros((2, 2), dtype=np.int64)
    initial_d[0, 1] = initial_d[1, 0] = 1
    initial_q[0, 1] = initial_q[1, 0] = 1
    trace_path = tmp_path / "max-min-instant-service.jsonl.zst"

    sim = GillespieQBPSimulator(config, seed=107, initial_d=initial_d, initial_q=initial_q)
    with EventTraceWriter(trace_path) as trace_writer:
        sampled = sim.step(trace_writer=trace_writer)

    with EventTraceReader(trace_path) as trace_reader:
        event_types = [event.event_type for event in trace_reader]

    assert sampled is not None
    assert sampled.event_type == "service_request"
    assert event_types == ["service_request", "physical_service"]
    assert sim.state.d[0, 1] == 0
    assert sim.state.h_r[0, 1] == 0
    assert sim.state.q[0, 1] == 0
    assert sim.services_completed == 1
    _assert_state_invariants(sim)


def test_max_min_trace_replays_without_virtual_swap_deficits(tmp_path) -> None:
    config = replace(
        _empty_config(4),
        generation_rates=np.array(
            [
                [0.0, 4.0, 4.0, 0.0],
                [4.0, 0.0, 0.0, 4.0],
                [4.0, 0.0, 0.0, 4.0],
                [0.0, 4.0, 4.0, 0.0],
            ],
            dtype=np.float64,
        ),
        demand_rates=np.array(
            [
                [0.0, 0.0, 0.0, 1.0],
                [0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0, 0.0],
            ],
            dtype=np.float64,
        ),
        service_rates=np.ones((4, 4), dtype=np.float64),
        swap_rates=np.array([0.0, 3.0, 3.0, 0.0], dtype=np.float64),
    )
    np.fill_diagonal(config.service_rates, 0.0)
    trace_path = tmp_path / "max-min.jsonl.zst"
    sim = GillespieQBPSimulator(config, seed=104)

    with EventTraceWriter(trace_path) as trace_writer:
        original = sim.run(until_time=10.0, max_events=300, sample_every=25, trace_writer=trace_writer, progress=False)

    with EventTraceReader(trace_path) as trace_reader:
        replayed = replay_event_stream(config=config, events=trace_reader, final_time=original.final_time, sample_every=25)

    assert original.events_processed == replayed.events_processed
    assert original.demand_arrivals == replayed.demand_arrivals
    assert original.services_completed == replayed.services_completed
    assert original.swaps_completed == replayed.swaps_completed
    assert original.virtual_swap_requests == 0
    assert replayed.virtual_swap_requests == 0


def test_max_min_config_and_cli_override_are_accepted() -> None:
    input_config = SimulationInputConfig(
        generation_rates=[[0.0, 1.0], [1.0, 0.0]],
        consumption_rates=[[0.0, 1.0], [1.0, 0.0]],
        swap_rates=[1.0, 1.0],
        virtual_swap_policy={"mode": "max-min"},
    )
    runtime = input_config.to_runtime_config()

    assert runtime.virtual_swap_policy.mode == "max_min"

    parser = _build_parser()
    args = parser.parse_args(["example", "--virtual-swap-policy", "max-min"])
    overridden = _apply_virtual_swap_policy_args(_empty_config(2, max_min=False), args)

    assert overridden.virtual_swap_policy.mode == "max_min"


def test_max_min_policy_rejects_unused_limited_information_parameters() -> None:
    try:
        GillespieQBPSimulator(
            replace(_empty_config(3), virtual_swap_policy=VirtualSwapPolicy(mode="max_min", k=1)),
            seed=105,
        )
    except ValueError as exc:
        assert "does not use k or memory" in str(exc)
    else:
        raise AssertionError("Expected max_min policy with k to fail validation.")


def test_limited_info_max_min_selects_best_sampled_preferred_candidate(monkeypatch) -> None:
    config = replace(
        _empty_config(4),
        virtual_swap_policy=VirtualSwapPolicy(mode="limited_info_max_min", k=2, memory=2),
    )
    config.swap_rates[0] = 1.0
    initial_q = np.zeros((4, 4), dtype=np.int64)
    initial_q[0, 1] = initial_q[1, 0] = 5
    initial_q[0, 2] = initial_q[2, 0] = 5
    initial_q[0, 3] = initial_q[3, 0] = 5
    initial_q[1, 2] = initial_q[2, 1] = 3
    initial_q[1, 3] = initial_q[3, 1] = 1
    initial_q[2, 3] = initial_q[3, 2] = 2

    sim = GillespieQBPSimulator(config, seed=108, initial_q=initial_q)
    producer = sim.producer
    sampled = np.asarray(
        [
            _swap_index(sim, 0, 1, 2),
            _swap_index(sim, 0, 1, 3),
        ],
        dtype=np.int64,
    )

    def fixed_sample(node: int) -> np.ndarray:
        return sampled if node == 0 else np.empty(0, dtype=np.int64)

    monkeypatch.setattr(producer, "_sample_limited_max_min_swap_candidates", fixed_sample)
    producer._refresh_limited_max_min_swap_node(sim.state, 0)

    assert producer.max_min_swap_memory_idx[0].tolist() == sampled[::-1].tolist()
    assert producer.max_min_swap_memory_output[0].tolist() == [1, 3]
    assert producer.max_min_swap_best_idx[0] == _swap_index(sim, 0, 1, 3)


def test_limited_info_max_min_does_not_use_global_max_min_scan(monkeypatch) -> None:
    def fail_global_scan(*args, **kwargs):
        raise AssertionError("limited-info max-min must not use the global max-min scan")

    monkeypatch.setattr(producer_module, "_compute_active_max_min_swap_rates", fail_global_scan)
    config = replace(
        _empty_config(4),
        virtual_swap_policy=VirtualSwapPolicy(mode="limited_info_max_min", k=1, memory=1),
    )
    config.swap_rates[0] = 1.0
    initial_q = np.zeros((4, 4), dtype=np.int64)
    initial_q[0, 1] = initial_q[1, 0] = 3
    initial_q[0, 2] = initial_q[2, 0] = 3

    sim = GillespieQBPSimulator(config, seed=109, initial_q=initial_q)

    assert sim.producer.active_max_min_swap_total == 1.0
    event = sim.produce_next_event(until_time=10.0)
    assert event is not None
    assert event.event_type in {"max_min_swap", "max_min_swap_idle"}


def test_limited_info_max_min_idle_event_replays() -> None:
    config = replace(
        _empty_config(3),
        virtual_swap_policy=VirtualSwapPolicy(mode="limited_info_max_min", k=1, memory=0),
    )
    event = QBPEvent(
        event_index=1,
        time=0.0,
        dt=0.0,
        total_rate=0.0,
        event_rate=0.0,
        event_type="max_min_swap_idle",
        i=0,
    )

    sim = GillespieQBPSimulator(config, seed=110)
    sim.apply_event(event)

    assert sim.events_processed == 1
    assert sim.swaps_completed == 0
    assert sim.total_inventory == 0


def test_max_min_event_applies_as_concrete_replayable_swap() -> None:
    config = _empty_config(3)
    initial_q = np.zeros((3, 3), dtype=np.int64)
    initial_q[0, 1] = initial_q[1, 0] = 3
    initial_q[0, 2] = initial_q[2, 0] = 3
    sim = GillespieQBPSimulator(config, seed=106, initial_q=initial_q)

    event = QBPEvent(
        event_index=1,
        time=0.0,
        dt=0.0,
        total_rate=0.0,
        event_rate=0.0,
        event_type="max_min_swap",
        swap_idx=_swap_index(sim, 0, 1, 2),
        i=0,
        y=1,
        z=2,
    )
    sim.apply_event(event)

    assert sim.state.q[0, 1] == 2
    assert sim.state.q[0, 2] == 2
    assert sim.state.q[1, 2] == 1
    assert sim.state.total_inventory == 5
    assert sim.state.total_swap_deficit == 0
