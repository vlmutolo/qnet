from __future__ import annotations

import json
import importlib.util
import os
from copy import deepcopy
from dataclasses import replace
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from qbp_sim.config import SimulationInputConfig
from qbp_sim.events import QBPEvent
from qbp_sim.examples import build_four_node_counterexample
from qbp_sim.experiments import (
    _apply_capacity_headroom,
    _cycle_consumption_edge_fraction,
    plot_limited_info_service_ratio_runs,
    run_limited_info_service_ratio_experiment,
)
from qbp_sim.analysis import plot_snapshot_metric, plot_snapshot_metric_series, summarize_snapshots
from qbp_sim.cli import _apply_instant_service_fulfillment_arg, _build_parser
from qbp_sim.progress import should_use_progress
from qbp_sim.snapshots import SnapshotReader, SnapshotWriter
import qbp_sim.simulator as simulator_module
from qbp_sim.simulator import GillespieQBPConfig, GillespieQBPSimulator, VirtualSwapPolicy, replay_event_stream
from qbp_sim.trace import EventTraceReader, EventTraceWriter, open_event_trace_reader, open_event_trace_writer


RUN_GATED_TESTS = os.environ.get("QBP_SIM_RUN_GATED_TESTS") == "1"
gated_test = pytest.mark.skipif(
    not RUN_GATED_TESTS,
    reason="set QBP_SIM_RUN_GATED_TESTS=1 to run gated stochastic simulator checks",
)


def _limited_policy(k: int, memory: int) -> VirtualSwapPolicy:
    return VirtualSwapPolicy(mode="power_of_k_memory", k=k, memory=memory)


def _collect_event_records(sim: GillespieQBPSimulator, count: int) -> list[dict[str, int | float | str]]:
    records: list[dict[str, int | float | str]] = []
    for _ in range(count):
        event = sim.step()
        if event is None:
            break
        records.append(event.to_dict())
    return records


def _upper_triangle_sum(matrix: np.ndarray) -> int:
    return int(np.triu(matrix, k=1).sum())


def _assert_state_invariants(sim: GillespieQBPSimulator) -> None:
    state = sim.state
    for matrix in (state.q, state.d, state.alpha, state.h_r):
        assert np.array_equal(matrix, matrix.T)
        assert np.all(np.diag(matrix) == 0)
        assert np.all(matrix >= 0)
    assert np.all(state.h_mu >= 0)
    assert state.total_inventory == _upper_triangle_sum(state.q)
    assert state.total_virtual_backlog == _upper_triangle_sum(state.d)
    assert state.total_scarcity == _upper_triangle_sum(state.alpha)
    assert state.total_service_deficit == _upper_triangle_sum(state.h_r)
    assert state.total_swap_deficit == int(state.h_mu.sum())
    assert state.total_backlog == state.total_virtual_backlog + state.total_service_deficit
    if state.demand_arrivals > 0:
        assert np.isclose(state.service_ratio, state.services_completed / state.demand_arrivals)
        assert 0.0 <= state.service_ratio <= 1.0


def _cycle_runtime_config(num_nodes: int, seed: int, tmp_path: Path) -> GillespieQBPConfig:
    linear_path = Path(__file__).resolve().parents[1] / "linear.py"
    spec = importlib.util.spec_from_file_location(f"linear_module_cycle_{num_nodes}_{seed}", linear_path)
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)

    config_path = tmp_path / f"cycle{num_nodes}_lp_config.json"
    module.single_run_topology(
        topology="cycle",
        num_nodes=num_nodes,
        seed=seed,
        output_mode="simulation-config",
        simulation_config_output_path=str(config_path),
    )
    return SimulationInputConfig.from_json_file(config_path).to_runtime_config()


def _single_edge_generation_config(*, instant_service_fulfillment: bool = False) -> GillespieQBPConfig:
    generation_rates = np.array([[0.0, 1.0], [1.0, 0.0]], dtype=np.float64)
    demand_rates = np.zeros((2, 2), dtype=np.float64)
    service_rates = np.zeros((2, 2), dtype=np.float64)
    swap_rates = np.zeros(2, dtype=np.float64)
    return GillespieQBPConfig(
        generation_rates=generation_rates,
        demand_rates=demand_rates,
        swap_rates=swap_rates,
        service_rates=service_rates,
        instant_service_fulfillment=instant_service_fulfillment,
    )


def _pending_service_matrix(n_nodes: int, x: int, y: int) -> np.ndarray:
    h_r = np.zeros((n_nodes, n_nodes), dtype=np.int64)
    h_r[x, y] = 1
    h_r[y, x] = 1
    return h_r


def test_service_event_consumes_inventory_and_backlog() -> None:
    generation_rates = np.zeros((2, 2), dtype=np.float64)
    demand_rates = np.zeros((2, 2), dtype=np.float64)
    service_rates = np.array([[0.0, 1.0], [1.0, 0.0]], dtype=np.float64)
    swap_rates = np.zeros(2, dtype=np.float64)

    initial_q = np.array([[0, 1], [1, 0]], dtype=np.int64)
    initial_d = np.array([[0, 1], [1, 0]], dtype=np.int64)
    initial_alpha = np.zeros((2, 2), dtype=np.int64)

    sim = GillespieQBPSimulator(
        config=GillespieQBPConfig(
            generation_rates=generation_rates,
            demand_rates=demand_rates,
            swap_rates=swap_rates,
            service_rates=service_rates,
        ),
        seed=7,
        initial_q=initial_q,
        initial_d=initial_d,
        initial_alpha=initial_alpha,
    )

    first = sim.step()
    second = sim.step()

    assert first is not None
    assert second is not None
    assert first.event_type == "virtual_service"
    assert second.event_type == "physical_service"
    assert sim.virtual_service_requests == 1
    assert sim.services_completed == 1
    assert sim.total_backlog == 0
    assert sim.total_inventory == 0


def test_instant_service_after_generation_fulfills_pending_service(tmp_path) -> None:
    config = _single_edge_generation_config(instant_service_fulfillment=True)
    initial_h_r = _pending_service_matrix(2, 0, 1)
    trace_path = tmp_path / "instant-generation.jsonl.zst"
    sim = GillespieQBPSimulator(config, seed=5, initial_h_r=initial_h_r)

    with EventTraceWriter(trace_path) as trace_writer:
        sampled = sim.step(trace_writer=trace_writer)

    with EventTraceReader(trace_path) as trace_reader:
        events = list(trace_reader)

    assert sampled is not None
    assert sampled.event_type == "pair_generation"
    assert [event.event_type for event in events] == ["pair_generation", "physical_service"]
    assert events[1].time == events[0].time
    assert events[1].dt == 0.0
    assert events[1].total_rate == 0.0
    assert events[1].event_rate == 0.0
    assert sim.pair_generations == 1
    assert sim.services_completed == 1
    assert sim.events_processed == 2
    assert sim.state.q[0, 1] == 0
    assert sim.state.h_r[0, 1] == 0
    assert sim.total_inventory == 0
    assert sim.state.total_service_deficit == 0


def test_instant_service_does_not_emit_without_pending_service(tmp_path) -> None:
    config = _single_edge_generation_config(instant_service_fulfillment=True)
    trace_path = tmp_path / "instant-negative.jsonl.zst"
    sim = GillespieQBPSimulator(config, seed=5)

    with EventTraceWriter(trace_path) as trace_writer:
        sampled = sim.step(trace_writer=trace_writer)

    with EventTraceReader(trace_path) as trace_reader:
        events = list(trace_reader)

    assert sampled is not None
    assert sampled.event_type == "pair_generation"
    assert [event.event_type for event in events] == ["pair_generation"]
    assert sim.pair_generations == 1
    assert sim.services_completed == 0
    assert sim.events_processed == 1
    assert sim.state.q[0, 1] == 1
    assert sim.total_inventory == 1


def test_instant_service_after_physical_swap_fulfills_output_edge(tmp_path) -> None:
    generation_rates = np.zeros((3, 3), dtype=np.float64)
    demand_rates = np.zeros((3, 3), dtype=np.float64)
    service_rates = np.zeros((3, 3), dtype=np.float64)
    swap_rates = np.array([1.0, 0.0, 0.0], dtype=np.float64)
    config = GillespieQBPConfig(
        generation_rates=generation_rates,
        demand_rates=demand_rates,
        swap_rates=swap_rates,
        service_rates=service_rates,
        instant_service_fulfillment=True,
    )
    probe = GillespieQBPSimulator(config, seed=0)
    matches = np.flatnonzero((probe.swap_i == 0) & (probe.swap_y == 1) & (probe.swap_z == 2))
    assert len(matches) == 1
    swap_idx = int(matches[0])

    initial_q = np.zeros((3, 3), dtype=np.int64)
    initial_q[0, 1] = initial_q[1, 0] = 1
    initial_q[0, 2] = initial_q[2, 0] = 1
    initial_h_mu = np.zeros_like(probe.state.h_mu)
    initial_h_mu[swap_idx] = 1
    initial_h_r = _pending_service_matrix(3, 1, 2)
    trace_path = tmp_path / "instant-swap.jsonl.zst"
    sim = GillespieQBPSimulator(
        config,
        seed=5,
        initial_q=initial_q,
        initial_h_r=initial_h_r,
        initial_h_mu=initial_h_mu,
    )

    with EventTraceWriter(trace_path) as trace_writer:
        sampled = sim.step(trace_writer=trace_writer)

    with EventTraceReader(trace_path) as trace_reader:
        events = list(trace_reader)

    assert sampled is not None
    assert sampled.event_type == "physical_swap"
    assert sampled.swap_idx == swap_idx
    assert [event.event_type for event in events] == ["physical_swap", "physical_service"]
    assert events[1].time == events[0].time
    assert events[1].dt == 0.0
    assert events[1].x == 1
    assert events[1].y == 2
    assert sim.swaps_completed == 1
    assert sim.services_completed == 1
    assert sim.events_processed == 2
    assert sim.state.q[1, 2] == 0
    assert sim.state.h_r[1, 2] == 0
    assert sim.state.h_mu[swap_idx] == 0
    assert sim.total_inventory == 0
    assert sim.state.total_service_deficit == 0


def test_instant_service_replay_is_literal_without_double_fulfillment(tmp_path) -> None:
    config = _single_edge_generation_config(instant_service_fulfillment=True)
    initial_h_r = _pending_service_matrix(2, 0, 1)
    trace_path = tmp_path / "instant-replay.jsonl.zst"
    sim = GillespieQBPSimulator(config, seed=5, initial_h_r=initial_h_r)

    with EventTraceWriter(trace_path) as trace_writer:
        original = sim.run(
            until_time=10.0,
            max_events=1,
            sample_every=0,
            trace_writer=trace_writer,
            progress=False,
        )

    with EventTraceReader(trace_path) as trace_reader:
        replayed = replay_event_stream(
            config=config,
            events=trace_reader,
            sample_every=0,
            final_time=original.final_time,
            initial_h_r=_pending_service_matrix(2, 0, 1),
        )

    assert original.events_processed == 2
    assert replayed.events_processed == original.events_processed
    assert replayed.final_time == original.final_time
    assert replayed.pair_generations == original.pair_generations
    assert replayed.services_completed == original.services_completed
    assert replayed.total_inventory == original.total_inventory
    assert replayed.total_backlog == original.total_backlog
    assert replayed.total_scarcity == original.total_scarcity


def test_instant_service_max_events_caps_sampled_events_not_companion_events() -> None:
    config = _single_edge_generation_config(instant_service_fulfillment=True)
    sim = GillespieQBPSimulator(config, seed=5, initial_h_r=_pending_service_matrix(2, 0, 1))

    result = sim.run(until_time=10.0, max_events=1, sample_every=0, progress=False)

    assert result.pair_generations == 1
    assert result.services_completed == 1
    assert result.events_processed == 2


def test_demand_only_network_accumulates_backlog() -> None:
    generation_rates = np.zeros((2, 2), dtype=np.float64)
    demand_rates = np.array([[0.0, 3.0], [3.0, 0.0]], dtype=np.float64)
    service_rates = np.zeros((2, 2), dtype=np.float64)
    swap_rates = np.zeros(2, dtype=np.float64)

    sim = GillespieQBPSimulator(
        config=GillespieQBPConfig(
            generation_rates=generation_rates,
            demand_rates=demand_rates,
            swap_rates=swap_rates,
            service_rates=service_rates,
        ),
        seed=5,
    )

    result = sim.run(until_time=1.0, max_events=64, sample_every=0)

    assert result.events_processed > 0
    assert result.demand_arrivals == result.events_processed
    assert result.total_backlog == result.demand_arrivals
    assert result.services_completed == 0


def test_four_node_example_builds_and_runs() -> None:
    sim = GillespieQBPSimulator(build_four_node_counterexample(), seed=11)
    result = sim.run(until_time=2.0, max_events=2_000, sample_every=100)

    assert result.events_processed > 0
    assert result.demand_arrivals > 0


def test_run_can_stop_by_time_without_event_cap() -> None:
    sim = GillespieQBPSimulator(build_four_node_counterexample(), seed=11)
    result = sim.run(until_time=0.5, max_events=None, sample_every=0, progress=False)

    assert result.final_time == 0.5
    assert result.events_processed > 0


def test_default_virtual_swap_policy_matches_explicit_global_policy() -> None:
    default_config = build_four_node_counterexample()
    explicit_global_config = replace(default_config, virtual_swap_policy=VirtualSwapPolicy(mode="global"))

    default_records = _collect_event_records(GillespieQBPSimulator(default_config, seed=47), 200)
    explicit_global_records = _collect_event_records(GillespieQBPSimulator(explicit_global_config, seed=47), 200)

    assert explicit_global_records == default_records


def test_limited_information_virtual_swap_policy_builds_and_runs() -> None:
    config = replace(
        build_four_node_counterexample(),
        virtual_swap_policy=_limited_policy(k=2, memory=2),
    )
    sim = GillespieQBPSimulator(config, seed=31)

    result = sim.run(until_time=2.0, max_events=2_000, sample_every=100, progress=False)

    assert result.events_processed > 0
    assert result.demand_arrivals > 0


def test_limited_information_replay_reproduces_summary_from_trace(tmp_path) -> None:
    config = replace(
        build_four_node_counterexample(),
        virtual_swap_policy=_limited_policy(k=2, memory=2),
    )
    trace_path = tmp_path / "limited-replay.jsonl.zst"
    sim = GillespieQBPSimulator(config, seed=37)

    with EventTraceWriter(trace_path) as trace_writer:
        original = sim.run(until_time=2.0, max_events=400, sample_every=25, trace_writer=trace_writer)

    with EventTraceReader(trace_path) as trace_reader:
        replayed = replay_event_stream(
            config=config,
            events=trace_reader,
            sample_every=25,
            final_time=original.final_time,
        )

    assert replayed.final_time == original.final_time
    assert replayed.events_processed == original.events_processed
    assert replayed.total_backlog == original.total_backlog
    assert replayed.total_inventory == original.total_inventory
    assert replayed.total_scarcity == original.total_scarcity
    assert replayed.virtual_swap_requests == original.virtual_swap_requests
    assert replayed.swaps_completed == original.swaps_completed


def test_limited_information_policy_matches_global_when_it_samples_every_candidate() -> None:
    generation_rates = np.zeros((4, 4), dtype=np.float64)
    demand_rates = np.zeros((4, 4), dtype=np.float64)
    service_rates = np.zeros((4, 4), dtype=np.float64)
    swap_rates = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
    initial_alpha = np.zeros((4, 4), dtype=np.int64)
    initial_alpha[2, 3] = 9
    initial_alpha[3, 2] = 9

    global_config = GillespieQBPConfig(
        generation_rates=generation_rates,
        demand_rates=demand_rates,
        swap_rates=swap_rates,
        service_rates=service_rates,
    )
    limited_config = replace(
        global_config,
        virtual_swap_policy=_limited_policy(k=3, memory=3),
    )

    global_sim = GillespieQBPSimulator(global_config, seed=41, initial_alpha=initial_alpha)
    limited_sim = GillespieQBPSimulator(limited_config, seed=41, initial_alpha=initial_alpha)
    limited_sim.producer._refresh_limited_virtual_swap_node(limited_sim.state, 0)

    assert limited_sim.producer.virtual_swap_memory_idx.shape == (4, 3)
    assert limited_sim.producer.virtual_swap_best_idx[0] == global_sim.producer.virtual_swap_best_idx[0]
    assert limited_sim.producer.virtual_swap_best_weight[0] == global_sim.producer.virtual_swap_best_weight[0]


def test_limited_information_all_candidate_refresh_matches_global_best_on_shared_state() -> None:
    global_sim = GillespieQBPSimulator(build_four_node_counterexample(), seed=53)
    for _ in range(100):
        assert global_sim.step(until_time=20.0) is not None

    limited_config = replace(global_sim.config, virtual_swap_policy=_limited_policy(k=3, memory=3))
    limited_sim = GillespieQBPSimulator(
        limited_config,
        seed=53,
        initial_q=global_sim.state.q,
        initial_d=global_sim.state.d,
        initial_alpha=global_sim.state.alpha,
        initial_h_r=global_sim.state.h_r,
        initial_h_mu=global_sim.state.h_mu,
    )
    for node in range(limited_sim.n_nodes):
        limited_sim.producer._refresh_limited_virtual_swap_node(limited_sim.state, node)
        assert limited_sim.producer.virtual_swap_best_idx[node] == global_sim.producer.virtual_swap_best_idx[node]
        assert limited_sim.producer.virtual_swap_best_weight[node] == global_sim.producer.virtual_swap_best_weight[node]


def test_limited_information_policy_bounds_candidate_score_reads(monkeypatch) -> None:
    def fail_global_scan(*args, **kwargs):
        raise AssertionError("limited-information policy must not use the global swap scan")

    monkeypatch.setattr(simulator_module, "_best_virtual_swap_for_node", fail_global_scan)
    config = replace(build_four_node_counterexample(), virtual_swap_policy=_limited_policy(k=2, memory=3))
    sim = GillespieQBPSimulator(config, seed=59)

    counts = dict.fromkeys(range(sim.n_nodes), 0)
    original_weight = sim.producer._virtual_swap_weight

    def counting_weight(state, swap_idx):
        node = int(sim.producer.swap_i[swap_idx])
        counts[node] += 1
        return original_weight(state, swap_idx)

    monkeypatch.setattr(sim.producer, "_virtual_swap_weight", counting_weight)
    sim.produce_next_event()

    assert sum(counts.values()) > 0
    assert all(count <= 5 for count in counts.values())


def test_limited_information_candidate_sampling_uses_independent_rng() -> None:
    config = replace(build_four_node_counterexample(), virtual_swap_policy=_limited_policy(k=1, memory=1))
    sim = GillespieQBPSimulator(config, seed=60)

    event_rng_state = deepcopy(sim.producer.rng.bit_generator.state)
    policy_rng_state = deepcopy(sim.producer.policy_rng.bit_generator.state)

    sim.producer._sample_limited_virtual_swap_candidates(1)

    assert sim.producer.rng.bit_generator.state == event_rng_state
    assert sim.producer.policy_rng.bit_generator.state != policy_rng_state


def test_limited_information_memory_keeps_best_sampled_candidates_not_global_best(monkeypatch) -> None:
    config = replace(
        GillespieQBPConfig(
            generation_rates=np.zeros((5, 5), dtype=np.float64),
            demand_rates=np.zeros((5, 5), dtype=np.float64),
            swap_rates=np.array([1.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float64),
            service_rates=np.zeros((5, 5), dtype=np.float64),
        ),
        virtual_swap_policy=_limited_policy(k=3, memory=2),
    )
    sim = GillespieQBPSimulator(config, seed=61)
    producer = sim.producer
    start = int(producer.swap_node_starts[0])
    sampled = np.asarray([start, start + 1, start + 2], dtype=np.int64)
    unsampled_global_best = start + 5

    sim.state.alpha[1, 2] = sim.state.alpha[2, 1] = 2
    sim.state.alpha[1, 3] = sim.state.alpha[3, 1] = 7
    sim.state.alpha[1, 4] = sim.state.alpha[4, 1] = 4
    sim.state.alpha[3, 4] = sim.state.alpha[4, 3] = 99
    producer.virtual_swap_memory_idx[0, :] = -1
    producer.virtual_swap_memory_weight[0, :] = 0

    def fixed_sample(node: int) -> np.ndarray:
        return sampled if node == 0 else np.empty(0, dtype=np.int64)

    monkeypatch.setattr(producer, "_sample_limited_virtual_swap_candidates", fixed_sample)
    producer._refresh_limited_virtual_swap_node(sim.state, 0)

    assert producer.virtual_swap_memory_idx[0].tolist() == [start + 1, start + 2]
    assert producer.virtual_swap_memory_weight[0].tolist() == [7, 4]
    assert unsampled_global_best not in producer.virtual_swap_memory_idx[0].tolist()


def test_limited_information_memory_rescores_stale_candidates(monkeypatch) -> None:
    config = replace(
        GillespieQBPConfig(
            generation_rates=np.zeros((4, 4), dtype=np.float64),
            demand_rates=np.zeros((4, 4), dtype=np.float64),
            swap_rates=np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64),
            service_rates=np.zeros((4, 4), dtype=np.float64),
        ),
        virtual_swap_policy=_limited_policy(k=1, memory=1),
    )
    sim = GillespieQBPSimulator(config, seed=67)
    producer = sim.producer
    candidate = int(producer.swap_node_starts[0] + 2)

    def first_sample(node: int) -> np.ndarray:
        return np.asarray([candidate], dtype=np.int64) if node == 0 else np.empty(0, dtype=np.int64)

    monkeypatch.setattr(producer, "_sample_limited_virtual_swap_candidates", first_sample)
    sim.state.alpha[2, 3] = sim.state.alpha[3, 2] = 5
    producer.virtual_swap_memory_idx[0, :] = -1
    producer.virtual_swap_memory_weight[0, :] = 0
    producer._refresh_limited_virtual_swap_node(sim.state, 0)

    assert producer.virtual_swap_best_idx[0] == candidate
    assert producer.virtual_swap_node_rates[0] == 1.0

    sim.state.alpha[2, 3] = sim.state.alpha[3, 2] = 0
    sim.state.alpha[0, 2] = sim.state.alpha[2, 0] = 3
    monkeypatch.setattr(
        producer,
        "_sample_limited_virtual_swap_candidates",
        lambda node: np.empty(0, dtype=np.int64),
    )
    producer._refresh_limited_virtual_swap_node(sim.state, 0)

    assert producer.virtual_swap_memory_idx[0, 0] == candidate
    assert producer.virtual_swap_memory_weight[0, 0] < 0
    assert producer.virtual_swap_best_idx[0] == -1
    assert producer.virtual_swap_node_rates[0] == 1.0


def test_limited_information_virtual_swap_event_is_valid_and_applies_expected_state_changes() -> None:
    initial_alpha = np.zeros((4, 4), dtype=np.int64)
    initial_alpha[2, 3] = initial_alpha[3, 2] = 5
    config = replace(
        GillespieQBPConfig(
            generation_rates=np.zeros((4, 4), dtype=np.float64),
            demand_rates=np.zeros((4, 4), dtype=np.float64),
            swap_rates=np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64),
            service_rates=np.zeros((4, 4), dtype=np.float64),
        ),
        virtual_swap_policy=_limited_policy(k=3, memory=3),
    )
    sim = GillespieQBPSimulator(config, seed=71, initial_alpha=initial_alpha)
    event = sim.produce_next_event(until_time=20.0)
    assert event is not None
    assert event.event_type == "virtual_swap"
    assert event.swap_idx is not None and event.swap_idx >= 0
    assert event.i == int(sim.swap_i[event.swap_idx])
    assert event.y == int(sim.swap_y[event.swap_idx])
    assert event.z == int(sim.swap_z[event.swap_idx])
    assert event.event_rate > 0.0

    i = int(event.i)
    y = int(event.y)
    z = int(event.z)
    idx = int(event.swap_idx)
    old_h_mu = int(sim.state.h_mu[idx])
    old_alpha_iy = int(sim.state.alpha[i, y])
    old_alpha_iz = int(sim.state.alpha[i, z])
    old_alpha_yz = int(sim.state.alpha[y, z])
    old_swap_requests = sim.state.virtual_swap_requests
    old_swap_deficit = sim.state.total_swap_deficit

    sim.apply_event(event)

    assert sim.state.h_mu[idx] == old_h_mu + 1
    assert sim.state.alpha[i, y] == old_alpha_iy + 1
    assert sim.state.alpha[i, z] == old_alpha_iz + 1
    assert sim.state.alpha[y, z] == max(old_alpha_yz - 1, 0)
    assert sim.state.virtual_swap_requests == old_swap_requests + 1
    assert sim.state.total_swap_deficit == old_swap_deficit + 1
    _assert_state_invariants(sim)


@pytest.mark.gated
@gated_test
def test_gated_limited_information_parameter_sweep_sanity(tmp_path) -> None:
    global_config = _cycle_runtime_config(num_nodes=5, seed=7, tmp_path=tmp_path)
    global_result = GillespieQBPSimulator(global_config, seed=79).run(
        until_time=250.0,
        max_events=250_000,
        sample_every=0,
        progress=False,
    )
    global_ratio = global_result.services_completed / global_result.demand_arrivals
    ratios: dict[tuple[int, int], float] = {}

    for k, memory in ((1, 1), (2, 2), (6, 6)):
        config = replace(global_config, virtual_swap_policy=_limited_policy(k=k, memory=memory))
        sim = GillespieQBPSimulator(config, seed=79)
        result = sim.run(
            until_time=250.0,
            max_events=250_000,
            sample_every=0,
            progress=False,
        )
        ratios[(k, memory)] = result.services_completed / result.demand_arrivals
        _assert_state_invariants(sim)

    assert all(0.0 <= ratio <= 1.0 for ratio in ratios.values())
    assert abs(ratios[(6, 6)] - global_ratio) < 0.02


@pytest.mark.gated
@gated_test
def test_gated_limited_information_long_run_preserves_state_invariants() -> None:
    config = replace(build_four_node_counterexample(), virtual_swap_policy=_limited_policy(k=2, memory=2))
    sim = GillespieQBPSimulator(config, seed=83)

    for event_idx in range(5_000):
        event = sim.step(until_time=500.0)
        if event is None:
            break
        if event_idx % 100 == 0:
            _assert_state_invariants(sim)

    _assert_state_invariants(sim)
    assert sim.events_processed > 0


@pytest.mark.gated
@gated_test
def test_gated_limited_information_four_node_counterexample_makes_progress() -> None:
    config = replace(build_four_node_counterexample(), virtual_swap_policy=_limited_policy(k=2, memory=2))
    result = GillespieQBPSimulator(config, seed=89).run(
        until_time=1_000.0,
        max_events=500_000,
        sample_every=0,
        progress=False,
    )
    service_ratio = result.services_completed / result.demand_arrivals

    assert result.virtual_swap_requests > 0
    assert result.swaps_completed > 0
    assert result.services_completed > 0
    assert service_ratio > 0.35


def test_limited_information_policy_allows_query_only_memory_zero() -> None:
    initial_alpha = np.zeros((4, 4), dtype=np.int64)
    initial_alpha[2, 3] = initial_alpha[3, 2] = 5
    config = replace(
        GillespieQBPConfig(
            generation_rates=np.zeros((4, 4), dtype=np.float64),
            demand_rates=np.zeros((4, 4), dtype=np.float64),
            swap_rates=np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64),
            service_rates=np.zeros((4, 4), dtype=np.float64),
        ),
        virtual_swap_policy=_limited_policy(k=3, memory=0),
    )

    sim = GillespieQBPSimulator(config, seed=43, initial_alpha=initial_alpha)
    assert sim.producer.virtual_swap_memory_idx.shape == (4, 0)

    event = sim.produce_next_event(until_time=20.0)

    assert event is not None
    assert event.event_type == "virtual_swap"
    assert event.i == 0


def test_limited_information_policy_rejects_missing_positive_k() -> None:
    config = replace(
        build_four_node_counterexample(),
        virtual_swap_policy=_limited_policy(k=0, memory=2),
    )

    try:
        GillespieQBPSimulator(config, seed=43)
    except Exception as exc:
        assert "positive k" in str(exc)
    else:
        raise AssertionError("Expected invalid limited-information policy to fail validation.")


def test_reset_measurements_keeps_state_but_resets_counters_and_time() -> None:
    sim = GillespieQBPSimulator(build_four_node_counterexample(), seed=29)
    sim.run(until_time=2.0, max_events=2_000, sample_every=0, progress=False)

    backlog_before = sim.total_backlog
    inventory_before = sim.total_inventory
    scarcity_before = sim.total_scarcity
    assert sim.events_processed > 0
    assert sim.time > 0.0

    sim.reset_measurements(reset_time_origin=True)

    assert sim.events_processed == 0
    assert sim.time == 0.0
    assert sim.demand_arrivals == 0
    assert sim.pair_generations == 0
    assert sim.virtual_service_requests == 0
    assert sim.virtual_swap_requests == 0
    assert sim.services_completed == 0
    assert sim.swaps_completed == 0
    assert sim.total_backlog == backlog_before
    assert sim.total_inventory == inventory_before
    assert sim.total_scarcity == scarcity_before


def test_trace_writer_records_every_event(tmp_path) -> None:
    trace_path = tmp_path / "events.jsonl.zst"
    sim = GillespieQBPSimulator(build_four_node_counterexample(), seed=13)

    with EventTraceWriter(trace_path) as trace_writer:
        result = sim.run(until_time=1.0, max_events=200, sample_every=0, trace_writer=trace_writer)

    with EventTraceReader(trace_path) as trace_reader:
        lines = [event.to_dict() for event in trace_reader]

    assert len(lines) == result.events_processed
    assert all("event_type" in line for line in lines)
    assert all("time" in line for line in lines)


def test_parquet_trace_writer_records_every_event_and_replays(tmp_path) -> None:
    trace_path = tmp_path / "events.parquet"
    config = build_four_node_counterexample()
    sim = GillespieQBPSimulator(config, seed=131)

    with open_event_trace_writer(trace_path) as trace_writer:
        original = sim.run(until_time=2.0, max_events=400, sample_every=25, trace_writer=trace_writer)

    with open_event_trace_reader(trace_path) as trace_reader:
        events = list(trace_reader)

    parquet_columns = set(pq.ParquetFile(trace_path).schema_arrow.names)
    assert "time" in parquet_columns
    assert "dt" not in parquet_columns
    assert pq.ParquetFile(trace_path).schema_arrow.field("time").type == pa.float32()
    assert len(events) == original.events_processed
    assert events[0].event_index == 1
    assert events[0].dt == events[0].time
    assert events[-1].event_index == original.events_processed
    assert events[-1].inventory_total == original.total_inventory

    replayed = replay_event_stream(
        config=config,
        events=events,
        sample_every=25,
        final_time=original.final_time,
    )
    assert replayed.final_time == original.final_time
    assert replayed.events_processed == original.events_processed
    assert replayed.total_backlog == original.total_backlog
    assert replayed.total_inventory == original.total_inventory
    assert replayed.total_scarcity == original.total_scarcity


def test_trace_float16_rejects_out_of_range_time(tmp_path) -> None:
    trace_path = tmp_path / "events.vortex"
    writer = open_event_trace_writer(trace_path, float_precision="float16")
    with pytest.raises(ValueError, match="exceeds float16 range"):
        with writer as trace_writer:
            trace_writer.write(
                QBPEvent(
                    event_index=1,
                    time=70_000.0,
                    dt=70_000.0,
                    total_rate=1.0,
                    event_type="demand_arrival",
                    event_rate=1.0,
                    x=0,
                    y=1,
                    backlog_total=1,
                    inventory_total=0,
                    scarcity_total=0,
                )
            )


def test_vortex_trace_writer_records_every_event_and_replays(tmp_path) -> None:
    trace_path = tmp_path / "events.vortex"
    config = build_four_node_counterexample()
    sim = GillespieQBPSimulator(config, seed=132)

    with open_event_trace_writer(trace_path) as trace_writer:
        original = sim.run(until_time=2.0, max_events=400, sample_every=25, trace_writer=trace_writer)

    with open_event_trace_reader(trace_path) as trace_reader:
        events = list(trace_reader)

    assert trace_path.exists()
    assert len(events) == original.events_processed
    assert events[0].dt == events[0].time
    assert events[-1].event_index == original.events_processed

    replayed = replay_event_stream(
        config=config,
        events=events,
        sample_every=25,
        final_time=original.final_time,
    )
    assert replayed.final_time == original.final_time
    assert replayed.events_processed == original.events_processed
    assert replayed.total_backlog == original.total_backlog
    assert replayed.total_inventory == original.total_inventory
    assert replayed.total_scarcity == original.total_scarcity


def test_replay_reproduces_summary_from_trace(tmp_path) -> None:
    trace_path = tmp_path / "replay.jsonl.zst"
    sim = GillespieQBPSimulator(build_four_node_counterexample(), seed=17)

    with EventTraceWriter(trace_path) as trace_writer:
        original = sim.run(until_time=2.0, max_events=400, sample_every=25, trace_writer=trace_writer)

    with EventTraceReader(trace_path) as trace_reader:
        replayed = replay_event_stream(
            config=build_four_node_counterexample(),
            events=trace_reader,
            sample_every=25,
            final_time=original.final_time,
        )

    assert replayed.final_time == original.final_time
    assert replayed.events_processed == original.events_processed
    assert replayed.total_backlog == original.total_backlog
    assert replayed.total_inventory == original.total_inventory
    assert replayed.total_scarcity == original.total_scarcity
    assert replayed.demand_arrivals == original.demand_arrivals
    assert replayed.pair_generations == original.pair_generations
    assert replayed.virtual_service_requests == original.virtual_service_requests
    assert replayed.virtual_swap_requests == original.virtual_swap_requests
    assert replayed.services_completed == original.services_completed
    assert replayed.swaps_completed == original.swaps_completed


def test_snapshot_writer_records_service_ratio_and_summary(tmp_path) -> None:
    snapshot_path = tmp_path / "snapshots.jsonl.zst"
    sim = GillespieQBPSimulator(build_four_node_counterexample(), seed=19)

    with SnapshotWriter(snapshot_path) as snapshot_writer:
        result = sim.run(
            until_time=2.0,
            max_events=300,
            sample_every=1,
            snapshot_writer=snapshot_writer,
        )

    with SnapshotReader(snapshot_path) as snapshot_reader:
        snapshots = list(snapshot_reader)

    assert snapshots
    assert snapshots[-1].event_index == result.events_processed
    assert snapshots[-1].service_ratio == (
        0.0 if result.demand_arrivals == 0 else result.services_completed / result.demand_arrivals
    )

    summary = summarize_snapshots(snapshots)
    assert summary.final_service_ratio == snapshots[-1].service_ratio
    assert summary.demand_arrivals == result.demand_arrivals


def test_altair_snapshot_plots_render_to_png(tmp_path) -> None:
    snapshot_path = tmp_path / "snapshots.jsonl.zst"
    sim = GillespieQBPSimulator(build_four_node_counterexample(), seed=23)

    with SnapshotWriter(snapshot_path) as snapshot_writer:
        sim.run(
            until_time=2.0,
            max_events=300,
            sample_every=5,
            snapshot_writer=snapshot_writer,
            progress=False,
        )

    with SnapshotReader(snapshot_path) as snapshot_reader:
        snapshots = list(snapshot_reader)

    single_plot_path = tmp_path / "service_ratio.png"
    series_plot_path = tmp_path / "service_ratio_series.png"
    plot_snapshot_metric(snapshots, "service_ratio", single_plot_path)
    plot_snapshot_metric_series(
        [("n=4", snapshots), ("n=8", snapshots)],
        "service_ratio",
        series_plot_path,
    )

    assert single_plot_path.exists()
    assert single_plot_path.stat().st_size > 0
    assert series_plot_path.exists()
    assert series_plot_path.stat().st_size > 0


def test_limited_info_service_ratio_experiment_writes_events_metadata_and_plot(tmp_path) -> None:
    runs = run_limited_info_service_ratio_experiment(
        n_nodes=4,
        limited_policies=[(1, 1)],
        output_dir=tmp_path / "limited-info",
        until_time=200.0,
        max_events=100_000,
        sample_every=100,
        seed_base=3,
        progress=False,
    )
    plot_path = tmp_path / "limited-info-service-ratio.png"
    plot_limited_info_service_ratio_runs(runs, plot_path)

    assert [run.policy_label for run in runs] == ["full info", "limited k=1, m=1"]
    assert all(run.snapshots for run in runs)
    assert all(run.trace_path.exists() for run in runs)
    assert all(run.trace_path.suffix == ".vortex" for run in runs)
    assert all(run.metadata_path.exists() for run in runs)
    assert all(run.simulation_config_path.exists() for run in runs)
    assert all(0.0 <= run.summary.final_service_ratio <= 1.0 for run in runs)
    assert plot_path.exists()
    assert plot_path.stat().st_size > 0


def test_cycle_consumption_edge_fraction_scales_with_graph_size() -> None:
    assert np.isclose(_cycle_consumption_edge_fraction(4, None), 2.0 / 6.0)
    assert np.isclose(_cycle_consumption_edge_fraction(8, None), 4.0 / 28.0)
    assert np.isclose(_cycle_consumption_edge_fraction(64, None), 32.0 / 2016.0)
    assert _cycle_consumption_edge_fraction(16, 0.125) == 0.125


def test_capacity_headroom_scales_capacity_rates_not_demand_rates() -> None:
    input_config = SimulationInputConfig(
        generation_rates=[
            [0.0, 2.0, 3.0],
            [2.0, 0.0, 5.0],
            [3.0, 5.0, 0.0],
        ],
        consumption_rates=[
            [0.0, 7.0, 11.0],
            [7.0, 0.0, 13.0],
            [11.0, 13.0, 0.0],
        ],
        swap_rates=[17.0, 19.0, 23.0],
    )

    scaled = _apply_capacity_headroom(input_config, 1.05)
    runtime = scaled.to_runtime_config()

    assert scaled.capacity_headroom == 1.05
    assert np.allclose(runtime.generation_rates, np.asarray(input_config.generation_rates) * 1.05)
    assert np.allclose(runtime.demand_rates, np.asarray(input_config.consumption_rates))
    assert np.allclose(runtime.service_rates, np.asarray(input_config.consumption_rates) * 1.05)
    assert np.allclose(runtime.swap_rates, np.asarray(input_config.swap_rates) * 1.05)


def test_simulation_input_config_defaults_capacity_headroom_to_one_percent() -> None:
    input_config = SimulationInputConfig(
        generation_rates=[[0.0, 2.0], [2.0, 0.0]],
        consumption_rates=[[0.0, 3.0], [3.0, 0.0]],
        swap_rates=[5.0, 7.0],
    )
    runtime = input_config.to_runtime_config()

    assert input_config.capacity_headroom == 1.01
    assert input_config.instant_service_fulfillment is False
    assert runtime.instant_service_fulfillment is False
    assert np.allclose(runtime.generation_rates, np.asarray(input_config.generation_rates) * 1.01)
    assert np.allclose(runtime.demand_rates, np.asarray(input_config.consumption_rates))
    assert np.allclose(runtime.service_rates, np.asarray(input_config.consumption_rates) * 1.01)
    assert np.allclose(runtime.swap_rates, np.asarray(input_config.swap_rates) * 1.01)


def test_cli_instant_service_fulfillment_flag_enables_runtime_config() -> None:
    parser = _build_parser()
    args = parser.parse_args(["example", "--instant-service-fulfillment"])
    config = _apply_instant_service_fulfillment_arg(build_four_node_counterexample(), args)

    assert config.instant_service_fulfillment is True


def test_simulation_input_config_loads_json_and_infers_runtime_config(tmp_path) -> None:
    config_path = tmp_path / "config.json"
    config_path.write_text(
        json.dumps(
            {
                "generation_rates": [
                    [0.0, 1.0, 1.0, 0.0],
                    [1.0, 0.0, 0.0, 1.0],
                    [1.0, 0.0, 0.0, 1.0],
                    [0.0, 1.0, 1.0, 0.0],
                ],
                "consumption_rates": [
                    [0.0, 0.0, 0.0, 2.0],
                    [0.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0],
                    [2.0, 0.0, 0.0, 0.0],
                ],
                "swap_rates": [0.0, 1.0, 1.0, 0.0],
                "capacity_headroom": 1.25,
                "instant_service_fulfillment": True,
                "virtual_swap_policy": {
                    "mode": "power_of_k_memory",
                    "k": 2,
                    "memory": 0,
                },
            }
        )
    )

    input_config = SimulationInputConfig.from_json_file(config_path)
    runtime = input_config.to_runtime_config()

    assert input_config.num_nodes == 4
    assert input_config.capacity_headroom == 1.25
    assert input_config.instant_service_fulfillment is True
    assert runtime.generation_rates[0, 1] == 1.25
    assert runtime.generation_rates[0, 3] == 0.0
    assert runtime.demand_rates[0, 3] == 2.0
    assert runtime.service_rates[0, 1] == 0.0
    assert runtime.service_rates[0, 3] == 2.5
    assert runtime.swap_rates[1] == 1.25
    assert runtime.virtual_swap_policy.mode == "power_of_k_memory"
    assert runtime.virtual_swap_policy.k == 2
    assert runtime.virtual_swap_policy.memory == 0
    assert runtime.instant_service_fulfillment is True


def test_simulation_input_config_rejects_asymmetric_rates() -> None:
    try:
        SimulationInputConfig(
            generation_rates=[[0.0, 1.0], [0.0, 0.0]],
            consumption_rates=[[0.0, 0.0], [0.0, 0.0]],
            swap_rates=[0.0, 1.0],
        )
    except Exception as exc:
        assert "symmetric" in str(exc)
    else:
        raise AssertionError("Expected asymmetric generation rates to fail validation.")


def test_linear_module_imports_without_visualization_helpers() -> None:
    linear_path = Path(__file__).resolve().parents[1] / "linear.py"
    spec = importlib.util.spec_from_file_location("linear_module_under_test", linear_path)
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)

    assert hasattr(module, "LinearSpec")
    assert hasattr(module, "build_simulation_input_config")
    assert hasattr(module, "build_lp_solution_simulation_input_config")


def test_linear_module_emits_shared_simulation_config_json(tmp_path) -> None:
    linear_path = Path(__file__).resolve().parents[1] / "linear.py"
    spec = importlib.util.spec_from_file_location("linear_module_under_test_emit", linear_path)
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)

    payload = module.build_simulation_input_config(
        generation_graph=np.array([[0.0, 1.0], [1.0, 0.0]], dtype=float),
        consumption_graph=np.array([[0.0, 2.0], [2.0, 0.0]], dtype=float),
        swap_rates=[1.5, 1.5],
    )
    output_path = tmp_path / "sim_config.json"
    module.emit_simulation_input_json(
        payload=payload,
        json_output_path=str(output_path),
        json_pretty=True,
    )

    loaded = SimulationInputConfig.from_json_file(output_path)
    assert loaded.generation_rates[0][1] == 1.0
    assert loaded.consumption_rates[0][1] == 2.0
    assert loaded.swap_rates == [1.5, 1.5]


def test_linear_module_enforces_per_node_swap_caps() -> None:
    linear_path = Path(__file__).resolve().parents[1] / "linear.py"
    spec = importlib.util.spec_from_file_location("linear_module_under_test_caps", linear_path)
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)

    generation_graph = module.create_chain_adjacency_matrix(3, edge_weight=1.0)
    consumption_graph = np.array(
        [
            [0.0, 0.0, 1.0],
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
        ],
        dtype=float,
    )

    infeasible = module.LinearSpec(3)
    infeasible.add_generate_constraints(generation_graph)
    infeasible.add_consume_constraints(consumption_graph)
    infeasible.add_swap_capacity_constraints([0.0, 0.0, 0.0])
    infeasible_result = infeasible.solve()
    assert infeasible_result.status != 0

    feasible = module.LinearSpec(3)
    feasible.add_generate_constraints(generation_graph)
    feasible.add_consume_constraints(consumption_graph)
    feasible.add_swap_capacity_constraints([0.0, 1.0, 0.0])
    feasible_result = feasible.solve()
    assert feasible_result.status == 0
    assert feasible.total_swap_undirected(feasible_result) > 0.0


def test_lp_cycle_bp_service_ratio_converges_near_one(tmp_path) -> None:
    linear_path = Path(__file__).resolve().parents[1] / "linear.py"
    spec = importlib.util.spec_from_file_location("linear_module_under_test_cycle", linear_path)
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)

    config_path = tmp_path / "cycle5_lp_config.json"
    module.single_run_topology(
        topology="cycle",
        num_nodes=5,
        seed=5,
        output_mode="simulation-config",
        simulation_config_output_path=str(config_path),
    )

    runtime_config = SimulationInputConfig.from_json_file(config_path).to_runtime_config()

    short_run = GillespieQBPSimulator(runtime_config, seed=1).run(
        until_time=1_000.0,
        max_events=1_000_000,
        sample_every=0,
        progress=False,
    )
    short_ratio = short_run.services_completed / short_run.demand_arrivals

    long_run = GillespieQBPSimulator(runtime_config, seed=1).run(
        until_time=10_000.0,
        max_events=6_000_000,
        sample_every=0,
        progress=False,
    )
    long_ratio = long_run.services_completed / long_run.demand_arrivals

    assert short_ratio > 0.90
    assert long_ratio > 0.97
    assert long_ratio > short_ratio


def test_single_run_topology_returns_solution_config(tmp_path) -> None:
    linear_path = Path(__file__).resolve().parents[1] / "linear.py"
    spec = importlib.util.spec_from_file_location("linear_module_under_test_return", linear_path)
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)

    config_path = tmp_path / "cycle4_lp_config.json"
    returned = module.single_run_topology(
        topology="cycle",
        num_nodes=4,
        seed=4,
        output_mode="simulation-config",
        simulation_config_output_path=str(config_path),
    )

    assert returned is not None
    loaded = SimulationInputConfig.from_json_file(config_path)
    assert returned.model_dump() == loaded.model_dump()


def test_should_use_progress_respects_tty_and_term(monkeypatch) -> None:
    class DummyStream:
        def __init__(self, is_tty: bool):
            self._is_tty = is_tty

        def isatty(self) -> bool:
            return self._is_tty

    monkeypatch.delenv("CI", raising=False)
    monkeypatch.setenv("TERM", "xterm-256color")
    assert should_use_progress(DummyStream(True)) is True
    assert should_use_progress(DummyStream(False)) is False

    monkeypatch.setenv("TERM", "dumb")
    assert should_use_progress(DummyStream(True)) is False
