from __future__ import annotations

import json
from copy import deepcopy
from dataclasses import replace
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from qbp_sim.analysis import plot_snapshot_metric, plot_snapshot_metric_series, summarize_snapshots
from qbp_sim.cli import _apply_instant_service_fulfillment_arg, _build_parser
from qbp_sim.config import SimulationInputConfig
from qbp_sim.events import QBPEvent
from qbp_sim.examples import build_four_node_counterexample
from qbp_sim.experiments import (
    _apply_capacity_headroom,
    _cycle_consumption_edge_fraction,
    plot_limited_info_service_ratio_runs,
    run_limited_info_service_ratio_experiment,
)
from qbp_sim.lp import linear as linear_module
from qbp_sim.progress import should_use_progress
from qbp_sim.snapshots import SnapshotReader, SnapshotWriter
import qbp_sim.core.producer as producer_module
from qbp_sim.simulator import GillespieQBPConfig, GillespieQBPSimulator, VirtualSwapPolicy, replay_event_stream
from qbp_sim.trace import EventTraceReader, EventTraceWriter, open_event_trace_reader, open_event_trace_writer
from tests.support import (
    _assert_state_invariants,
    _collect_event_records,
    _cycle_runtime_config,
    _limited_policy,
    _pending_service_matrix,
    _single_edge_generation_config,
    _swap_index,
    gated_test,
)


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

    monkeypatch.setattr(producer_module, "_best_virtual_swap_for_node", fail_global_scan)
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
