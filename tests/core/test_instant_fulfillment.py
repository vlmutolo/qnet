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
    swap_idx = _swap_index(probe, 0, 1, 2)

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


def test_instant_service_after_virtual_service_fulfills_existing_inventory(tmp_path) -> None:
    generation_rates = np.zeros((2, 2), dtype=np.float64)
    demand_rates = np.zeros((2, 2), dtype=np.float64)
    service_rates = np.array([[0.0, 1.0], [1.0, 0.0]], dtype=np.float64)
    swap_rates = np.zeros(2, dtype=np.float64)
    config = GillespieQBPConfig(
        generation_rates=generation_rates,
        demand_rates=demand_rates,
        swap_rates=swap_rates,
        service_rates=service_rates,
        instant_service_fulfillment=True,
    )
    initial_q = np.array([[0, 1], [1, 0]], dtype=np.int64)
    initial_d = np.array([[0, 1], [1, 0]], dtype=np.int64)
    trace_path = tmp_path / "instant-virtual-service.jsonl.zst"
    sim = GillespieQBPSimulator(config, seed=5, initial_q=initial_q, initial_d=initial_d)

    with EventTraceWriter(trace_path) as trace_writer:
        sampled = sim.step(trace_writer=trace_writer)

    with EventTraceReader(trace_path) as trace_reader:
        events = list(trace_reader)

    assert sampled is not None
    assert sampled.event_type == "virtual_service"
    assert [event.event_type for event in events] == ["virtual_service", "physical_service"]
    assert events[1].time == events[0].time
    assert events[1].dt == 0.0
    assert events[1].total_rate == 0.0
    assert events[1].event_rate == 0.0
    assert sim.virtual_service_requests == 1
    assert sim.services_completed == 1
    assert sim.events_processed == 2
    assert sim.state.q[0, 1] == 0
    assert sim.state.h_r[0, 1] == 0
    assert sim.total_inventory == 0
    assert sim.total_backlog == 0


def test_instant_swap_after_generation_uses_frontier_edge_and_largest_pending_swap(tmp_path) -> None:
    generation_rates = np.zeros((4, 4), dtype=np.float64)
    generation_rates[0, 1] = generation_rates[1, 0] = 1.0
    demand_rates = np.zeros((4, 4), dtype=np.float64)
    service_rates = np.zeros((4, 4), dtype=np.float64)
    swap_rates = np.zeros(4, dtype=np.float64)
    config = GillespieQBPConfig(
        generation_rates=generation_rates,
        demand_rates=demand_rates,
        swap_rates=swap_rates,
        service_rates=service_rates,
        instant_swap_fulfillment=True,
    )
    probe = GillespieQBPSimulator(config, seed=0)
    lower_deficit_idx = _swap_index(probe, 0, 1, 2)
    higher_deficit_idx = _swap_index(probe, 1, 0, 3)
    initial_q = np.zeros((4, 4), dtype=np.int64)
    initial_q[0, 2] = initial_q[2, 0] = 1
    initial_q[1, 3] = initial_q[3, 1] = 1
    initial_h_mu = np.zeros_like(probe.state.h_mu)
    initial_h_mu[lower_deficit_idx] = 1
    initial_h_mu[higher_deficit_idx] = 3
    trace_path = tmp_path / "instant-generation-swap.jsonl.zst"
    sim = GillespieQBPSimulator(
        config,
        seed=5,
        initial_q=initial_q,
        initial_h_mu=initial_h_mu,
    )

    with EventTraceWriter(trace_path) as trace_writer:
        sampled = sim.step(trace_writer=trace_writer)

    with EventTraceReader(trace_path) as trace_reader:
        events = list(trace_reader)

    assert sampled is not None
    assert sampled.event_type == "pair_generation"
    assert [event.event_type for event in events] == ["pair_generation", "physical_swap"]
    assert events[1].swap_idx == higher_deficit_idx
    assert events[1].dt == 0.0
    assert events[1].total_rate == 0.0
    assert events[1].event_rate == 0.0
    assert sim.swaps_completed == 1
    assert sim.events_processed == 2
    assert sim.state.q[0, 1] == 0
    assert sim.state.q[1, 3] == 0
    assert sim.state.q[0, 2] == 1
    assert sim.state.q[0, 3] == 1
    assert sim.state.h_mu[higher_deficit_idx] == 2
    assert sim.state.h_mu[lower_deficit_idx] == 1


def test_instant_frontier_prioritizes_service_over_swap(tmp_path) -> None:
    generation_rates = np.zeros((3, 3), dtype=np.float64)
    generation_rates[0, 1] = generation_rates[1, 0] = 1.0
    demand_rates = np.zeros((3, 3), dtype=np.float64)
    service_rates = np.zeros((3, 3), dtype=np.float64)
    swap_rates = np.zeros(3, dtype=np.float64)
    config = GillespieQBPConfig(
        generation_rates=generation_rates,
        demand_rates=demand_rates,
        swap_rates=swap_rates,
        service_rates=service_rates,
        instant_service_fulfillment=True,
        instant_swap_fulfillment=True,
    )
    probe = GillespieQBPSimulator(config, seed=0)
    swap_idx = _swap_index(probe, 0, 1, 2)
    initial_q = np.zeros((3, 3), dtype=np.int64)
    initial_q[0, 2] = initial_q[2, 0] = 1
    initial_h_mu = np.zeros_like(probe.state.h_mu)
    initial_h_mu[swap_idx] = 1
    initial_h_r = _pending_service_matrix(3, 0, 1)
    trace_path = tmp_path / "instant-service-priority.jsonl.zst"
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
    assert sampled.event_type == "pair_generation"
    assert [event.event_type for event in events] == ["pair_generation", "physical_service"]
    assert sim.services_completed == 1
    assert sim.swaps_completed == 0
    assert sim.state.q[0, 1] == 0
    assert sim.state.q[0, 2] == 1
    assert sim.state.h_mu[swap_idx] == 1


def test_instant_swap_after_virtual_swap_cascades_to_output_service(tmp_path) -> None:
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
        instant_swap_fulfillment=True,
    )
    probe = GillespieQBPSimulator(config, seed=0)
    swap_idx = _swap_index(probe, 0, 1, 2)
    initial_q = np.zeros((3, 3), dtype=np.int64)
    initial_q[0, 1] = initial_q[1, 0] = 1
    initial_q[0, 2] = initial_q[2, 0] = 1
    initial_alpha = np.zeros((3, 3), dtype=np.int64)
    initial_alpha[1, 2] = initial_alpha[2, 1] = 5
    initial_h_r = _pending_service_matrix(3, 1, 2)
    trace_path = tmp_path / "instant-virtual-swap-cascade.jsonl.zst"
    sim = GillespieQBPSimulator(
        config,
        seed=5,
        initial_q=initial_q,
        initial_alpha=initial_alpha,
        initial_h_r=initial_h_r,
    )

    with EventTraceWriter(trace_path) as trace_writer:
        sampled = sim.step(trace_writer=trace_writer)

    with EventTraceReader(trace_path) as trace_reader:
        events = list(trace_reader)

    assert sampled is not None
    assert sampled.event_type == "virtual_swap"
    assert sampled.swap_idx == swap_idx
    assert [event.event_type for event in events] == ["virtual_swap", "physical_swap", "physical_service"]
    assert events[1].swap_idx == swap_idx
    assert events[1].dt == 0.0
    assert events[2].dt == 0.0
    assert events[2].x == 1
    assert events[2].y == 2
    assert sim.virtual_swap_requests == 1
    assert sim.swaps_completed == 1
    assert sim.services_completed == 1
    assert sim.events_processed == 3
    assert sim.state.q[1, 2] == 0
    assert sim.state.h_r[1, 2] == 0
    assert sim.state.h_mu[swap_idx] == 0
    assert sim.total_inventory == 0


def test_instant_swap_removes_physical_swap_from_stochastic_sampler() -> None:
    generation_rates = np.zeros((3, 3), dtype=np.float64)
    demand_rates = np.zeros((3, 3), dtype=np.float64)
    service_rates = np.zeros((3, 3), dtype=np.float64)
    swap_rates = np.array([1.0, 0.0, 0.0], dtype=np.float64)
    config = GillespieQBPConfig(
        generation_rates=generation_rates,
        demand_rates=demand_rates,
        swap_rates=swap_rates,
        service_rates=service_rates,
        instant_swap_fulfillment=True,
    )
    probe = GillespieQBPSimulator(config, seed=0)
    swap_idx = _swap_index(probe, 0, 1, 2)
    initial_q = np.zeros((3, 3), dtype=np.int64)
    initial_q[0, 1] = initial_q[1, 0] = 1
    initial_q[0, 2] = initial_q[2, 0] = 1
    initial_h_mu = np.zeros_like(probe.state.h_mu)
    initial_h_mu[swap_idx] = 1
    sim = GillespieQBPSimulator(
        config,
        seed=5,
        initial_q=initial_q,
        initial_h_mu=initial_h_mu,
    )

    assert sim.step() is None
    assert sim.swaps_completed == 0
    assert sim.events_processed == 0


def test_instant_swap_cascade_replay_is_literal_without_double_fulfillment(tmp_path) -> None:
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
        instant_swap_fulfillment=True,
    )
    probe = GillespieQBPSimulator(config, seed=0)
    swap_idx = _swap_index(probe, 0, 1, 2)
    initial_q = np.zeros((3, 3), dtype=np.int64)
    initial_q[0, 1] = initial_q[1, 0] = 1
    initial_q[0, 2] = initial_q[2, 0] = 1
    initial_alpha = np.zeros((3, 3), dtype=np.int64)
    initial_alpha[1, 2] = initial_alpha[2, 1] = 5
    initial_h_r = _pending_service_matrix(3, 1, 2)
    trace_path = tmp_path / "instant-cascade-replay.jsonl.zst"
    sim = GillespieQBPSimulator(
        config,
        seed=5,
        initial_q=initial_q,
        initial_alpha=initial_alpha,
        initial_h_r=initial_h_r,
    )

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
            initial_q=initial_q,
            initial_alpha=initial_alpha,
            initial_h_r=initial_h_r,
        )

    assert original.events_processed == 3
    assert replayed.events_processed == original.events_processed
    assert replayed.final_time == original.final_time
    assert replayed.virtual_swap_requests == original.virtual_swap_requests
    assert replayed.swaps_completed == original.swaps_completed
    assert replayed.services_completed == original.services_completed
    assert replayed.total_inventory == original.total_inventory
    assert replayed.total_backlog == original.total_backlog
    assert replayed.total_scarcity == original.total_scarcity
    assert replayed.total_backlog == 0
    assert replayed.swaps_completed == 1
    assert replayed.services_completed == 1
    assert swap_idx >= 0


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
