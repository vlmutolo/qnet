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
