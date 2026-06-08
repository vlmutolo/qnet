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


def test_linear_module_imports_without_visualization_helpers() -> None:
    module = linear_module

    assert hasattr(module, "LinearSpec")
    assert hasattr(module, "build_simulation_input_config")
    assert hasattr(module, "build_lp_solution_simulation_input_config")


def test_linear_module_emits_shared_simulation_config_json(tmp_path) -> None:
    module = linear_module

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
    module = linear_module

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
    module = linear_module

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
    module = linear_module

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
