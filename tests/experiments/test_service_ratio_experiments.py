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
    plot_limited_info_service_ratio_runs,
    run_limited_info_service_ratio_experiment,
)
from qbp_sim.experiments.common import (
    _apply_capacity_headroom,
    _cycle_consumption_edge_fraction,
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
