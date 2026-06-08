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


def test_simulation_input_config_defaults_capacity_headroom_to_one_percent() -> None:
    input_config = SimulationInputConfig(
        generation_rates=[[0.0, 2.0], [2.0, 0.0]],
        consumption_rates=[[0.0, 3.0], [3.0, 0.0]],
        swap_rates=[5.0, 7.0],
    )
    runtime = input_config.to_runtime_config()

    assert input_config.capacity_headroom == 1.01
    assert input_config.instant_service_fulfillment is False
    assert input_config.instant_swap_fulfillment is False
    assert runtime.instant_service_fulfillment is False
    assert runtime.instant_swap_fulfillment is False
    assert np.allclose(runtime.generation_rates, np.asarray(input_config.generation_rates) * 1.01)
    assert np.allclose(runtime.demand_rates, np.asarray(input_config.consumption_rates))
    assert np.allclose(runtime.service_rates, np.asarray(input_config.consumption_rates) * 1.01)
    assert np.allclose(runtime.swap_rates, np.asarray(input_config.swap_rates) * 1.01)


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
                "instant_swap_fulfillment": True,
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
    assert input_config.instant_swap_fulfillment is True
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
    assert runtime.instant_swap_fulfillment is True


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
