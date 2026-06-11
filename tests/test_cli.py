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


def test_cli_instant_service_fulfillment_flag_enables_runtime_config() -> None:
    parser = _build_parser()
    args = parser.parse_args(["example", "--instant-service-fulfillment", "--instant-swap-fulfillment"])
    config = _apply_instant_service_fulfillment_arg(build_four_node_counterexample(), args)

    assert config.instant_service_fulfillment is True
    assert config.instant_swap_fulfillment is True


def test_cli_trace_time_mode_flag_is_available_for_trace_writers() -> None:
    parser = _build_parser()
    args = parser.parse_args(["example", "--trace", "events.vortex", "--trace-time-mode", "none"])

    assert args.trace_time_mode == "none"
