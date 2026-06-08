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
