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
