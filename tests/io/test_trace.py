from __future__ import annotations

import json
from copy import deepcopy
from dataclasses import replace
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import vortex as vx
import zstandard as zstd

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


def test_jsonl_timeless_trace_omits_timing_payload(tmp_path) -> None:
    trace_path = tmp_path / "events.jsonl.zst"
    event = QBPEvent(
        event_index=1,
        time=12.5,
        dt=12.5,
        total_rate=4.0,
        event_type="demand_arrival",
        event_rate=1.5,
        x=0,
        y=1,
        backlog_total=1,
        inventory_total=0,
        scarcity_total=0,
    )

    with EventTraceWriter(trace_path, time_mode="none") as trace_writer:
        trace_writer.write(event)

    with trace_path.open("rb") as raw_handle:
        payload = zstd.ZstdDecompressor().stream_reader(raw_handle).read().decode("utf-8")
    record = json.loads(payload.strip())

    assert "time" not in record
    assert "dt" not in record
    assert "total_rate" not in record
    assert "event_rate" not in record
    assert record["event_type"] == "demand_arrival"

    with EventTraceReader(trace_path) as trace_reader:
        [read_event] = list(trace_reader)
    assert read_event.time == 0.0
    assert read_event.dt == 0.0
    assert read_event.total_rate == 0.0
    assert read_event.event_rate == 0.0


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


def test_parquet_timeless_trace_omits_time_columns_and_replays_state(tmp_path) -> None:
    trace_path = tmp_path / "events.parquet"
    config = build_four_node_counterexample()
    sim = GillespieQBPSimulator(config, seed=133)

    with open_event_trace_writer(trace_path, time_mode="none") as trace_writer:
        original = sim.run(until_time=2.0, max_events=400, sample_every=25, trace_writer=trace_writer)

    parquet_columns = set(pq.ParquetFile(trace_path).schema_arrow.names)
    assert {"time", "total_rate", "event_rate"}.isdisjoint(parquet_columns)
    assert "event_index" in parquet_columns
    assert "event_type" in parquet_columns

    with open_event_trace_reader(trace_path) as trace_reader:
        events = list(trace_reader)

    assert len(events) == original.events_processed
    assert all(event.time == 0.0 for event in events)
    assert all(event.dt == 0.0 for event in events)
    assert all(event.total_rate == 0.0 for event in events)
    assert all(event.event_rate == 0.0 for event in events)

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


def test_vortex_timeless_trace_omits_time_columns_and_replays_state(tmp_path) -> None:
    trace_path = tmp_path / "events.vortex"
    config = build_four_node_counterexample()
    sim = GillespieQBPSimulator(config, seed=134)

    with open_event_trace_writer(trace_path, time_mode="none") as trace_writer:
        original = sim.run(until_time=2.0, max_events=400, sample_every=25, trace_writer=trace_writer)

    batches = list(vx.open(str(trace_path)).to_arrow(batch_size=65_536))
    table = pa.Table.from_batches(batches)
    assert {"time", "total_rate", "event_rate"}.isdisjoint(table.column_names)
    assert "event_index" in table.column_names
    assert "event_type" in table.column_names

    with open_event_trace_reader(trace_path) as trace_reader:
        events = list(trace_reader)

    assert len(events) == original.events_processed
    assert all(event.time == 0.0 for event in events)
    assert all(event.dt == 0.0 for event in events)

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
