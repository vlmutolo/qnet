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
from qbp_sim.cli import _apply_instant_service_fulfillment_arg, _build_parser, main
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


def test_cli_help_is_available(capsys) -> None:
    parser = _build_parser()
    try:
        parser.parse_args(["--help"])
    except SystemExit as exc:
        assert exc.code == 0
    else:
        raise AssertionError("Expected argparse help to exit.")

    captured = capsys.readouterr()
    assert "Continuous-time Gillespie simulation" in captured.out


def test_cli_matrix_dry_run_expands_cases(tmp_path, capsys) -> None:
    config_path = tmp_path / "matrix.json"
    config_path.write_text(
        json.dumps(
            {
                "topologies": ["cycle"],
                "graph_sizes": [3],
                "policies": [
                    {"mode": "bp"},
                    {"mode": "limited_info_bp", "k": 1, "memory": 1},
                ],
                "until_time": 1.0,
                "sample_every": 10,
            }
        ),
        encoding="utf-8",
    )

    main(["matrix", "--config", str(config_path), "--output-dir", str(tmp_path / "runs"), "--dry-run"])

    payload = json.loads(capsys.readouterr().out)
    assert payload["case_count"] == 2
    assert [case["policy_mode"] for case in payload["cases"]] == ["bp", "limited_info_bp"]
    assert not (tmp_path / "runs").exists()


def test_cli_matrix_tiny_run_writes_artifacts(tmp_path) -> None:
    config_path = tmp_path / "matrix.json"
    output_dir = tmp_path / "runs"
    config_path.write_text(
        json.dumps(
            {
                "topologies": ["cycle"],
                "graph_sizes": [3],
                "consumption_edge_fractions": [None],
                "headrooms": [1.01],
                "policies": [{"mode": "bp"}],
                "gen_scales": [10.0],
                "cons_scales": [0.1],
                "swap_rates": [20.0],
                "until_time": 1.0,
                "max_events": 200,
                "sample_every": 10,
                "trace_time_mode": "none",
            }
        ),
        encoding="utf-8",
    )

    main(["matrix", "--config", str(config_path), "--output-dir", str(output_dir)])

    summary_path = output_dir / "summary.csv"
    case_dirs = [path for path in output_dir.iterdir() if path.is_dir()]
    assert summary_path.exists()
    assert len(case_dirs) == 1
    assert (case_dirs[0] / "simulation_config.json").exists()
    assert (case_dirs[0] / "events.vortex").exists()
    assert (case_dirs[0] / "run_metadata.json").exists()
