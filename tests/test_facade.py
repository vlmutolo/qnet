from __future__ import annotations

import json

from qbp_sim import (
    RunOptions,
    TraceFloatPrecision,
    TraceFormat,
    TraceTimeMode,
    VirtualSwapPolicyMode,
    build_four_node_example_config,
    load_simulation_config,
    replay_trace,
    run_simulation,
)
from qbp_sim.cli import _build_parser


def test_run_simulation_from_input_config() -> None:
    output = run_simulation(
        build_four_node_example_config(),
        RunOptions(until_time=1.0, max_events=100, sample_every=0, seed=1, progress=False),
    )

    assert output.result.final_time <= 1.0
    assert output.result.events_processed <= 100
    assert 0.0 <= output.service_ratio <= 1.0


def test_run_simulation_from_json_path_and_replay_trace(tmp_path) -> None:
    config = build_four_node_example_config()
    config_path = tmp_path / "config.json"
    trace_path = tmp_path / "events.vortex"
    config_path.write_text(config.model_dump_json(indent=2), encoding="utf-8")

    run_output = run_simulation(
        config_path,
        RunOptions(
            until_time=2.0,
            max_events=200,
            sample_every=25,
            seed=2,
            trace_path=trace_path,
            trace_format=TraceFormat.VORTEX,
            trace_float_precision=TraceFloatPrecision.FLOAT32,
            trace_time_mode=TraceTimeMode.FULL,
            progress=False,
        ),
    )
    replay_output = replay_trace(
        trace_path,
        config=load_simulation_config(config_path),
        final_time=run_output.result.final_time,
        sample_every=25,
    )

    assert run_output.trace_path == trace_path
    assert replay_output.result.events_processed == run_output.result.events_processed
    assert replay_output.result.services_completed == run_output.result.services_completed
    assert replay_output.result.demand_arrivals == run_output.result.demand_arrivals


def test_run_simulation_can_write_explicit_parquet_trace(tmp_path) -> None:
    trace_path = tmp_path / "events.parquet"
    run_output = run_simulation(
        build_four_node_example_config(),
        RunOptions(
            until_time=1.0,
            max_events=100,
            sample_every=0,
            seed=3,
            trace_path=trace_path,
            trace_format=TraceFormat.PARQUET,
            progress=False,
        ),
    )

    replay_output = replay_trace(
        trace_path,
        config=build_four_node_example_config(),
        final_time=run_output.result.final_time,
        sample_every=0,
    )

    assert run_output.trace_path == trace_path
    assert replay_output.result.events_processed == run_output.result.events_processed
    assert replay_output.result.services_completed == run_output.result.services_completed


def test_policy_aliases_validate_through_json_and_cli(tmp_path) -> None:
    config_path = tmp_path / "config.json"
    payload = build_four_node_example_config().model_dump()
    payload["virtual_swap_policy"] = {"mode": "max-min"}
    config_path.write_text(json.dumps(payload), encoding="utf-8")

    config = load_simulation_config(config_path)
    parser = _build_parser()
    args = parser.parse_args(["example", "--virtual-swap-policy", "max-min"])

    assert config.virtual_swap_policy.mode == VirtualSwapPolicyMode.MAX_MIN
    assert args.virtual_swap_policy == "max-min"
