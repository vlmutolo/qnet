from __future__ import annotations

import json
import importlib.util
from pathlib import Path

import numpy as np

from qbp_sim.config import SimulationInputConfig
from qbp_sim.examples import build_four_node_counterexample
from qbp_sim.analysis import summarize_snapshots
from qbp_sim.progress import should_use_progress
from qbp_sim.snapshots import SnapshotReader, SnapshotWriter
from qbp_sim.simulator import GillespieQBPConfig, GillespieQBPSimulator, replay_event_stream
from qbp_sim.trace import EventTraceReader, EventTraceWriter


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


def test_snapshot_writer_records_service_ratio_and_summary(tmp_path) -> None:
    snapshot_path = tmp_path / "snapshots.jsonl.zst"
    sim = GillespieQBPSimulator(build_four_node_counterexample(), seed=19)

    with SnapshotWriter(snapshot_path) as snapshot_writer:
        result = sim.run(
            until_time=2.0,
            max_events=300,
            sample_every=1,
            snapshot_writer=snapshot_writer,
        )

    with SnapshotReader(snapshot_path) as snapshot_reader:
        snapshots = list(snapshot_reader)

    assert snapshots
    assert snapshots[-1].event_index == result.events_processed
    assert snapshots[-1].service_ratio == (
        0.0 if result.demand_arrivals == 0 else result.services_completed / result.demand_arrivals
    )

    summary = summarize_snapshots(snapshots)
    assert summary.final_service_ratio == snapshots[-1].service_ratio
    assert summary.demand_arrivals == result.demand_arrivals


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
            }
        )
    )

    input_config = SimulationInputConfig.from_json_file(config_path)
    runtime = input_config.to_runtime_config()

    assert input_config.num_nodes == 4
    assert runtime.generation_rates[0, 1] == 1.0
    assert runtime.generation_rates[0, 3] == 0.0
    assert runtime.demand_rates[0, 3] == 2.0
    assert runtime.service_rates[0, 1] == 0.0
    assert runtime.service_rates[0, 3] == 2.0


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


def test_linear_module_imports_without_visualization_helpers() -> None:
    linear_path = Path(__file__).resolve().parents[1] / "linear.py"
    spec = importlib.util.spec_from_file_location("linear_module_under_test", linear_path)
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)

    assert hasattr(module, "LinearSpec")
    assert hasattr(module, "build_simulation_input_config")
    assert hasattr(module, "build_lp_solution_simulation_input_config")


def test_linear_module_emits_shared_simulation_config_json(tmp_path) -> None:
    linear_path = Path(__file__).resolve().parents[1] / "linear.py"
    spec = importlib.util.spec_from_file_location("linear_module_under_test_emit", linear_path)
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)

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


def test_lp_cycle_bp_service_ratio_converges_near_one(tmp_path) -> None:
    linear_path = Path(__file__).resolve().parents[1] / "linear.py"
    spec = importlib.util.spec_from_file_location("linear_module_under_test_cycle", linear_path)
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)

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


def test_should_use_progress_respects_tty_and_term(monkeypatch) -> None:
    class DummyStream:
        def __init__(self, is_tty: bool):
            self._is_tty = is_tty

        def isatty(self) -> bool:
            return self._is_tty

    monkeypatch.delenv("CI", raising=False)
    monkeypatch.setenv("TERM", "xterm-256color")
    assert should_use_progress(DummyStream(True)) is True
    assert should_use_progress(DummyStream(False)) is False

    monkeypatch.setenv("TERM", "dumb")
    assert should_use_progress(DummyStream(True)) is False
