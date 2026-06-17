from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

from qbp_sim.core.simulator import GillespieQBPSimulator
from qbp_sim.experiments.common import (
    _MemorySnapshotWriter,
    _apply_capacity_headroom,
    _apply_instant_service_fulfillment,
    _cycle_consumption_edge_fraction,
    _load_linear_module,
    _result_payload,
    _simulator_state_payload,
    _write_run_metadata,
)
from qbp_sim.experiments.matrix import ExperimentMatrixCase, ExperimentMatrixConfig
from qbp_sim.io.trace import open_event_trace_writer


@dataclass(frozen=True, slots=True)
class ExperimentMatrixRun:
    case: ExperimentMatrixCase
    case_dir: Path
    lp_json_path: Path
    simulation_config_path: Path
    trace_path: Path
    metadata_path: Path
    result: dict[str, int | float | bool]


def _run_matrix_case(
    case: ExperimentMatrixCase,
    *,
    output_dir: Path,
    progress: bool | None,
) -> ExperimentMatrixRun:
    case_dir = output_dir / case.slug
    case_dir.mkdir(parents=True, exist_ok=True)
    lp_json_path = case_dir / "lp_solution.json"
    simulation_config_path = case_dir / "simulation_config.json"
    trace_path = case_dir / "events.vortex"
    metadata_path = case_dir / "run_metadata.json"
    resolved_cons_edge_fraction = _cycle_consumption_edge_fraction(
        case.n_nodes,
        case.consumption_edge_fraction,
    )

    linear_module = _load_linear_module()
    simulation_input = linear_module.single_run_topology(
        topology=case.topology,
        num_nodes=case.n_nodes,
        edge_weight=case.edge_weight,
        gen_scale=case.gen_scale,
        cons_scale=case.cons_scale,
        cons_edge_fraction=resolved_cons_edge_fraction,
        cons_max_edge_weight=case.cons_max_edge_weight,
        seed=case.seed,
        objective=case.objective,
        swap_rate=case.swap_rate,
        json_output_path=str(lp_json_path),
        simulation_config_output_path=str(simulation_config_path),
        json_pretty=True,
        json_emit_full_matrices=True,
        output_mode="json+simulation-config",
    )
    if simulation_input is None:
        raise RuntimeError(f"LP solve failed for matrix case {case.slug}.")

    simulation_input = _apply_capacity_headroom(simulation_input, case.capacity_headroom)
    simulation_input = simulation_input.model_copy(
        update={"virtual_swap_policy": case.virtual_swap_policy}
    )
    simulation_input = _apply_instant_service_fulfillment(
        simulation_input,
        case.instant_service_fulfillment,
        case.instant_swap_fulfillment,
    )
    simulation_config_path.write_text(
        simulation_input.model_dump_json(indent=2),
        encoding="utf-8",
    )

    simulator = GillespieQBPSimulator(simulation_input.to_runtime_config(), seed=case.seed)
    initial_state = None
    if case.burn_in_time > 0.0:
        simulator.run(
            until_time=case.burn_in_time,
            max_events=case.max_events,
            sample_every=0,
            progress=progress,
        )
        simulator.reset_measurements(reset_time_origin=True)
        initial_state = _simulator_state_payload(simulator)

    snapshot_writer = _MemorySnapshotWriter()
    with open_event_trace_writer(
        trace_path,
        float_precision=case.trace_float_precision,
        time_mode=case.trace_time_mode,
    ) as trace_writer:
        result = simulator.run(
            until_time=case.until_time,
            max_events=case.max_events,
            sample_every=case.sample_every,
            trace_writer=trace_writer,
            snapshot_writer=snapshot_writer,
            progress=progress,
        )

    _write_run_metadata(
        metadata_path,
        command="matrix",
        n_nodes=case.n_nodes,
        seed=case.seed,
        until_time=case.until_time,
        max_events=case.max_events,
        sample_every=case.sample_every,
        burn_in_time=case.burn_in_time,
        trace_float_precision=case.trace_float_precision,
        trace_time_mode=case.trace_time_mode,
        simulation_config_path=simulation_config_path,
        trace_path=trace_path,
        lp_json_path=lp_json_path,
        result=result,
        initial_state=initial_state,
        extra={
            "topology": case.topology,
            "capacity_headroom": case.capacity_headroom,
            "policy_label": case.policy_label,
            "policy_mode": case.policy_mode,
            "k": case.k,
            "memory": case.memory,
            "edge_weight": case.edge_weight,
            "gen_scale": case.gen_scale,
            "cons_scale": case.cons_scale,
            "consumption_edge_fraction": case.consumption_edge_fraction,
            "resolved_consumption_edge_fraction": resolved_cons_edge_fraction,
            "swap_rate": case.swap_rate,
            "instant_service_fulfillment": case.instant_service_fulfillment,
            "instant_swap_fulfillment": case.instant_swap_fulfillment,
        },
    )

    return ExperimentMatrixRun(
        case=case,
        case_dir=case_dir,
        lp_json_path=lp_json_path,
        simulation_config_path=simulation_config_path,
        trace_path=trace_path,
        metadata_path=metadata_path,
        result=_result_payload(result),
    )


def write_matrix_summary(
    runs: list[ExperimentMatrixRun],
    summary_path: str | Path,
) -> None:
    path = Path(summary_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "slug",
        "topology",
        "n_nodes",
        "capacity_headroom",
        "policy_label",
        "policy_mode",
        "k",
        "memory",
        "seed",
        "until_time",
        "final_time",
        "events_processed",
        "pair_generations",
        "virtual_service_requests",
        "virtual_swap_requests",
        "demand_arrivals",
        "services_completed",
        "swaps_completed",
        "service_ratio",
        "total_backlog",
        "total_inventory",
        "total_scarcity",
        "trace_path",
        "metadata_path",
        "simulation_config_path",
    ]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for run in runs:
            case = run.case
            row = {
                "slug": case.slug,
                "topology": case.topology,
                "n_nodes": case.n_nodes,
                "capacity_headroom": case.capacity_headroom,
                "policy_label": case.policy_label,
                "policy_mode": case.policy_mode,
                "k": "" if case.k is None else case.k,
                "memory": "" if case.memory is None else case.memory,
                "seed": case.seed,
                "until_time": case.until_time,
                "trace_path": str(run.trace_path),
                "metadata_path": str(run.metadata_path),
                "simulation_config_path": str(run.simulation_config_path),
            }
            row.update(run.result)
            writer.writerow(row)


def run_experiment_matrix(
    matrix: ExperimentMatrixConfig,
    *,
    output_dir: str | Path,
    progress: bool | None = None,
) -> list[ExperimentMatrixRun]:
    base_dir = Path(output_dir)
    base_dir.mkdir(parents=True, exist_ok=True)
    runs = [
        _run_matrix_case(case, output_dir=base_dir, progress=progress)
        for case in matrix.cases()
    ]
    write_matrix_summary(runs, base_dir / "summary.csv")
    return runs
