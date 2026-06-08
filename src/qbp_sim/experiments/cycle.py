
from __future__ import annotations

from pathlib import Path

from qbp_sim.core.simulator import GillespieQBPSimulator
from qbp_sim.experiments.common import (
    CycleServiceRatioRun,
    _MemorySnapshotWriter,
    _apply_instant_service_fulfillment,
    _cycle_consumption_edge_fraction,
    _load_linear_module,
    _simulator_state_payload,
    _write_run_metadata,
)
from qbp_sim.io.trace import open_event_trace_writer

def run_cycle_service_ratio_experiment(
    *,
    cycle_sizes: list[int],
    output_dir: str | Path,
    burn_in_time: float,
    until_time: float,
    max_events: int | None,
    sample_every: int,
    seed_base: int = 0,
    edge_weight: float = 10.0,
    gen_scale: float = 10.0,
    cons_scale: float = 1.0,
    cons_edge_fraction: float | None = None,
    cons_max_edge_weight: float = 7.0,
    objective: str = "min_sum_generate",
    swap_rate: float = 100.0,
    trace_float_precision: str = "float32",
    instant_service_fulfillment: bool = False,
    instant_swap_fulfillment: bool = False,
    progress: bool | None = None,
) -> list[CycleServiceRatioRun]:
    if sample_every <= 0:
        raise ValueError("sample_every must be positive for cycle service-ratio experiments.")

    base_dir = Path(output_dir)
    base_dir.mkdir(parents=True, exist_ok=True)
    runs: list[CycleServiceRatioRun] = []
    linear_module = _load_linear_module()

    for n_nodes in cycle_sizes:
        case_dir = base_dir / f"cycle_n{n_nodes}"
        case_dir.mkdir(parents=True, exist_ok=True)
        lp_json_path = case_dir / "lp_solution.json"
        simulation_config_path = case_dir / "simulation_config.json"
        trace_path = case_dir / "events.vortex"
        metadata_path = case_dir / "run_metadata.json"
        run_seed = seed_base + n_nodes
        run_cons_edge_fraction = _cycle_consumption_edge_fraction(n_nodes, cons_edge_fraction)

        simulation_input = linear_module.single_run_topology(
            topology="cycle",
            num_nodes=n_nodes,
            edge_weight=edge_weight,
            gen_scale=gen_scale,
            cons_scale=cons_scale,
            cons_edge_fraction=run_cons_edge_fraction,
            cons_max_edge_weight=cons_max_edge_weight,
            seed=run_seed,
            objective=objective,
            swap_rate=swap_rate,
            json_output_path=str(lp_json_path),
            simulation_config_output_path=str(simulation_config_path),
            json_pretty=True,
            json_emit_full_matrices=True,
            output_mode="json+simulation-config",
        )
        if simulation_input is None:
            raise RuntimeError(f"LP solve failed for cycle n={n_nodes}.")
        simulation_input = _apply_instant_service_fulfillment(
            simulation_input,
            instant_service_fulfillment,
            instant_swap_fulfillment,
        )
        simulation_config_path.write_text(simulation_input.model_dump_json(indent=2), encoding="utf-8")

        simulator = GillespieQBPSimulator(config=simulation_input.to_runtime_config(), seed=run_seed)
        initial_state = None
        if burn_in_time > 0.0:
            simulator.run(
                until_time=burn_in_time,
                max_events=max_events,
                sample_every=0,
                progress=progress,
            )
            simulator.reset_measurements(reset_time_origin=True)
            initial_state = _simulator_state_payload(simulator)
        snapshot_writer = _MemorySnapshotWriter()
        with open_event_trace_writer(trace_path, float_precision=trace_float_precision) as trace_writer:
            result = simulator.run(
                until_time=until_time,
                max_events=max_events,
                sample_every=sample_every,
                trace_writer=trace_writer,
                snapshot_writer=snapshot_writer,
                progress=progress,
            )
        _write_run_metadata(
            metadata_path,
            command="cycle-service-ratio",
            n_nodes=n_nodes,
            seed=run_seed,
            until_time=until_time,
            max_events=max_events,
            sample_every=sample_every,
            burn_in_time=burn_in_time,
            trace_float_precision=trace_float_precision,
            simulation_config_path=simulation_config_path,
            trace_path=trace_path,
            lp_json_path=lp_json_path,
            result=result,
            initial_state=initial_state,
            extra={
                "instant_service_fulfillment": simulation_input.instant_service_fulfillment,
                "instant_swap_fulfillment": simulation_input.instant_swap_fulfillment,
            },
        )

        runs.append(
            CycleServiceRatioRun(
                n_nodes=n_nodes,
                lp_json_path=lp_json_path,
                simulation_config_path=simulation_config_path,
                trace_path=trace_path,
                metadata_path=metadata_path,
                snapshots=snapshot_writer.snapshots,
            )
        )

    return runs
