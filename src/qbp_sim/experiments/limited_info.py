
from __future__ import annotations

from pathlib import Path

from qbp_sim.config import SimulationInputConfig, VirtualSwapPolicyConfig
from qbp_sim.core.simulator import GillespieQBPSimulator
from qbp_sim.experiments.common import (
    LimitedInfoServiceRatioRun,
    _MemorySnapshotWriter,
    _apply_capacity_headroom,
    _apply_instant_service_fulfillment,
    _cycle_consumption_edge_fraction,
    _load_linear_module,
    _policy_slug,
    _write_run_metadata,
)
from qbp_sim.io.trace import open_event_trace_writer

def run_limited_info_service_ratio_experiment(
    *,
    n_nodes: int,
    limited_policies: list[tuple[int, int]],
    output_dir: str | Path,
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
    capacity_headroom: float = 1.01,
    trace_float_precision: str = "float32",
    instant_service_fulfillment: bool = False,
    instant_swap_fulfillment: bool = False,
    progress: bool | None = None,
) -> list[LimitedInfoServiceRatioRun]:
    if sample_every <= 0:
        raise ValueError("sample_every must be positive for limited-info service-ratio experiments.")
    if not limited_policies:
        raise ValueError("limited_policies must not be empty.")

    base_dir = Path(output_dir)
    base_dir.mkdir(parents=True, exist_ok=True)
    linear_module = _load_linear_module()
    lp_json_path = base_dir / "lp_solution.json"
    base_simulation_config_path = base_dir / "base_simulation_config.json"
    run_seed = seed_base + n_nodes
    run_cons_edge_fraction = _cycle_consumption_edge_fraction(n_nodes, cons_edge_fraction)

    base_simulation_input = linear_module.single_run_topology(
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
        simulation_config_output_path=str(base_simulation_config_path),
        json_pretty=True,
        json_emit_full_matrices=True,
        output_mode="json+simulation-config",
    )
    if base_simulation_input is None:
        raise RuntimeError(f"LP solve failed for cycle n={n_nodes}.")
    base_simulation_input = _apply_instant_service_fulfillment(
        _apply_capacity_headroom(base_simulation_input, capacity_headroom),
        instant_service_fulfillment,
        instant_swap_fulfillment,
    )
    base_simulation_config_path.write_text(base_simulation_input.model_dump_json(indent=2), encoding="utf-8")

    variants: list[tuple[str, str, int | None, int | None, SimulationInputConfig]] = [
        ("full info", "global", None, None, base_simulation_input)
    ]
    for k, memory in limited_policies:
        if k <= 0 or memory < 0:
            raise ValueError("limited policy k must be positive and memory must be non-negative.")
        policy = VirtualSwapPolicyConfig(mode="power_of_k_memory", k=k, memory=memory)
        variants.append(
            (
                f"limited k={k}, m={memory}",
                "power_of_k_memory",
                k,
                memory,
                base_simulation_input.model_copy(update={"virtual_swap_policy": policy}),
            )
        )

    runs: list[LimitedInfoServiceRatioRun] = []
    for policy_label, policy_mode, k, memory, simulation_input in variants:
        case_dir = base_dir / _policy_slug(policy_label)
        case_dir.mkdir(parents=True, exist_ok=True)
        simulation_config_path = case_dir / "simulation_config.json"
        trace_path = case_dir / "events.vortex"
        metadata_path = case_dir / "run_metadata.json"
        simulation_config_path.write_text(simulation_input.model_dump_json(indent=2), encoding="utf-8")

        simulator = GillespieQBPSimulator(config=simulation_input.to_runtime_config(), seed=run_seed)
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
            command="limited-info-service-ratio",
            n_nodes=n_nodes,
            seed=run_seed,
            until_time=until_time,
            max_events=max_events,
            sample_every=sample_every,
            burn_in_time=0.0,
            trace_float_precision=trace_float_precision,
            simulation_config_path=simulation_config_path,
            trace_path=trace_path,
            lp_json_path=lp_json_path,
            result=result,
            extra={
                "policy_label": policy_label,
                "policy_mode": policy_mode,
                "k": k,
                "memory": memory,
                "capacity_headroom": simulation_input.capacity_headroom,
                "instant_service_fulfillment": simulation_input.instant_service_fulfillment,
                "instant_swap_fulfillment": simulation_input.instant_swap_fulfillment,
            },
        )

        runs.append(
            LimitedInfoServiceRatioRun(
                n_nodes=n_nodes,
                policy_label=policy_label,
                policy_mode=policy_mode,
                k=k,
                memory=memory,
                lp_json_path=lp_json_path,
                simulation_config_path=simulation_config_path,
                trace_path=trace_path,
                metadata_path=metadata_path,
                snapshots=snapshot_writer.snapshots,
            )
        )

    return runs
