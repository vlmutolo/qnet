from __future__ import annotations

import importlib.util
import json
import math
from dataclasses import dataclass
from pathlib import Path

import altair as alt
import numpy as np

from qbp_sim.analysis import save_chart, summarize_snapshots
from qbp_sim.config import SimulationInputConfig, VirtualSwapPolicyConfig
from qbp_sim.simulator import GillespieQBPResult, GillespieQBPSimulator
from qbp_sim.snapshots import QBPSnapshot
from qbp_sim.trace import open_event_trace_writer


@dataclass(slots=True)
class CycleServiceRatioRun:
    n_nodes: int
    lp_json_path: Path
    simulation_config_path: Path
    trace_path: Path
    metadata_path: Path
    snapshots: list[QBPSnapshot]

    @property
    def summary(self):
        return summarize_snapshots(self.snapshots)


@dataclass(slots=True)
class HeadroomRun:
    n_nodes: int
    capacity_headroom: float
    lp_json_path: Path
    simulation_config_path: Path
    trace_path: Path
    metadata_path: Path
    snapshots: list[QBPSnapshot]

    @property
    def summary(self):
        return summarize_snapshots(self.snapshots)


@dataclass(slots=True)
class LimitedInfoServiceRatioRun:
    n_nodes: int
    policy_label: str
    policy_mode: str
    k: int | None
    memory: int | None
    lp_json_path: Path
    simulation_config_path: Path
    trace_path: Path
    metadata_path: Path
    snapshots: list[QBPSnapshot]

    @property
    def summary(self):
        return summarize_snapshots(self.snapshots)


class _MemorySnapshotWriter:
    def __init__(self) -> None:
        self.snapshots: list[QBPSnapshot] = []

    def write(self, snapshot: QBPSnapshot) -> None:
        self.snapshots.append(snapshot)


def _simulator_state_payload(simulator: GillespieQBPSimulator) -> dict[str, list]:
    state = simulator.state
    return {
        "q": state.q.tolist(),
        "d": state.d.tolist(),
        "alpha": state.alpha.tolist(),
        "h_r": state.h_r.tolist(),
        "h_mu": state.h_mu.tolist(),
    }


def _result_payload(result: GillespieQBPResult) -> dict[str, int | float | bool]:
    return {
        "final_time": result.final_time,
        "events_processed": result.events_processed,
        "total_backlog": result.total_backlog,
        "total_inventory": result.total_inventory,
        "total_scarcity": result.total_scarcity,
        "demand_arrivals": result.demand_arrivals,
        "pair_generations": result.pair_generations,
        "virtual_service_requests": result.virtual_service_requests,
        "virtual_swap_requests": result.virtual_swap_requests,
        "services_completed": result.services_completed,
        "swaps_completed": result.swaps_completed,
        "service_ratio": (
            0.0
            if result.demand_arrivals == 0
            else float(result.services_completed) / float(result.demand_arrivals)
        ),
    }


def _write_run_metadata(
    path: str | Path,
    *,
    command: str,
    n_nodes: int,
    seed: int,
    until_time: float,
    max_events: int | None,
    sample_every: int,
    burn_in_time: float,
    trace_float_precision: str,
    simulation_config_path: Path,
    trace_path: Path,
    lp_json_path: Path,
    result: GillespieQBPResult,
    initial_state: dict[str, list] | None = None,
    extra: dict[str, int | float | str | bool | None] | None = None,
) -> None:
    metadata = {
        "schema_version": 1,
        "command": command,
        "n_nodes": n_nodes,
        "seed": seed,
        "until_time": until_time,
        "max_events": max_events,
        "sample_every": sample_every,
        "burn_in_time": burn_in_time,
        "trace_float_precision": trace_float_precision,
        "hit_time_horizon": result.final_time >= until_time,
        "hit_event_cap": (
            False if max_events is None else result.events_processed >= max_events and result.final_time < until_time
        ),
        "simulation_config_path": str(simulation_config_path),
        "trace_path": str(trace_path),
        "lp_json_path": str(lp_json_path),
        "trace_format": trace_path.suffix.lstrip("."),
        "result": _result_payload(result),
        "initial_state": initial_state,
        "extra": extra or {},
    }
    metadata_path = Path(path)
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")


def _load_linear_module():
    linear_path = Path(__file__).resolve().parents[2] / "linear.py"
    spec = importlib.util.spec_from_file_location("qbp_sim_linear_module", linear_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load LP module from {linear_path}.")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _cycle_consumption_edge_fraction(
    n_nodes: int,
    cons_edge_fraction: float | None,
) -> float:
    if cons_edge_fraction is not None:
        return cons_edge_fraction
    total_pairs = math.comb(n_nodes, 2)
    target_pairs = max(1, n_nodes // 2)
    return min(1.0, float(target_pairs) / float(total_pairs))


def _headroom_slug(headroom: float) -> str:
    return f"{headroom:.6f}".rstrip("0").rstrip(".").replace(".", "p")


def _policy_slug(policy_label: str) -> str:
    return (
        policy_label.lower()
        .replace(" ", "_")
        .replace("=", "")
        .replace(",", "")
        .replace("/", "_")
    )


def _apply_capacity_headroom(
    simulation_input: SimulationInputConfig,
    headroom: float,
) -> SimulationInputConfig:
    if headroom <= 0.0:
        raise ValueError("capacity headroom must be positive.")
    return simulation_input.model_copy(update={"capacity_headroom": float(headroom)})


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
    base_simulation_input = _apply_capacity_headroom(base_simulation_input, capacity_headroom)
    base_simulation_config_path.write_text(base_simulation_input.model_dump_json(indent=2), encoding="utf-8")

    variants: list[tuple[str, str, int | None, int | None, SimulationInputConfig]] = [
        ("full info", "global", None, None, base_simulation_input)
    ]
    for k, memory in limited_policies:
        if k <= 0 or memory <= 0:
            raise ValueError("limited policy k and memory must be positive.")
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


def run_headroom_experiment(
    *,
    n_nodes: int,
    capacity_headrooms: list[float],
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
    progress: bool | None = None,
) -> list[HeadroomRun]:
    if sample_every <= 0:
        raise ValueError("sample_every must be positive for headroom experiments.")
    if not capacity_headrooms:
        raise ValueError("capacity_headrooms must not be empty.")

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

    runs: list[HeadroomRun] = []
    for headroom in capacity_headrooms:
        case_dir = base_dir / f"headroom_{_headroom_slug(headroom)}"
        case_dir.mkdir(parents=True, exist_ok=True)
        simulation_config_path = case_dir / "simulation_config.json"
        trace_path = case_dir / "events.vortex"
        metadata_path = case_dir / "run_metadata.json"
        headroom_input = _apply_capacity_headroom(base_simulation_input, headroom)
        simulation_config_path.write_text(headroom_input.model_dump_json(indent=2))

        simulator = GillespieQBPSimulator(config=headroom_input.to_runtime_config(), seed=run_seed)
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
            command="headroom-service-ratio",
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
            extra={"capacity_headroom": headroom},
        )

        runs.append(
            HeadroomRun(
                n_nodes=n_nodes,
                capacity_headroom=headroom,
                lp_json_path=lp_json_path,
                simulation_config_path=simulation_config_path,
                trace_path=trace_path,
                metadata_path=metadata_path,
                snapshots=snapshot_writer.snapshots,
            )
        )

    return runs


def plot_cycle_service_ratio_runs(
    runs: list[CycleServiceRatioRun],
    output_path: str | Path,
) -> None:
    ordered_runs = sorted(runs, key=lambda run: run.n_nodes)
    rows: list[dict[str, float | int | str]] = []
    series_order = [f"n={run.n_nodes}" for run in ordered_runs]
    for order, run in enumerate(ordered_runs):
        label = f"n={run.n_nodes}"
        for snapshot in run.snapshots:
            rows.append(
                {
                    "series": label,
                    "series_order": order,
                    "time": max(1e-12, snapshot.time),
                    "event_index": snapshot.event_index,
                    "service_gap": max(1e-12, 1.0 - snapshot.service_ratio),
                }
            )

    chart = (
        alt.Chart(alt.Data(values=rows))
        .mark_line(strokeWidth=2.2)
        .encode(
            x=alt.X("time:Q", title="time", scale=alt.Scale(type="log")),
            y=alt.Y(
                "service_gap:Q",
                title="1 - service_ratio",
                scale=alt.Scale(type="log"),
            ),
            color=alt.Color("series:N", title="cycle size", sort=series_order),
            detail="series:N",
            tooltip=[
                alt.Tooltip("series:N"),
                alt.Tooltip("time:Q", format=".4f"),
                alt.Tooltip("service_gap:Q", format=".6e"),
                alt.Tooltip("event_index:Q"),
            ],
        )
        .properties(
            width=860,
            height=480,
            title="BP service-gap decay on LP-derived cycle topologies (log-log)",
        )
    )
    save_chart(chart, output_path)


def plot_headroom_runs(
    runs: list[HeadroomRun],
    output_path: str | Path,
) -> None:
    ordered_runs = sorted(runs, key=lambda run: run.capacity_headroom)
    rows: list[dict[str, float | int | str]] = []
    series_order = [f"x{run.capacity_headroom:g}" for run in ordered_runs]
    for order, run in enumerate(ordered_runs):
        label = f"x{run.capacity_headroom:g}"
        for snapshot in run.snapshots:
            rows.append(
                {
                    "series": label,
                    "series_order": order,
                    "time": max(1e-12, snapshot.time),
                    "event_index": snapshot.event_index,
                    "service_gap": max(1e-12, 1.0 - snapshot.service_ratio),
                }
            )

    chart = (
        alt.Chart(alt.Data(values=rows))
        .mark_line(strokeWidth=2.2)
        .encode(
            x=alt.X("time:Q", title="time", scale=alt.Scale(type="log")),
            y=alt.Y(
                "service_gap:Q",
                title="1 - service_ratio",
                scale=alt.Scale(type="log"),
            ),
            color=alt.Color("series:N", title="capacity headroom", sort=series_order),
            detail="series:N",
            tooltip=[
                alt.Tooltip("series:N"),
                alt.Tooltip("time:Q", format=".4f"),
                alt.Tooltip("service_gap:Q", format=".6e"),
                alt.Tooltip("event_index:Q"),
            ],
        )
        .properties(
            width=860,
            height=480,
            title=f"BP service-gap decay on LP-derived cycle n={ordered_runs[0].n_nodes} with capacity headroom (log-log)",
        )
    )
    save_chart(chart, output_path)


def plot_limited_info_service_ratio_runs(
    runs: list[LimitedInfoServiceRatioRun],
    output_path: str | Path,
    *,
    plot_start_time: float = 100.0,
) -> None:
    if not runs:
        raise ValueError("No limited-info service-ratio runs were provided.")
    if plot_start_time <= 0.0:
        raise ValueError("plot_start_time must be positive for the log-scaled time axis.")

    rows: list[dict[str, float | int | str]] = []
    series_order = [run.policy_label for run in runs]
    for order, run in enumerate(runs):
        for snapshot in run.snapshots:
            if snapshot.time < plot_start_time:
                continue
            rows.append(
                {
                    "series": run.policy_label,
                    "series_order": order,
                    "policy_mode": run.policy_mode,
                    "time": snapshot.time,
                    "plot_time": snapshot.time,
                    "event_index": snapshot.event_index,
                    "service_ratio": snapshot.service_ratio,
                    "demand_arrivals": snapshot.demand_arrivals,
                    "services_completed": snapshot.services_completed,
                }
            )
    if not rows:
        raise ValueError(f"No snapshots at or after plot_start_time={plot_start_time}.")
    max_plot_time = max(float(row["plot_time"]) for row in rows)

    chart = (
        alt.Chart(alt.Data(values=rows))
        .mark_line(strokeWidth=2.3)
        .encode(
            x=alt.X(
                "plot_time:Q",
                title="simulation time since t=0 (log scale)",
                scale=alt.Scale(type="log", domain=[plot_start_time, max_plot_time]),
            ),
            y=alt.Y(
                "service_ratio:Q",
                title="service_ratio = services_completed / demand_arrivals",
                scale=alt.Scale(domain=[0.0, 1.0]),
            ),
            color=alt.Color("series:N", title="policy", sort=series_order),
            detail="series:N",
            tooltip=[
                alt.Tooltip("series:N", title="policy"),
                alt.Tooltip("time:Q", format=".3f"),
                alt.Tooltip("service_ratio:Q", format=".4f"),
                alt.Tooltip("demand_arrivals:Q"),
                alt.Tooltip("services_completed:Q"),
                alt.Tooltip("event_index:Q"),
            ],
        )
        .properties(
            width=900,
            height=500,
            title=f"Limited-info vs full-info BP service ratio from t={plot_start_time:g} on cycle n={runs[0].n_nodes}",
        )
    )
    save_chart(chart, output_path)
