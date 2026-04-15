from __future__ import annotations

import importlib.util
import math
from dataclasses import dataclass
from pathlib import Path

import altair as alt

from qbp_sim.analysis import save_chart, summarize_snapshots
from qbp_sim.simulator import GillespieQBPSimulator
from qbp_sim.snapshots import QBPSnapshot, SnapshotReader, SnapshotWriter


@dataclass(slots=True)
class CycleServiceRatioRun:
    n_nodes: int
    lp_json_path: Path
    simulation_config_path: Path
    snapshots_path: Path
    snapshots: list[QBPSnapshot]

    @property
    def summary(self):
        return summarize_snapshots(self.snapshots)


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


def run_cycle_service_ratio_experiment(
    *,
    cycle_sizes: list[int],
    output_dir: str | Path,
    until_time: float,
    max_events: int,
    sample_every: int,
    seed_base: int = 0,
    edge_weight: float = 10.0,
    gen_scale: float = 10.0,
    cons_scale: float = 1.0,
    cons_edge_fraction: float | None = None,
    cons_max_edge_weight: float = 7.0,
    objective: str = "min_sum_generate",
    swap_rate: float = 100.0,
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
        snapshots_path = case_dir / "bp_snapshots.jsonl.zst"
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
        with SnapshotWriter(snapshots_path) as snapshot_writer:
            simulator.run(
                until_time=until_time,
                max_events=max_events,
                sample_every=sample_every,
                snapshot_writer=snapshot_writer,
                progress=progress,
            )
        with SnapshotReader(snapshots_path) as snapshot_reader:
            snapshots = list(snapshot_reader)

        runs.append(
            CycleServiceRatioRun(
                n_nodes=n_nodes,
                lp_json_path=lp_json_path,
                simulation_config_path=simulation_config_path,
                snapshots_path=snapshots_path,
                snapshots=snapshots,
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
                    "time": snapshot.time,
                    "event_index": snapshot.event_index,
                    "service_gap": max(1e-12, 1.0 - snapshot.service_ratio),
                }
            )

    chart = (
        alt.Chart(alt.Data(values=rows))
        .mark_line(strokeWidth=2.2)
        .encode(
            x=alt.X("time:Q", title="time"),
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
            title="BP service-gap decay on LP-derived cycle topologies",
        )
    )
    save_chart(chart, output_path)
