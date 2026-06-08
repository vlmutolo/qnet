
from __future__ import annotations

from pathlib import Path

import altair as alt

from qbp_sim.analysis import save_chart
from qbp_sim.experiments.common import CycleServiceRatioRun, HeadroomRun, LimitedInfoServiceRatioRun

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
