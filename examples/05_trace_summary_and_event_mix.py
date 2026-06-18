from __future__ import annotations

from pathlib import Path

import altair as alt
import polars as pl
import vortex as vx

from qbp_sim import RunOptions, SimulationInputConfig, VirtualSwapPolicyConfig, run_simulation


def build_config() -> SimulationInputConfig:
    return SimulationInputConfig(
        generation_rates=[
            [0.0, 1.0, 1.0, 0.0],
            [1.0, 0.0, 0.0, 1.0],
            [1.0, 0.0, 0.0, 1.0],
            [0.0, 1.0, 1.0, 0.0],
        ],
        consumption_rates=[
            [0.0, 0.0, 0.0, 2.0],
            [0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0],
            [2.0, 0.0, 0.0, 0.0],
        ],
        swap_rates=[0.0, 1.0, 1.0, 0.0],
        capacity_headroom=1.01,
        virtual_swap_policy=VirtualSwapPolicyConfig(mode="limited_info_max_min", k=1, memory=1),
    )


def load_trace(trace_path: Path) -> pl.LazyFrame:
    return vx.open(str(trace_path)).to_polars()


def event_mix_frame(lf: pl.LazyFrame) -> pl.DataFrame:
    return (
        lf.group_by("event_type")
        .agg(pl.len().alias("count"))
        .with_columns(fraction=pl.col("count") / pl.col("count").sum())
        .sort("count", descending=True)
        .collect()
    )


def aggregate_state_frame(lf: pl.LazyFrame, sample_every: int = 25) -> pl.DataFrame:
    return (
        lf.select("event_index", "time", "backlog_total", "inventory_total", "scarcity_total")
        .collect()
        .filter((pl.col("event_index") % sample_every) == 0)
        .unpivot(
            index=["event_index", "time"],
            on=["backlog_total", "inventory_total", "scarcity_total"],
            variable_name="metric",
            value_name="value",
        )
    )


def service_summary(lf: pl.LazyFrame) -> pl.DataFrame:
    return (
        lf.select("event_type")
        .select(
            demand_arrivals=(pl.col("event_type") == "demand_arrival").sum(),
            services_completed=(pl.col("event_type") == "physical_service").sum(),
            swaps_completed=pl.col("event_type").is_in(["physical_swap", "max_min_swap"]).sum(),
        )
        .with_columns(
            service_ratio=pl.when(pl.col("demand_arrivals") > 0)
            .then(pl.col("services_completed") / pl.col("demand_arrivals"))
            .otherwise(0.0)
        )
        .collect()
    )


def main() -> None:
    output_dir = Path("output/examples/05_trace_summary_and_event_mix")
    output_dir.mkdir(parents=True, exist_ok=True)
    trace_path = output_dir / "events.vortex"

    run_simulation(
        build_config(),
        RunOptions(
            until_time=50.0,
            max_events=20_000,
            sample_every=0,
            seed=70,
            trace_path=trace_path,
            progress=False,
        ),
    )
    lf = load_trace(trace_path)
    event_mix = event_mix_frame(lf)
    aggregate_state = aggregate_state_frame(lf)
    summary = service_summary(lf)

    event_chart = (
        alt.Chart(event_mix)
        .mark_bar()
        .encode(
            x=alt.X("count:Q", title="events"),
            y=alt.Y("event_type:N", title="event type", sort="-x"),
            tooltip=["event_type:N", "count:Q", alt.Tooltip("fraction:Q", format=".2%")],
        )
        .properties(width=760, height=260, title="Event Mix")
    )
    state_chart = (
        alt.Chart(aggregate_state)
        .mark_line(strokeWidth=2)
        .encode(
            x=alt.X("time:Q", title="simulation time"),
            y=alt.Y("value:Q", title="aggregate count"),
            color=alt.Color("metric:N", title="metric"),
            tooltip=["metric:N", "event_index:Q", "time:Q", "value:Q"],
        )
        .properties(width=760, height=320, title="Aggregate State")
    )
    plot_path = output_dir / "trace_summary.png"
    (event_chart & state_chart).save(plot_path, ppi=300)

    print(summary)
    print(f"trace={trace_path}")
    print(f"plot={plot_path}")


if __name__ == "__main__":
    main()
