from __future__ import annotations

from pathlib import Path

import altair as alt
import polars as pl
import vortex as vx

from qbp_sim import RunOptions, SimulationInputConfig, VirtualSwapPolicyConfig, run_simulation


def build_config(headroom: float) -> SimulationInputConfig:
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
        capacity_headroom=headroom,
        virtual_swap_policy=VirtualSwapPolicyConfig(mode="bp"),
    )


def service_gap_frame(trace_path: Path, headroom: float, sample_every: int = 25) -> pl.DataFrame:
    lf = vx.open(str(trace_path)).to_polars()
    return (
        lf.select("event_index", "time", "event_type")
        .with_columns(
            demand_arrivals=(pl.col("event_type") == "demand_arrival").cast(pl.Int64).cum_sum(),
            services_completed=(pl.col("event_type") == "physical_service").cast(pl.Int64).cum_sum(),
        )
        .filter((pl.col("event_index") % sample_every) == 0)
        .with_columns(
            headroom=pl.lit(f"{headroom:g}"),
            service_ratio=pl.when(pl.col("demand_arrivals") > 0)
            .then(pl.col("services_completed") / pl.col("demand_arrivals"))
            .otherwise(0.0),
        )
        .with_columns(service_gap=pl.max_horizontal(pl.lit(1.0e-6), 1.0 - pl.col("service_ratio")))
        .select("headroom", "event_index", "time", "service_gap")
        .collect()
    )


def main() -> None:
    output_dir = Path("output/examples/03_headroom_sweep")
    output_dir.mkdir(parents=True, exist_ok=True)
    frames: list[pl.DataFrame] = []

    for offset, headroom in enumerate([1.0, 1.01, 1.05]):
        trace_path = output_dir / f"headroom_{str(headroom).replace('.', 'p')}.vortex"
        run_simulation(
            build_config(headroom),
            RunOptions(
                until_time=75.0,
                max_events=30_000,
                sample_every=0,
                seed=30 + offset,
                trace_path=trace_path,
                progress=False,
            ),
        )
        frames.append(service_gap_frame(trace_path, headroom))

    df = pl.concat(frames)
    plot_path = output_dir / "headroom_service_gap.html"
    chart = (
        alt.Chart(df)
        .mark_line(strokeWidth=2)
        .encode(
            x=alt.X("time:Q", title="simulation time"),
            y=alt.Y("service_gap:Q", title="1 - service ratio", scale=alt.Scale(type="log")),
            color=alt.Color("headroom:N", title="capacity headroom"),
            tooltip=["headroom:N", "event_index:Q", "time:Q", "service_gap:Q"],
        )
        .properties(width=860, height=460, title="Headroom Sweep")
    )
    chart.save(plot_path)
    print(f"plot={plot_path}")


if __name__ == "__main__":
    main()
