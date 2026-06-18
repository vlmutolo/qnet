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
        virtual_swap_policy=VirtualSwapPolicyConfig(mode="bp"),
    )


def service_ratio_frame(trace_path: Path, sample_every: int = 25) -> pl.DataFrame:
    lf = vx.open(str(trace_path)).to_polars()
    return (
        lf.select("event_index", "time", "event_type")
        .with_columns(
            demand_arrivals=(pl.col("event_type") == "demand_arrival").cast(pl.Int64).cum_sum(),
            services_completed=(pl.col("event_type") == "physical_service").cast(pl.Int64).cum_sum(),
        )
        .filter((pl.col("event_index") % sample_every) == 0)
        .with_columns(
            service_ratio=pl.when(pl.col("demand_arrivals") > 0)
            .then(pl.col("services_completed") / pl.col("demand_arrivals"))
            .otherwise(0.0)
        )
        .select("event_index", "time", "demand_arrivals", "services_completed", "service_ratio")
        .collect()
    )


def main() -> None:
    output_dir = Path("output/examples/01_basic_run_service_ratio")
    output_dir.mkdir(parents=True, exist_ok=True)
    trace_path = output_dir / "events.vortex"
    plot_path = output_dir / "service_ratio.html"

    result = run_simulation(
        build_config(),
        RunOptions(
            until_time=50.0,
            max_events=20_000,
            sample_every=0,
            seed=1,
            trace_path=trace_path,
            progress=False,
        ),
    )
    df = service_ratio_frame(trace_path)

    chart = (
        alt.Chart(df)
        .mark_line(strokeWidth=2)
        .encode(
            x=alt.X("time:Q", title="simulation time"),
            y=alt.Y("service_ratio:Q", title="service ratio", scale=alt.Scale(domain=[0, 1])),
            tooltip=["event_index:Q", "time:Q", "demand_arrivals:Q", "services_completed:Q", "service_ratio:Q"],
        )
        .properties(width=800, height=420, title="Basic BP Service Ratio")
    )
    chart.save(plot_path)

    print(f"trace={trace_path}")
    print(f"plot={plot_path}")
    print(f"final_service_ratio={result.service_ratio:.6f}")


if __name__ == "__main__":
    main()
