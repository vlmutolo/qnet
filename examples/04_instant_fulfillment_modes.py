from __future__ import annotations

from pathlib import Path

import altair as alt
import polars as pl
import vortex as vx

from qbp_sim import RunOptions, SimulationInputConfig, VirtualSwapPolicyConfig, run_simulation


def build_config(*, instant_service: bool, instant_swap: bool) -> SimulationInputConfig:
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
        instant_service_fulfillment=instant_service,
        instant_swap_fulfillment=instant_swap,
    )


def service_ratio_frame(trace_path: Path, mode_label: str, sample_every: int = 25) -> pl.DataFrame:
    lf = vx.open(str(trace_path)).to_polars()
    return (
        lf.select("event_index", "time", "event_type")
        .with_columns(
            demand_arrivals=(pl.col("event_type") == "demand_arrival").cast(pl.Int64).cum_sum(),
            services_completed=(pl.col("event_type") == "physical_service").cast(pl.Int64).cum_sum(),
        )
        .filter((pl.col("event_index") % sample_every) == 0)
        .with_columns(
            mode=pl.lit(mode_label),
            service_ratio=pl.when(pl.col("demand_arrivals") > 0)
            .then(pl.col("services_completed") / pl.col("demand_arrivals"))
            .otherwise(0.0),
        )
        .select("mode", "event_index", "time", "service_ratio")
        .collect()
    )


def main() -> None:
    output_dir = Path("output/examples/04_instant_fulfillment_modes")
    output_dir.mkdir(parents=True, exist_ok=True)
    modes = [
        ("sampled service and swaps", False, False),
        ("instant service", True, False),
        ("instant service and swaps", True, True),
    ]

    frames: list[pl.DataFrame] = []
    for offset, (label, instant_service, instant_swap) in enumerate(modes):
        trace_path = output_dir / f"mode_{offset}.vortex"
        run_simulation(
            build_config(instant_service=instant_service, instant_swap=instant_swap),
            RunOptions(
                until_time=50.0,
                max_events=20_000,
                sample_every=0,
                seed=50 + offset,
                trace_path=trace_path,
                progress=False,
            ),
        )
        frames.append(service_ratio_frame(trace_path, label))

    df = pl.concat(frames)
    plot_path = output_dir / "instant_fulfillment_service_ratio.html"
    chart = (
        alt.Chart(df)
        .mark_line(strokeWidth=2)
        .encode(
            x=alt.X("time:Q", title="simulation time"),
            y=alt.Y("service_ratio:Q", title="service ratio", scale=alt.Scale(domain=[0, 1])),
            color=alt.Color("mode:N", title="mode"),
            tooltip=["mode:N", "event_index:Q", "time:Q", "service_ratio:Q"],
        )
        .properties(width=860, height=460, title="Instant Fulfillment Modes")
    )
    chart.save(plot_path)
    print(f"plot={plot_path}")


if __name__ == "__main__":
    main()
