from __future__ import annotations

from pathlib import Path

import altair as alt
import polars as pl
import vortex as vx

from qbp_sim import RunOptions, SimulationInputConfig, VirtualSwapPolicyConfig, run_simulation


def build_config(policy: VirtualSwapPolicyConfig) -> SimulationInputConfig:
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
        virtual_swap_policy=policy,
    )


def service_ratio_frame(trace_path: Path, policy_label: str, sample_every: int = 25) -> pl.DataFrame:
    lf = vx.open(str(trace_path)).to_polars()
    return (
        lf.select("event_index", "time", "event_type")
        .with_columns(
            demand_arrivals=(pl.col("event_type") == "demand_arrival").cast(pl.Int64).cum_sum(),
            services_completed=(pl.col("event_type") == "physical_service").cast(pl.Int64).cum_sum(),
        )
        .filter((pl.col("event_index") % sample_every) == 0)
        .with_columns(
            policy=pl.lit(policy_label),
            service_ratio=pl.when(pl.col("demand_arrivals") > 0)
            .then(pl.col("services_completed") / pl.col("demand_arrivals"))
            .otherwise(0.0),
        )
        .select("policy", "event_index", "time", "service_ratio")
        .collect()
    )


def final_service_summary(trace_path: Path, policy_label: str) -> pl.DataFrame:
    lf = vx.open(str(trace_path)).to_polars()
    return (
        lf.select("event_type")
        .select(
            demand_arrivals=(pl.col("event_type") == "demand_arrival").sum(),
            services_completed=(pl.col("event_type") == "physical_service").sum(),
        )
        .with_columns(
            policy=pl.lit(policy_label),
            final_service_ratio=pl.when(pl.col("demand_arrivals") > 0)
            .then(pl.col("services_completed") / pl.col("demand_arrivals"))
            .otherwise(0.0),
        )
        .select("policy", "demand_arrivals", "services_completed", "final_service_ratio")
        .collect()
    )


def main() -> None:
    output_dir = Path("output/examples/02_compare_policies")
    output_dir.mkdir(parents=True, exist_ok=True)
    policies = [
        ("bp", VirtualSwapPolicyConfig(mode="bp")),
        ("limited_info_bp", VirtualSwapPolicyConfig(mode="limited_info_bp", k=1, memory=1)),
        ("max_min", VirtualSwapPolicyConfig(mode="max_min")),
        ("limited_info_max_min", VirtualSwapPolicyConfig(mode="limited_info_max_min", k=1, memory=1)),
    ]

    frames: list[pl.DataFrame] = []
    summaries: list[pl.DataFrame] = []
    for offset, (label, policy) in enumerate(policies):
        trace_path = output_dir / f"{label}.vortex"
        run_simulation(
            build_config(policy),
            RunOptions(
                until_time=50.0,
                max_events=20_000,
                sample_every=0,
                seed=10 + offset,
                trace_path=trace_path,
                progress=False,
            ),
        )
        frames.append(service_ratio_frame(trace_path, label))
        summaries.append(final_service_summary(trace_path, label))

    df = pl.concat(frames)
    plot_path = output_dir / "policy_service_ratio.png"
    chart = (
        alt.Chart(df)
        .mark_line(strokeWidth=2)
        .encode(
            x=alt.X("time:Q", title="simulation time"),
            y=alt.Y("service_ratio:Q", title="service ratio", scale=alt.Scale(domain=[0, 1])),
            color=alt.Color("policy:N", title="policy"),
            tooltip=["policy:N", "event_index:Q", "time:Q", "service_ratio:Q"],
        )
        .properties(width=860, height=460, title="Policy Comparison")
    )
    chart.save(plot_path, ppi=300)

    final = pl.concat(summaries).sort("policy")
    print(final)
    print(f"plot={plot_path}")


if __name__ == "__main__":
    main()
