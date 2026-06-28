from __future__ import annotations

from pathlib import Path

import altair as alt
import numpy as np
import polars as pl

from qbp_sim import RunOptions, SimulationInputConfig, VirtualSwapPolicyConfig, run_simulation
from qbp_sim.lp import linear


def _simulation_config(
    lp_config: SimulationInputConfig,
    *,
    swap_rates: list[float],
    policy_mode: str,
) -> SimulationInputConfig:
    return lp_config.model_copy(
        update={
            "swap_rates": swap_rates,
            "capacity_headroom": 1.01,
            "virtual_swap_policy": VirtualSwapPolicyConfig(mode=policy_mode),
        }
    )


def build_lp_configs() -> list[tuple[str, str, str, SimulationInputConfig]]:
    num_nodes = 8
    generation_capacity = linear.create_cycle_adjacency_matrix(num_nodes, edge_weight=10.0)
    consumption_demand = linear.create_sparse_symmetric_adjacency_matrix(
        num_nodes,
        edge_fraction=0.3,
        max_edge_weight=7.0,
        seed=3,
        min_positive_edges=1,
    )
    generation_capacity = generation_capacity.astype(float) * 10.0
    consumption_demand = consumption_demand.astype(float)
    swap_caps = np.full(num_nodes, 20.0, dtype=float)

    spec = linear.LinearSpec(num_nodes)
    spec.add_generate_constraints(generation_capacity)
    spec.add_consume_constraints(consumption_demand)
    spec.add_swap_capacity_constraints(swap_caps)
    lp_result = spec.solve(objective="min_sum_generate")
    if lp_result.status != 0:
        raise RuntimeError(f"LP solve failed: {lp_result.message}")

    lp_config = linear.build_lp_solution_simulation_input_config(spec=spec, lp_result=lp_result)
    lp_swap_rates = np.asarray(lp_config.swap_rates, dtype=float)
    uniform_swap_rate = float(lp_swap_rates.sum() / num_nodes)
    uniform_swap_rates = [uniform_swap_rate] * num_nodes

    print(f"lp_node_swap_rates={lp_swap_rates.tolist()}")
    print(f"uniform_swap_rate={uniform_swap_rate:.6f}")
    return [
        (
            "bp_lp_node_totals",
            "bp",
            "lp_node_totals",
            _simulation_config(lp_config, swap_rates=lp_swap_rates.tolist(), policy_mode="bp"),
        ),
        (
            "max_min_lp_node_totals",
            "max_min",
            "lp_node_totals",
            _simulation_config(lp_config, swap_rates=lp_swap_rates.tolist(), policy_mode="max_min"),
        ),
        (
            "bp_uniform_lp_total_per_node",
            "bp",
            "uniform_lp_total_per_node",
            _simulation_config(lp_config, swap_rates=uniform_swap_rates, policy_mode="bp"),
        ),
        (
            "max_min_uniform_lp_total_per_node",
            "max_min",
            "uniform_lp_total_per_node",
            _simulation_config(lp_config, swap_rates=uniform_swap_rates, policy_mode="max_min"),
        ),
    ]


def service_ratio_frame(
    trace_path: Path,
    run_label: str,
    policy: str,
    swap_rate_mode: str,
    sample_every: int = 50,
) -> pl.DataFrame:
    lf = pl.scan_parquet(trace_path)
    return (
        lf.select("event_index", "time", "event_type")
        .with_columns(
            demand_arrivals=(pl.col("event_type") == "demand_arrival").cast(pl.Int64).cum_sum(),
            services_completed=(pl.col("event_type") == "physical_service").cast(pl.Int64).cum_sum(),
        )
        .filter((pl.col("event_index") % sample_every) == 0)
        .with_columns(
            run=pl.lit(run_label),
            policy=pl.lit(policy),
            swap_rate_mode=pl.lit(swap_rate_mode),
            service_ratio=pl.when(pl.col("demand_arrivals") > 0)
            .then(pl.col("services_completed") / pl.col("demand_arrivals"))
            .otherwise(0.0),
        )
        .select(
            "run",
            "policy",
            "swap_rate_mode",
            "event_index",
            "time",
            "demand_arrivals",
            "services_completed",
            "service_ratio",
        )
        .collect()
    )


def final_service_summary(trace_path: Path, run_label: str, policy: str, swap_rate_mode: str) -> pl.DataFrame:
    lf = pl.scan_parquet(trace_path)
    return (
        lf.select("event_type")
        .select(
            demand_arrivals=(pl.col("event_type") == "demand_arrival").sum(),
            services_completed=(pl.col("event_type") == "physical_service").sum(),
        )
        .with_columns(
            run=pl.lit(run_label),
            policy=pl.lit(policy),
            swap_rate_mode=pl.lit(swap_rate_mode),
            final_service_ratio=pl.when(pl.col("demand_arrivals") > 0)
            .then(pl.col("services_completed") / pl.col("demand_arrivals"))
            .otherwise(0.0),
        )
        .select("run", "policy", "swap_rate_mode", "demand_arrivals", "services_completed", "final_service_ratio")
        .collect()
    )


def main() -> None:
    output_dir = Path("output/examples/07_uniform_lp_swap_rate")
    output_dir.mkdir(parents=True, exist_ok=True)

    frames: list[pl.DataFrame] = []
    summaries: list[pl.DataFrame] = []
    for offset, (run_label, policy, swap_rate_mode, config) in enumerate(build_lp_configs()):
        trace_path = output_dir / f"{run_label}.parquet"
        run_simulation(
            config,
            RunOptions(
                until_time=50.0,
                max_events=30_000,
                sample_every=0,
                seed=100 + offset,
                trace_path=trace_path,
                trace_format="parquet",
                progress=False,
            ),
        )
        frames.append(service_ratio_frame(trace_path, run_label, policy, swap_rate_mode))
        summaries.append(final_service_summary(trace_path, run_label, policy, swap_rate_mode))

    df = pl.concat(frames)
    plot_path = output_dir / "uniform_lp_swap_rate_service_ratio.png"
    chart = (
        alt.Chart(df)
        .mark_line(strokeWidth=2)
        .encode(
            x=alt.X("time:Q", title="simulation time"),
            y=alt.Y("service_ratio:Q", title="service ratio", scale=alt.Scale(domain=[0, 1])),
            color=alt.Color("policy:N", title="policy"),
            strokeDash=alt.StrokeDash("swap_rate_mode:N", title="swap rate mode"),
            detail="run:N",
            tooltip=[
                "run:N",
                "policy:N",
                "swap_rate_mode:N",
                "event_index:Q",
                "time:Q",
                "demand_arrivals:Q",
                "services_completed:Q",
                "service_ratio:Q",
            ],
        )
        .properties(width=860, height=460, title="BP and Max-Min from LP-Derived Swap Rates")
    )
    chart.save(plot_path, ppi=300)

    final = pl.concat(summaries).sort(["policy", "swap_rate_mode"])
    print(final)
    print(f"plot={plot_path}")


if __name__ == "__main__":
    main()
