from __future__ import annotations

from pathlib import Path

import altair as alt
import numpy as np
import polars as pl
import vortex as vx

from qbp_sim import RunOptions, SimulationInputConfig, VirtualSwapPolicyConfig, run_simulation
from qbp_sim.lp import linear


def build_lp_derived_config() -> SimulationInputConfig:
    num_nodes = 6
    generation_capacity = linear.create_cycle_adjacency_matrix(num_nodes, edge_weight=10.0)
    consumption_demand = linear.create_sparse_symmetric_adjacency_matrix(
        num_nodes,
        edge_fraction=0.3,
        max_edge_weight=7.0,
        seed=0,
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

    config = linear.build_lp_solution_simulation_input_config(spec=spec, lp_result=lp_result)
    return config.model_copy(
        update={
            "capacity_headroom": 1.01,
            "virtual_swap_policy": VirtualSwapPolicyConfig(mode="bp"),
        }
    )


def service_ratio_frame(trace_path: Path, sample_every: int = 50) -> pl.DataFrame:
    lf = vx.open(str(trace_path)).to_polars()
    return (
        lf.select("event_index", "time", "event_type", "backlog_total", "inventory_total", "scarcity_total")
        .with_columns(
            demand_arrivals=(pl.col("event_type") == "demand_arrival").cast(pl.Int64).cum_sum(),
            services_completed=(pl.col("event_type") == "physical_service").cast(pl.Int64).cum_sum(),
        )
        .filter((pl.col("event_index") % sample_every) == 0)
        .with_columns(
            service_ratio=pl.when(pl.col("demand_arrivals") > 0)
            .then(pl.col("services_completed") / pl.col("demand_arrivals"))
            .otherwise(0.0),
        )
        .select(
            "event_index",
            "time",
            "demand_arrivals",
            "services_completed",
            "service_ratio",
            "backlog_total",
            "inventory_total",
            "scarcity_total",
        )
        .collect()
    )


def main() -> None:
    output_dir = Path("output/examples/06_lp_derived_config")
    output_dir.mkdir(parents=True, exist_ok=True)
    trace_path = output_dir / "events.vortex"
    plot_path = output_dir / "lp_derived_service_ratio.png"

    config = build_lp_derived_config()
    run = run_simulation(
        config,
        RunOptions(
            until_time=50.0,
            max_events=30_000,
            sample_every=0,
            seed=90,
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
        .properties(width=800, height=420, title="LP-Derived BP Config")
    )
    chart.save(plot_path, ppi=300)

    print(f"trace={trace_path}")
    print(f"plot={plot_path}")
    print(f"num_nodes={config.num_nodes}")
    print(f"final_service_ratio={run.service_ratio:.6f}")
    print(f"nonzero_generation_edges={int(np.count_nonzero(np.triu(np.asarray(config.generation_rates), k=1)))}")
    print(f"nonzero_consumption_edges={int(np.count_nonzero(np.triu(np.asarray(config.consumption_rates), k=1)))}")


if __name__ == "__main__":
    main()
