from __future__ import annotations

import numpy as np

from qbp_sim.simulator import GillespieQBPConfig


def build_four_node_counterexample() -> GillespieQBPConfig:
    """Continuous-time analogue of the four-node counterexample in the paper."""
    n_nodes = 4
    generation_rates = np.zeros((n_nodes, n_nodes), dtype=np.float64)
    demand_rates = np.zeros((n_nodes, n_nodes), dtype=np.float64)
    service_rates = np.ones((n_nodes, n_nodes), dtype=np.float64)
    swap_rates = np.zeros(n_nodes, dtype=np.float64)

    def connect(a: int, b: int, rate: float) -> None:
        generation_rates[a, b] = rate
        generation_rates[b, a] = rate

    connect(0, 1, 1.0)  # A-B
    connect(0, 2, 1.0)  # A-C
    connect(1, 3, 1.0)  # B-D
    connect(2, 3, 1.0)  # C-D

    demand_rates[0, 3] = 2.0
    demand_rates[3, 0] = 2.0

    swap_rates[1] = 1.0  # B
    swap_rates[2] = 1.0  # C

    np.fill_diagonal(service_rates, 0.0)

    return GillespieQBPConfig(
        generation_rates=generation_rates,
        demand_rates=demand_rates,
        swap_rates=swap_rates,
        service_rates=service_rates,
    )
