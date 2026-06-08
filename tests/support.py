
from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest

from qbp_sim.config import SimulationInputConfig
from qbp_sim.core.types import GillespieQBPConfig, VirtualSwapPolicy
from qbp_sim.examples import build_four_node_counterexample
from qbp_sim.lp import linear as linear_module
from qbp_sim.simulator import GillespieQBPSimulator

RUN_GATED_TESTS = os.environ.get("QBP_SIM_RUN_GATED_TESTS") == "1"
gated_test = pytest.mark.skipif(
    not RUN_GATED_TESTS,
    reason="set QBP_SIM_RUN_GATED_TESTS=1 to run gated stochastic simulator checks",
)


def _limited_policy(k: int, memory: int) -> VirtualSwapPolicy:
    return VirtualSwapPolicy(mode="power_of_k_memory", k=k, memory=memory)


def _collect_event_records(sim: GillespieQBPSimulator, count: int) -> list[dict[str, int | float | str]]:
    records: list[dict[str, int | float | str]] = []
    for _ in range(count):
        event = sim.step()
        if event is None:
            break
        records.append(event.to_dict())
    return records


def _upper_triangle_sum(matrix: np.ndarray) -> int:
    return int(np.triu(matrix, k=1).sum())


def _assert_state_invariants(sim: GillespieQBPSimulator) -> None:
    state = sim.state
    for matrix in (state.q, state.d, state.alpha, state.h_r):
        assert np.array_equal(matrix, matrix.T)
        assert np.all(np.diag(matrix) == 0)
        assert np.all(matrix >= 0)
    assert np.all(state.h_mu >= 0)
    assert state.total_inventory == _upper_triangle_sum(state.q)
    assert state.total_virtual_backlog == _upper_triangle_sum(state.d)
    assert state.total_scarcity == _upper_triangle_sum(state.alpha)
    assert state.total_service_deficit == _upper_triangle_sum(state.h_r)
    assert state.total_swap_deficit == int(state.h_mu.sum())
    assert state.total_backlog == state.total_virtual_backlog + state.total_service_deficit
    if state.demand_arrivals > 0:
        assert np.isclose(state.service_ratio, state.services_completed / state.demand_arrivals)
        assert 0.0 <= state.service_ratio <= 1.0


def _cycle_runtime_config(num_nodes: int, seed: int, tmp_path: Path) -> GillespieQBPConfig:
    config_path = tmp_path / f"cycle{num_nodes}_lp_config.json"
    linear_module.single_run_topology(
        topology="cycle",
        num_nodes=num_nodes,
        seed=seed,
        output_mode="simulation-config",
        simulation_config_output_path=str(config_path),
    )
    return SimulationInputConfig.from_json_file(config_path).to_runtime_config()


def _single_edge_generation_config(
    *,
    instant_service_fulfillment: bool = False,
    instant_swap_fulfillment: bool = False,
) -> GillespieQBPConfig:
    generation_rates = np.array([[0.0, 1.0], [1.0, 0.0]], dtype=np.float64)
    demand_rates = np.zeros((2, 2), dtype=np.float64)
    service_rates = np.zeros((2, 2), dtype=np.float64)
    swap_rates = np.zeros(2, dtype=np.float64)
    return GillespieQBPConfig(
        generation_rates=generation_rates,
        demand_rates=demand_rates,
        swap_rates=swap_rates,
        service_rates=service_rates,
        instant_service_fulfillment=instant_service_fulfillment,
        instant_swap_fulfillment=instant_swap_fulfillment,
    )


def _swap_index(sim: GillespieQBPSimulator, i: int, y: int, z: int) -> int:
    matches = np.flatnonzero((sim.swap_i == i) & (sim.swap_y == y) & (sim.swap_z == z))
    assert len(matches) == 1
    return int(matches[0])


def _pending_service_matrix(n_nodes: int, x: int, y: int) -> np.ndarray:
    h_r = np.zeros((n_nodes, n_nodes), dtype=np.int64)
    h_r[x, y] = 1
    h_r[y, x] = 1
    return h_r
