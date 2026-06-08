
from __future__ import annotations

import numpy as np
from numba import njit
from numpy.typing import NDArray

from qbp_sim.core.types import Array1D, IntArray1D, IntMatrix

@njit(cache=True)
def _compute_active_virtual_service_rates(
    d: IntMatrix,
    alpha: IntMatrix,
    pair_u: IntArray1D,
    pair_v: IntArray1D,
    service_pair_rates: Array1D,
    out: Array1D,
) -> float:
    total = 0.0
    for idx in range(pair_u.shape[0]):
        x = pair_u[idx]
        y = pair_v[idx]
        rate = 0.0
        if d[x, y] > 0 and d[x, y] >= alpha[x, y]:
            rate = service_pair_rates[idx]
        out[idx] = rate
        total += rate
    return total


@njit(cache=True)
def _compute_active_physical_service_rates(
    q: IntMatrix,
    h_r: IntMatrix,
    pair_u: IntArray1D,
    pair_v: IntArray1D,
    service_pair_rates: Array1D,
    out: Array1D,
) -> float:
    total = 0.0
    for idx in range(pair_u.shape[0]):
        x = pair_u[idx]
        y = pair_v[idx]
        rate = 0.0
        if h_r[x, y] > 0 and q[x, y] > 0:
            rate = service_pair_rates[idx]
        out[idx] = rate
        total += rate
    return total


@njit(cache=True)
def _best_virtual_swap_for_node(
    alpha: IntMatrix,
    node: int,
    swap_node_starts: IntArray1D,
    swap_y: IntArray1D,
    swap_z: IntArray1D,
) -> tuple[int, int]:
    best_weight = 0
    best_idx = -1
    start = swap_node_starts[node]
    stop = swap_node_starts[node + 1]
    for idx in range(start, stop):
        y = swap_y[idx]
        z = swap_z[idx]
        weight = alpha[y, z] - alpha[node, y] - alpha[node, z]
        if weight > best_weight:
            best_weight = weight
            best_idx = idx
    return best_weight, best_idx


@njit(cache=True)
def _best_physical_swap_for_node(
    q: IntMatrix,
    h_mu: IntArray1D,
    node: int,
    swap_node_starts: IntArray1D,
    swap_y: IntArray1D,
    swap_z: IntArray1D,
) -> tuple[int, int]:
    best_deficit = 0
    best_idx = -1
    start = swap_node_starts[node]
    stop = swap_node_starts[node + 1]
    for idx in range(start, stop):
        if h_mu[idx] <= 0:
            continue
        y = swap_y[idx]
        z = swap_z[idx]
        if q[node, y] <= 0 or q[node, z] <= 0:
            continue
        deficit = h_mu[idx]
        if deficit > best_deficit:
            best_deficit = deficit
            best_idx = idx
    return best_deficit, best_idx


@njit(cache=True)
def _physical_swap_is_feasible(
    q: IntMatrix,
    h_mu: IntArray1D,
    swap_idx: int,
    swap_i: IntArray1D,
    swap_y: IntArray1D,
    swap_z: IntArray1D,
) -> bool:
    if swap_idx < 0 or h_mu[swap_idx] <= 0:
        return False
    i = swap_i[swap_idx]
    y = swap_y[swap_idx]
    z = swap_z[swap_idx]
    return q[i, y] > 0 and q[i, z] > 0


@njit(cache=True)
def _best_physical_swap_using_edge(
    q: IntMatrix,
    h_mu: IntArray1D,
    edge_x: int,
    edge_y: int,
    swap_lookup: NDArray[np.int64],
    swap_i: IntArray1D,
    swap_y: IntArray1D,
    swap_z: IntArray1D,
) -> int:
    best_deficit = 0
    best_idx = -1
    n_nodes = swap_lookup.shape[0]
    for other in range(n_nodes):
        if other == edge_x or other == edge_y:
            continue

        idx = swap_lookup[edge_x, edge_y, other]
        if _physical_swap_is_feasible(q, h_mu, idx, swap_i, swap_y, swap_z):
            deficit = h_mu[idx]
            if deficit > best_deficit or (deficit == best_deficit and (best_idx < 0 or idx < best_idx)):
                best_deficit = deficit
                best_idx = idx

        idx = swap_lookup[edge_y, edge_x, other]
        if _physical_swap_is_feasible(q, h_mu, idx, swap_i, swap_y, swap_z):
            deficit = h_mu[idx]
            if deficit > best_deficit or (deficit == best_deficit and (best_idx < 0 or idx < best_idx)):
                best_deficit = deficit
                best_idx = idx

    return best_idx


@njit(cache=True)
def _update_virtual_swap_alpha_change_nonendpoints(
    alpha: IntMatrix,
    a: int,
    b: int,
    delta: int,
    swap_lookup: NDArray[np.int64],
    swap_rates: Array1D,
    best_weight: IntArray1D,
    best_idx: IntArray1D,
    node_rates: Array1D,
    rescan_nodes_out: IntArray1D,
) -> tuple[float, int]:
    total_delta = 0.0
    rescan_count = 0
    n_nodes = best_weight.shape[0]
    for node in range(n_nodes):
        if node == a or node == b:
            continue
        candidate_idx = swap_lookup[node, a, b]
        if candidate_idx < 0:
            continue
        current_idx = best_idx[node]
        new_weight = alpha[a, b] - alpha[node, a] - alpha[node, b]
        if current_idx == candidate_idx:
            if delta < 0:
                rescan_nodes_out[rescan_count] = node
                rescan_count += 1
            else:
                best_weight[node] = new_weight
        elif delta > 0 and new_weight > best_weight[node]:
            old_rate = node_rates[node]
            new_rate = swap_rates[node]
            best_weight[node] = new_weight
            best_idx[node] = candidate_idx
            node_rates[node] = new_rate
            total_delta += new_rate - old_rate
    return total_delta, rescan_count


@njit(cache=True)
def _apply_demand_arrival(d: IntMatrix, x: int, y: int) -> None:
    new_value = d[x, y] + 1
    d[x, y] = new_value
    d[y, x] = new_value


@njit(cache=True)
def _apply_pair_generation(q: IntMatrix, alpha: IntMatrix, x: int, y: int) -> None:
    inventory = q[x, y] + 1
    q[x, y] = inventory
    q[y, x] = inventory

    scarcity = alpha[x, y] - 1
    if scarcity < 0:
        scarcity = 0
    alpha[x, y] = scarcity
    alpha[y, x] = scarcity


@njit(cache=True)
def _apply_virtual_service(d: IntMatrix, alpha: IntMatrix, h_r: IntMatrix, x: int, y: int) -> None:
    backlog = d[x, y] - 1
    scarcity = alpha[x, y] + 1
    pending = h_r[x, y] + 1

    d[x, y] = backlog
    d[y, x] = backlog
    alpha[x, y] = scarcity
    alpha[y, x] = scarcity
    h_r[x, y] = pending
    h_r[y, x] = pending


@njit(cache=True)
def _apply_virtual_swap(alpha: IntMatrix, h_mu: IntArray1D, swap_idx: int, i: int, y: int, z: int) -> None:
    alpha_iy = alpha[i, y] + 1
    alpha_iz = alpha[i, z] + 1
    alpha_yz = alpha[y, z] - 1
    if alpha_yz < 0:
        alpha_yz = 0

    alpha[i, y] = alpha_iy
    alpha[y, i] = alpha_iy
    alpha[i, z] = alpha_iz
    alpha[z, i] = alpha_iz
    alpha[y, z] = alpha_yz
    alpha[z, y] = alpha_yz
    h_mu[swap_idx] += 1


@njit(cache=True)
def _apply_physical_service(q: IntMatrix, h_r: IntMatrix, x: int, y: int) -> None:
    inventory = q[x, y] - 1
    pending = h_r[x, y] - 1

    q[x, y] = inventory
    q[y, x] = inventory
    h_r[x, y] = pending
    h_r[y, x] = pending


@njit(cache=True)
def _apply_physical_swap(q: IntMatrix, h_mu: IntArray1D, swap_idx: int, i: int, y: int, z: int) -> None:
    q_iy = q[i, y] - 1
    q_iz = q[i, z] - 1
    q_yz = q[y, z] + 1

    q[i, y] = q_iy
    q[y, i] = q_iy
    q[i, z] = q_iz
    q[z, i] = q_iz
    q[y, z] = q_yz
    q[z, y] = q_yz
    h_mu[swap_idx] -= 1



@njit(cache=True)
def _sample_index_from_threshold(rates: Array1D, threshold: float) -> int:
    cumulative = 0.0
    for idx, rate in enumerate(rates):
        cumulative += float(rate)
        if threshold <= cumulative:
            return idx
    return len(rates) - 1
