from __future__ import annotations

import numpy as np
from numba import njit
from numpy.typing import NDArray


Array1D = NDArray[np.float64]
IntArray1D = NDArray[np.int64]
IntMatrix = NDArray[np.int64]


@njit(cache=True)
def _build_pair_lookup(n_nodes: int, pair_u: IntArray1D, pair_v: IntArray1D) -> IntMatrix:
    lookup = np.empty((n_nodes, n_nodes), dtype=np.int64)
    lookup[:, :] = -1
    for idx in range(pair_u.shape[0]):
        u = pair_u[idx]
        v = pair_v[idx]
        lookup[u, v] = idx
        lookup[v, u] = idx
    return lookup


@njit(cache=True)
def _build_swap_lookup(
    n_nodes: int,
    swap_i: IntArray1D,
    swap_y: IntArray1D,
    swap_z: IntArray1D,
) -> NDArray[np.int64]:
    lookup = np.empty((n_nodes, n_nodes, n_nodes), dtype=np.int64)
    lookup[:, :, :] = -1
    for idx in range(swap_i.shape[0]):
        i = swap_i[idx]
        y = swap_y[idx]
        z = swap_z[idx]
        lookup[i, y, z] = idx
        lookup[i, z, y] = idx
    return lookup


@njit(cache=True)
def _sample_index(rates: Array1D, total_rate: float) -> int:
    threshold = np.random.random() * total_rate
    cumulative = 0.0
    last_positive = -1
    for idx in range(rates.shape[0]):
        rate = rates[idx]
        if rate > 0.0:
            last_positive = idx
        cumulative += rate
        if threshold < cumulative:
            return idx
    return last_positive


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
def _recompute_virtual_service_pair(
    d: IntMatrix,
    alpha: IntMatrix,
    x: int,
    y: int,
    pair_lookup: IntMatrix,
    service_pair_rates: Array1D,
    active_virtual_service_rates: Array1D,
    active_virtual_service_total: float,
) -> float:
    idx = pair_lookup[x, y]
    old_rate = active_virtual_service_rates[idx]
    new_rate = 0.0
    if d[x, y] > 0 and d[x, y] >= alpha[x, y]:
        new_rate = service_pair_rates[idx]
    active_virtual_service_rates[idx] = new_rate
    return active_virtual_service_total + new_rate - old_rate


@njit(cache=True)
def _recompute_physical_service_pair(
    q: IntMatrix,
    h_r: IntMatrix,
    x: int,
    y: int,
    pair_lookup: IntMatrix,
    service_pair_rates: Array1D,
    active_physical_service_rates: Array1D,
    active_physical_service_total: float,
) -> float:
    idx = pair_lookup[x, y]
    old_rate = active_physical_service_rates[idx]
    new_rate = 0.0
    if h_r[x, y] > 0 and q[x, y] > 0:
        new_rate = service_pair_rates[idx]
    active_physical_service_rates[idx] = new_rate
    return active_physical_service_total + new_rate - old_rate


@njit(cache=True)
def _recompute_virtual_swap_node(
    alpha: IntMatrix,
    node: int,
    swap_rates: Array1D,
    swap_node_starts: IntArray1D,
    swap_y: IntArray1D,
    swap_z: IntArray1D,
    virtual_swap_best_weight: IntArray1D,
    virtual_swap_best_idx: IntArray1D,
    virtual_swap_node_rates: Array1D,
    active_virtual_swap_total: float,
) -> float:
    old_rate = virtual_swap_node_rates[node]
    best_weight, best_idx = _best_virtual_swap_for_node(alpha, node, swap_node_starts, swap_y, swap_z)
    virtual_swap_best_weight[node] = best_weight
    virtual_swap_best_idx[node] = best_idx
    new_rate = swap_rates[node] if best_idx >= 0 else 0.0
    virtual_swap_node_rates[node] = new_rate
    return active_virtual_swap_total + new_rate - old_rate


@njit(cache=True)
def _recompute_physical_swap_node(
    q: IntMatrix,
    h_mu: IntArray1D,
    node: int,
    swap_rates: Array1D,
    swap_node_starts: IntArray1D,
    swap_y: IntArray1D,
    swap_z: IntArray1D,
    physical_swap_best_deficit: IntArray1D,
    physical_swap_best_idx: IntArray1D,
    physical_swap_node_rates: Array1D,
    active_physical_swap_total: float,
) -> float:
    old_rate = physical_swap_node_rates[node]
    best_deficit, best_idx = _best_physical_swap_for_node(q, h_mu, node, swap_node_starts, swap_y, swap_z)
    physical_swap_best_deficit[node] = best_deficit
    physical_swap_best_idx[node] = best_idx
    new_rate = swap_rates[node] if best_idx >= 0 else 0.0
    physical_swap_node_rates[node] = new_rate
    return active_physical_swap_total + new_rate - old_rate


@njit(cache=True)
def _update_virtual_swap_for_alpha_change(
    alpha: IntMatrix,
    a: int,
    b: int,
    delta: int,
    swap_lookup: NDArray[np.int64],
    swap_rates: Array1D,
    swap_node_starts: IntArray1D,
    swap_y: IntArray1D,
    swap_z: IntArray1D,
    virtual_swap_best_weight: IntArray1D,
    virtual_swap_best_idx: IntArray1D,
    virtual_swap_node_rates: Array1D,
    active_virtual_swap_total: float,
) -> float:
    active_virtual_swap_total = _recompute_virtual_swap_node(
        alpha,
        a,
        swap_rates,
        swap_node_starts,
        swap_y,
        swap_z,
        virtual_swap_best_weight,
        virtual_swap_best_idx,
        virtual_swap_node_rates,
        active_virtual_swap_total,
    )
    active_virtual_swap_total = _recompute_virtual_swap_node(
        alpha,
        b,
        swap_rates,
        swap_node_starts,
        swap_y,
        swap_z,
        virtual_swap_best_weight,
        virtual_swap_best_idx,
        virtual_swap_node_rates,
        active_virtual_swap_total,
    )

    n_nodes = virtual_swap_best_weight.shape[0]
    for node in range(n_nodes):
        if node == a or node == b:
            continue
        candidate_idx = swap_lookup[node, a, b]
        if candidate_idx < 0:
            continue
        current_idx = virtual_swap_best_idx[node]
        new_weight = alpha[a, b] - alpha[node, a] - alpha[node, b]
        if current_idx == candidate_idx:
            if delta < 0:
                active_virtual_swap_total = _recompute_virtual_swap_node(
                    alpha,
                    node,
                    swap_rates,
                    swap_node_starts,
                    swap_y,
                    swap_z,
                    virtual_swap_best_weight,
                    virtual_swap_best_idx,
                    virtual_swap_node_rates,
                    active_virtual_swap_total,
                )
            else:
                virtual_swap_best_weight[node] = new_weight
        elif delta > 0 and new_weight > virtual_swap_best_weight[node]:
            old_rate = virtual_swap_node_rates[node]
            new_rate = swap_rates[node]
            virtual_swap_best_weight[node] = new_weight
            virtual_swap_best_idx[node] = candidate_idx
            virtual_swap_node_rates[node] = new_rate
            active_virtual_swap_total += new_rate - old_rate
    return active_virtual_swap_total


@njit(cache=True)
def _record_sample(
    sample_times: Array1D,
    backlog_samples: IntArray1D,
    inventory_samples: IntArray1D,
    alpha_samples: IntArray1D,
    sample_count: int,
    time_value: float,
    total_virtual_backlog_count: int,
    total_service_deficit_count: int,
    total_inventory_count: int,
    total_scarcity_count: int,
) -> int:
    sample_times[sample_count] = time_value
    backlog_samples[sample_count] = total_virtual_backlog_count + total_service_deficit_count
    inventory_samples[sample_count] = total_inventory_count
    alpha_samples[sample_count] = total_scarcity_count
    return sample_count + 1


@njit(cache=True)
def run_global_event_loop(
    q: IntMatrix,
    d: IntMatrix,
    alpha: IntMatrix,
    h_r: IntMatrix,
    h_mu: IntArray1D,
    current_time: float,
    events_processed: int,
    demand_arrivals: int,
    pair_generations: int,
    virtual_service_requests: int,
    virtual_swap_requests: int,
    services_completed: int,
    swaps_completed: int,
    total_virtual_backlog_count: int,
    total_service_deficit_count: int,
    total_swap_deficit_count: int,
    total_inventory_count: int,
    total_scarcity_count: int,
    generation_pair_rates: Array1D,
    demand_pair_rates: Array1D,
    service_pair_rates: Array1D,
    swap_rates: Array1D,
    pair_u: IntArray1D,
    pair_v: IntArray1D,
    swap_i: IntArray1D,
    swap_y: IntArray1D,
    swap_z: IntArray1D,
    swap_node_starts: IntArray1D,
    active_virtual_service_rates: Array1D,
    active_physical_service_rates: Array1D,
    virtual_swap_best_weight: IntArray1D,
    virtual_swap_best_idx: IntArray1D,
    virtual_swap_node_rates: Array1D,
    physical_swap_best_deficit: IntArray1D,
    physical_swap_best_idx: IntArray1D,
    physical_swap_node_rates: Array1D,
    active_virtual_service_total: float,
    active_physical_service_total: float,
    active_virtual_swap_total: float,
    active_physical_swap_total: float,
    until_time: float,
    max_events: int,
    sample_every: int,
    rng_seed: int,
) -> tuple[
    float,
    int,
    int,
    int,
    int,
    int,
    int,
    int,
    int,
    int,
    int,
    float,
    float,
    float,
    float,
    Array1D,
    IntArray1D,
    IntArray1D,
    IntArray1D,
    int,
    int,
    float,
]:
    if rng_seed >= 0:
        np.random.seed(rng_seed)

    n_nodes = swap_rates.shape[0]
    n_pairs = pair_u.shape[0]
    pair_lookup = _build_pair_lookup(n_nodes, pair_u, pair_v)
    swap_lookup = _build_swap_lookup(n_nodes, swap_i, swap_y, swap_z)

    generation_total = 0.0
    demand_total = 0.0
    for idx in range(n_pairs):
        generation_total += generation_pair_rates[idx]
        demand_total += demand_pair_rates[idx]

    remaining_events = 0
    if max_events >= 0 and max_events > events_processed:
        remaining_events = max_events - events_processed
    sample_capacity = 1
    if sample_every > 0 and remaining_events > 0:
        sample_capacity = remaining_events // sample_every + 2
    sample_times = np.empty(sample_capacity, dtype=np.float64)
    backlog_samples = np.empty(sample_capacity, dtype=np.int64)
    inventory_samples = np.empty(sample_capacity, dtype=np.int64)
    alpha_samples = np.empty(sample_capacity, dtype=np.int64)
    sample_count = 0
    last_sample_event = -1
    last_sample_time = -1.0

    while current_time < until_time and (max_events < 0 or events_processed < max_events):
        total_rate = (
            demand_total
            + generation_total
            + active_virtual_service_total
            + active_virtual_swap_total
            + active_physical_service_total
            + active_physical_swap_total
        )
        if total_rate <= 0.0:
            break

        next_time = current_time + np.random.exponential(1.0 / total_rate)
        if next_time > until_time:
            current_time = until_time
            break
        current_time = next_time
        selector = np.random.random() * total_rate
        events_processed += 1

        if selector < demand_total:
            idx = _sample_index(demand_pair_rates, demand_total)
            x = pair_u[idx]
            y = pair_v[idx]
            new_value = d[x, y] + 1
            d[x, y] = new_value
            d[y, x] = new_value
            demand_arrivals += 1
            total_virtual_backlog_count += 1
            active_virtual_service_total = _recompute_virtual_service_pair(
                d,
                alpha,
                x,
                y,
                pair_lookup,
                service_pair_rates,
                active_virtual_service_rates,
                active_virtual_service_total,
            )

        elif selector - demand_total < generation_total:
            idx = _sample_index(generation_pair_rates, generation_total)
            x = pair_u[idx]
            y = pair_v[idx]
            q_value = q[x, y] + 1
            q[x, y] = q_value
            q[y, x] = q_value
            old_scarcity = alpha[x, y]
            alpha_value = old_scarcity - 1
            if alpha_value < 0:
                alpha_value = 0
            alpha[x, y] = alpha_value
            alpha[y, x] = alpha_value
            pair_generations += 1
            total_inventory_count += 1
            if old_scarcity > 0:
                total_scarcity_count -= 1
            active_physical_service_total = _recompute_physical_service_pair(
                q,
                h_r,
                x,
                y,
                pair_lookup,
                service_pair_rates,
                active_physical_service_rates,
                active_physical_service_total,
            )
            active_physical_swap_total = _recompute_physical_swap_node(
                q,
                h_mu,
                x,
                swap_rates,
                swap_node_starts,
                swap_y,
                swap_z,
                physical_swap_best_deficit,
                physical_swap_best_idx,
                physical_swap_node_rates,
                active_physical_swap_total,
            )
            active_physical_swap_total = _recompute_physical_swap_node(
                q,
                h_mu,
                y,
                swap_rates,
                swap_node_starts,
                swap_y,
                swap_z,
                physical_swap_best_deficit,
                physical_swap_best_idx,
                physical_swap_node_rates,
                active_physical_swap_total,
            )
            active_virtual_service_total = _recompute_virtual_service_pair(
                d,
                alpha,
                x,
                y,
                pair_lookup,
                service_pair_rates,
                active_virtual_service_rates,
                active_virtual_service_total,
            )
            active_virtual_swap_total = _update_virtual_swap_for_alpha_change(
                alpha,
                x,
                y,
                -1,
                swap_lookup,
                swap_rates,
                swap_node_starts,
                swap_y,
                swap_z,
                virtual_swap_best_weight,
                virtual_swap_best_idx,
                virtual_swap_node_rates,
                active_virtual_swap_total,
            )

        elif selector - demand_total - generation_total < active_virtual_service_total:
            idx = _sample_index(active_virtual_service_rates, active_virtual_service_total)
            x = pair_u[idx]
            y = pair_v[idx]
            d_value = d[x, y] - 1
            alpha_value = alpha[x, y] + 1
            hr_value = h_r[x, y] + 1
            d[x, y] = d_value
            d[y, x] = d_value
            alpha[x, y] = alpha_value
            alpha[y, x] = alpha_value
            h_r[x, y] = hr_value
            h_r[y, x] = hr_value
            virtual_service_requests += 1
            total_virtual_backlog_count -= 1
            total_service_deficit_count += 1
            total_scarcity_count += 1
            active_physical_service_total = _recompute_physical_service_pair(
                q,
                h_r,
                x,
                y,
                pair_lookup,
                service_pair_rates,
                active_physical_service_rates,
                active_physical_service_total,
            )
            active_virtual_service_total = _recompute_virtual_service_pair(
                d,
                alpha,
                x,
                y,
                pair_lookup,
                service_pair_rates,
                active_virtual_service_rates,
                active_virtual_service_total,
            )
            active_virtual_swap_total = _update_virtual_swap_for_alpha_change(
                alpha,
                x,
                y,
                1,
                swap_lookup,
                swap_rates,
                swap_node_starts,
                swap_y,
                swap_z,
                virtual_swap_best_weight,
                virtual_swap_best_idx,
                virtual_swap_node_rates,
                active_virtual_swap_total,
            )

        elif selector - demand_total - generation_total - active_virtual_service_total < active_virtual_swap_total:
            node = _sample_index(virtual_swap_node_rates, active_virtual_swap_total)
            idx = virtual_swap_best_idx[node]
            i = swap_i[idx]
            y = swap_y[idx]
            z = swap_z[idx]
            old_output_scarcity = alpha[y, z]
            alpha_iy = alpha[i, y] + 1
            alpha_iz = alpha[i, z] + 1
            alpha_yz = old_output_scarcity - 1
            if alpha_yz < 0:
                alpha_yz = 0
            alpha[i, y] = alpha_iy
            alpha[y, i] = alpha_iy
            alpha[i, z] = alpha_iz
            alpha[z, i] = alpha_iz
            alpha[y, z] = alpha_yz
            alpha[z, y] = alpha_yz
            h_mu[idx] += 1
            virtual_swap_requests += 1
            total_swap_deficit_count += 1
            total_scarcity_count += 2
            if old_output_scarcity > 0:
                total_scarcity_count -= 1
            active_physical_swap_total = _recompute_physical_swap_node(
                q,
                h_mu,
                i,
                swap_rates,
                swap_node_starts,
                swap_y,
                swap_z,
                physical_swap_best_deficit,
                physical_swap_best_idx,
                physical_swap_node_rates,
                active_physical_swap_total,
            )
            active_virtual_service_total = _recompute_virtual_service_pair(
                d,
                alpha,
                i,
                y,
                pair_lookup,
                service_pair_rates,
                active_virtual_service_rates,
                active_virtual_service_total,
            )
            active_virtual_swap_total = _update_virtual_swap_for_alpha_change(
                alpha,
                i,
                y,
                1,
                swap_lookup,
                swap_rates,
                swap_node_starts,
                swap_y,
                swap_z,
                virtual_swap_best_weight,
                virtual_swap_best_idx,
                virtual_swap_node_rates,
                active_virtual_swap_total,
            )
            active_virtual_service_total = _recompute_virtual_service_pair(
                d,
                alpha,
                i,
                z,
                pair_lookup,
                service_pair_rates,
                active_virtual_service_rates,
                active_virtual_service_total,
            )
            active_virtual_swap_total = _update_virtual_swap_for_alpha_change(
                alpha,
                i,
                z,
                1,
                swap_lookup,
                swap_rates,
                swap_node_starts,
                swap_y,
                swap_z,
                virtual_swap_best_weight,
                virtual_swap_best_idx,
                virtual_swap_node_rates,
                active_virtual_swap_total,
            )
            active_virtual_service_total = _recompute_virtual_service_pair(
                d,
                alpha,
                y,
                z,
                pair_lookup,
                service_pair_rates,
                active_virtual_service_rates,
                active_virtual_service_total,
            )
            active_virtual_swap_total = _update_virtual_swap_for_alpha_change(
                alpha,
                y,
                z,
                -1,
                swap_lookup,
                swap_rates,
                swap_node_starts,
                swap_y,
                swap_z,
                virtual_swap_best_weight,
                virtual_swap_best_idx,
                virtual_swap_node_rates,
                active_virtual_swap_total,
            )

        elif (
            selector
            - demand_total
            - generation_total
            - active_virtual_service_total
            - active_virtual_swap_total
            < active_physical_service_total
        ):
            idx = _sample_index(active_physical_service_rates, active_physical_service_total)
            x = pair_u[idx]
            y = pair_v[idx]
            q_value = q[x, y] - 1
            hr_value = h_r[x, y] - 1
            q[x, y] = q_value
            q[y, x] = q_value
            h_r[x, y] = hr_value
            h_r[y, x] = hr_value
            services_completed += 1
            total_inventory_count -= 1
            total_service_deficit_count -= 1
            active_physical_service_total = _recompute_physical_service_pair(
                q,
                h_r,
                x,
                y,
                pair_lookup,
                service_pair_rates,
                active_physical_service_rates,
                active_physical_service_total,
            )
            active_physical_swap_total = _recompute_physical_swap_node(
                q,
                h_mu,
                x,
                swap_rates,
                swap_node_starts,
                swap_y,
                swap_z,
                physical_swap_best_deficit,
                physical_swap_best_idx,
                physical_swap_node_rates,
                active_physical_swap_total,
            )
            active_physical_swap_total = _recompute_physical_swap_node(
                q,
                h_mu,
                y,
                swap_rates,
                swap_node_starts,
                swap_y,
                swap_z,
                physical_swap_best_deficit,
                physical_swap_best_idx,
                physical_swap_node_rates,
                active_physical_swap_total,
            )

        else:
            node = _sample_index(physical_swap_node_rates, active_physical_swap_total)
            idx = physical_swap_best_idx[node]
            i = swap_i[idx]
            y = swap_y[idx]
            z = swap_z[idx]
            q_iy = q[i, y] - 1
            q_iz = q[i, z] - 1
            q_yz = q[y, z] + 1
            q[i, y] = q_iy
            q[y, i] = q_iy
            q[i, z] = q_iz
            q[z, i] = q_iz
            q[y, z] = q_yz
            q[z, y] = q_yz
            h_mu[idx] -= 1
            swaps_completed += 1
            total_inventory_count -= 1
            total_swap_deficit_count -= 1
            active_physical_service_total = _recompute_physical_service_pair(
                q,
                h_r,
                i,
                y,
                pair_lookup,
                service_pair_rates,
                active_physical_service_rates,
                active_physical_service_total,
            )
            active_physical_service_total = _recompute_physical_service_pair(
                q,
                h_r,
                i,
                z,
                pair_lookup,
                service_pair_rates,
                active_physical_service_rates,
                active_physical_service_total,
            )
            active_physical_service_total = _recompute_physical_service_pair(
                q,
                h_r,
                y,
                z,
                pair_lookup,
                service_pair_rates,
                active_physical_service_rates,
                active_physical_service_total,
            )
            active_physical_swap_total = _recompute_physical_swap_node(
                q,
                h_mu,
                i,
                swap_rates,
                swap_node_starts,
                swap_y,
                swap_z,
                physical_swap_best_deficit,
                physical_swap_best_idx,
                physical_swap_node_rates,
                active_physical_swap_total,
            )
            active_physical_swap_total = _recompute_physical_swap_node(
                q,
                h_mu,
                y,
                swap_rates,
                swap_node_starts,
                swap_y,
                swap_z,
                physical_swap_best_deficit,
                physical_swap_best_idx,
                physical_swap_node_rates,
                active_physical_swap_total,
            )
            active_physical_swap_total = _recompute_physical_swap_node(
                q,
                h_mu,
                z,
                swap_rates,
                swap_node_starts,
                swap_y,
                swap_z,
                physical_swap_best_deficit,
                physical_swap_best_idx,
                physical_swap_node_rates,
                active_physical_swap_total,
            )

        if sample_every > 0 and events_processed % sample_every == 0:
            sample_count = _record_sample(
                sample_times,
                backlog_samples,
                inventory_samples,
                alpha_samples,
                sample_count,
                current_time,
                total_virtual_backlog_count,
                total_service_deficit_count,
                total_inventory_count,
                total_scarcity_count,
            )
            last_sample_event = events_processed
            last_sample_time = current_time

    return (
        current_time,
        events_processed,
        demand_arrivals,
        pair_generations,
        virtual_service_requests,
        virtual_swap_requests,
        services_completed,
        swaps_completed,
        total_virtual_backlog_count,
        total_service_deficit_count,
        total_swap_deficit_count,
        total_inventory_count,
        total_scarcity_count,
        active_virtual_service_total,
        active_physical_service_total,
        active_virtual_swap_total,
        active_physical_swap_total,
        sample_times,
        backlog_samples,
        inventory_samples,
        alpha_samples,
        sample_count,
        last_sample_event,
        last_sample_time,
    )
