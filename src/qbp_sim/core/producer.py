
from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from qbp_sim.core.indexing import (
    _build_pair_lookup,
    _build_swap_lookup,
    _build_swap_node_starts,
    _matrix_to_pair_vector,
)
from qbp_sim.core.kernels import (
    _compute_active_max_min_swap_rates,
    _compute_active_service_request_rates,
    _best_physical_swap_for_node,
    _best_virtual_swap_for_node,
    _compute_active_physical_service_rates,
    _compute_active_virtual_service_rates,
    _sample_index_from_threshold,
    _update_virtual_swap_alpha_change_nonendpoints,
)
from qbp_sim.core.types import (
    VIRTUAL_SWAP_POLICY_LIMITED_INFO_BP,
    VIRTUAL_SWAP_POLICY_LIMITED_INFO_MAX_MIN,
    VIRTUAL_SWAP_POLICY_MAX_MIN,
    Array1D,
    GillespieQBPConfig,
    IntArray1D,
    QBPState,
    _require,
)
from qbp_sim.io.events import QBPEvent


def _sample_index(rng: np.random.Generator, rates: Array1D, total_rate: float) -> int:
    threshold = float(rng.random() * total_rate)
    return _sample_index_from_threshold(rates, threshold)


class GillespieQBPEventProducer:
    """Sample virtual requests and physical realizations from current state."""

    def __init__(
        self,
        config: GillespieQBPConfig,
        pair_u: IntArray1D,
        pair_v: IntArray1D,
        swap_i: IntArray1D,
        swap_y: IntArray1D,
        swap_z: IntArray1D,
        seed: int | None = None,
    ) -> None:
        self.config = config
        self.n_nodes = config.swap_rates.shape[0]
        self.pair_u = pair_u
        self.pair_v = pair_v
        self.swap_i = swap_i
        self.swap_y = swap_y
        self.swap_z = swap_z
        self.pair_lookup = _build_pair_lookup(self.n_nodes, pair_u, pair_v)
        self.swap_node_starts = _build_swap_node_starts(self.n_nodes, swap_i)
        self.swap_lookup = _build_swap_lookup(self.n_nodes, swap_i, swap_y, swap_z)
        self.demand_pair_rates = _matrix_to_pair_vector(config.demand_rates, pair_u, pair_v)
        self.generation_pair_rates = _matrix_to_pair_vector(config.generation_rates, pair_u, pair_v)
        self.service_pair_rates = _matrix_to_pair_vector(config.service_rates, pair_u, pair_v)
        self.demand_total = float(self.demand_pair_rates.sum())
        self.generation_total = float(self.generation_pair_rates.sum())
        self.active_virtual_service_rates = np.zeros_like(self.service_pair_rates)
        self.active_physical_service_rates = np.zeros_like(self.service_pair_rates)
        self.active_virtual_service_total = 0.0
        self.active_physical_service_total = 0.0
        self.virtual_swap_best_weight = np.zeros(self.n_nodes, dtype=np.int64)
        self.virtual_swap_best_idx = np.full(self.n_nodes, -1, dtype=np.int64)
        self.virtual_swap_node_rates = np.zeros(self.n_nodes, dtype=np.float64)
        self.active_virtual_swap_total = 0.0
        self.virtual_swap_rescan_nodes = np.empty(self.n_nodes, dtype=np.int64)
        self.virtual_swap_memory_idx = np.empty((0, 0), dtype=np.int64)
        self.virtual_swap_memory_weight = np.empty((0, 0), dtype=np.int64)
        if self._uses_limited_virtual_swap_policy():
            memory_size = int(config.virtual_swap_policy.memory)
            self.virtual_swap_memory_idx = np.full((self.n_nodes, memory_size), -1, dtype=np.int64)
            self.virtual_swap_memory_weight = np.zeros((self.n_nodes, memory_size), dtype=np.int64)
        self.physical_swap_best_deficit = np.zeros(self.n_nodes, dtype=np.int64)
        self.physical_swap_best_idx = np.full(self.n_nodes, -1, dtype=np.int64)
        self.physical_swap_node_rates = np.zeros(self.n_nodes, dtype=np.float64)
        self.active_physical_swap_total = 0.0
        self.max_min_swap_best_output = np.full(self.n_nodes, -1, dtype=np.int64)
        self.max_min_swap_best_idx = np.full(self.n_nodes, -1, dtype=np.int64)
        self.max_min_swap_node_rates = np.zeros(self.n_nodes, dtype=np.float64)
        self.active_max_min_swap_total = 0.0
        self.max_min_swap_memory_idx = np.empty((0, 0), dtype=np.int64)
        self.max_min_swap_memory_output = np.empty((0, 0), dtype=np.int64)
        if self._uses_limited_max_min_policy():
            memory_size = int(config.virtual_swap_policy.memory)
            self.max_min_swap_memory_idx = np.full((self.n_nodes, memory_size), -1, dtype=np.int64)
            self.max_min_swap_memory_output = np.full((self.n_nodes, memory_size), -1, dtype=np.int64)
        self.rng = np.random.default_rng(seed)
        policy_seed = None if seed is None else int(seed) + 1_000_003
        self.policy_rng = np.random.default_rng(policy_seed)

    def _uses_limited_virtual_swap_policy(self) -> bool:
        return self.config.virtual_swap_policy.mode == VIRTUAL_SWAP_POLICY_LIMITED_INFO_BP

    def _uses_max_min_policy(self) -> bool:
        return self.config.virtual_swap_policy.mode in {
            VIRTUAL_SWAP_POLICY_MAX_MIN,
            VIRTUAL_SWAP_POLICY_LIMITED_INFO_MAX_MIN,
        }

    def _uses_limited_max_min_policy(self) -> bool:
        return self.config.virtual_swap_policy.mode == VIRTUAL_SWAP_POLICY_LIMITED_INFO_MAX_MIN

    def initialize(self, state: QBPState) -> None:
        if self._uses_max_min_policy():
            self.active_virtual_service_total = _compute_active_service_request_rates(
                state.d,
                self.pair_u,
                self.pair_v,
                self.service_pair_rates,
                self.active_virtual_service_rates,
            )
        else:
            self.active_virtual_service_total = _compute_active_virtual_service_rates(
                state.d,
                state.alpha,
                self.pair_u,
                self.pair_v,
                self.service_pair_rates,
                self.active_virtual_service_rates,
            )
        self.active_physical_service_total = _compute_active_physical_service_rates(
            state.q,
            state.h_r,
            self.pair_u,
            self.pair_v,
            self.service_pair_rates,
            self.active_physical_service_rates,
        )
        self.active_virtual_swap_total = 0.0
        self.active_physical_swap_total = 0.0
        self.active_max_min_swap_total = 0.0
        if self._uses_limited_max_min_policy():
            for node in range(self.n_nodes):
                self._initialize_limited_max_min_swap_node(node)
        elif self._uses_max_min_policy():
            self.active_max_min_swap_total = _compute_active_max_min_swap_rates(
                state.q,
                self.swap_node_starts,
                self.swap_y,
                self.swap_z,
                self.config.swap_rates,
                self.max_min_swap_best_output,
                self.max_min_swap_best_idx,
                self.max_min_swap_node_rates,
            )
        else:
            for node in range(self.n_nodes):
                if self._uses_limited_virtual_swap_policy():
                    self._initialize_limited_virtual_swap_node(node)
                else:
                    self._recompute_virtual_swap_node(state, node)
                self._recompute_physical_swap_node(state, node)

    def _recompute_virtual_service_pair(self, state: QBPState, x: int, y: int) -> None:
        idx = int(self.pair_lookup[x, y])
        old_rate = float(self.active_virtual_service_rates[idx])
        new_rate = 0.0
        if state.d[x, y] > 0:
            if self._uses_max_min_policy() or state.d[x, y] >= state.alpha[x, y]:
                new_rate = float(self.service_pair_rates[idx])
        self.active_virtual_service_rates[idx] = new_rate
        self.active_virtual_service_total += new_rate - old_rate

    def _recompute_physical_service_pair(self, state: QBPState, x: int, y: int) -> None:
        idx = int(self.pair_lookup[x, y])
        old_rate = float(self.active_physical_service_rates[idx])
        new_rate = 0.0
        if state.h_r[x, y] > 0 and state.q[x, y] > 0:
            new_rate = float(self.service_pair_rates[idx])
        self.active_physical_service_rates[idx] = new_rate
        self.active_physical_service_total += new_rate - old_rate

    def _recompute_virtual_swap_node(self, state: QBPState, node: int) -> None:
        old_rate = float(self.virtual_swap_node_rates[node])
        best_weight, best_idx = _best_virtual_swap_for_node(
            state.alpha,
            node,
            self.swap_node_starts,
            self.swap_y,
            self.swap_z,
        )
        self.virtual_swap_best_weight[node] = best_weight
        self.virtual_swap_best_idx[node] = best_idx
        new_rate = float(self.config.swap_rates[node]) if best_idx >= 0 else 0.0
        self.virtual_swap_node_rates[node] = new_rate
        self.active_virtual_swap_total += new_rate - old_rate

    def _initialize_limited_virtual_swap_node(self, node: int) -> None:
        old_rate = float(self.virtual_swap_node_rates[node])
        self.virtual_swap_best_weight[node] = 0
        self.virtual_swap_best_idx[node] = -1
        has_candidates = self.swap_node_starts[node] < self.swap_node_starts[node + 1]
        new_rate = float(self.config.swap_rates[node]) if has_candidates else 0.0
        self.virtual_swap_node_rates[node] = new_rate
        self.active_virtual_swap_total += new_rate - old_rate

    def _virtual_swap_weight(self, state: QBPState, swap_idx: int) -> int:
        i = int(self.swap_i[swap_idx])
        y = int(self.swap_y[swap_idx])
        z = int(self.swap_z[swap_idx])
        return int(state.alpha[y, z] - state.alpha[i, y] - state.alpha[i, z])

    def _sample_limited_virtual_swap_candidates(self, node: int) -> NDArray[np.int64]:
        start = int(self.swap_node_starts[node])
        stop = int(self.swap_node_starts[node + 1])
        count = stop - start
        if count <= 0:
            return np.empty(0, dtype=np.int64)

        k = int(self.config.virtual_swap_policy.k)
        if k >= count:
            return np.arange(start, stop, dtype=np.int64)

        offsets = self.policy_rng.choice(count, size=k, replace=False)
        return np.asarray(offsets, dtype=np.int64) + start

    def _sample_limited_max_min_swap_candidates(self, node: int) -> NDArray[np.int64]:
        return self._sample_limited_virtual_swap_candidates(node)

    def _refresh_limited_virtual_swap_node(self, state: QBPState, node: int) -> None:
        old_rate = float(self.virtual_swap_node_rates[node])
        memory_size = int(self.config.virtual_swap_policy.memory)
        candidates: dict[int, int] = {}

        for slot in range(memory_size):
            remembered_idx = int(self.virtual_swap_memory_idx[node, slot])
            if remembered_idx >= 0:
                candidates[remembered_idx] = self._virtual_swap_weight(state, remembered_idx)

        for sampled_idx in self._sample_limited_virtual_swap_candidates(node):
            idx = int(sampled_idx)
            candidates[idx] = self._virtual_swap_weight(state, idx)

        ranked = sorted(candidates.items(), key=lambda item: (-item[1], item[0]))
        self.virtual_swap_memory_idx[node, :] = -1
        self.virtual_swap_memory_weight[node, :] = 0
        for slot, (idx, weight) in enumerate(ranked[:memory_size]):
            self.virtual_swap_memory_idx[node, slot] = idx
            self.virtual_swap_memory_weight[node, slot] = weight

        best_idx = -1
        best_weight = 0
        if ranked and ranked[0][1] > 0:
            best_idx = int(ranked[0][0])
            best_weight = int(ranked[0][1])

        self.virtual_swap_best_weight[node] = best_weight
        self.virtual_swap_best_idx[node] = best_idx
        has_candidates = self.swap_node_starts[node] < self.swap_node_starts[node + 1]
        new_rate = float(self.config.swap_rates[node]) if has_candidates else 0.0
        self.virtual_swap_node_rates[node] = new_rate
        self.active_virtual_swap_total += new_rate - old_rate

    def _recompute_physical_swap_node(self, state: QBPState, node: int) -> None:
        old_rate = float(self.physical_swap_node_rates[node])
        best_deficit, best_idx = _best_physical_swap_for_node(
            state.q,
            state.h_mu,
            node,
            self.swap_node_starts,
            self.swap_y,
            self.swap_z,
        )
        self.physical_swap_best_deficit[node] = best_deficit
        self.physical_swap_best_idx[node] = best_idx
        new_rate = float(self.config.swap_rates[node]) if best_idx >= 0 else 0.0
        self.physical_swap_node_rates[node] = new_rate
        self.active_physical_swap_total += new_rate - old_rate

    def _initialize_limited_max_min_swap_node(self, node: int) -> None:
        old_rate = float(self.max_min_swap_node_rates[node])
        self.max_min_swap_best_output[node] = -1
        self.max_min_swap_best_idx[node] = -1
        has_candidates = self.swap_node_starts[node] < self.swap_node_starts[node + 1]
        new_rate = float(self.config.swap_rates[node]) if has_candidates else 0.0
        self.max_min_swap_node_rates[node] = new_rate
        self.active_max_min_swap_total += new_rate - old_rate

    def _max_min_swap_output(self, state: QBPState, swap_idx: int) -> int:
        i = int(self.swap_i[swap_idx])
        y = int(self.swap_y[swap_idx])
        z = int(self.swap_z[swap_idx])
        output_count = int(state.q[y, z])
        if state.q[i, y] > output_count + 1 and state.q[i, z] > output_count + 1:
            return output_count
        return -1

    def _refresh_limited_max_min_swap_node(self, state: QBPState, node: int) -> None:
        old_rate = float(self.max_min_swap_node_rates[node])
        memory_size = int(self.config.virtual_swap_policy.memory)
        candidates: dict[int, int] = {}

        for slot in range(memory_size):
            remembered_idx = int(self.max_min_swap_memory_idx[node, slot])
            if remembered_idx >= 0:
                candidates[remembered_idx] = self._max_min_swap_output(state, remembered_idx)

        for sampled_idx in self._sample_limited_max_min_swap_candidates(node):
            idx = int(sampled_idx)
            candidates[idx] = self._max_min_swap_output(state, idx)

        ranked = sorted(
            candidates.items(),
            key=lambda item: (
                0 if item[1] >= 0 else 1,
                item[1] if item[1] >= 0 else 9223372036854775807,
                item[0],
            ),
        )
        self.max_min_swap_memory_idx[node, :] = -1
        self.max_min_swap_memory_output[node, :] = -1
        for slot, (idx, output_count) in enumerate(ranked[:memory_size]):
            self.max_min_swap_memory_idx[node, slot] = idx
            self.max_min_swap_memory_output[node, slot] = output_count

        best_idx = -1
        best_output = -1
        if ranked and ranked[0][1] >= 0:
            best_idx = int(ranked[0][0])
            best_output = int(ranked[0][1])

        self.max_min_swap_best_output[node] = best_output
        self.max_min_swap_best_idx[node] = best_idx
        has_candidates = self.swap_node_starts[node] < self.swap_node_starts[node + 1]
        new_rate = float(self.config.swap_rates[node]) if has_candidates else 0.0
        self.max_min_swap_node_rates[node] = new_rate
        self.active_max_min_swap_total += new_rate - old_rate

    def _recompute_max_min_swaps(self, state: QBPState) -> None:
        if not self._uses_max_min_policy():
            return
        if self._uses_limited_max_min_policy():
            return
        self.active_max_min_swap_total = _compute_active_max_min_swap_rates(
            state.q,
            self.swap_node_starts,
            self.swap_y,
            self.swap_z,
            self.config.swap_rates,
            self.max_min_swap_best_output,
            self.max_min_swap_best_idx,
            self.max_min_swap_node_rates,
        )

    def _update_virtual_swap_for_alpha_pair_change(self, state: QBPState, a: int, b: int, delta: int) -> None:
        self._recompute_virtual_service_pair(state, a, b)
        if self._uses_max_min_policy():
            return
        if self._uses_limited_virtual_swap_policy():
            return
        self._recompute_virtual_swap_node(state, a)
        self._recompute_virtual_swap_node(state, b)
        total_delta, rescan_count = _update_virtual_swap_alpha_change_nonendpoints(
            state.alpha,
            a,
            b,
            delta,
            self.swap_lookup,
            self.config.swap_rates,
            self.virtual_swap_best_weight,
            self.virtual_swap_best_idx,
            self.virtual_swap_node_rates,
            self.virtual_swap_rescan_nodes,
        )
        self.active_virtual_swap_total += total_delta
        for idx in range(rescan_count):
            self._recompute_virtual_swap_node(state, int(self.virtual_swap_rescan_nodes[idx]))

    def on_event_applied(self, state: QBPState, event: QBPEvent) -> None:
        if event.event_type == "demand_arrival":
            x = _require(event.x, "x")
            y = _require(event.y, "y")
            self._recompute_virtual_service_pair(state, x, y)
            return

        if event.event_type == "pair_generation":
            x = _require(event.x, "x")
            y = _require(event.y, "y")
            self._recompute_physical_service_pair(state, x, y)
            if self._uses_max_min_policy():
                self._recompute_max_min_swaps(state)
            else:
                self._recompute_physical_swap_node(state, x)
                self._recompute_physical_swap_node(state, y)
            self._update_virtual_swap_for_alpha_pair_change(state, x, y, -1)
            return

        if event.event_type == "virtual_service":
            x = _require(event.x, "x")
            y = _require(event.y, "y")
            self._recompute_physical_service_pair(state, x, y)
            self._update_virtual_swap_for_alpha_pair_change(state, x, y, +1)
            return

        if event.event_type == "service_request":
            x = _require(event.x, "x")
            y = _require(event.y, "y")
            self._recompute_virtual_service_pair(state, x, y)
            self._recompute_physical_service_pair(state, x, y)
            return

        if event.event_type == "virtual_swap":
            swap_idx = _require(event.swap_idx, "swap_idx")
            i = _require(event.i, "i")
            y = _require(event.y, "y")
            z = _require(event.z, "z")
            self._recompute_physical_swap_node(state, i)
            self._update_virtual_swap_for_alpha_pair_change(state, i, y, +1)
            self._update_virtual_swap_for_alpha_pair_change(state, i, z, +1)
            self._update_virtual_swap_for_alpha_pair_change(state, y, z, -1)
            return

        if event.event_type == "virtual_swap_idle":
            return

        if event.event_type == "physical_service":
            x = _require(event.x, "x")
            y = _require(event.y, "y")
            self._recompute_physical_service_pair(state, x, y)
            if self._uses_max_min_policy():
                self._recompute_max_min_swaps(state)
            else:
                self._recompute_physical_swap_node(state, x)
                self._recompute_physical_swap_node(state, y)
            return

        if event.event_type == "physical_swap":
            swap_idx = _require(event.swap_idx, "swap_idx")
            i = _require(event.i, "i")
            y = _require(event.y, "y")
            z = _require(event.z, "z")
            self._recompute_physical_service_pair(state, i, y)
            self._recompute_physical_service_pair(state, i, z)
            self._recompute_physical_service_pair(state, y, z)
            self._recompute_physical_swap_node(state, i)
            self._recompute_physical_swap_node(state, y)
            self._recompute_physical_swap_node(state, z)
            return

        if event.event_type == "max_min_swap":
            i = _require(event.i, "i")
            y = _require(event.y, "y")
            z = _require(event.z, "z")
            self._recompute_physical_service_pair(state, i, y)
            self._recompute_physical_service_pair(state, i, z)
            self._recompute_physical_service_pair(state, y, z)
            self._recompute_max_min_swaps(state)
            return

    def produce(self, state: QBPState, until_time: float | None = None) -> tuple[QBPEvent | None, bool]:
        demand_total = self.demand_total
        generation_total = self.generation_total
        virtual_service_total = self.active_virtual_service_total
        virtual_swap_total = 0.0 if self._uses_max_min_policy() else self.active_virtual_swap_total
        physical_service_total = 0.0 if self.config.instant_service_fulfillment else self.active_physical_service_total
        physical_swap_total = (
            0.0 if self.config.instant_swap_fulfillment or self._uses_max_min_policy() else self.active_physical_swap_total
        )
        max_min_swap_total = self.active_max_min_swap_total if self._uses_max_min_policy() else 0.0
        total_rate = (
            demand_total
            + generation_total
            + virtual_service_total
            + virtual_swap_total
            + physical_service_total
            + physical_swap_total
            + max_min_swap_total
        )
        if total_rate <= 0.0:
            return None, False

        dt = float(self.rng.exponential(scale=1.0 / total_rate))
        next_time = state.time + dt
        if until_time is not None and next_time > until_time:
            return None, True

        selector = float(self.rng.random() * total_rate)
        event_index = state.events_processed + 1

        if selector < demand_total:
            idx = _sample_index(self.rng, self.demand_pair_rates, demand_total)
            return (
                QBPEvent(
                    event_index=event_index,
                    time=next_time,
                    dt=dt,
                    total_rate=total_rate,
                    event_type="demand_arrival",
                    event_rate=float(self.demand_pair_rates[idx]),
                    x=int(self.pair_u[idx]),
                    y=int(self.pair_v[idx]),
                ),
                False,
            )

        selector -= demand_total
        if selector < generation_total:
            idx = _sample_index(self.rng, self.generation_pair_rates, generation_total)
            return (
                QBPEvent(
                    event_index=event_index,
                    time=next_time,
                    dt=dt,
                    total_rate=total_rate,
                    event_type="pair_generation",
                    event_rate=float(self.generation_pair_rates[idx]),
                    x=int(self.pair_u[idx]),
                    y=int(self.pair_v[idx]),
                ),
                False,
            )

        selector -= generation_total
        if selector < virtual_service_total:
            idx = _sample_index(self.rng, self.active_virtual_service_rates, virtual_service_total)
            return (
                QBPEvent(
                    event_index=event_index,
                    time=next_time,
                    dt=dt,
                    total_rate=total_rate,
                    event_type="service_request" if self._uses_max_min_policy() else "virtual_service",
                    event_rate=float(self.active_virtual_service_rates[idx]),
                    x=int(self.pair_u[idx]),
                    y=int(self.pair_v[idx]),
                ),
                False,
            )

        selector -= virtual_service_total
        if selector < virtual_swap_total:
            node = _sample_index(self.rng, self.virtual_swap_node_rates, virtual_swap_total)
            if self._uses_limited_virtual_swap_policy():
                self._refresh_limited_virtual_swap_node(state, node)
            idx = int(self.virtual_swap_best_idx[node])
            if idx < 0:
                return (
                    QBPEvent(
                        event_index=event_index,
                        time=next_time,
                        dt=dt,
                        total_rate=total_rate,
                        event_type="virtual_swap_idle",
                        event_rate=float(self.virtual_swap_node_rates[node]),
                        i=int(node),
                    ),
                    False,
                )
            return (
                QBPEvent(
                    event_index=event_index,
                    time=next_time,
                    dt=dt,
                    total_rate=total_rate,
                    event_type="virtual_swap",
                    event_rate=float(self.virtual_swap_node_rates[node]),
                    swap_idx=int(idx),
                    i=int(self.swap_i[idx]),
                    y=int(self.swap_y[idx]),
                    z=int(self.swap_z[idx]),
                ),
                False,
            )

        selector -= virtual_swap_total
        if selector < physical_service_total:
            idx = _sample_index(self.rng, self.active_physical_service_rates, physical_service_total)
            return (
                QBPEvent(
                    event_index=event_index,
                    time=next_time,
                    dt=dt,
                    total_rate=total_rate,
                    event_type="physical_service",
                    event_rate=float(self.active_physical_service_rates[idx]),
                    x=int(self.pair_u[idx]),
                    y=int(self.pair_v[idx]),
                ),
                False,
            )

        selector -= physical_service_total
        if selector < physical_swap_total:
            node = _sample_index(self.rng, self.physical_swap_node_rates, physical_swap_total)
            idx = int(self.physical_swap_best_idx[node])
            return (
                QBPEvent(
                    event_index=event_index,
                    time=next_time,
                    dt=dt,
                    total_rate=total_rate,
                    event_type="physical_swap",
                    event_rate=float(self.physical_swap_node_rates[node]),
                    swap_idx=int(idx),
                    i=int(self.swap_i[idx]),
                    y=int(self.swap_y[idx]),
                    z=int(self.swap_z[idx]),
                ),
                False,
            )

        node = _sample_index(self.rng, self.max_min_swap_node_rates, max_min_swap_total)
        if self._uses_limited_max_min_policy():
            self._refresh_limited_max_min_swap_node(state, node)
        idx = int(self.max_min_swap_best_idx[node])
        if idx < 0:
            return (
                QBPEvent(
                    event_index=event_index,
                    time=next_time,
                    dt=dt,
                    total_rate=total_rate,
                    event_type="max_min_swap_idle",
                    event_rate=float(self.max_min_swap_node_rates[node]),
                    i=int(node),
                ),
                False,
            )
        return (
            QBPEvent(
                event_index=event_index,
                time=next_time,
                dt=dt,
                total_rate=total_rate,
                event_type="max_min_swap",
                event_rate=float(self.max_min_swap_node_rates[node]),
                swap_idx=int(idx),
                i=int(self.swap_i[idx]),
                y=int(self.swap_y[idx]),
                z=int(self.swap_z[idx]),
            ),
            False,
        )
