from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

import numpy as np
from numba import njit
from numpy.typing import NDArray
from tqdm import tqdm

from qbp_sim.events import QBPEvent
from qbp_sim.progress import should_use_progress
from qbp_sim.snapshots import QBPSnapshot, SnapshotWriter
from qbp_sim.trace import EventTraceWriter


Array1D = NDArray[np.float64]
IntArray1D = NDArray[np.int64]
FloatMatrix = NDArray[np.float64]
IntMatrix = NDArray[np.int64]


@dataclass(slots=True)
class GillespieQBPConfig:
    generation_rates: FloatMatrix
    demand_rates: FloatMatrix
    swap_rates: IntArray1D | Array1D
    service_rates: FloatMatrix


@dataclass(slots=True)
class GillespieQBPResult:
    final_time: float
    events_processed: int
    total_backlog: int
    total_inventory: int
    total_scarcity: int
    demand_arrivals: int
    pair_generations: int
    virtual_service_requests: int
    virtual_swap_requests: int
    services_completed: int
    swaps_completed: int
    sample_times: list[float]
    total_backlog_samples: list[int]
    total_inventory_samples: list[int]
    total_alpha_samples: list[int]

    def format_summary(self) -> str:
        return "\n".join(
            [
                "QBP Gillespie simulation",
                f"time={self.final_time:.6f}",
                f"events={self.events_processed}",
                f"backlog={self.total_backlog}",
                f"inventory={self.total_inventory}",
                f"scarcity={self.total_scarcity}",
                f"demand_arrivals={self.demand_arrivals}",
                f"pair_generations={self.pair_generations}",
                f"virtual_service_requests={self.virtual_service_requests}",
                f"virtual_swap_requests={self.virtual_swap_requests}",
                f"services_completed={self.services_completed}",
                f"swaps_completed={self.swaps_completed}",
            ]
        )


@dataclass(slots=True)
class QBPState:
    q: IntMatrix
    d: IntMatrix
    alpha: IntMatrix
    h_r: IntMatrix
    h_mu: IntArray1D
    total_virtual_backlog_count: int = 0
    total_service_deficit_count: int = 0
    total_swap_deficit_count: int = 0
    total_inventory_count: int = 0
    total_scarcity_count: int = 0
    time: float = 0.0
    events_processed: int = 0
    demand_arrivals: int = 0
    pair_generations: int = 0
    virtual_service_requests: int = 0
    virtual_swap_requests: int = 0
    services_completed: int = 0
    swaps_completed: int = 0

    @property
    def total_virtual_backlog(self) -> int:
        return self.total_virtual_backlog_count

    @property
    def total_service_deficit(self) -> int:
        return self.total_service_deficit_count

    @property
    def total_swap_deficit(self) -> int:
        return self.total_swap_deficit_count

    @property
    def total_backlog(self) -> int:
        return self.total_virtual_backlog_count + self.total_service_deficit_count

    @property
    def total_inventory(self) -> int:
        return self.total_inventory_count

    @property
    def total_scarcity(self) -> int:
        return self.total_scarcity_count

    @property
    def service_ratio(self) -> float:
        if self.demand_arrivals <= 0:
            return 0.0
        return float(self.services_completed) / float(self.demand_arrivals)


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
def _compute_active_virtual_swap_choices(
    alpha: IntMatrix,
    swap_i: IntArray1D,
    swap_y: IntArray1D,
    swap_z: IntArray1D,
    swap_rates: Array1D,
    best_weight: IntArray1D,
    best_idx: IntArray1D,
    choice_idx_out: IntArray1D,
    choice_rate_out: Array1D,
) -> tuple[float, int]:
    total = 0.0
    n_nodes = swap_rates.shape[0]

    for i in range(n_nodes):
        best_weight[i] = 0
        best_idx[i] = -1

    for idx in range(swap_i.shape[0]):
        i = swap_i[idx]
        y = swap_y[idx]
        z = swap_z[idx]
        weight = alpha[y, z] - alpha[i, y] - alpha[i, z]
        if weight > best_weight[i]:
            best_weight[i] = weight
            best_idx[i] = idx

    active_count = 0
    for i in range(n_nodes):
        idx = best_idx[i]
        if idx >= 0:
            rate = swap_rates[i]
            choice_idx_out[active_count] = idx
            choice_rate_out[active_count] = rate
            total += rate
            active_count += 1
    return total, active_count


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
def _compute_active_physical_swap_choices(
    q: IntMatrix,
    h_mu: IntArray1D,
    swap_i: IntArray1D,
    swap_y: IntArray1D,
    swap_z: IntArray1D,
    swap_rates: Array1D,
    best_deficit: IntArray1D,
    best_idx: IntArray1D,
    choice_idx_out: IntArray1D,
    choice_rate_out: Array1D,
) -> tuple[float, int]:
    total = 0.0
    n_nodes = swap_rates.shape[0]

    for i in range(n_nodes):
        best_deficit[i] = 0
        best_idx[i] = -1

    for idx in range(swap_i.shape[0]):
        if h_mu[idx] <= 0:
            continue
        i = swap_i[idx]
        y = swap_y[idx]
        z = swap_z[idx]
        if q[i, y] <= 0 or q[i, z] <= 0:
            continue
        deficit = h_mu[idx]
        if deficit > best_deficit[i]:
            best_deficit[i] = deficit
            best_idx[i] = idx

    active_count = 0
    for i in range(n_nodes):
        idx = best_idx[i]
        if idx >= 0:
            rate = swap_rates[i]
            choice_idx_out[active_count] = idx
            choice_rate_out[active_count] = rate
            total += rate
            active_count += 1
    return total, active_count


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


class QBPEventApplier:
    """Apply concrete events to mutable QBP state."""

    def apply(self, state: QBPState, event: QBPEvent) -> QBPEvent:
        expected_index = state.events_processed + 1
        if event.event_index != expected_index:
            raise ValueError(f"Expected event index {expected_index}, got {event.event_index}.")
        if event.time < state.time:
            raise ValueError("Event time cannot go backwards.")

        state.time = event.time
        if event.event_type == "demand_arrival":
            _apply_demand_arrival(state.d, _require(event.x, "x"), _require(event.y, "y"))
            state.demand_arrivals += 1
            state.total_virtual_backlog_count += 1
        elif event.event_type == "pair_generation":
            x = _require(event.x, "x")
            y = _require(event.y, "y")
            old_scarcity = state.alpha[x, y]
            _apply_pair_generation(state.q, state.alpha, x, y)
            state.pair_generations += 1
            state.total_inventory_count += 1
            if old_scarcity > 0:
                state.total_scarcity_count -= 1
        elif event.event_type == "virtual_service":
            x = _require(event.x, "x")
            y = _require(event.y, "y")
            _apply_virtual_service(
                state.d,
                state.alpha,
                state.h_r,
                x,
                y,
            )
            state.virtual_service_requests += 1
            state.total_virtual_backlog_count -= 1
            state.total_service_deficit_count += 1
            state.total_scarcity_count += 1
        elif event.event_type == "virtual_swap":
            swap_idx = _require(event.swap_idx, "swap_idx")
            i = _require(event.i, "i")
            y = _require(event.y, "y")
            z = _require(event.z, "z")
            old_output_scarcity = state.alpha[y, z]
            _apply_virtual_swap(
                state.alpha,
                state.h_mu,
                swap_idx,
                i,
                y,
                z,
            )
            state.virtual_swap_requests += 1
            state.total_swap_deficit_count += 1
            state.total_scarcity_count += 2
            if old_output_scarcity > 0:
                state.total_scarcity_count -= 1
        elif event.event_type == "physical_service":
            _apply_physical_service(state.q, state.h_r, _require(event.x, "x"), _require(event.y, "y"))
            state.services_completed += 1
            state.total_inventory_count -= 1
            state.total_service_deficit_count -= 1
        elif event.event_type == "physical_swap":
            swap_idx = _require(event.swap_idx, "swap_idx")
            _apply_physical_swap(
                state.q,
                state.h_mu,
                swap_idx,
                _require(event.i, "i"),
                _require(event.y, "y"),
                _require(event.z, "z"),
            )
            state.swaps_completed += 1
            state.total_inventory_count -= 1
            state.total_swap_deficit_count -= 1
        else:
            raise ValueError(f"Unknown event type: {event.event_type}")

        state.events_processed += 1
        event.backlog_total = state.total_backlog
        event.inventory_total = state.total_inventory
        event.scarcity_total = state.total_scarcity
        return event


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
        self.pair_u = pair_u
        self.pair_v = pair_v
        self.swap_i = swap_i
        self.swap_y = swap_y
        self.swap_z = swap_z
        self.demand_pair_rates = _matrix_to_pair_vector(config.demand_rates, pair_u, pair_v)
        self.generation_pair_rates = _matrix_to_pair_vector(config.generation_rates, pair_u, pair_v)
        self.service_pair_rates = _matrix_to_pair_vector(config.service_rates, pair_u, pair_v)
        self.demand_total = float(self.demand_pair_rates.sum())
        self.generation_total = float(self.generation_pair_rates.sum())
        self.active_virtual_service_rates = np.zeros_like(self.service_pair_rates)
        self.active_physical_service_rates = np.zeros_like(self.service_pair_rates)
        n_nodes = config.swap_rates.shape[0]
        self.virtual_swap_best_weight = np.zeros(n_nodes, dtype=np.int64)
        self.virtual_swap_best_idx = np.full(n_nodes, -1, dtype=np.int64)
        self.active_virtual_swap_choice_idx = np.empty(n_nodes, dtype=np.int64)
        self.active_virtual_swap_choice_rates = np.zeros(n_nodes, dtype=np.float64)
        self.physical_swap_best_deficit = np.zeros(n_nodes, dtype=np.int64)
        self.physical_swap_best_idx = np.full(n_nodes, -1, dtype=np.int64)
        self.active_physical_swap_choice_idx = np.empty(n_nodes, dtype=np.int64)
        self.active_physical_swap_choice_rates = np.zeros(n_nodes, dtype=np.float64)
        self.rng = np.random.default_rng(seed)

    def produce(self, state: QBPState, until_time: float | None = None) -> tuple[QBPEvent | None, bool]:
        virtual_service_total = _compute_active_virtual_service_rates(
            state.d,
            state.alpha,
            self.pair_u,
            self.pair_v,
            self.service_pair_rates,
            self.active_virtual_service_rates,
        )
        virtual_swap_total, virtual_swap_count = _compute_active_virtual_swap_choices(
            state.alpha,
            self.swap_i,
            self.swap_y,
            self.swap_z,
            self.config.swap_rates,
            self.virtual_swap_best_weight,
            self.virtual_swap_best_idx,
            self.active_virtual_swap_choice_idx,
            self.active_virtual_swap_choice_rates,
        )
        physical_service_total = _compute_active_physical_service_rates(
            state.q,
            state.h_r,
            self.pair_u,
            self.pair_v,
            self.service_pair_rates,
            self.active_physical_service_rates,
        )
        physical_swap_total, physical_swap_count = _compute_active_physical_swap_choices(
            state.q,
            state.h_mu,
            self.swap_i,
            self.swap_y,
            self.swap_z,
            self.config.swap_rates,
            self.physical_swap_best_deficit,
            self.physical_swap_best_idx,
            self.active_physical_swap_choice_idx,
            self.active_physical_swap_choice_rates,
        )

        demand_total = self.demand_total
        generation_total = self.generation_total
        total_rate = (
            demand_total
            + generation_total
            + virtual_service_total
            + virtual_swap_total
            + physical_service_total
            + physical_swap_total
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
                    event_type="virtual_service",
                    event_rate=float(self.active_virtual_service_rates[idx]),
                    x=int(self.pair_u[idx]),
                    y=int(self.pair_v[idx]),
                ),
                False,
            )

        selector -= virtual_service_total
        if selector < virtual_swap_total:
            compact_idx = _sample_index(
                self.rng,
                self.active_virtual_swap_choice_rates[:virtual_swap_count],
                virtual_swap_total,
            )
            idx = int(self.active_virtual_swap_choice_idx[compact_idx])
            return (
                QBPEvent(
                    event_index=event_index,
                    time=next_time,
                    dt=dt,
                    total_rate=total_rate,
                    event_type="virtual_swap",
                    event_rate=float(self.active_virtual_swap_choice_rates[compact_idx]),
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

        compact_idx = _sample_index(
            self.rng,
            self.active_physical_swap_choice_rates[:physical_swap_count],
            physical_swap_total,
        )
        idx = int(self.active_physical_swap_choice_idx[compact_idx])
        return (
            QBPEvent(
                event_index=event_index,
                time=next_time,
                dt=dt,
                total_rate=total_rate,
                event_type="physical_swap",
                event_rate=float(self.active_physical_swap_choice_rates[compact_idx]),
                swap_idx=int(idx),
                i=int(self.swap_i[idx]),
                y=int(self.swap_y[idx]),
                z=int(self.swap_z[idx]),
            ),
            False,
        )


class GillespieQBPSimulator:
    """Continuous-time analogue of the paper's virtual/physical QBP model."""

    def __init__(
        self,
        config: GillespieQBPConfig,
        seed: int | None = None,
        initial_q: IntMatrix | None = None,
        initial_d: IntMatrix | None = None,
        initial_alpha: IntMatrix | None = None,
        initial_h_r: IntMatrix | None = None,
        initial_h_mu: IntArray1D | None = None,
    ) -> None:
        self.config = self._validate_config(config)
        self.n_nodes = self.config.generation_rates.shape[0]
        self.pair_u, self.pair_v = _build_pair_index(self.n_nodes)
        self.swap_i, self.swap_y, self.swap_z = _build_swap_candidates(self.n_nodes)
        initial_q_matrix = _init_state_matrix(self.n_nodes, initial_q)
        initial_d_matrix = _init_state_matrix(self.n_nodes, initial_d)
        initial_alpha_matrix = _init_state_matrix(self.n_nodes, initial_alpha)
        initial_h_r_matrix = _init_state_matrix(self.n_nodes, initial_h_r)
        initial_h_mu_array = _init_swap_counter(self.swap_i.shape[0], initial_h_mu)
        self.state = QBPState(
            q=initial_q_matrix,
            d=initial_d_matrix,
            alpha=initial_alpha_matrix,
            h_r=initial_h_r_matrix,
            h_mu=initial_h_mu_array,
            total_virtual_backlog_count=_upper_triangle_sum(initial_d_matrix),
            total_service_deficit_count=_upper_triangle_sum(initial_h_r_matrix),
            total_swap_deficit_count=int(initial_h_mu_array.sum()),
            total_inventory_count=_upper_triangle_sum(initial_q_matrix),
            total_scarcity_count=_upper_triangle_sum(initial_alpha_matrix),
        )
        self.producer = GillespieQBPEventProducer(
            config=self.config,
            pair_u=self.pair_u,
            pair_v=self.pair_v,
            swap_i=self.swap_i,
            swap_y=self.swap_y,
            swap_z=self.swap_z,
            seed=seed,
        )
        self.applier = QBPEventApplier()

    def produce_next_event(self, until_time: float | None = None) -> QBPEvent | None:
        event, hit_limit = self.producer.produce(self.state, until_time=until_time)
        if event is None and hit_limit and until_time is not None:
            self.state.time = until_time
        return event

    def apply_event(self, event: QBPEvent, trace_writer: EventTraceWriter | None = None) -> QBPEvent:
        applied = self.applier.apply(self.state, event)
        if trace_writer is not None:
            trace_writer.write(applied)
        return applied

    def step(
        self,
        until_time: float | None = None,
        trace_writer: EventTraceWriter | None = None,
    ) -> QBPEvent | None:
        event = self.produce_next_event(until_time=until_time)
        if event is None:
            return None
        return self.apply_event(event, trace_writer=trace_writer)

    def run(
        self,
        until_time: float,
        max_events: int = 100_000,
        sample_every: int = 500,
        trace_writer: EventTraceWriter | None = None,
        snapshot_writer: SnapshotWriter | None = None,
        progress: bool | None = None,
    ) -> GillespieQBPResult:
        sample_times: list[float] = []
        backlog_samples: list[int] = []
        inventory_samples: list[int] = []
        alpha_samples: list[int] = []
        last_snapshot_key: tuple[int, float] | None = None
        use_progress = should_use_progress() if progress is None else progress

        if use_progress:
            with tqdm(
                total=max_events,
                desc="simulate",
                unit="event",
            ) as progress_bar:
                while self.state.time < until_time and self.state.events_processed < max_events:
                    event = self.step(until_time=until_time, trace_writer=trace_writer)
                    if event is None:
                        break
                    progress_bar.update(1)
                    if self.state.events_processed % 1000 == 0 or self.state.time >= until_time:
                        progress_bar.set_postfix(
                            time=f"{self.state.time:.3f}",
                            backlog=self.state.total_backlog,
                        )
                    if sample_every > 0 and self.state.events_processed % sample_every == 0:
                        if snapshot_writer is not None:
                            snapshot_writer.write(self.build_snapshot())
                        last_snapshot_key = (self.state.events_processed, self.state.time)
                        sample_times.append(self.state.time)
                        backlog_samples.append(self.state.total_backlog)
                        inventory_samples.append(self.state.total_inventory)
                        alpha_samples.append(self.state.total_scarcity)
        else:
            while self.state.time < until_time and self.state.events_processed < max_events:
                event = self.step(until_time=until_time, trace_writer=trace_writer)
                if event is None:
                    break
                if sample_every > 0 and self.state.events_processed % sample_every == 0:
                    if snapshot_writer is not None:
                        snapshot_writer.write(self.build_snapshot())
                    last_snapshot_key = (self.state.events_processed, self.state.time)
                    sample_times.append(self.state.time)
                    backlog_samples.append(self.state.total_backlog)
                    inventory_samples.append(self.state.total_inventory)
                    alpha_samples.append(self.state.total_scarcity)

        final_snapshot_key = (self.state.events_processed, self.state.time)
        if sample_every > 0 and final_snapshot_key != last_snapshot_key:
            if snapshot_writer is not None:
                snapshot_writer.write(self.build_snapshot())
            sample_times.append(self.state.time)
            backlog_samples.append(self.state.total_backlog)
            inventory_samples.append(self.state.total_inventory)
            alpha_samples.append(self.state.total_scarcity)

        return self._build_result(sample_times, backlog_samples, inventory_samples, alpha_samples)

    def replay(
        self,
        events: Iterable[QBPEvent],
        sample_every: int = 500,
        trace_writer: EventTraceWriter | None = None,
        snapshot_writer: SnapshotWriter | None = None,
        final_time: float | None = None,
        progress: bool | None = None,
    ) -> GillespieQBPResult:
        sample_times: list[float] = []
        backlog_samples: list[int] = []
        inventory_samples: list[int] = []
        alpha_samples: list[int] = []
        last_snapshot_key: tuple[int, float] | None = None
        use_progress = should_use_progress() if progress is None else progress

        if use_progress:
            with tqdm(
                desc="replay",
                unit="event",
            ) as progress_bar:
                for event in events:
                    self.apply_event(event, trace_writer=trace_writer)
                    progress_bar.update(1)
                    if self.state.events_processed % 1000 == 0:
                        progress_bar.set_postfix(
                            time=f"{self.state.time:.3f}",
                            backlog=self.state.total_backlog,
                        )
                    if sample_every > 0 and self.state.events_processed % sample_every == 0:
                        if snapshot_writer is not None:
                            snapshot_writer.write(self.build_snapshot())
                        last_snapshot_key = (self.state.events_processed, self.state.time)
                        sample_times.append(self.state.time)
                        backlog_samples.append(self.state.total_backlog)
                        inventory_samples.append(self.state.total_inventory)
                        alpha_samples.append(self.state.total_scarcity)
        else:
            for event in events:
                self.apply_event(event, trace_writer=trace_writer)
                if sample_every > 0 and self.state.events_processed % sample_every == 0:
                    if snapshot_writer is not None:
                        snapshot_writer.write(self.build_snapshot())
                    last_snapshot_key = (self.state.events_processed, self.state.time)
                    sample_times.append(self.state.time)
                    backlog_samples.append(self.state.total_backlog)
                    inventory_samples.append(self.state.total_inventory)
                    alpha_samples.append(self.state.total_scarcity)

        if final_time is not None and final_time >= self.state.time:
            self.state.time = final_time

        final_snapshot_key = (self.state.events_processed, self.state.time)
        if sample_every > 0 and final_snapshot_key != last_snapshot_key:
            if snapshot_writer is not None:
                snapshot_writer.write(self.build_snapshot())
            sample_times.append(self.state.time)
            backlog_samples.append(self.state.total_backlog)
            inventory_samples.append(self.state.total_inventory)
            alpha_samples.append(self.state.total_scarcity)

        return self._build_result(sample_times, backlog_samples, inventory_samples, alpha_samples)

    @property
    def total_backlog(self) -> int:
        return self.state.total_backlog

    @property
    def total_inventory(self) -> int:
        return self.state.total_inventory

    @property
    def total_scarcity(self) -> int:
        return self.state.total_scarcity

    @property
    def time(self) -> float:
        return self.state.time

    @property
    def events_processed(self) -> int:
        return self.state.events_processed

    @property
    def demand_arrivals(self) -> int:
        return self.state.demand_arrivals

    @property
    def pair_generations(self) -> int:
        return self.state.pair_generations

    @property
    def virtual_service_requests(self) -> int:
        return self.state.virtual_service_requests

    @property
    def virtual_swap_requests(self) -> int:
        return self.state.virtual_swap_requests

    @property
    def services_completed(self) -> int:
        return self.state.services_completed

    @property
    def swaps_completed(self) -> int:
        return self.state.swaps_completed

    def _build_result(
        self,
        sample_times: list[float],
        backlog_samples: list[int],
        inventory_samples: list[int],
        alpha_samples: list[int],
    ) -> GillespieQBPResult:
        return GillespieQBPResult(
            final_time=self.state.time,
            events_processed=self.state.events_processed,
            total_backlog=self.state.total_backlog,
            total_inventory=self.state.total_inventory,
            total_scarcity=self.state.total_scarcity,
            demand_arrivals=self.state.demand_arrivals,
            pair_generations=self.state.pair_generations,
            virtual_service_requests=self.state.virtual_service_requests,
            virtual_swap_requests=self.state.virtual_swap_requests,
            services_completed=self.state.services_completed,
            swaps_completed=self.state.swaps_completed,
            sample_times=sample_times,
            total_backlog_samples=backlog_samples,
            total_inventory_samples=inventory_samples,
            total_alpha_samples=alpha_samples,
        )

    def build_snapshot(self) -> QBPSnapshot:
        return QBPSnapshot(
            event_index=self.state.events_processed,
            time=self.state.time,
            total_backlog=self.state.total_backlog,
            total_inventory=self.state.total_inventory,
            total_scarcity=self.state.total_scarcity,
            demand_arrivals=self.state.demand_arrivals,
            pair_generations=self.state.pair_generations,
            services_completed=self.state.services_completed,
            swaps_completed=self.state.swaps_completed,
            service_ratio=self.state.service_ratio,
        )

    def _validate_config(self, config: GillespieQBPConfig) -> GillespieQBPConfig:
        generation_rates = np.asarray(config.generation_rates, dtype=np.float64)
        demand_rates = np.asarray(config.demand_rates, dtype=np.float64)
        service_rates = np.asarray(config.service_rates, dtype=np.float64)
        swap_rates = np.asarray(config.swap_rates, dtype=np.float64)

        for name, matrix in (
            ("generation_rates", generation_rates),
            ("demand_rates", demand_rates),
            ("service_rates", service_rates),
        ):
            if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
                raise ValueError(f"{name} must be a square matrix.")
            if not np.allclose(matrix, matrix.T):
                raise ValueError(f"{name} must be symmetric.")
        if swap_rates.ndim != 1 or swap_rates.shape[0] != generation_rates.shape[0]:
            raise ValueError("swap_rates must be a vector with one entry per node.")
        if np.any(generation_rates < 0.0) or np.any(demand_rates < 0.0) or np.any(service_rates < 0.0):
            raise ValueError("All rates must be non-negative.")
        if np.any(swap_rates < 0.0):
            raise ValueError("swap_rates must be non-negative.")

        generation_rates = generation_rates.copy()
        demand_rates = demand_rates.copy()
        service_rates = service_rates.copy()

        np.fill_diagonal(generation_rates, 0.0)
        np.fill_diagonal(demand_rates, 0.0)
        np.fill_diagonal(service_rates, 0.0)

        return GillespieQBPConfig(
            generation_rates=generation_rates,
            demand_rates=demand_rates,
            swap_rates=swap_rates,
            service_rates=service_rates,
        )


def replay_event_stream(
    config: GillespieQBPConfig,
    events: Iterable[QBPEvent],
    sample_every: int = 500,
    final_time: float | None = None,
    snapshot_writer: SnapshotWriter | None = None,
    initial_q: IntMatrix | None = None,
    initial_d: IntMatrix | None = None,
    initial_alpha: IntMatrix | None = None,
    initial_h_r: IntMatrix | None = None,
    initial_h_mu: IntArray1D | None = None,
) -> GillespieQBPResult:
    simulator = GillespieQBPSimulator(
        config=config,
        seed=None,
        initial_q=initial_q,
        initial_d=initial_d,
        initial_alpha=initial_alpha,
        initial_h_r=initial_h_r,
        initial_h_mu=initial_h_mu,
    )
    return simulator.replay(
        events=events,
        sample_every=sample_every,
        final_time=final_time,
        snapshot_writer=snapshot_writer,
    )


def _build_pair_index(n_nodes: int) -> tuple[IntArray1D, IntArray1D]:
    pair_u: list[int] = []
    pair_v: list[int] = []
    for x in range(n_nodes):
        for y in range(x + 1, n_nodes):
            pair_u.append(x)
            pair_v.append(y)
    return np.asarray(pair_u, dtype=np.int64), np.asarray(pair_v, dtype=np.int64)


def _build_swap_candidates(n_nodes: int) -> tuple[IntArray1D, IntArray1D, IntArray1D]:
    swap_i: list[int] = []
    swap_y: list[int] = []
    swap_z: list[int] = []
    for i in range(n_nodes):
        others = [j for j in range(n_nodes) if j != i]
        for left in range(len(others)):
            for right in range(left + 1, len(others)):
                y = others[left]
                z = others[right]
                swap_i.append(i)
                swap_y.append(y)
                swap_z.append(z)
    return (
        np.asarray(swap_i, dtype=np.int64),
        np.asarray(swap_y, dtype=np.int64),
        np.asarray(swap_z, dtype=np.int64),
    )


def _matrix_to_pair_vector(matrix: FloatMatrix, pair_u: IntArray1D, pair_v: IntArray1D) -> Array1D:
    values = np.zeros(pair_u.shape[0], dtype=np.float64)
    for idx, (x, y) in enumerate(zip(pair_u, pair_v, strict=True)):
        values[idx] = float(matrix[x, y])
    return values


def _init_state_matrix(n_nodes: int, value: IntMatrix | None) -> IntMatrix:
    if value is None:
        return np.zeros((n_nodes, n_nodes), dtype=np.int64)
    matrix = np.asarray(value, dtype=np.int64)
    if matrix.shape != (n_nodes, n_nodes):
        raise ValueError("Initial state matrices must match the configured node count.")
    if not np.array_equal(matrix, matrix.T):
        raise ValueError("Initial state matrices must be symmetric.")
    matrix = matrix.copy()
    np.fill_diagonal(matrix, 0)
    return matrix


def _init_swap_counter(size: int, value: IntArray1D | None) -> IntArray1D:
    if value is None:
        return np.zeros(size, dtype=np.int64)
    counter = np.asarray(value, dtype=np.int64)
    if counter.shape != (size,):
        raise ValueError("Initial swap counters must match the number of swap candidates.")
    if np.any(counter < 0):
        raise ValueError("Initial swap counters must be non-negative.")
    return counter.copy()


def _upper_triangle_sum(matrix: IntMatrix) -> int:
    return int(np.triu(matrix, k=1).sum())


@njit(cache=True)
def _sample_index_from_threshold(rates: Array1D, threshold: float) -> int:
    cumulative = 0.0
    for idx, rate in enumerate(rates):
        cumulative += float(rate)
        if threshold <= cumulative:
            return idx
    return len(rates) - 1


def _sample_index(rng: np.random.Generator, rates: Array1D, total_rate: float) -> int:
    threshold = float(rng.random() * total_rate)
    return _sample_index_from_threshold(rates, threshold)


def _require(value: int | None, name: str) -> int:
    if value is None:
        raise ValueError(f"Event is missing required field {name}.")
    return value
