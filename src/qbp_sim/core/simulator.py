
from __future__ import annotations

import numpy as np
from tqdm import tqdm

from qbp_sim.core.applier import QBPEventApplier
from qbp_sim.core.frontier import InstantFulfillmentMixin
from qbp_sim.core.indexing import (
    _build_pair_index,
    _build_swap_candidates,
    _init_state_matrix,
    _init_swap_counter,
    _upper_triangle_sum,
)
from qbp_sim.core.producer import GillespieQBPEventProducer
from qbp_sim.core.types import (
    VIRTUAL_SWAP_POLICY_BP,
    VIRTUAL_SWAP_POLICY_LIMITED_INFO_BP,
    VIRTUAL_SWAP_POLICY_LIMITED_INFO_MAX_MIN,
    VIRTUAL_SWAP_POLICY_MAX_MIN,
    GillespieQBPConfig,
    GillespieQBPResult,
    IntArray1D,
    IntMatrix,
    QBPState,
    VirtualSwapPolicy,
    normalize_virtual_swap_policy_mode,
)
from qbp_sim.io.events import QBPEvent
from qbp_sim.io.snapshots import QBPSnapshot, SnapshotWriter
from qbp_sim.io.trace import EventTraceWriter
from qbp_sim.progress import should_use_progress

class GillespieQBPSimulator(InstantFulfillmentMixin):
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
        self.producer.initialize(self.state)
        self.applier = QBPEventApplier()

    def produce_next_event(self, until_time: float | None = None) -> QBPEvent | None:
        event, hit_limit = self.producer.produce(self.state, until_time=until_time)
        if event is None and hit_limit and until_time is not None:
            self.state.time = until_time
        return event

    def apply_event(self, event: QBPEvent, trace_writer: EventTraceWriter | None = None) -> QBPEvent:
        applied = self.applier.apply(self.state, event)
        self.producer.on_event_applied(self.state, applied)
        if trace_writer is not None:
            trace_writer.write(applied)
        return applied

    def reset_measurements(self, *, reset_time_origin: bool = True) -> None:
        self.state.events_processed = 0
        self.state.demand_arrivals = 0
        self.state.pair_generations = 0
        self.state.virtual_service_requests = 0
        self.state.virtual_swap_requests = 0
        self.state.services_completed = 0
        self.state.swaps_completed = 0
        if reset_time_origin:
            self.state.time = 0.0

    def step(
        self,
        until_time: float | None = None,
        trace_writer: EventTraceWriter | None = None,
    ) -> QBPEvent | None:
        event = self.produce_next_event(until_time=until_time)
        if event is None:
            return None
        applied = self.apply_event(event, trace_writer=trace_writer)
        self._run_instant_fulfillment_closure(applied, trace_writer=trace_writer)
        return applied

    def run(
        self,
        until_time: float,
        max_events: int | None = None,
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

        sampled_events_processed = 0

        if use_progress:
            with tqdm(
                total=(
                    None
                    if self.config.instant_service_fulfillment or self.config.instant_swap_fulfillment
                    else max_events
                ),
                desc="simulate",
                unit="event",
            ) as progress_bar:
                while self.state.time < until_time and (
                    max_events is None or sampled_events_processed < max_events
                ):
                    events_before = self.state.events_processed
                    event = self.step(until_time=until_time, trace_writer=trace_writer)
                    if event is None:
                        break
                    sampled_events_processed += 1
                    progress_bar.update(self.state.events_processed - events_before)
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
            while self.state.time < until_time and (
                max_events is None or sampled_events_processed < max_events
            ):
                event = self.step(until_time=until_time, trace_writer=trace_writer)
                if event is None:
                    break
                sampled_events_processed += 1
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
            virtual_swap_policy=self._validate_virtual_swap_policy(config.virtual_swap_policy),
            instant_service_fulfillment=bool(config.instant_service_fulfillment),
            instant_swap_fulfillment=bool(config.instant_swap_fulfillment),
        )

    def _validate_virtual_swap_policy(self, policy: VirtualSwapPolicy) -> VirtualSwapPolicy:
        if isinstance(policy, dict):
            policy = VirtualSwapPolicy(**policy)

        mode = normalize_virtual_swap_policy_mode(str(policy.mode))
        if mode not in {
            VIRTUAL_SWAP_POLICY_BP,
            VIRTUAL_SWAP_POLICY_LIMITED_INFO_BP,
            VIRTUAL_SWAP_POLICY_MAX_MIN,
            VIRTUAL_SWAP_POLICY_LIMITED_INFO_MAX_MIN,
        }:
            raise ValueError(
                "virtual_swap_policy.mode must be either "
                f"{VIRTUAL_SWAP_POLICY_BP!r}, {VIRTUAL_SWAP_POLICY_LIMITED_INFO_BP!r}, "
                f"{VIRTUAL_SWAP_POLICY_MAX_MIN!r}, or {VIRTUAL_SWAP_POLICY_LIMITED_INFO_MAX_MIN!r}."
            )

        k = int(policy.k)
        memory = int(policy.memory)
        if k < 0 or memory < 0:
            raise ValueError("virtual_swap_policy k and memory must be non-negative.")
        if mode in {VIRTUAL_SWAP_POLICY_LIMITED_INFO_BP, VIRTUAL_SWAP_POLICY_LIMITED_INFO_MAX_MIN} and k <= 0:
            raise ValueError(f"{mode} virtual swap policy requires positive k.")
        if mode in {VIRTUAL_SWAP_POLICY_BP, VIRTUAL_SWAP_POLICY_MAX_MIN} and (k != 0 or memory != 0):
            raise ValueError(f"{mode} virtual swap policy does not use k or memory.")

        return VirtualSwapPolicy(mode=mode, k=k, memory=memory)
