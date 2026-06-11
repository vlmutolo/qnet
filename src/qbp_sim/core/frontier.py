
from __future__ import annotations

from qbp_sim.core.kernels import _best_physical_swap_using_edge, _physical_swap_is_feasible
from qbp_sim.core.types import (
    INSTANT_FRONTIER_EDGE,
    INSTANT_FRONTIER_NONE,
    INSTANT_FRONTIER_SWAP,
    _require,
)
from qbp_sim.io.events import QBPEvent
from qbp_sim.io.trace import EventTraceWriter


class InstantFulfillmentMixin:
    def _instant_frontier_for_event(self, event: QBPEvent) -> tuple[int, int, int, int]:
        if event.event_type == "pair_generation":
            return INSTANT_FRONTIER_EDGE, _require(event.x, "x"), _require(event.y, "y"), -1
        if event.event_type in {"virtual_service", "service_request"} and self.config.instant_service_fulfillment:
            return INSTANT_FRONTIER_EDGE, _require(event.x, "x"), _require(event.y, "y"), -1
        if event.event_type == "virtual_swap" and self.config.instant_swap_fulfillment:
            return INSTANT_FRONTIER_SWAP, -1, -1, _require(event.swap_idx, "swap_idx")
        if event.event_type in {"physical_swap", "max_min_swap"}:
            return INSTANT_FRONTIER_EDGE, _require(event.y, "y"), _require(event.z, "z"), -1
        return INSTANT_FRONTIER_NONE, -1, -1, -1

    def _instant_physical_service_event(self, x: int, y: int) -> QBPEvent | None:
        if not self.config.instant_service_fulfillment:
            return None
        if self.state.h_r[x, y] <= 0 or self.state.q[x, y] <= 0:
            return None
        return QBPEvent(
            event_index=self.state.events_processed + 1,
            time=self.state.time,
            dt=0.0,
            total_rate=0.0,
            event_type="physical_service",
            event_rate=0.0,
            x=x,
            y=y,
        )

    def _instant_physical_swap_event(self, swap_idx: int) -> QBPEvent | None:
        if not self.config.instant_swap_fulfillment:
            return None
        if not _physical_swap_is_feasible(
            self.state.q,
            self.state.h_mu,
            swap_idx,
            self.swap_i,
            self.swap_y,
            self.swap_z,
        ):
            return None
        return QBPEvent(
            event_index=self.state.events_processed + 1,
            time=self.state.time,
            dt=0.0,
            total_rate=0.0,
            event_type="physical_swap",
            event_rate=0.0,
            swap_idx=int(swap_idx),
            i=int(self.swap_i[swap_idx]),
            y=int(self.swap_y[swap_idx]),
            z=int(self.swap_z[swap_idx]),
        )

    def _best_instant_swap_using_edge(self, x: int, y: int) -> int:
        if not self.config.instant_swap_fulfillment:
            return -1
        return int(
            _best_physical_swap_using_edge(
                self.state.q,
                self.state.h_mu,
                x,
                y,
                self.producer.swap_lookup,
                self.swap_i,
                self.swap_y,
                self.swap_z,
            )
        )

    def _run_instant_fulfillment_closure(
        self,
        trigger_event: QBPEvent,
        trace_writer: EventTraceWriter | None = None,
    ) -> None:
        if not self.config.instant_service_fulfillment and not self.config.instant_swap_fulfillment:
            return

        frontier_kind, frontier_x, frontier_y, frontier_swap_idx = self._instant_frontier_for_event(trigger_event)
        while frontier_kind != INSTANT_FRONTIER_NONE:
            if frontier_kind == INSTANT_FRONTIER_EDGE:
                service_event = self._instant_physical_service_event(frontier_x, frontier_y)
                if service_event is not None:
                    self.apply_event(service_event, trace_writer=trace_writer)
                    return

                swap_idx = self._best_instant_swap_using_edge(frontier_x, frontier_y)
                swap_event = self._instant_physical_swap_event(swap_idx)
                if swap_event is None:
                    return
                applied_swap = self.apply_event(swap_event, trace_writer=trace_writer)
                frontier_kind = INSTANT_FRONTIER_EDGE
                frontier_x = _require(applied_swap.y, "y")
                frontier_y = _require(applied_swap.z, "z")
                frontier_swap_idx = -1
                continue

            if frontier_kind == INSTANT_FRONTIER_SWAP:
                swap_event = self._instant_physical_swap_event(frontier_swap_idx)
                if swap_event is None:
                    return
                applied_swap = self.apply_event(swap_event, trace_writer=trace_writer)
                frontier_kind = INSTANT_FRONTIER_EDGE
                frontier_x = _require(applied_swap.y, "y")
                frontier_y = _require(applied_swap.z, "z")
                frontier_swap_idx = -1
                continue

            return
