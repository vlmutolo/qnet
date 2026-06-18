
from __future__ import annotations

from qbp_sim.core.kernels import (
    _apply_demand_arrival,
    _apply_direct_physical_swap,
    _apply_pair_generation,
    _apply_physical_service,
    _apply_physical_swap,
    _apply_service_request,
    _apply_virtual_service,
    _apply_virtual_swap,
)
from qbp_sim.core.types import QBPState, _require
from qbp_sim.io.events import QBPEvent

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
        elif event.event_type == "service_request":
            x = _require(event.x, "x")
            y = _require(event.y, "y")
            _apply_service_request(state.d, state.h_r, x, y)
            state.virtual_service_requests += 1
            state.total_virtual_backlog_count -= 1
            state.total_service_deficit_count += 1
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
        elif event.event_type == "virtual_swap_idle":
            _require(event.i, "i")
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
        elif event.event_type == "max_min_swap":
            _apply_direct_physical_swap(
                state.q,
                _require(event.i, "i"),
                _require(event.y, "y"),
                _require(event.z, "z"),
            )
            state.swaps_completed += 1
            state.total_inventory_count -= 1
        elif event.event_type == "max_min_swap_idle":
            _require(event.i, "i")
        else:
            raise ValueError(f"Unknown event type: {event.event_type}")

        state.events_processed += 1
        event.backlog_total = state.total_backlog
        event.inventory_total = state.total_inventory
        event.scarcity_total = state.total_scarcity
        return event
