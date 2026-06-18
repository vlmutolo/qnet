
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from numpy.typing import NDArray

Array1D = NDArray[np.float64]
IntArray1D = NDArray[np.int64]
FloatMatrix = NDArray[np.float64]
IntMatrix = NDArray[np.int64]

VIRTUAL_SWAP_POLICY_BP = "bp"
VIRTUAL_SWAP_POLICY_LIMITED_INFO_BP = "limited_info_bp"
VIRTUAL_SWAP_POLICY_MAX_MIN = "max_min"
VIRTUAL_SWAP_POLICY_LIMITED_INFO_MAX_MIN = "limited_info_max_min"

VIRTUAL_SWAP_POLICY_ALIASES = {
    "global": VIRTUAL_SWAP_POLICY_BP,
    "power_of_k_memory": VIRTUAL_SWAP_POLICY_LIMITED_INFO_BP,
}


def normalize_virtual_swap_policy_mode(mode: str) -> str:
    normalized = mode.replace("-", "_")
    return VIRTUAL_SWAP_POLICY_ALIASES.get(normalized, normalized)


@dataclass(frozen=True, slots=True)
class VirtualSwapPolicy:
    mode: str = VIRTUAL_SWAP_POLICY_BP
    k: int = 0
    memory: int = 0


@dataclass(slots=True)
class GillespieQBPConfig:
    generation_rates: FloatMatrix
    demand_rates: FloatMatrix
    swap_rates: IntArray1D | Array1D
    service_rates: FloatMatrix
    virtual_swap_policy: VirtualSwapPolicy = field(default_factory=VirtualSwapPolicy)
    instant_service_fulfillment: bool = False
    instant_swap_fulfillment: bool = False


INSTANT_FRONTIER_NONE = 0
INSTANT_FRONTIER_EDGE = 1
INSTANT_FRONTIER_SWAP = 2
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



def _require(value: int | None, name: str) -> int:
    if value is None:
        raise ValueError(f"Event is missing required field {name}.")
    return value
