from __future__ import annotations

from pathlib import Path

import numpy as np
from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

from qbp_sim.simulator import (
    VIRTUAL_SWAP_POLICY_GLOBAL,
    VIRTUAL_SWAP_POLICY_POWER_OF_K_MEMORY,
    GillespieQBPConfig,
    VirtualSwapPolicy,
)


class VirtualSwapPolicyConfig(BaseModel):
    """JSON-facing virtual swap scheduler policy."""

    model_config = ConfigDict(extra="forbid")

    mode: str = Field(
        default=VIRTUAL_SWAP_POLICY_GLOBAL,
        description="Virtual swap policy: global or power_of_k_memory.",
    )
    k: int = Field(
        default=0,
        description="Number of fresh candidate swaps queried per actor refresh.",
    )
    memory: int = Field(
        default=0,
        description="Number of best candidate swaps remembered per actor.",
    )

    @model_validator(mode="after")
    def _validate_policy(self) -> VirtualSwapPolicyConfig:
        mode = self.mode.replace("-", "_")
        if mode not in {VIRTUAL_SWAP_POLICY_GLOBAL, VIRTUAL_SWAP_POLICY_POWER_OF_K_MEMORY}:
            raise ValueError(
                "virtual_swap_policy.mode must be either "
                f"{VIRTUAL_SWAP_POLICY_GLOBAL!r} or {VIRTUAL_SWAP_POLICY_POWER_OF_K_MEMORY!r}."
            )
        if self.k < 0 or self.memory < 0:
            raise ValueError("virtual_swap_policy k and memory must be non-negative.")
        if mode == VIRTUAL_SWAP_POLICY_POWER_OF_K_MEMORY and self.k <= 0:
            raise ValueError("power_of_k_memory virtual swap policy requires positive k.")
        self.mode = mode
        return self


class SimulationInputConfig(BaseModel):
    """Typed JSON-facing config for a Gillespie simulation run."""

    model_config = ConfigDict(extra="forbid")

    generation_rates: list[list[float]] = Field(
        description="Symmetric generation-rate matrix. Positive entries imply a physical edge."
    )
    consumption_rates: list[list[float]] = Field(
        description="Symmetric demand/consumption-rate matrix."
    )
    swap_rates: list[float] = Field(
        description="Per-node swap hazard rates."
    )
    capacity_headroom: float = Field(
        default=1.01,
        description=(
            "Multiplier applied to controllable capacity/opportunity rates at runtime: "
            "generation, swap, and service rates. Demand/consumption rates are not scaled."
        ),
    )
    virtual_swap_policy: VirtualSwapPolicyConfig = Field(
        default_factory=VirtualSwapPolicyConfig,
        description="Optional virtual swap selection policy.",
    )
    instant_service_fulfillment: bool = Field(
        default=False,
        description=(
            "When true, a local deterministic frontier immediately realizes one pending "
            "physical service when inventory and H^R meet on the same edge."
        ),
    )
    instant_swap_fulfillment: bool = Field(
        default=False,
        description=(
            "When true, a local deterministic frontier immediately realizes pending physical "
            "swaps instead of sampling physical swap hazards."
        ),
    )

    @model_validator(mode="after")
    def _validate_shapes(self) -> SimulationInputConfig:
        generation = np.asarray(self.generation_rates, dtype=np.float64)
        consumption = np.asarray(self.consumption_rates, dtype=np.float64)
        swap = np.asarray(self.swap_rates, dtype=np.float64)

        for name, matrix in (("generation_rates", generation), ("consumption_rates", consumption)):
            if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
                raise ValueError(f"{name} must be a square matrix.")
            if not np.allclose(matrix, matrix.T):
                raise ValueError(f"{name} must be symmetric.")
            if np.any(matrix < 0.0):
                raise ValueError(f"{name} must be non-negative.")

        if generation.shape != consumption.shape:
            raise ValueError("generation_rates and consumption_rates must have the same shape.")
        if swap.ndim != 1 or swap.shape[0] != generation.shape[0]:
            raise ValueError("swap_rates must contain one rate per node.")
        if np.any(swap < 0.0):
            raise ValueError("swap_rates must be non-negative.")
        if self.capacity_headroom <= 0.0:
            raise ValueError("capacity_headroom must be positive.")
        return self

    @property
    def num_nodes(self) -> int:
        return len(self.swap_rates)

    def to_runtime_config(self) -> GillespieQBPConfig:
        generation = np.asarray(self.generation_rates, dtype=np.float64)
        consumption = np.asarray(self.consumption_rates, dtype=np.float64)
        swap = np.asarray(self.swap_rates, dtype=np.float64)

        generation = generation.copy()
        consumption = consumption.copy()
        np.fill_diagonal(generation, 0.0)
        np.fill_diagonal(consumption, 0.0)
        headroom = float(self.capacity_headroom)
        generation *= headroom
        swap *= headroom
        service_rates = consumption.copy() * headroom

        return GillespieQBPConfig(
            generation_rates=generation,
            demand_rates=consumption,
            swap_rates=swap,
            service_rates=service_rates,
            virtual_swap_policy=VirtualSwapPolicy(
                mode=self.virtual_swap_policy.mode,
                k=self.virtual_swap_policy.k,
                memory=self.virtual_swap_policy.memory,
            ),
            instant_service_fulfillment=self.instant_service_fulfillment,
            instant_swap_fulfillment=self.instant_swap_fulfillment,
        )

    @classmethod
    def from_json_file(cls, path: str | Path) -> SimulationInputConfig:
        config_path = Path(path)
        return cls.model_validate_json(config_path.read_text())


def load_simulation_config(path: str | Path) -> SimulationInputConfig:
    try:
        return SimulationInputConfig.from_json_file(path)
    except ValidationError as exc:
        raise ValueError(str(exc)) from exc
