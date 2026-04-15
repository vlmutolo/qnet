from __future__ import annotations

from pathlib import Path

import numpy as np
from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

from qbp_sim.simulator import GillespieQBPConfig


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
        service_rates = consumption.copy()

        return GillespieQBPConfig(
            generation_rates=generation,
            demand_rates=consumption,
            swap_rates=swap,
            service_rates=service_rates,
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
