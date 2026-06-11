from __future__ import annotations

from itertools import product
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

from qbp_sim.config import VirtualSwapPolicyConfig
from qbp_sim.core.types import (
    VIRTUAL_SWAP_POLICY_GLOBAL,
    VIRTUAL_SWAP_POLICY_MAX_MIN,
    VIRTUAL_SWAP_POLICY_POWER_OF_K_MEMORY,
)

TopologyName = Literal["cycle", "chain", "grid"]
TraceFloatPrecision = Literal["float16", "float32", "float64"]
TraceTimeMode = Literal["full", "none"]


def _slug_float(value: float) -> str:
    return f"{value:.6f}".rstrip("0").rstrip(".").replace(".", "p")


def _slug_text(value: str) -> str:
    return (
        value.lower()
        .replace(" ", "_")
        .replace("=", "")
        .replace(",", "")
        .replace("/", "_")
    )


class ExperimentPolicyConfig(BaseModel):
    """One virtual-swap policy variant in an experiment matrix."""

    model_config = ConfigDict(extra="forbid")

    mode: str = Field(
        default=VIRTUAL_SWAP_POLICY_GLOBAL,
        description="Virtual swap policy: global, power_of_k_memory, or max_min.",
    )
    k: int | None = Field(
        default=None,
        description="Fresh candidate swaps queried per actor refresh for power_of_k_memory.",
    )
    memory: int | None = Field(
        default=None,
        description="Remembered best candidates per actor for power_of_k_memory.",
    )
    label: str | None = Field(
        default=None,
        description="Optional plot/run label. Defaults to full info or limited k=<k>, m=<memory>.",
    )

    @model_validator(mode="after")
    def _validate_policy(self) -> ExperimentPolicyConfig:
        self.mode = self.mode.replace("-", "_")
        if self.mode not in {
            VIRTUAL_SWAP_POLICY_GLOBAL,
            VIRTUAL_SWAP_POLICY_POWER_OF_K_MEMORY,
            VIRTUAL_SWAP_POLICY_MAX_MIN,
        }:
            raise ValueError(
                "policy mode must be either "
                f"{VIRTUAL_SWAP_POLICY_GLOBAL!r}, {VIRTUAL_SWAP_POLICY_POWER_OF_K_MEMORY!r}, "
                f"or {VIRTUAL_SWAP_POLICY_MAX_MIN!r}."
            )
        if self.mode in {VIRTUAL_SWAP_POLICY_GLOBAL, VIRTUAL_SWAP_POLICY_MAX_MIN}:
            if self.k is not None or self.memory is not None:
                raise ValueError(f"{self.mode} policy must not set k or memory.")
            return self
        if self.k is None or self.k <= 0:
            raise ValueError("power_of_k_memory policy requires positive k.")
        if self.memory is None or self.memory < 0:
            raise ValueError("power_of_k_memory policy requires non-negative memory.")
        return self

    @property
    def resolved_label(self) -> str:
        if self.label is not None:
            return self.label
        if self.mode == VIRTUAL_SWAP_POLICY_GLOBAL:
            return "full info"
        if self.mode == VIRTUAL_SWAP_POLICY_MAX_MIN:
            return "max-min"
        return f"limited k={self.k}, m={self.memory}"

    def to_virtual_swap_policy_config(self) -> VirtualSwapPolicyConfig:
        return VirtualSwapPolicyConfig(
            mode=self.mode,
            k=0 if self.k is None else self.k,
            memory=0 if self.memory is None else self.memory,
        )


class ExperimentMatrixCase(BaseModel):
    """A fully resolved experiment case produced by an ExperimentMatrixConfig."""

    model_config = ConfigDict(extra="forbid")

    topology: TopologyName
    n_nodes: int
    consumption_edge_fraction: float | None
    capacity_headroom: float
    policy_label: str
    policy_mode: str
    k: int | None
    memory: int | None
    virtual_swap_policy: VirtualSwapPolicyConfig
    seed: int
    edge_weight: float
    gen_scale: float
    cons_scale: float
    cons_max_edge_weight: float
    objective: str
    swap_rate: float
    burn_in_time: float
    until_time: float
    max_events: int | None
    sample_every: int
    trace_float_precision: TraceFloatPrecision
    trace_time_mode: TraceTimeMode
    instant_service_fulfillment: bool
    instant_swap_fulfillment: bool

    @property
    def slug(self) -> str:
        sparsity = "default_sparsity"
        if self.consumption_edge_fraction is not None:
            sparsity = f"cons_frac_{_slug_float(self.consumption_edge_fraction)}"
        return "_".join(
            [
                self.topology,
                f"n{self.n_nodes}",
                f"headroom_{_slug_float(self.capacity_headroom)}",
                sparsity,
                _slug_text(self.policy_label),
                f"seed{self.seed}",
            ]
        )


class ExperimentMatrixConfig(BaseModel):
    """JSON-facing Cartesian product for backpressure experiment sweeps."""

    model_config = ConfigDict(extra="forbid")

    topologies: list[TopologyName] = Field(
        default_factory=lambda: ["cycle"],
        description="Physical generation topologies to test. Supported LP topologies: cycle, chain, grid.",
    )
    graph_sizes: list[int] = Field(
        default_factory=lambda: [4],
        description="Node counts to test.",
    )
    consumption_edge_fractions: list[float | None] = Field(
        default_factory=lambda: [None],
        description=(
            "Fraction of node pairs with nonzero consumption demand. "
            "Use null to keep the size-aware default used by the experiment helpers."
        ),
    )
    headrooms: list[float] = Field(
        default_factory=lambda: [1.01],
        description="Capacity multipliers applied to generation, swap, and service opportunity rates.",
    )
    policies: list[ExperimentPolicyConfig] = Field(
        default_factory=lambda: [
            ExperimentPolicyConfig(mode=VIRTUAL_SWAP_POLICY_GLOBAL),
            ExperimentPolicyConfig(mode=VIRTUAL_SWAP_POLICY_POWER_OF_K_MEMORY, k=1, memory=1),
        ],
        description="Virtual-swap policy variants to test.",
    )
    edge_weights: list[float] = Field(
        default_factory=lambda: [10.0],
        description="Physical edge weights/capacities used before solving the LP.",
    )
    gen_scales: list[float] = Field(
        default_factory=lambda: [10.0],
        description="Generation-capacity scale factors used before solving the LP.",
    )
    cons_scales: list[float] = Field(
        default_factory=lambda: [1.0],
        description="Consumption-demand scale factors used before solving the LP.",
    )
    swap_rates: list[float] = Field(
        default_factory=lambda: [100.0],
        description="Uniform per-node LP swap caps / simulator swap opportunity rates before headroom.",
    )
    seed_offsets: list[int] = Field(
        default_factory=lambda: [0],
        description="Offsets added to seed_base + n_nodes for repeated stochastic replicates.",
    )
    cons_max_edge_weight: float = Field(
        default=7.0,
        description="Maximum sampled consumption edge weight before cons_scale.",
    )
    objective: str = Field(
        default="min_sum_generate",
        description="LP objective name passed through to the LP benchmark builder.",
    )
    burn_in_time: float = Field(
        default=0.0,
        description="Optional warmup time before resetting measurements.",
    )
    until_time: float = Field(
        default=1_000.0,
        description="Measured simulation horizon.",
    )
    max_events: int | None = Field(
        default=None,
        description="Optional sampled-event cap. Null means no event cap.",
    )
    sample_every: int = Field(
        default=1_000,
        description="Snapshot cadence in applied events.",
    )
    seed_base: int = Field(
        default=0,
        description="Base seed. Concrete case seed is seed_base + n_nodes + seed_offset.",
    )
    trace_float_precision: TraceFloatPrecision = Field(
        default="float32",
        description="Floating-point precision for columnar event traces.",
    )
    trace_time_mode: TraceTimeMode = Field(
        default="full",
        description=(
            "Whether traces persist timing/rate fields. Use none for smaller traces that preserve event order "
            "but cannot reconstruct simulation time."
        ),
    )
    instant_service_fulfillment: list[bool] = Field(
        default_factory=lambda: [False],
        description="Whether to test deterministic immediate physical service fulfillment.",
    )
    instant_swap_fulfillment: list[bool] = Field(
        default_factory=lambda: [False],
        description="Whether to test deterministic immediate physical swap fulfillment.",
    )

    @model_validator(mode="after")
    def _validate_matrix(self) -> ExperimentMatrixConfig:
        for field_name in (
            "topologies",
            "graph_sizes",
            "consumption_edge_fractions",
            "headrooms",
            "policies",
            "edge_weights",
            "gen_scales",
            "cons_scales",
            "swap_rates",
            "seed_offsets",
            "instant_service_fulfillment",
            "instant_swap_fulfillment",
        ):
            if not getattr(self, field_name):
                raise ValueError(f"{field_name} must not be empty.")
        if any(n_nodes < 2 for n_nodes in self.graph_sizes):
            raise ValueError("graph_sizes must contain node counts >= 2.")
        if any(headroom <= 0.0 for headroom in self.headrooms):
            raise ValueError("headrooms must be positive.")
        for fraction in self.consumption_edge_fractions:
            if fraction is not None and not (0.0 < fraction <= 1.0):
                raise ValueError("consumption_edge_fractions must be null or in (0, 1].")
        if any(edge_weight <= 0.0 for edge_weight in self.edge_weights):
            raise ValueError("edge_weights must be positive.")
        if any(gen_scale <= 0.0 for gen_scale in self.gen_scales):
            raise ValueError("gen_scales must be positive.")
        if any(cons_scale <= 0.0 for cons_scale in self.cons_scales):
            raise ValueError("cons_scales must be positive.")
        if any(swap_rate < 0.0 for swap_rate in self.swap_rates):
            raise ValueError("swap_rates must be non-negative.")
        if self.cons_max_edge_weight <= 0.0:
            raise ValueError("cons_max_edge_weight must be positive.")
        if self.burn_in_time < 0.0:
            raise ValueError("burn_in_time must be non-negative.")
        if self.until_time <= 0.0:
            raise ValueError("until_time must be positive.")
        if self.max_events is not None and self.max_events <= 0:
            raise ValueError("max_events must be null or positive.")
        if self.sample_every <= 0:
            raise ValueError("sample_every must be positive.")
        return self

    @property
    def case_count(self) -> int:
        return len(self.cases())

    def cases(self) -> list[ExperimentMatrixCase]:
        rows: list[ExperimentMatrixCase] = []
        for (
            topology,
            n_nodes,
            consumption_edge_fraction,
            headroom,
            policy,
            edge_weight,
            gen_scale,
            cons_scale,
            swap_rate,
            seed_offset,
            instant_service,
            instant_swap,
        ) in product(
            self.topologies,
            self.graph_sizes,
            self.consumption_edge_fractions,
            self.headrooms,
            self.policies,
            self.edge_weights,
            self.gen_scales,
            self.cons_scales,
            self.swap_rates,
            self.seed_offsets,
            self.instant_service_fulfillment,
            self.instant_swap_fulfillment,
        ):
            rows.append(
                ExperimentMatrixCase(
                    topology=topology,
                    n_nodes=n_nodes,
                    consumption_edge_fraction=consumption_edge_fraction,
                    capacity_headroom=headroom,
                    policy_label=policy.resolved_label,
                    policy_mode=policy.mode,
                    k=policy.k,
                    memory=policy.memory,
                    virtual_swap_policy=policy.to_virtual_swap_policy_config(),
                    seed=self.seed_base + n_nodes + seed_offset,
                    edge_weight=edge_weight,
                    gen_scale=gen_scale,
                    cons_scale=cons_scale,
                    cons_max_edge_weight=self.cons_max_edge_weight,
                    objective=self.objective,
                    swap_rate=swap_rate,
                    burn_in_time=self.burn_in_time,
                    until_time=self.until_time,
                    max_events=self.max_events,
                    sample_every=self.sample_every,
                    trace_float_precision=self.trace_float_precision,
                    trace_time_mode=self.trace_time_mode,
                    instant_service_fulfillment=instant_service,
                    instant_swap_fulfillment=instant_swap,
                )
            )
        return rows

    @classmethod
    def from_json_file(cls, path: str | Path) -> ExperimentMatrixConfig:
        config_path = Path(path)
        return cls.model_validate_json(config_path.read_text())


def load_experiment_matrix_config(path: str | Path) -> ExperimentMatrixConfig:
    try:
        return ExperimentMatrixConfig.from_json_file(path)
    except ValidationError as exc:
        raise ValueError(str(exc)) from exc
