
from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path

from qbp_sim.analysis import summarize_snapshots
from qbp_sim.config import SimulationInputConfig
from qbp_sim.core.types import GillespieQBPResult
from qbp_sim.core.simulator import GillespieQBPSimulator
from qbp_sim.io.snapshots import QBPSnapshot

@dataclass(slots=True)
class CycleServiceRatioRun:
    n_nodes: int
    lp_json_path: Path
    simulation_config_path: Path
    trace_path: Path
    metadata_path: Path
    snapshots: list[QBPSnapshot]

    @property
    def summary(self):
        return summarize_snapshots(self.snapshots)


@dataclass(slots=True)
class HeadroomRun:
    n_nodes: int
    capacity_headroom: float
    lp_json_path: Path
    simulation_config_path: Path
    trace_path: Path
    metadata_path: Path
    snapshots: list[QBPSnapshot]

    @property
    def summary(self):
        return summarize_snapshots(self.snapshots)


@dataclass(slots=True)
class LimitedInfoServiceRatioRun:
    n_nodes: int
    policy_label: str
    policy_mode: str
    k: int | None
    memory: int | None
    lp_json_path: Path
    simulation_config_path: Path
    trace_path: Path
    metadata_path: Path
    snapshots: list[QBPSnapshot]

    @property
    def summary(self):
        return summarize_snapshots(self.snapshots)


class _MemorySnapshotWriter:
    def __init__(self) -> None:
        self.snapshots: list[QBPSnapshot] = []

    def write(self, snapshot: QBPSnapshot) -> None:
        self.snapshots.append(snapshot)


def _simulator_state_payload(simulator: GillespieQBPSimulator) -> dict[str, list]:
    state = simulator.state
    return {
        "q": state.q.tolist(),
        "d": state.d.tolist(),
        "alpha": state.alpha.tolist(),
        "h_r": state.h_r.tolist(),
        "h_mu": state.h_mu.tolist(),
    }


def _result_payload(result: GillespieQBPResult) -> dict[str, int | float | bool]:
    return {
        "final_time": result.final_time,
        "events_processed": result.events_processed,
        "total_backlog": result.total_backlog,
        "total_inventory": result.total_inventory,
        "total_scarcity": result.total_scarcity,
        "demand_arrivals": result.demand_arrivals,
        "pair_generations": result.pair_generations,
        "virtual_service_requests": result.virtual_service_requests,
        "virtual_swap_requests": result.virtual_swap_requests,
        "services_completed": result.services_completed,
        "swaps_completed": result.swaps_completed,
        "service_ratio": (
            0.0
            if result.demand_arrivals == 0
            else float(result.services_completed) / float(result.demand_arrivals)
        ),
    }


def _write_run_metadata(
    path: str | Path,
    *,
    command: str,
    n_nodes: int,
    seed: int,
    until_time: float,
    max_events: int | None,
    sample_every: int,
    burn_in_time: float,
    trace_float_precision: str,
    trace_format: str,
    trace_time_mode: str,
    simulation_config_path: Path,
    trace_path: Path,
    lp_json_path: Path,
    result: GillespieQBPResult,
    initial_state: dict[str, list] | None = None,
    extra: dict[str, int | float | str | bool | None] | None = None,
) -> None:
    metadata = {
        "schema_version": 1,
        "command": command,
        "n_nodes": n_nodes,
        "seed": seed,
        "until_time": until_time,
        "max_events": max_events,
        "sample_every": sample_every,
        "burn_in_time": burn_in_time,
        "trace_float_precision": trace_float_precision,
        "trace_format": trace_format,
        "trace_time_mode": trace_time_mode,
        "hit_time_horizon": result.final_time >= until_time,
        "hit_event_cap": (
            False if max_events is None else result.events_processed >= max_events and result.final_time < until_time
        ),
        "simulation_config_path": str(simulation_config_path),
        "trace_path": str(trace_path),
        "lp_json_path": str(lp_json_path),
        "result": _result_payload(result),
        "initial_state": initial_state,
        "extra": extra or {},
    }
    metadata_path = Path(path)
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")


def _load_linear_module():
    from qbp_sim.lp import linear

    return linear


def _cycle_consumption_edge_fraction(
    n_nodes: int,
    cons_edge_fraction: float | None,
) -> float:
    if cons_edge_fraction is not None:
        return cons_edge_fraction
    total_pairs = math.comb(n_nodes, 2)
    target_pairs = max(1, n_nodes // 2)
    return min(1.0, float(target_pairs) / float(total_pairs))


def _headroom_slug(headroom: float) -> str:
    return f"{headroom:.6f}".rstrip("0").rstrip(".").replace(".", "p")


def _policy_slug(policy_label: str) -> str:
    return (
        policy_label.lower()
        .replace(" ", "_")
        .replace("=", "")
        .replace(",", "")
        .replace("/", "_")
    )


def _apply_capacity_headroom(
    simulation_input: SimulationInputConfig,
    headroom: float,
) -> SimulationInputConfig:
    if headroom <= 0.0:
        raise ValueError("capacity headroom must be positive.")
    return simulation_input.model_copy(update={"capacity_headroom": float(headroom)})


def _apply_instant_service_fulfillment(
    simulation_input: SimulationInputConfig,
    service_enabled: bool,
    swap_enabled: bool = False,
) -> SimulationInputConfig:
    return simulation_input.model_copy(
        update={
            "instant_service_fulfillment": bool(service_enabled),
            "instant_swap_fulfillment": bool(swap_enabled),
        }
    )
