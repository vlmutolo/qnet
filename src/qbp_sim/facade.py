from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path

import numpy as np

from qbp_sim.config import SimulationInputConfig, load_simulation_config
from qbp_sim.core.replay import replay_event_stream
from qbp_sim.core.simulator import GillespieQBPSimulator
from qbp_sim.core.types import GillespieQBPResult
from qbp_sim.io.snapshots import SnapshotWriter
from qbp_sim.io.trace import open_event_trace_reader, open_event_trace_writer


class VirtualSwapPolicyMode(StrEnum):
    BP = "bp"
    LIMITED_INFO_BP = "limited_info_bp"
    MAX_MIN = "max_min"
    LIMITED_INFO_MAX_MIN = "limited_info_max_min"


class TopologyName(StrEnum):
    CYCLE = "cycle"
    CHAIN = "chain"
    GRID = "grid"


class TraceFloatPrecision(StrEnum):
    FLOAT16 = "float16"
    FLOAT32 = "float32"
    FLOAT64 = "float64"


class TraceFormat(StrEnum):
    VORTEX = "vortex"
    PARQUET = "parquet"
    JSONL_ZST = "jsonl_zst"


class TraceTimeMode(StrEnum):
    FULL = "full"
    NONE = "none"


@dataclass(frozen=True, slots=True)
class RunOptions:
    until_time: float = 50.0
    max_events: int | None = None
    sample_every: int = 500
    seed: int = 0
    trace_path: str | Path | None = None
    snapshots_path: str | Path | None = None
    trace_format: TraceFormat | str | None = None
    trace_float_precision: TraceFloatPrecision | str = TraceFloatPrecision.FLOAT32
    trace_time_mode: TraceTimeMode | str = TraceTimeMode.FULL
    progress: bool | None = None


@dataclass(frozen=True, slots=True)
class RunOutput:
    result: GillespieQBPResult
    trace_path: Path | None = None
    snapshots_path: Path | None = None

    @property
    def service_ratio(self) -> float:
        if self.result.demand_arrivals == 0:
            return 0.0
        return self.result.services_completed / self.result.demand_arrivals


def build_four_node_example_config() -> SimulationInputConfig:
    """Return the documented four-node example as a JSON-facing config object."""
    n_nodes = 4
    generation_rates = np.zeros((n_nodes, n_nodes), dtype=float)
    consumption_rates = np.zeros((n_nodes, n_nodes), dtype=float)
    swap_rates = np.zeros(n_nodes, dtype=float)

    for a, b in ((0, 1), (0, 2), (1, 3), (2, 3)):
        generation_rates[a, b] = 1.0
        generation_rates[b, a] = 1.0

    consumption_rates[0, 3] = 2.0
    consumption_rates[3, 0] = 2.0
    swap_rates[1] = 1.0
    swap_rates[2] = 1.0

    return SimulationInputConfig(
        generation_rates=generation_rates.tolist(),
        consumption_rates=consumption_rates.tolist(),
        swap_rates=swap_rates.tolist(),
    )


def _coerce_input_config(config: SimulationInputConfig | str | Path) -> SimulationInputConfig:
    if isinstance(config, SimulationInputConfig):
        return config
    return load_simulation_config(config)


def run_simulation(
    config: SimulationInputConfig | str | Path,
    options: RunOptions | None = None,
) -> RunOutput:
    opts = options or RunOptions()
    input_config = _coerce_input_config(config)
    simulator = GillespieQBPSimulator(input_config.to_runtime_config(), seed=opts.seed)

    trace_path = None if opts.trace_path is None else Path(opts.trace_path)
    snapshots_path = None if opts.snapshots_path is None else Path(opts.snapshots_path)

    if trace_path is None and snapshots_path is None:
        result = simulator.run(
            until_time=opts.until_time,
            max_events=opts.max_events,
            sample_every=opts.sample_every,
            progress=opts.progress,
        )
    elif trace_path is not None and snapshots_path is None:
        with open_event_trace_writer(
            trace_path,
            trace_format=None if opts.trace_format is None else str(opts.trace_format),
            float_precision=str(opts.trace_float_precision),
            time_mode=str(opts.trace_time_mode),
        ) as trace_writer:
            result = simulator.run(
                until_time=opts.until_time,
                max_events=opts.max_events,
                sample_every=opts.sample_every,
                trace_writer=trace_writer,
                progress=opts.progress,
            )
    elif trace_path is None and snapshots_path is not None:
        with SnapshotWriter(snapshots_path) as snapshot_writer:
            result = simulator.run(
                until_time=opts.until_time,
                max_events=opts.max_events,
                sample_every=opts.sample_every,
                snapshot_writer=snapshot_writer,
                progress=opts.progress,
            )
    else:
        assert trace_path is not None
        assert snapshots_path is not None
        with open_event_trace_writer(
            trace_path,
            trace_format=None if opts.trace_format is None else str(opts.trace_format),
            float_precision=str(opts.trace_float_precision),
            time_mode=str(opts.trace_time_mode),
        ) as trace_writer, SnapshotWriter(snapshots_path) as snapshot_writer:
            result = simulator.run(
                until_time=opts.until_time,
                max_events=opts.max_events,
                sample_every=opts.sample_every,
                trace_writer=trace_writer,
                snapshot_writer=snapshot_writer,
                progress=opts.progress,
            )

    return RunOutput(result=result, trace_path=trace_path, snapshots_path=snapshots_path)


def replay_trace(
    trace_path: str | Path,
    *,
    config: SimulationInputConfig | str | Path | None = None,
    final_time: float | None = None,
    sample_every: int = 500,
    snapshots_path: str | Path | None = None,
) -> RunOutput:
    input_config = build_four_node_example_config() if config is None else _coerce_input_config(config)
    resolved_trace_path = Path(trace_path)
    resolved_snapshots_path = None if snapshots_path is None else Path(snapshots_path)

    if resolved_snapshots_path is None:
        with open_event_trace_reader(resolved_trace_path) as trace_reader:
            result = replay_event_stream(
                config=input_config.to_runtime_config(),
                events=trace_reader,
                sample_every=sample_every,
                final_time=final_time,
            )
    else:
        with open_event_trace_reader(resolved_trace_path) as trace_reader, SnapshotWriter(
            resolved_snapshots_path
        ) as snapshot_writer:
            result = replay_event_stream(
                config=input_config.to_runtime_config(),
                events=trace_reader,
                sample_every=sample_every,
                final_time=final_time,
                snapshot_writer=snapshot_writer,
            )

    return RunOutput(
        result=result,
        trace_path=resolved_trace_path,
        snapshots_path=resolved_snapshots_path,
    )
