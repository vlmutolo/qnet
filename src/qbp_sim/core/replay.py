
from __future__ import annotations

from collections.abc import Iterable

from qbp_sim.core.simulator import GillespieQBPSimulator
from qbp_sim.core.types import GillespieQBPConfig, GillespieQBPResult, IntArray1D, IntMatrix
from qbp_sim.io.events import QBPEvent
from qbp_sim.io.snapshots import SnapshotWriter

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
