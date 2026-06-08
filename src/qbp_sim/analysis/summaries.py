
from __future__ import annotations

from dataclasses import dataclass

from qbp_sim.io.snapshots import QBPSnapshot

@dataclass(slots=True)
class SnapshotSummary:
    num_snapshots: int
    final_time: float
    final_service_ratio: float
    max_backlog: int
    max_inventory: int
    max_scarcity: int
    final_backlog: int
    final_inventory: int
    final_scarcity: int
    demand_arrivals: int
    services_completed: int

    def format_summary(self) -> str:
        return "\n".join(
            [
                "QBP snapshot analysis",
                f"snapshots={self.num_snapshots}",
                f"final_time={self.final_time:.6f}",
                f"final_service_ratio={self.final_service_ratio:.6f}",
                f"max_backlog={self.max_backlog}",
                f"max_inventory={self.max_inventory}",
                f"max_scarcity={self.max_scarcity}",
                f"final_backlog={self.final_backlog}",
                f"final_inventory={self.final_inventory}",
                f"final_scarcity={self.final_scarcity}",
                f"demand_arrivals={self.demand_arrivals}",
                f"services_completed={self.services_completed}",
            ]
        )


def summarize_snapshots(snapshots: list[QBPSnapshot]) -> SnapshotSummary:
    if not snapshots:
        raise ValueError("No snapshots were provided for analysis.")
    final = snapshots[-1]
    return SnapshotSummary(
        num_snapshots=len(snapshots),
        final_time=final.time,
        final_service_ratio=final.service_ratio,
        max_backlog=max(snapshot.total_backlog for snapshot in snapshots),
        max_inventory=max(snapshot.total_inventory for snapshot in snapshots),
        max_scarcity=max(snapshot.total_scarcity for snapshot in snapshots),
        final_backlog=final.total_backlog,
        final_inventory=final.total_inventory,
        final_scarcity=final.total_scarcity,
        demand_arrivals=final.demand_arrivals,
        services_completed=final.services_completed,
    )
