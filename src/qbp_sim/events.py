from __future__ import annotations

from dataclasses import dataclass


@dataclass(slots=True)
class QBPEvent:
    event_index: int
    time: float
    dt: float
    total_rate: float
    event_type: str
    event_rate: float
    swap_idx: int | None = None
    x: int | None = None
    y: int | None = None
    z: int | None = None
    i: int | None = None
    backlog_total: int | None = None
    inventory_total: int | None = None
    scarcity_total: int | None = None

    def to_dict(self) -> dict[str, int | float | str]:
        record: dict[str, int | float | str] = {
            "event_index": self.event_index,
            "time": self.time,
            "dt": self.dt,
            "total_rate": self.total_rate,
            "event_type": self.event_type,
            "event_rate": self.event_rate,
        }
        for name in (
            "swap_idx",
            "x",
            "y",
            "z",
            "i",
            "backlog_total",
            "inventory_total",
            "scarcity_total",
        ):
            value = getattr(self, name)
            if value is not None:
                record[name] = value
        return record

    @classmethod
    def from_dict(cls, record: dict[str, int | float | str]) -> QBPEvent:
        return cls(
            event_index=int(record["event_index"]),
            time=float(record["time"]),
            dt=float(record["dt"]),
            total_rate=float(record["total_rate"]),
            event_type=str(record["event_type"]),
            event_rate=float(record["event_rate"]),
            swap_idx=_optional_int(record.get("swap_idx")),
            x=_optional_int(record.get("x")),
            y=_optional_int(record.get("y")),
            z=_optional_int(record.get("z")),
            i=_optional_int(record.get("i")),
            backlog_total=_optional_int(record.get("backlog_total")),
            inventory_total=_optional_int(record.get("inventory_total")),
            scarcity_total=_optional_int(record.get("scarcity_total")),
        )


def _optional_int(value: object) -> int | None:
    if value is None:
        return None
    return int(value)
