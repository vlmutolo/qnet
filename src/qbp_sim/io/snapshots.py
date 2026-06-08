from __future__ import annotations

import io
import json
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

import zstandard as zstd


@dataclass(slots=True)
class QBPSnapshot:
    event_index: int
    time: float
    total_backlog: int
    total_inventory: int
    total_scarcity: int
    demand_arrivals: int
    pair_generations: int
    services_completed: int
    swaps_completed: int
    service_ratio: float

    def to_dict(self) -> dict[str, int | float]:
        return {
            "event_index": self.event_index,
            "time": self.time,
            "total_backlog": self.total_backlog,
            "total_inventory": self.total_inventory,
            "total_scarcity": self.total_scarcity,
            "demand_arrivals": self.demand_arrivals,
            "pair_generations": self.pair_generations,
            "services_completed": self.services_completed,
            "swaps_completed": self.swaps_completed,
            "service_ratio": self.service_ratio,
        }

    @classmethod
    def from_dict(cls, record: dict[str, int | float]) -> QBPSnapshot:
        return cls(
            event_index=int(record["event_index"]),
            time=float(record["time"]),
            total_backlog=int(record["total_backlog"]),
            total_inventory=int(record["total_inventory"]),
            total_scarcity=int(record["total_scarcity"]),
            demand_arrivals=int(record["demand_arrivals"]),
            pair_generations=int(record["pair_generations"]),
            services_completed=int(record["services_completed"]),
            swaps_completed=int(record["swaps_completed"]),
            service_ratio=float(record["service_ratio"]),
        )


class SnapshotWriter:
    """Write sampled simulation snapshots into a compressed JSONL file."""

    def __init__(self, path: str | Path, level: int = 3) -> None:
        self.path = Path(path)
        self.level = level
        self._raw_handle: io.BufferedWriter | None = None
        self._zstd_handle: io.BufferedWriter | None = None
        self._text_handle: io.TextIOWrapper | None = None

    def __enter__(self) -> SnapshotWriter:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._raw_handle = self.path.open("wb")
        compressor = zstd.ZstdCompressor(level=self.level)
        self._zstd_handle = compressor.stream_writer(self._raw_handle)
        self._text_handle = io.TextIOWrapper(self._zstd_handle, encoding="utf-8", write_through=True)
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    def write(self, snapshot: QBPSnapshot) -> None:
        if self._text_handle is None:
            raise RuntimeError("Snapshot writer must be opened with a context manager before use.")
        payload = json.dumps(snapshot.to_dict(), separators=(",", ":"))
        self._text_handle.write(payload)
        self._text_handle.write("\n")

    def close(self) -> None:
        if self._text_handle is not None:
            self._text_handle.close()
            self._text_handle = None
            self._zstd_handle = None
            self._raw_handle = None


class SnapshotReader:
    """Read compressed snapshot JSONL files as QBPSnapshot records."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self._raw_handle: io.BufferedReader | None = None
        self._zstd_handle: io.BufferedReader | None = None
        self._text_handle: io.TextIOWrapper | None = None

    def __enter__(self) -> SnapshotReader:
        self._raw_handle = self.path.open("rb")
        decompressor = zstd.ZstdDecompressor()
        self._zstd_handle = decompressor.stream_reader(self._raw_handle)
        self._text_handle = io.TextIOWrapper(self._zstd_handle, encoding="utf-8")
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    def __iter__(self) -> Iterator[QBPSnapshot]:
        if self._text_handle is None:
            raise RuntimeError("Snapshot reader must be opened with a context manager before use.")
        for line in self._text_handle:
            stripped = line.strip()
            if stripped:
                yield QBPSnapshot.from_dict(json.loads(stripped))

    def close(self) -> None:
        if self._text_handle is not None:
            self._text_handle.close()
            self._text_handle = None
            self._zstd_handle = None
            self._raw_handle = None
