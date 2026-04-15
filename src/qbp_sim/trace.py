from __future__ import annotations

import io
import json
from collections.abc import Iterator
from pathlib import Path

import zstandard as zstd

from qbp_sim.events import QBPEvent


class EventTraceWriter:
    """Write one JSON object per line into a Zstandard-compressed file."""

    def __init__(self, path: str | Path, level: int = 3) -> None:
        self.path = Path(path)
        self.level = level
        self._raw_handle: io.BufferedWriter | None = None
        self._zstd_handle: io.BufferedWriter | None = None
        self._text_handle: io.TextIOWrapper | None = None

    def __enter__(self) -> EventTraceWriter:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._raw_handle = self.path.open("wb")
        compressor = zstd.ZstdCompressor(level=self.level)
        self._zstd_handle = compressor.stream_writer(self._raw_handle)
        self._text_handle = io.TextIOWrapper(self._zstd_handle, encoding="utf-8", write_through=True)
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    def write(self, event: QBPEvent) -> None:
        if self._text_handle is None:
            raise RuntimeError("Trace writer must be opened with a context manager before use.")
        payload = json.dumps(event.to_dict(), separators=(",", ":"))
        self._text_handle.write(payload)
        self._text_handle.write("\n")

    def close(self) -> None:
        if self._text_handle is not None:
            self._text_handle.close()
            self._text_handle = None
            self._zstd_handle = None
            self._raw_handle = None


class EventTraceReader:
    """Read a Zstandard-compressed JSONL trace as QBPEvent records."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self._raw_handle: io.BufferedReader | None = None
        self._zstd_handle: io.BufferedReader | None = None
        self._text_handle: io.TextIOWrapper | None = None

    def __enter__(self) -> EventTraceReader:
        self._raw_handle = self.path.open("rb")
        decompressor = zstd.ZstdDecompressor()
        self._zstd_handle = decompressor.stream_reader(self._raw_handle)
        self._text_handle = io.TextIOWrapper(self._zstd_handle, encoding="utf-8")
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    def __iter__(self) -> Iterator[QBPEvent]:
        if self._text_handle is None:
            raise RuntimeError("Trace reader must be opened with a context manager before use.")
        for line in self._text_handle:
            stripped = line.strip()
            if stripped:
                yield QBPEvent.from_dict(json.loads(stripped))

    def close(self) -> None:
        if self._text_handle is not None:
            self._text_handle.close()
            self._text_handle = None
            self._zstd_handle = None
            self._raw_handle = None
