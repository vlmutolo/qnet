from __future__ import annotations

import io
import json
from collections.abc import Iterator
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import zstandard as zstd

from qbp_sim.events import QBPEvent

TRACE_COLUMNS = (
    "event_index",
    "time",
    "dt",
    "total_rate",
    "event_type",
    "event_rate",
    "swap_idx",
    "x",
    "y",
    "z",
    "i",
    "backlog_total",
    "inventory_total",
    "scarcity_total",
)

TRACE_SCHEMA = pa.schema(
    [
        ("event_index", pa.int64()),
        ("time", pa.float64()),
        ("dt", pa.float64()),
        ("total_rate", pa.float64()),
        ("event_type", pa.string()),
        ("event_rate", pa.float64()),
        ("swap_idx", pa.int64()),
        ("x", pa.int64()),
        ("y", pa.int64()),
        ("z", pa.int64()),
        ("i", pa.int64()),
        ("backlog_total", pa.int64()),
        ("inventory_total", pa.int64()),
        ("scarcity_total", pa.int64()),
    ]
)


def _event_to_trace_row(event: QBPEvent) -> dict[str, int | float | str | None]:
    return {column: getattr(event, column) for column in TRACE_COLUMNS}


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


class ParquetEventTraceWriter:
    """Write events into a buffered Parquet file for replay and columnar analysis."""

    def __init__(
        self,
        path: str | Path,
        *,
        buffer_size: int = 65_536,
        compression: str = "zstd",
    ) -> None:
        if buffer_size <= 0:
            raise ValueError("buffer_size must be positive.")
        self.path = Path(path)
        self.buffer_size = buffer_size
        self.compression = compression
        self._writer: pq.ParquetWriter | None = None
        self._rows: list[dict[str, int | float | str | None]] = []

    def __enter__(self) -> ParquetEventTraceWriter:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._writer = pq.ParquetWriter(
            self.path,
            TRACE_SCHEMA,
            compression=self.compression,
            use_dictionary=["event_type"],
        )
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    def write(self, event: QBPEvent) -> None:
        if self._writer is None:
            raise RuntimeError("Parquet trace writer must be opened with a context manager before use.")
        self._rows.append(_event_to_trace_row(event))
        if len(self._rows) >= self.buffer_size:
            self.flush()

    def flush(self) -> None:
        if self._writer is None:
            raise RuntimeError("Parquet trace writer must be opened with a context manager before use.")
        if not self._rows:
            return
        table = pa.Table.from_pylist(self._rows, schema=TRACE_SCHEMA)
        self._writer.write_table(table)
        self._rows.clear()

    def close(self) -> None:
        if self._writer is not None:
            self.flush()
            self._writer.close()
            self._writer = None


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


class ParquetEventTraceReader:
    """Read a Parquet event trace as QBPEvent records."""

    def __init__(self, path: str | Path, batch_size: int = 65_536) -> None:
        if batch_size <= 0:
            raise ValueError("batch_size must be positive.")
        self.path = Path(path)
        self.batch_size = batch_size
        self._parquet_file: pq.ParquetFile | None = None

    def __enter__(self) -> ParquetEventTraceReader:
        self._parquet_file = pq.ParquetFile(self.path)
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    def __iter__(self) -> Iterator[QBPEvent]:
        if self._parquet_file is None:
            raise RuntimeError("Parquet trace reader must be opened with a context manager before use.")
        for batch in self._parquet_file.iter_batches(batch_size=self.batch_size, columns=list(TRACE_COLUMNS)):
            for record in batch.to_pylist():
                yield QBPEvent.from_dict(record)

    def close(self) -> None:
        self._parquet_file = None


def open_event_trace_writer(path: str | Path):
    trace_path = Path(path)
    if trace_path.suffix.lower() == ".parquet":
        return ParquetEventTraceWriter(trace_path)
    return EventTraceWriter(trace_path)


def open_event_trace_reader(path: str | Path):
    trace_path = Path(path)
    if trace_path.suffix.lower() == ".parquet":
        return ParquetEventTraceReader(trace_path)
    return EventTraceReader(trace_path)
