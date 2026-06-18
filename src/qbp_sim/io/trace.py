from __future__ import annotations

import io
import json
import queue
import threading
from collections.abc import Iterator
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import vortex as vx
import zstandard as zstd

from qbp_sim.io.events import QBPEvent

TRACE_COLUMNS = (
    "event_index",
    "time",
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
TIME_TRACE_COLUMNS = ("time", "total_rate", "event_rate")

DEFAULT_TRACE_FLOAT_PRECISION = "float32"
DEFAULT_TRACE_FORMAT = "vortex"
TRACE_FORMAT_VORTEX = "vortex"
TRACE_FORMAT_PARQUET = "parquet"
TRACE_FORMAT_JSONL_ZST = "jsonl_zst"
TRACE_FORMATS = {TRACE_FORMAT_VORTEX, TRACE_FORMAT_PARQUET, TRACE_FORMAT_JSONL_ZST}
TRACE_TIME_MODE_FULL = "full"
TRACE_TIME_MODE_NONE = "none"
TRACE_TIME_MODES = {TRACE_TIME_MODE_FULL, TRACE_TIME_MODE_NONE}
TRACE_FLOAT_TYPES = {
    "float16": pa.float16(),
    "float32": pa.float32(),
    "float64": pa.float64(),
}
FLOAT16_MAX_FINITE = 65_504.0


def trace_schema(
    float_precision: str = DEFAULT_TRACE_FLOAT_PRECISION,
    *,
    time_mode: str = TRACE_TIME_MODE_FULL,
) -> pa.Schema:
    float_type = _trace_float_type(float_precision)
    normalized_time_mode = _trace_time_mode(time_mode)
    fields: list[tuple[str, pa.DataType]] = [
        ("event_index", pa.int64()),
    ]
    if normalized_time_mode == TRACE_TIME_MODE_FULL:
        fields.extend(
            [
                ("time", float_type),
                ("total_rate", float_type),
            ]
        )
    fields.append(("event_type", pa.string()))
    if normalized_time_mode == TRACE_TIME_MODE_FULL:
        fields.append(("event_rate", float_type))
    fields.extend(
        [
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
    return pa.schema(fields)


def trace_columns(*, time_mode: str = TRACE_TIME_MODE_FULL) -> tuple[str, ...]:
    normalized_time_mode = _trace_time_mode(time_mode)
    if normalized_time_mode == TRACE_TIME_MODE_FULL:
        return TRACE_COLUMNS
    return tuple(column for column in TRACE_COLUMNS if column not in TIME_TRACE_COLUMNS)


def _trace_time_mode(time_mode: str) -> str:
    normalized = time_mode.lower().replace("-", "_")
    if normalized in {"timeless", "no_time", "notime"}:
        normalized = TRACE_TIME_MODE_NONE
    if normalized not in TRACE_TIME_MODES:
        allowed = ", ".join(sorted(TRACE_TIME_MODES))
        raise ValueError(f"time_mode must be one of: {allowed}.")
    return normalized


def _trace_float_type(float_precision: str) -> pa.DataType:
    normalized = float_precision.lower()
    try:
        return TRACE_FLOAT_TYPES[normalized]
    except KeyError as exc:
        allowed = ", ".join(sorted(TRACE_FLOAT_TYPES))
        raise ValueError(f"float_precision must be one of: {allowed}.") from exc


def normalize_trace_format(trace_format: str) -> str:
    normalized = str(trace_format).lower().replace("-", "_").replace(".", "_")
    if normalized in {"json", "jsonl", "jsonl_zstd", "zstd_jsonl"}:
        normalized = TRACE_FORMAT_JSONL_ZST
    if normalized not in TRACE_FORMATS:
        allowed = ", ".join(sorted(TRACE_FORMATS))
        raise ValueError(f"trace_format must be one of: {allowed}.")
    return normalized


def trace_file_extension(trace_format: str) -> str:
    normalized = normalize_trace_format(trace_format)
    if normalized == TRACE_FORMAT_VORTEX:
        return ".vortex"
    if normalized == TRACE_FORMAT_PARQUET:
        return ".parquet"
    return ".jsonl.zst"


def _trace_format_from_path(path: Path) -> str:
    if path.name.lower().endswith(".jsonl.zst"):
        return TRACE_FORMAT_JSONL_ZST
    suffix = path.suffix.lower()
    if suffix == ".vortex":
        return TRACE_FORMAT_VORTEX
    if suffix == ".parquet":
        return TRACE_FORMAT_PARQUET
    return TRACE_FORMAT_JSONL_ZST


def _validate_float_columns(
    columns: dict[str, list[int | float | str | None]],
    *,
    float_precision: str,
) -> None:
    if float_precision != "float16":
        return
    for column in TIME_TRACE_COLUMNS:
        if column not in columns:
            continue
        for value in columns[column]:
            if value is not None and abs(float(value)) > FLOAT16_MAX_FINITE:
                raise ValueError(
                    f"{column} value {value!r} exceeds float16 range; use float32 or float64 trace precision."
                )


TRACE_SCHEMA = trace_schema("float64")


class EventTraceWriter:
    """Write one JSON object per line into a Zstandard-compressed file."""

    def __init__(
        self,
        path: str | Path,
        level: int = 3,
        *,
        time_mode: str = TRACE_TIME_MODE_FULL,
    ) -> None:
        self.path = Path(path)
        self.level = level
        self.time_mode = _trace_time_mode(time_mode)
        self._raw_handle: io.BufferedWriter | None = None
        self._zstd_handle: io.BufferedWriter | None = None
        self._text_handle: io.TextIOWrapper | None = None

    def __enter__(self) -> EventTraceWriter:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._raw_handle = self.path.open("wb")
        compressor = zstd.ZstdCompressor(level=self.level)
        self._zstd_handle = compressor.stream_writer(self._raw_handle)
        self._text_handle = io.TextIOWrapper(self._zstd_handle, encoding="utf-8")
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    def write(self, event: QBPEvent) -> None:
        if self._text_handle is None:
            raise RuntimeError("Trace writer must be opened with a context manager before use.")
        payload = json.dumps(
            event.to_dict(include_time_values=self.time_mode == TRACE_TIME_MODE_FULL),
            separators=(",", ":"),
        )
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
        float_precision: str = DEFAULT_TRACE_FLOAT_PRECISION,
        time_mode: str = TRACE_TIME_MODE_FULL,
    ) -> None:
        if buffer_size <= 0:
            raise ValueError("buffer_size must be positive.")
        self.path = Path(path)
        self.buffer_size = buffer_size
        self.compression = compression
        self.float_precision = float_precision.lower()
        self.time_mode = _trace_time_mode(time_mode)
        self.columns = trace_columns(time_mode=self.time_mode)
        self.schema = trace_schema(self.float_precision, time_mode=self.time_mode)
        self._writer: pq.ParquetWriter | None = None
        self._columns: dict[str, list[int | float | str | None]] = self._new_column_buffer()
        self._row_count = 0

    def _new_column_buffer(self) -> dict[str, list[int | float | str | None]]:
        return {column: [] for column in self.columns}

    def __enter__(self) -> ParquetEventTraceWriter:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._writer = pq.ParquetWriter(
            self.path,
            self.schema,
            compression=self.compression,
            use_dictionary=["event_type"],
        )
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    def write(self, event: QBPEvent) -> None:
        if self._writer is None:
            raise RuntimeError("Parquet trace writer must be opened with a context manager before use.")
        columns = self._columns
        columns["event_index"].append(event.event_index)
        if self.time_mode == TRACE_TIME_MODE_FULL:
            columns["time"].append(event.time)
            columns["total_rate"].append(event.total_rate)
        columns["event_type"].append(event.event_type)
        if self.time_mode == TRACE_TIME_MODE_FULL:
            columns["event_rate"].append(event.event_rate)
        columns["swap_idx"].append(event.swap_idx)
        columns["x"].append(event.x)
        columns["y"].append(event.y)
        columns["z"].append(event.z)
        columns["i"].append(event.i)
        columns["backlog_total"].append(event.backlog_total)
        columns["inventory_total"].append(event.inventory_total)
        columns["scarcity_total"].append(event.scarcity_total)
        self._row_count += 1
        if self._row_count >= self.buffer_size:
            self.flush()

    def flush(self) -> None:
        if self._writer is None:
            raise RuntimeError("Parquet trace writer must be opened with a context manager before use.")
        if self._row_count == 0:
            return
        _validate_float_columns(self._columns, float_precision=self.float_precision)
        table = pa.Table.from_pydict(self._columns, schema=self.schema)
        self._writer.write_table(table)
        self._columns = self._new_column_buffer()
        self._row_count = 0

    def close(self) -> None:
        if self._writer is not None:
            self.flush()
            self._writer.close()
            self._writer = None


class VortexEventTraceWriter:
    """Write events into a buffered compact Vortex file for replay and columnar analysis."""

    def __init__(
        self,
        path: str | Path,
        *,
        buffer_size: int = 65_536,
        queue_size: int = 8,
        float_precision: str = DEFAULT_TRACE_FLOAT_PRECISION,
        time_mode: str = TRACE_TIME_MODE_FULL,
    ) -> None:
        if buffer_size <= 0:
            raise ValueError("buffer_size must be positive.")
        if queue_size <= 0:
            raise ValueError("queue_size must be positive.")
        self.path = Path(path)
        self.buffer_size = buffer_size
        self.queue_size = queue_size
        self.float_precision = float_precision.lower()
        self.time_mode = _trace_time_mode(time_mode)
        self.columns = trace_columns(time_mode=self.time_mode)
        self.schema = trace_schema(self.float_precision, time_mode=self.time_mode)
        self._columns: dict[str, list[int | float | str | None]] = self._new_column_buffer()
        self._row_count = 0
        self._batch_queue: queue.Queue[pa.RecordBatch | None] | None = None
        self._writer_thread: threading.Thread | None = None
        self._writer_error: BaseException | None = None
        self._closed = True

    def _new_column_buffer(self) -> dict[str, list[int | float | str | None]]:
        return {column: [] for column in self.columns}

    def __enter__(self) -> VortexEventTraceWriter:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._batch_queue = queue.Queue(maxsize=self.queue_size)
        self._closed = False
        self._writer_thread = threading.Thread(target=self._write_batches, name="vortex-event-trace-writer")
        self._writer_thread.start()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    def write(self, event: QBPEvent) -> None:
        if self._closed or self._batch_queue is None:
            raise RuntimeError("Vortex trace writer must be opened with a context manager before use.")
        self._raise_writer_error()
        columns = self._columns
        columns["event_index"].append(event.event_index)
        if self.time_mode == TRACE_TIME_MODE_FULL:
            columns["time"].append(event.time)
            columns["total_rate"].append(event.total_rate)
        columns["event_type"].append(event.event_type)
        if self.time_mode == TRACE_TIME_MODE_FULL:
            columns["event_rate"].append(event.event_rate)
        columns["swap_idx"].append(event.swap_idx)
        columns["x"].append(event.x)
        columns["y"].append(event.y)
        columns["z"].append(event.z)
        columns["i"].append(event.i)
        columns["backlog_total"].append(event.backlog_total)
        columns["inventory_total"].append(event.inventory_total)
        columns["scarcity_total"].append(event.scarcity_total)
        self._row_count += 1
        if self._row_count >= self.buffer_size:
            self.flush()

    def flush(self) -> None:
        if self._closed or self._batch_queue is None:
            raise RuntimeError("Vortex trace writer must be opened with a context manager before use.")
        self._raise_writer_error()
        if self._row_count == 0:
            return
        _validate_float_columns(self._columns, float_precision=self.float_precision)
        table = pa.Table.from_pydict(self._columns, schema=self.schema)
        batches = table.to_batches(max_chunksize=self._row_count)
        if len(batches) != 1:
            raise RuntimeError("Expected a single event-trace batch.")
        self._batch_queue.put(batches[0])
        self._columns = self._new_column_buffer()
        self._row_count = 0
        self._raise_writer_error()

    def close(self) -> None:
        if self._closed:
            return
        if self._batch_queue is None:
            self._closed = True
            return
        try:
            self.flush()
        finally:
            self._batch_queue.put(None)
            if self._writer_thread is not None:
                self._writer_thread.join()
            self._writer_thread = None
            self._batch_queue = None
            self._closed = True
        self._raise_writer_error()

    def _iter_batches(self) -> Iterator[pa.RecordBatch]:
        if self._batch_queue is None:
            raise RuntimeError("Vortex trace writer queue is not initialized.")
        while True:
            batch = self._batch_queue.get()
            if batch is None:
                return
            yield batch

    def _write_batches(self) -> None:
        try:
            reader = pa.RecordBatchReader.from_batches(self.schema, self._iter_batches())
            vx.io.VortexWriteOptions.compact().write(reader, str(self.path))
        except BaseException as exc:
            self._writer_error = exc

    def _raise_writer_error(self) -> None:
        if self._writer_error is not None:
            raise RuntimeError("Vortex trace writer failed.") from self._writer_error


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
        previous_time = 0.0
        columns = [column for column in TRACE_COLUMNS if column in self._parquet_file.schema_arrow.names]
        for batch in self._parquet_file.iter_batches(batch_size=self.batch_size, columns=columns):
            for record in batch.to_pylist():
                if "time" in record:
                    event_time = float(record["time"])
                    record["dt"] = event_time - previous_time
                    previous_time = event_time
                yield QBPEvent.from_dict(record)

    def close(self) -> None:
        self._parquet_file = None


class VortexEventTraceReader:
    """Read a Vortex event trace as QBPEvent records."""

    def __init__(self, path: str | Path, batch_size: int = 65_536) -> None:
        if batch_size <= 0:
            raise ValueError("batch_size must be positive.")
        self.path = Path(path)
        self.batch_size = batch_size
        self._reader: pa.RecordBatchReader | None = None

    def __enter__(self) -> VortexEventTraceReader:
        self._reader = vx.open(str(self.path)).to_arrow(batch_size=self.batch_size)
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    def __iter__(self) -> Iterator[QBPEvent]:
        if self._reader is None:
            raise RuntimeError("Vortex trace reader must be opened with a context manager before use.")
        previous_time = 0.0
        for batch in self._reader:
            table = pa.Table.from_batches([batch])
            columns = [column for column in TRACE_COLUMNS if column in table.column_names]
            for record in table.select(columns).to_pylist():
                if "time" in record:
                    event_time = float(record["time"])
                    record["dt"] = event_time - previous_time
                    previous_time = event_time
                yield QBPEvent.from_dict(record)

    def close(self) -> None:
        self._reader = None


def open_event_trace_writer(
    path: str | Path,
    *,
    trace_format: str | None = None,
    float_precision: str = DEFAULT_TRACE_FLOAT_PRECISION,
    time_mode: str = TRACE_TIME_MODE_FULL,
):
    trace_path = Path(path)
    resolved_trace_format = _trace_format_from_path(trace_path) if trace_format is None else normalize_trace_format(trace_format)
    if resolved_trace_format == TRACE_FORMAT_VORTEX:
        return VortexEventTraceWriter(trace_path, float_precision=float_precision, time_mode=time_mode)
    if resolved_trace_format == TRACE_FORMAT_PARQUET:
        return ParquetEventTraceWriter(trace_path, float_precision=float_precision, time_mode=time_mode)
    return EventTraceWriter(trace_path, time_mode=time_mode)


def open_event_trace_reader(path: str | Path, *, trace_format: str | None = None):
    trace_path = Path(path)
    resolved_trace_format = _trace_format_from_path(trace_path) if trace_format is None else normalize_trace_format(trace_format)
    if resolved_trace_format == TRACE_FORMAT_VORTEX:
        return VortexEventTraceReader(trace_path)
    if resolved_trace_format == TRACE_FORMAT_PARQUET:
        return ParquetEventTraceReader(trace_path)
    return EventTraceReader(trace_path)
