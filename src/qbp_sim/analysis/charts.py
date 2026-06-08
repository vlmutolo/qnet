
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import altair as alt
import vl_convert as vlc

from qbp_sim.io.snapshots import QBPSnapshot

alt.data_transformers.disable_max_rows()
PNG_PPI = 300

def snapshot_metric_rows(
    snapshots: list[QBPSnapshot],
    metric: str,
    *,
    series: str | None = None,
    series_order: int | None = None,
) -> list[dict[str, Any]]:
    if not snapshots:
        raise ValueError("No snapshots were provided for plotting.")
    if not hasattr(snapshots[0], metric):
        raise ValueError(f"Unknown snapshot metric: {metric}")

    rows: list[dict[str, Any]] = []
    for snapshot in snapshots:
        row: dict[str, Any] = {
            "time": snapshot.time,
            "event_index": snapshot.event_index,
            metric: getattr(snapshot, metric),
        }
        if series is not None:
            row["series"] = series
        if series_order is not None:
            row["series_order"] = series_order
        rows.append(row)
    return rows


def build_snapshot_metric_chart(
    snapshots: list[QBPSnapshot],
    metric: str,
    *,
    title: str | None = None,
) -> alt.Chart:
    rows = snapshot_metric_rows(snapshots, metric)
    return (
        alt.Chart(alt.Data(values=rows))
        .mark_line(point=False, strokeWidth=2.0)
        .encode(
            x=alt.X("time:Q", title="time"),
            y=alt.Y(f"{metric}:Q", title=metric),
            tooltip=[
                alt.Tooltip("time:Q", format=".4f"),
                alt.Tooltip(f"{metric}:Q"),
                alt.Tooltip("event_index:Q"),
            ],
        )
        .properties(width=760, height=420, title=title or f"{metric} over time")
    )


def build_snapshot_metric_series_chart(
    series_snapshots: list[tuple[str, list[QBPSnapshot]]],
    metric: str,
    *,
    title: str | None = None,
) -> alt.Chart:
    if not series_snapshots:
        raise ValueError("No snapshot series were provided for plotting.")

    rows: list[dict[str, Any]] = []
    series_order = [label for label, _ in series_snapshots]
    for order, (label, snapshots) in enumerate(series_snapshots):
        rows.extend(
            snapshot_metric_rows(
                snapshots,
                metric,
                series=label,
                series_order=order,
            )
        )

    return (
        alt.Chart(alt.Data(values=rows))
        .mark_line(strokeWidth=2.2)
        .encode(
            x=alt.X("time:Q", title="time"),
            y=alt.Y(f"{metric}:Q", title=metric),
            color=alt.Color(
                "series:N",
                title="series",
                sort=series_order,
            ),
            detail="series:N",
            tooltip=[
                alt.Tooltip("series:N"),
                alt.Tooltip("time:Q", format=".4f"),
                alt.Tooltip(f"{metric}:Q"),
                alt.Tooltip("event_index:Q"),
            ],
        )
        .properties(width=860, height=480, title=title or f"{metric} over time")
    )


def save_chart(chart: alt.Chart, output_path: str | Path) -> None:
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    suffix = output.suffix.lower()
    if suffix == ".html":
        output.write_text(chart.to_html(), encoding="utf-8")
        return
    if suffix == ".json":
        output.write_text(chart.to_json(indent=None), encoding="utf-8")
        return

    spec = json.loads(chart.to_json(indent=None))
    if suffix == ".png":
        output.write_bytes(vlc.vegalite_to_png(spec, ppi=PNG_PPI))
        return
    if suffix == ".svg":
        output.write_text(vlc.vegalite_to_svg(spec), encoding="utf-8")
        return
    raise ValueError(f"Unsupported chart output format: {output.suffix}")


def plot_snapshot_metric(
    snapshots: list[QBPSnapshot],
    metric: str,
    output_path: str | Path,
) -> None:
    chart = build_snapshot_metric_chart(snapshots, metric)
    save_chart(chart, output_path)


def plot_snapshot_metric_series(
    series_snapshots: list[tuple[str, list[QBPSnapshot]]],
    metric: str,
    output_path: str | Path,
    *,
    title: str | None = None,
) -> None:
    chart = build_snapshot_metric_series_chart(series_snapshots, metric, title=title)
    save_chart(chart, output_path)
