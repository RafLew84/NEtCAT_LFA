"""Plot-series models and helpers for AtomMapper analytical visualizations."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import math
from statistics import fmean, pstdev
from typing import Sequence

from .models import AtomPoint, AtomRow


class RowPlotMode(str, Enum):
    """Supported modes for the single-row plot."""

    X_PX = "x_px"
    Y_PX = "y_px"
    DISTANCE_PX = "distance_px"


@dataclass(frozen=True)
class RowSeriesSample:
    """Single sample used by the selected-row plot."""

    x_value: float
    y_value: float
    point_id: str
    point_index: int
    row_id: str


@dataclass(frozen=True)
class RowMetricSeries:
    """Prepared series for a single selected atom row."""

    row_id: str
    row_display_name: str
    mode: RowPlotMode
    x_label: str
    y_label: str
    samples: tuple[RowSeriesSample, ...]


@dataclass(frozen=True)
class GlobalScatterSample:
    """Single point sample used by the global scatter plot."""

    x_px: float
    y_px: float
    point_id: str
    point_index: int
    row_id: str
    row_display_name: str
    image_id: str
    source_group_id: str
    is_manual_override: bool
    color_hex: str | None


@dataclass(frozen=True)
class GlobalScatterSeries:
    """Prepared scatter payload for all rows in the active image family."""

    x_label: str
    y_label: str
    samples: tuple[GlobalScatterSample, ...]


@dataclass(frozen=True)
class RowDistanceMetrics:
    """Distance summary statistics for a single atom row."""

    row_id: str
    row_display_name: str
    point_count: int
    distance_count: int
    mean_distance_px: float | None
    std_distance_px: float | None
    min_distance_px: float | None
    max_distance_px: float | None


def sorted_row_points(row: AtomRow) -> tuple[AtomPoint, ...]:
    """Return row points sorted by point index and stable point id."""

    return tuple(sorted(row.points, key=lambda point: (point.point_index, point.point_id)))


def build_row_metric_series(row: AtomRow, mode: RowPlotMode) -> RowMetricSeries:
    """Build a plot-ready series for a single row and selected metric."""

    ordered_points = sorted_row_points(row)

    if mode is RowPlotMode.X_PX:
        samples = tuple(
            RowSeriesSample(
                x_value=float(point.point_index),
                y_value=float(point.x_px),
                point_id=point.point_id,
                point_index=point.point_index,
                row_id=row.row_id,
            )
            for point in ordered_points
        )
        return RowMetricSeries(
            row_id=row.row_id,
            row_display_name=row.display_name,
            mode=mode,
            x_label="point index",
            y_label="x (px)",
            samples=samples,
        )

    if mode is RowPlotMode.Y_PX:
        samples = tuple(
            RowSeriesSample(
                x_value=float(point.point_index),
                y_value=float(point.y_px),
                point_id=point.point_id,
                point_index=point.point_index,
                row_id=row.row_id,
            )
            for point in ordered_points
        )
        return RowMetricSeries(
            row_id=row.row_id,
            row_display_name=row.display_name,
            mode=mode,
            x_label="point index",
            y_label="y (px)",
            samples=samples,
        )

    if mode is RowPlotMode.DISTANCE_PX:
        distance_samples: list[RowSeriesSample] = []
        for left_point, right_point in zip(ordered_points, ordered_points[1:]):
            delta_x = float(right_point.x_px) - float(left_point.x_px)
            delta_y = float(right_point.y_px) - float(left_point.y_px)
            distance_px = math.hypot(delta_x, delta_y)
            distance_samples.append(
                RowSeriesSample(
                    x_value=float(left_point.point_index),
                    y_value=distance_px,
                    point_id=right_point.point_id,
                    point_index=right_point.point_index,
                    row_id=row.row_id,
                )
            )
        return RowMetricSeries(
            row_id=row.row_id,
            row_display_name=row.display_name,
            mode=mode,
            x_label="segment index",
            y_label="distance (px)",
            samples=tuple(distance_samples),
        )

    raise ValueError(f"Unsupported RowPlotMode: {mode!r}")


def build_global_scatter_series(rows: Sequence[AtomRow]) -> GlobalScatterSeries:
    """Build a plot-ready global scatter series across multiple rows."""

    samples: list[GlobalScatterSample] = []
    for row in rows:
        for point in sorted_row_points(row):
            samples.append(
                GlobalScatterSample(
                    x_px=float(point.x_px),
                    y_px=float(point.y_px),
                    point_id=point.point_id,
                    point_index=point.point_index,
                    row_id=row.row_id,
                    row_display_name=row.display_name,
                    image_id=point.image_id,
                    source_group_id=point.source_group_id,
                    is_manual_override=bool(point.manual_override),
                    color_hex=row.color_hex,
                )
            )

    samples.sort(
        key=lambda sample: (
            sample.row_display_name.lower(),
            sample.point_index,
            sample.point_id,
        )
    )
    return GlobalScatterSeries(
        x_label="x (px)",
        y_label="y (px)",
        samples=tuple(samples),
    )


def build_row_distance_metrics(row: AtomRow) -> RowDistanceMetrics:
    """Build basic distance statistics for consecutive points in a row."""

    ordered_points = sorted_row_points(row)
    distances: list[float] = []
    for left_point, right_point in zip(ordered_points, ordered_points[1:]):
        delta_x = float(right_point.x_px) - float(left_point.x_px)
        delta_y = float(right_point.y_px) - float(left_point.y_px)
        distances.append(math.hypot(delta_x, delta_y))

    if not distances:
        return RowDistanceMetrics(
            row_id=row.row_id,
            row_display_name=row.display_name,
            point_count=len(ordered_points),
            distance_count=0,
            mean_distance_px=None,
            std_distance_px=None,
            min_distance_px=None,
            max_distance_px=None,
        )

    std_distance_px = 0.0 if len(distances) == 1 else float(pstdev(distances))
    return RowDistanceMetrics(
        row_id=row.row_id,
        row_display_name=row.display_name,
        point_count=len(ordered_points),
        distance_count=len(distances),
        mean_distance_px=float(fmean(distances)),
        std_distance_px=std_distance_px,
        min_distance_px=float(min(distances)),
        max_distance_px=float(max(distances)),
    )
