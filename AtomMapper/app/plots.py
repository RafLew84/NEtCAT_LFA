"""Plot-series models and helpers for AtomMapper analytical visualizations."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import math
from statistics import fmean, pstdev
from typing import Sequence

from .models import AtomPoint, AtomRow


class PlotUnit(str, Enum):
    """Display unit supported by AtomMapper analytical plots."""

    PX = "px"
    NM = "nm"


class RowPlotMode(str, Enum):
    """Supported modes for the single-row plot."""

    X_PX = "x_px"
    Y_PX = "y_px"
    DISTANCE_PX = "distance_px"
    X_NM = "x_nm"
    Y_NM = "y_nm"
    DISTANCE_NM = "distance_nm"


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

    x_value: float
    y_value: float
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

    unit: PlotUnit
    x_label: str
    y_label: str
    samples: tuple[GlobalScatterSample, ...]


@dataclass(frozen=True)
class RowDistanceMetrics:
    """Distance summary statistics for a single atom row."""

    row_id: str
    row_display_name: str
    point_count: int
    distance_count_px: int
    distance_count_nm: int
    mean_distance_px: float | None
    std_distance_px: float | None
    min_distance_px: float | None
    max_distance_px: float | None
    mean_distance_nm: float | None
    std_distance_nm: float | None
    min_distance_nm: float | None
    max_distance_nm: float | None

    @property
    def distance_count(self) -> int:
        """Backwards-compatible distance count in pixels."""

        return self.distance_count_px

    def distance_count_for_unit(self, unit: PlotUnit) -> int:
        """Return the number of valid consecutive segments for the selected unit."""

        if unit is PlotUnit.NM:
            return self.distance_count_nm
        return self.distance_count_px

    def mean_distance_for_unit(self, unit: PlotUnit) -> float | None:
        """Return the mean distance in the selected unit."""

        if unit is PlotUnit.NM:
            return self.mean_distance_nm
        return self.mean_distance_px

    def std_distance_for_unit(self, unit: PlotUnit) -> float | None:
        """Return the standard deviation in the selected unit."""

        if unit is PlotUnit.NM:
            return self.std_distance_nm
        return self.std_distance_px

    def min_distance_for_unit(self, unit: PlotUnit) -> float | None:
        """Return the minimum distance in the selected unit."""

        if unit is PlotUnit.NM:
            return self.min_distance_nm
        return self.min_distance_px

    def max_distance_for_unit(self, unit: PlotUnit) -> float | None:
        """Return the maximum distance in the selected unit."""

        if unit is PlotUnit.NM:
            return self.max_distance_nm
        return self.max_distance_px


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

    if mode is RowPlotMode.X_NM:
        samples = tuple(
            RowSeriesSample(
                x_value=float(point.point_index),
                y_value=float(point.x_nm),
                point_id=point.point_id,
                point_index=point.point_index,
                row_id=row.row_id,
            )
            for point in ordered_points
            if point.x_nm is not None
        )
        return RowMetricSeries(
            row_id=row.row_id,
            row_display_name=row.display_name,
            mode=mode,
            x_label="point index",
            y_label="x (nm)",
            samples=samples,
        )

    if mode is RowPlotMode.Y_NM:
        samples = tuple(
            RowSeriesSample(
                x_value=float(point.point_index),
                y_value=float(point.y_nm),
                point_id=point.point_id,
                point_index=point.point_index,
                row_id=row.row_id,
            )
            for point in ordered_points
            if point.y_nm is not None
        )
        return RowMetricSeries(
            row_id=row.row_id,
            row_display_name=row.display_name,
            mode=mode,
            x_label="point index",
            y_label="y (nm)",
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

    if mode is RowPlotMode.DISTANCE_NM:
        distance_samples: list[RowSeriesSample] = []
        for left_point, right_point in zip(ordered_points, ordered_points[1:]):
            if (
                left_point.x_nm is None
                or left_point.y_nm is None
                or right_point.x_nm is None
                or right_point.y_nm is None
            ):
                continue
            delta_x = float(right_point.x_nm) - float(left_point.x_nm)
            delta_y = float(right_point.y_nm) - float(left_point.y_nm)
            distance_nm = math.hypot(delta_x, delta_y)
            distance_samples.append(
                RowSeriesSample(
                    x_value=float(left_point.point_index),
                    y_value=distance_nm,
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
            y_label="distance (nm)",
            samples=tuple(distance_samples),
        )

    raise ValueError(f"Unsupported RowPlotMode: {mode!r}")


def build_global_scatter_series(
    rows: Sequence[AtomRow],
    unit: PlotUnit = PlotUnit.PX,
) -> GlobalScatterSeries:
    """Build a plot-ready global scatter series across multiple rows."""

    samples: list[GlobalScatterSample] = []
    for row in rows:
        for point in sorted_row_points(row):
            if unit is PlotUnit.NM:
                if point.x_nm is None or point.y_nm is None:
                    continue
                x_value = float(point.x_nm)
                y_value = float(point.y_nm)
            else:
                x_value = float(point.x_px)
                y_value = float(point.y_px)
            samples.append(
                GlobalScatterSample(
                    x_value=x_value,
                    y_value=y_value,
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
        unit=unit,
        x_label=f"x ({unit.value})",
        y_label=f"y ({unit.value})",
        samples=tuple(samples),
    )


def build_row_distance_metrics(row: AtomRow) -> RowDistanceMetrics:
    """Build basic distance statistics for consecutive points in a row."""

    ordered_points = sorted_row_points(row)
    distances_px: list[float] = []
    distances_nm: list[float] = []
    for left_point, right_point in zip(ordered_points, ordered_points[1:]):
        delta_x_px = float(right_point.x_px) - float(left_point.x_px)
        delta_y_px = float(right_point.y_px) - float(left_point.y_px)
        distances_px.append(math.hypot(delta_x_px, delta_y_px))
        if (
            left_point.x_nm is not None
            and left_point.y_nm is not None
            and right_point.x_nm is not None
            and right_point.y_nm is not None
        ):
            delta_x_nm = float(right_point.x_nm) - float(left_point.x_nm)
            delta_y_nm = float(right_point.y_nm) - float(left_point.y_nm)
            distances_nm.append(math.hypot(delta_x_nm, delta_y_nm))

    if not distances_px:
        return RowDistanceMetrics(
            row_id=row.row_id,
            row_display_name=row.display_name,
            point_count=len(ordered_points),
            distance_count_px=0,
            distance_count_nm=0,
            mean_distance_px=None,
            std_distance_px=None,
            min_distance_px=None,
            max_distance_px=None,
            mean_distance_nm=None,
            std_distance_nm=None,
            min_distance_nm=None,
            max_distance_nm=None,
        )

    std_distance_px = 0.0 if len(distances_px) == 1 else float(pstdev(distances_px))
    std_distance_nm = None
    if distances_nm:
        std_distance_nm = 0.0 if len(distances_nm) == 1 else float(pstdev(distances_nm))
    return RowDistanceMetrics(
        row_id=row.row_id,
        row_display_name=row.display_name,
        point_count=len(ordered_points),
        distance_count_px=len(distances_px),
        distance_count_nm=len(distances_nm),
        mean_distance_px=float(fmean(distances_px)),
        std_distance_px=std_distance_px,
        min_distance_px=float(min(distances_px)),
        max_distance_px=float(max(distances_px)),
        mean_distance_nm=float(fmean(distances_nm)) if distances_nm else None,
        std_distance_nm=std_distance_nm,
        min_distance_nm=float(min(distances_nm)) if distances_nm else None,
        max_distance_nm=float(max(distances_nm)) if distances_nm else None,
    )
