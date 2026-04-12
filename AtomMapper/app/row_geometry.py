"""Data contracts and fitting helpers for geometric analysis of atom rows."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, Optional

import math
import numpy as np

from .models import AtomRow


def _normalize_optional_float(value: Optional[float], field_name: str) -> Optional[float]:
    """Normalize optional float payloads while rejecting non-finite values."""

    if value is None:
        return None
    normalized = float(value)
    if not math.isfinite(normalized):
        raise ValueError(f"{field_name} must be finite when provided.")
    return normalized


class RowGeometryUnit(str, Enum):
    """Supported units for row-geometry helpers."""

    PX = "px"
    NM = "nm"


class RowProjectionSortMode(str, Enum):
    """Supported ordering modes for projected row samples."""

    POINT_INDEX = "point_index"
    ALONG_AXIS = "along_axis"


@dataclass(frozen=True)
class RowGeometry:
    """Serializable geometry contract for a single atom row."""

    row_id: str
    row_display_name: str
    point_count: int
    fitted_point_count: int
    reference_x_px: float
    reference_y_px: float
    direction_x_px: float
    direction_y_px: float
    span_length_px: float
    reference_x_nm: Optional[float] = None
    reference_y_nm: Optional[float] = None
    direction_x_nm: Optional[float] = None
    direction_y_nm: Optional[float] = None
    span_length_nm: Optional[float] = None
    metadata: Dict[str, Any] = field(default_factory=dict, repr=False)

    def __post_init__(self) -> None:
        row_id = str(self.row_id).strip()
        row_display_name = str(self.row_display_name).strip()
        if not row_id:
            raise ValueError("row_id must be a non-empty string.")
        if not row_display_name:
            raise ValueError("row_display_name must be a non-empty string.")

        point_count = int(self.point_count)
        fitted_point_count = int(self.fitted_point_count)
        if point_count < 1:
            raise ValueError("point_count must be >= 1.")
        if fitted_point_count < 1:
            raise ValueError("fitted_point_count must be >= 1.")
        if fitted_point_count > point_count:
            raise ValueError("fitted_point_count cannot exceed point_count.")

        reference_x_px = _normalize_optional_float(self.reference_x_px, "reference_x_px")
        reference_y_px = _normalize_optional_float(self.reference_y_px, "reference_y_px")
        direction_x_px = _normalize_optional_float(self.direction_x_px, "direction_x_px")
        direction_y_px = _normalize_optional_float(self.direction_y_px, "direction_y_px")
        span_length_px = _normalize_optional_float(self.span_length_px, "span_length_px")
        if span_length_px is None or span_length_px < 0.0:
            raise ValueError("span_length_px must be a finite value >= 0.")

        px_norm = math.hypot(direction_x_px, direction_y_px)
        if px_norm <= 0.0:
            raise ValueError("Pixel-space direction vector must be non-zero.")

        reference_x_nm = _normalize_optional_float(self.reference_x_nm, "reference_x_nm")
        reference_y_nm = _normalize_optional_float(self.reference_y_nm, "reference_y_nm")
        direction_x_nm = _normalize_optional_float(self.direction_x_nm, "direction_x_nm")
        direction_y_nm = _normalize_optional_float(self.direction_y_nm, "direction_y_nm")
        span_length_nm = _normalize_optional_float(self.span_length_nm, "span_length_nm")

        nm_fields = (
            reference_x_nm,
            reference_y_nm,
            direction_x_nm,
            direction_y_nm,
            span_length_nm,
        )
        if any(value is not None for value in nm_fields):
            if any(value is None for value in nm_fields):
                raise ValueError(
                    "NM geometry must either provide all nm fields or leave all of them unset."
                )
            if span_length_nm is None or span_length_nm < 0.0:
                raise ValueError("span_length_nm must be a finite value >= 0.")
            nm_norm = math.hypot(direction_x_nm, direction_y_nm)
            if nm_norm <= 0.0:
                raise ValueError("NM-space direction vector must be non-zero.")
            object.__setattr__(self, "reference_x_nm", reference_x_nm)
            object.__setattr__(self, "reference_y_nm", reference_y_nm)
            object.__setattr__(self, "direction_x_nm", direction_x_nm / nm_norm)
            object.__setattr__(self, "direction_y_nm", direction_y_nm / nm_norm)
            object.__setattr__(self, "span_length_nm", span_length_nm)
        else:
            object.__setattr__(self, "reference_x_nm", None)
            object.__setattr__(self, "reference_y_nm", None)
            object.__setattr__(self, "direction_x_nm", None)
            object.__setattr__(self, "direction_y_nm", None)
            object.__setattr__(self, "span_length_nm", None)

        object.__setattr__(self, "row_id", row_id)
        object.__setattr__(self, "row_display_name", row_display_name)
        object.__setattr__(self, "point_count", point_count)
        object.__setattr__(self, "fitted_point_count", fitted_point_count)
        object.__setattr__(self, "reference_x_px", reference_x_px)
        object.__setattr__(self, "reference_y_px", reference_y_px)
        object.__setattr__(self, "direction_x_px", direction_x_px / px_norm)
        object.__setattr__(self, "direction_y_px", direction_y_px / px_norm)
        object.__setattr__(self, "span_length_px", span_length_px)

    @property
    def has_nm_geometry(self) -> bool:
        """Return ``True`` when the row geometry includes physical-space values."""

        return self.reference_x_nm is not None

    def to_dict(self) -> Dict[str, Any]:
        """Return a JSON-serializable representation of the row geometry."""

        return {
            "row_id": self.row_id,
            "row_display_name": self.row_display_name,
            "point_count": self.point_count,
            "fitted_point_count": self.fitted_point_count,
            "reference_x_px": self.reference_x_px,
            "reference_y_px": self.reference_y_px,
            "direction_x_px": self.direction_x_px,
            "direction_y_px": self.direction_y_px,
            "span_length_px": self.span_length_px,
            "reference_x_nm": self.reference_x_nm,
            "reference_y_nm": self.reference_y_nm,
            "direction_x_nm": self.direction_x_nm,
            "direction_y_nm": self.direction_y_nm,
            "span_length_nm": self.span_length_nm,
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_dict(cls, payload: Dict[str, Any]) -> "RowGeometry":
        """Rebuild a row-geometry payload from serialized data."""

        return cls(
            row_id=payload["row_id"],
            row_display_name=payload["row_display_name"],
            point_count=payload["point_count"],
            fitted_point_count=payload["fitted_point_count"],
            reference_x_px=payload["reference_x_px"],
            reference_y_px=payload["reference_y_px"],
            direction_x_px=payload["direction_x_px"],
            direction_y_px=payload["direction_y_px"],
            span_length_px=payload["span_length_px"],
            reference_x_nm=payload.get("reference_x_nm"),
            reference_y_nm=payload.get("reference_y_nm"),
            direction_x_nm=payload.get("direction_x_nm"),
            direction_y_nm=payload.get("direction_y_nm"),
            span_length_nm=payload.get("span_length_nm"),
            metadata=dict(payload.get("metadata", {})),
        )


@dataclass(frozen=True)
class RowProjectionSample:
    """Single projected point sample in row-aligned coordinates."""

    row_id: str
    point_id: str
    point_index: int
    image_id: str
    source_group_id: str
    x_value: float
    y_value: float
    along_value: float
    transverse_value: float
    is_manual_override: bool


@dataclass(frozen=True)
class RowProjectionSeries:
    """Projected row samples for a single unit and ordering mode."""

    row_id: str
    row_display_name: str
    unit: RowGeometryUnit
    sort_mode: RowProjectionSortMode
    geometry: RowGeometry
    samples: tuple[RowProjectionSample, ...]


@dataclass(frozen=True)
class RowDisturbanceSample:
    """Single local-disturbance sample centered on one interior row point."""

    row_id: str
    point_id: str
    point_index: int
    image_id: str
    source_group_id: str
    previous_point_id: str
    next_point_id: str
    along_value: float
    transverse_value: float
    spacing_before: float
    spacing_after: float
    spacing_jump_abs: float
    transverse_jump_abs: float
    local_direction_change_deg: float
    candidate_score: float
    is_candidate_spacing: bool
    is_candidate_transverse: bool
    is_candidate_direction: bool

    @property
    def is_candidate(self) -> bool:
        """Return ``True`` when at least one disturbance marker exceeded its threshold."""

        return (
            self.is_candidate_spacing
            or self.is_candidate_transverse
            or self.is_candidate_direction
        )


@dataclass(frozen=True)
class RowDisturbanceSeries:
    """Local-disturbance markers for one row in one selected unit."""

    row_id: str
    row_display_name: str
    unit: RowGeometryUnit
    geometry: RowGeometry
    spacing_jump_threshold: float
    transverse_jump_threshold: float
    direction_change_threshold_deg: float
    samples: tuple[RowDisturbanceSample, ...]

    @property
    def candidate_count(self) -> int:
        """Return the number of samples flagged as local-disturbance candidates."""

        return sum(1 for sample in self.samples if sample.is_candidate)


def _orient_direction(
    direction: np.ndarray,
    endpoint_vector: np.ndarray,
) -> np.ndarray:
    """Orient a direction vector deterministically using the row endpoint ordering."""

    oriented = np.asarray(direction, dtype=float)
    endpoint = np.asarray(endpoint_vector, dtype=float)
    endpoint_norm = float(np.linalg.norm(endpoint))
    if endpoint_norm > 0.0 and float(np.dot(oriented, endpoint)) < 0.0:
        oriented = -oriented

    if endpoint_norm <= 0.0:
        if oriented[0] < 0.0 or (math.isclose(float(oriented[0]), 0.0) and oriented[1] < 0.0):
            oriented = -oriented
    return oriented


def _fit_axis_for_coordinates(coordinates: np.ndarray) -> tuple[float, float, float, float, float] | None:
    """Fit a stable axis to ordered 2D coordinates using a PCA/SVD principal direction."""

    array = np.asarray(coordinates, dtype=float)
    if array.ndim != 2 or array.shape[1] != 2:
        raise ValueError("Row-axis fitting expects an array of shape (N, 2).")
    if array.shape[0] < 2:
        return None
    if not np.all(np.isfinite(array)):
        raise ValueError("Row-axis fitting coordinates must be finite.")

    centroid = array.mean(axis=0)
    centered = array - centroid
    endpoint_vector = array[-1] - array[0]

    singular_values: np.ndarray
    if array.shape[0] == 2:
        singular_values = np.asarray([np.linalg.norm(endpoint_vector)], dtype=float)
        principal_direction = endpoint_vector
    else:
        _, singular_values, right_vectors = np.linalg.svd(centered, full_matrices=False)
        principal_direction = right_vectors[0]

    principal_norm = float(np.linalg.norm(principal_direction))
    if principal_norm <= 0.0:
        return None

    if singular_values.size == 0 or float(singular_values[0]) <= 1e-12:
        return None

    principal_direction = principal_direction / principal_norm
    principal_direction = _orient_direction(principal_direction, endpoint_vector)

    projections = centered @ principal_direction
    span = float(projections.max() - projections.min())

    return (
        float(centroid[0]),
        float(centroid[1]),
        float(principal_direction[0]),
        float(principal_direction[1]),
        span,
    )


def fit_row_geometry(row: AtomRow, *, minimum_point_count: int = 2) -> RowGeometry | None:
    """Fit a single-row axis model from stored row points.

    The fit uses the current point coordinates, so manual point edits are reflected directly
    in the fitted geometry. Pixel-space geometry is always attempted first. Physical-space
    geometry is included only when every fitted point exposes both ``x_nm`` and ``y_nm``.
    """

    ordered_points = tuple(sorted(row.points, key=lambda point: (point.point_index, point.point_id)))
    if len(ordered_points) < int(minimum_point_count):
        return None

    coordinates_px = np.asarray(
        [(float(point.x_px), float(point.y_px)) for point in ordered_points],
        dtype=float,
    )
    px_fit = _fit_axis_for_coordinates(coordinates_px)
    if px_fit is None:
        return None

    (
        reference_x_px,
        reference_y_px,
        direction_x_px,
        direction_y_px,
        span_length_px,
    ) = px_fit

    reference_x_nm: Optional[float] = None
    reference_y_nm: Optional[float] = None
    direction_x_nm: Optional[float] = None
    direction_y_nm: Optional[float] = None
    span_length_nm: Optional[float] = None

    if all(point.x_nm is not None and point.y_nm is not None for point in ordered_points):
        coordinates_nm = np.asarray(
            [(float(point.x_nm), float(point.y_nm)) for point in ordered_points],
            dtype=float,
        )
        nm_fit = _fit_axis_for_coordinates(coordinates_nm)
        if nm_fit is not None:
            (
                reference_x_nm,
                reference_y_nm,
                direction_x_nm,
                direction_y_nm,
                span_length_nm,
            ) = nm_fit

    return RowGeometry(
        row_id=row.row_id,
        row_display_name=row.display_name,
        point_count=row.point_count,
        fitted_point_count=len(ordered_points),
        reference_x_px=reference_x_px,
        reference_y_px=reference_y_px,
        direction_x_px=direction_x_px,
        direction_y_px=direction_y_px,
        span_length_px=span_length_px,
        reference_x_nm=reference_x_nm,
        reference_y_nm=reference_y_nm,
        direction_x_nm=direction_x_nm,
        direction_y_nm=direction_y_nm,
        span_length_nm=span_length_nm,
        metadata={
            "fit_method": "svd-pca",
            "minimum_point_count": int(minimum_point_count),
            "used_point_ids": tuple(point.point_id for point in ordered_points),
            "uses_manual_override": any(point.manual_override for point in ordered_points),
        },
    )


def _normalize_projection_unit(unit: RowGeometryUnit | str) -> RowGeometryUnit:
    """Normalize a row-geometry unit selection."""

    if isinstance(unit, RowGeometryUnit):
        return unit
    return RowGeometryUnit(str(unit).strip().lower())


def _normalize_projection_sort_mode(
    sort_mode: RowProjectionSortMode | str,
) -> RowProjectionSortMode:
    """Normalize a row-projection ordering mode."""

    if isinstance(sort_mode, RowProjectionSortMode):
        return sort_mode
    return RowProjectionSortMode(str(sort_mode).strip().lower())


def project_row_points(
    row: AtomRow,
    geometry: RowGeometry | None = None,
    *,
    unit: RowGeometryUnit | str = RowGeometryUnit.PX,
    sort_mode: RowProjectionSortMode | str = RowProjectionSortMode.POINT_INDEX,
    minimum_point_count: int = 2,
) -> RowProjectionSeries | None:
    """Project row points to coordinates along and transverse to the fitted row axis."""

    normalized_unit = _normalize_projection_unit(unit)
    normalized_sort_mode = _normalize_projection_sort_mode(sort_mode)

    fitted_geometry = geometry
    if fitted_geometry is None:
        fitted_geometry = fit_row_geometry(row, minimum_point_count=minimum_point_count)
    if fitted_geometry is None:
        return None
    if fitted_geometry.row_id != row.row_id:
        raise ValueError("geometry.row_id must match row.row_id.")

    ordered_points = tuple(sorted(row.points, key=lambda point: (point.point_index, point.point_id)))
    if len(ordered_points) < int(minimum_point_count):
        return None

    if normalized_unit is RowGeometryUnit.PX:
        reference = np.asarray(
            [fitted_geometry.reference_x_px, fitted_geometry.reference_y_px],
            dtype=float,
        )
        direction = np.asarray(
            [fitted_geometry.direction_x_px, fitted_geometry.direction_y_px],
            dtype=float,
        )
        coordinates = tuple((float(point.x_px), float(point.y_px)) for point in ordered_points)
    else:
        if not fitted_geometry.has_nm_geometry:
            return None
        if not all(point.x_nm is not None and point.y_nm is not None for point in ordered_points):
            return None
        reference = np.asarray(
            [fitted_geometry.reference_x_nm, fitted_geometry.reference_y_nm],
            dtype=float,
        )
        direction = np.asarray(
            [fitted_geometry.direction_x_nm, fitted_geometry.direction_y_nm],
            dtype=float,
        )
        coordinates = tuple((float(point.x_nm), float(point.y_nm)) for point in ordered_points)

    perpendicular = np.asarray([-direction[1], direction[0]], dtype=float)
    samples: list[RowProjectionSample] = []
    for point, (x_value, y_value) in zip(ordered_points, coordinates):
        delta = np.asarray([x_value, y_value], dtype=float) - reference
        samples.append(
            RowProjectionSample(
                row_id=row.row_id,
                point_id=point.point_id,
                point_index=point.point_index,
                image_id=point.image_id,
                source_group_id=point.source_group_id,
                x_value=x_value,
                y_value=y_value,
                along_value=float(np.dot(delta, direction)),
                transverse_value=float(np.dot(delta, perpendicular)),
                is_manual_override=point.manual_override,
            )
        )

    if normalized_sort_mode is RowProjectionSortMode.ALONG_AXIS:
        samples.sort(key=lambda sample: (sample.along_value, sample.point_index, sample.point_id))
    else:
        samples.sort(key=lambda sample: (sample.point_index, sample.point_id))

    return RowProjectionSeries(
        row_id=row.row_id,
        row_display_name=row.display_name,
        unit=normalized_unit,
        sort_mode=normalized_sort_mode,
        geometry=fitted_geometry,
        samples=tuple(samples),
    )


def _local_angle_change_deg(
    previous_vector: np.ndarray,
    next_vector: np.ndarray,
) -> float:
    """Return the unsigned local direction change between two 2D segment vectors."""

    dot_product = float(np.dot(previous_vector, next_vector))
    determinant = float(previous_vector[0] * next_vector[1] - previous_vector[1] * next_vector[0])
    return abs(float(math.degrees(math.atan2(determinant, dot_product))))


def build_row_disturbance_series(
    row: AtomRow,
    geometry: RowGeometry | None = None,
    *,
    unit: RowGeometryUnit | str = RowGeometryUnit.PX,
    minimum_point_count: int = 3,
    spacing_jump_threshold: float | None = None,
    transverse_jump_threshold: float | None = None,
    direction_change_threshold_deg: float = 10.0,
) -> RowDisturbanceSeries | None:
    """Build minimal local-disturbance markers for one row.

    The helper is intentionally heuristic: it emits raw local markers and simple thresholded
    candidate flags, without claiming a full domain-wall classifier.
    """

    normalized_unit = _normalize_projection_unit(unit)
    fitted_geometry = geometry
    if fitted_geometry is None:
        fitted_geometry = fit_row_geometry(row, minimum_point_count=max(2, minimum_point_count - 1))
    if fitted_geometry is None:
        return None
    if fitted_geometry.row_id != row.row_id:
        raise ValueError("geometry.row_id must match row.row_id.")

    projection_series = project_row_points(
        row,
        geometry=fitted_geometry,
        unit=normalized_unit,
        sort_mode=RowProjectionSortMode.ALONG_AXIS,
        minimum_point_count=max(2, minimum_point_count - 1),
    )
    if projection_series is None or len(projection_series.samples) < int(minimum_point_count):
        return None

    samples = projection_series.samples
    spacing_values = [
        float(right_sample.along_value - left_sample.along_value)
        for left_sample, right_sample in zip(samples, samples[1:])
    ]
    if spacing_jump_threshold is None:
        baseline_spacing = float(np.median(spacing_values)) if spacing_values else 0.0
        spacing_jump_threshold = max(0.0, 0.5 * baseline_spacing)
    else:
        spacing_jump_threshold = float(spacing_jump_threshold)

    if transverse_jump_threshold is None:
        baseline_spacing = float(np.median(spacing_values)) if spacing_values else 0.0
        transverse_rms = float(
            math.sqrt(
                sum(float(sample.transverse_value) ** 2 for sample in samples) / len(samples)
            )
        )
        transverse_jump_threshold = max(3.0 * transverse_rms, 0.15 * baseline_spacing)
    else:
        transverse_jump_threshold = float(transverse_jump_threshold)

    direction_change_threshold_deg = float(direction_change_threshold_deg)

    disturbance_samples: list[RowDisturbanceSample] = []
    for previous_sample, current_sample, next_sample in zip(samples, samples[1:], samples[2:]):
        spacing_before = float(current_sample.along_value - previous_sample.along_value)
        spacing_after = float(next_sample.along_value - current_sample.along_value)
        spacing_jump_abs = abs(spacing_after - spacing_before)

        transverse_delta_before = float(current_sample.transverse_value - previous_sample.transverse_value)
        transverse_delta_after = float(next_sample.transverse_value - current_sample.transverse_value)
        transverse_jump_abs = max(abs(transverse_delta_before), abs(transverse_delta_after))

        previous_vector = np.asarray(
            [
                float(current_sample.x_value - previous_sample.x_value),
                float(current_sample.y_value - previous_sample.y_value),
            ],
            dtype=float,
        )
        next_vector = np.asarray(
            [
                float(next_sample.x_value - current_sample.x_value),
                float(next_sample.y_value - current_sample.y_value),
            ],
            dtype=float,
        )
        local_direction_change_deg = _local_angle_change_deg(previous_vector, next_vector)

        is_candidate_spacing = spacing_jump_abs > spacing_jump_threshold
        is_candidate_transverse = transverse_jump_abs > transverse_jump_threshold
        is_candidate_direction = local_direction_change_deg > direction_change_threshold_deg

        score_components = []
        score_components.append(
            0.0
            if spacing_jump_threshold <= 0.0
            else spacing_jump_abs / spacing_jump_threshold
        )
        score_components.append(
            0.0
            if transverse_jump_threshold <= 0.0
            else transverse_jump_abs / transverse_jump_threshold
        )
        score_components.append(
            0.0
            if direction_change_threshold_deg <= 0.0
            else local_direction_change_deg / direction_change_threshold_deg
        )

        disturbance_samples.append(
            RowDisturbanceSample(
                row_id=row.row_id,
                point_id=current_sample.point_id,
                point_index=current_sample.point_index,
                image_id=current_sample.image_id,
                source_group_id=current_sample.source_group_id,
                previous_point_id=previous_sample.point_id,
                next_point_id=next_sample.point_id,
                along_value=float(current_sample.along_value),
                transverse_value=float(current_sample.transverse_value),
                spacing_before=spacing_before,
                spacing_after=spacing_after,
                spacing_jump_abs=spacing_jump_abs,
                transverse_jump_abs=transverse_jump_abs,
                local_direction_change_deg=local_direction_change_deg,
                candidate_score=max(score_components),
                is_candidate_spacing=is_candidate_spacing,
                is_candidate_transverse=is_candidate_transverse,
                is_candidate_direction=is_candidate_direction,
            )
        )

    return RowDisturbanceSeries(
        row_id=row.row_id,
        row_display_name=row.display_name,
        unit=normalized_unit,
        geometry=fitted_geometry,
        spacing_jump_threshold=spacing_jump_threshold,
        transverse_jump_threshold=transverse_jump_threshold,
        direction_change_threshold_deg=direction_change_threshold_deg,
        samples=tuple(disturbance_samples),
    )
