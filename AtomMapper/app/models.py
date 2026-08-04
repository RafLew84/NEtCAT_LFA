"""Data models used by the AtomMapper application."""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Dict, Optional, Tuple
from uuid import uuid4

import numpy as np


@dataclass(frozen=True)
class PhysicalCalibration:
    """Physical calibration derived from STM scan metadata."""

    pixels_x: int
    pixels_y: int
    size_nm_x: float
    size_nm_y: float
    pixel_size_nm_x: float
    pixel_size_nm_y: float

    def x_px_to_nm(self, x_px: float) -> float:
        """Convert an X image coordinate from pixels to nanometers."""

        return float(x_px) * self.pixel_size_nm_x

    def y_px_to_nm(self, y_px: float) -> float:
        """Convert a Y image coordinate from pixels to nanometers."""

        return float(y_px) * self.pixel_size_nm_y

    def point_px_to_nm(self, x_px: float, y_px: float) -> tuple[float, float]:
        """Convert a 2D image coordinate from pixels to nanometers."""

        return (self.x_px_to_nm(x_px), self.y_px_to_nm(y_px))


@dataclass(frozen=True)
class ROIState:
    """Image-space rectangular ROI stored in pixel coordinates."""

    x: int
    y: int
    width: int
    height: int

    def clamped(self, image_width: int, image_height: int, minimum_size: int = 4) -> "ROIState":
        """Return an ROI clamped to image bounds."""

        max_width = max(1, int(image_width))
        max_height = max(1, int(image_height))
        min_size = max(1, int(minimum_size))

        width = min(max_width, max(min_size, int(round(self.width))))
        height = min(max_height, max(min_size, int(round(self.height))))
        x = min(max(0, int(round(self.x))), max(0, max_width - width))
        y = min(max(0, int(round(self.y))), max(0, max_height - height))
        return ROIState(x=x, y=y, width=width, height=height)

    def to_dict(self) -> Dict[str, int]:
        """Return a JSON-serializable dictionary representation of the ROI."""

        return {
            "x": int(self.x),
            "y": int(self.y),
            "width": int(self.width),
            "height": int(self.height),
        }

    @classmethod
    def from_dict(cls, payload: Dict[str, Any]) -> "ROIState":
        """Rebuild an ROIState from a serialized payload."""

        return cls(
            x=int(payload["x"]),
            y=int(payload["y"]),
            width=int(payload["width"]),
            height=int(payload["height"]),
        )


@dataclass(frozen=True)
class LoadedImage:
    """Normalized STM image payload used by AtomMapper."""

    source_path: str
    display_name: str
    file_extension: str
    image_data: np.ndarray = field(repr=False)
    pixels_x: int
    pixels_y: int
    size_nm_x: float
    size_nm_y: float
    image_id: str = ""
    source_group_id: str = ""
    parent_image_id: Optional[str] = None
    variant_name: str = "original"
    metadata: Dict[str, Any] = field(default_factory=dict, repr=False)
    raw_metadata: Dict[str, Any] = field(default_factory=dict, repr=False)

    def __post_init__(self) -> None:
        """Normalize variant identity fields for later session/preprocessing work."""

        image_id = str(self.image_id).strip() or uuid4().hex
        variant_name = str(self.variant_name).strip().lower() or "original"
        source_group_id = str(self.source_group_id).strip() or image_id
        pixels_x = int(self.pixels_x)
        pixels_y = int(self.pixels_y)
        size_nm_x = float(self.size_nm_x)
        size_nm_y = float(self.size_nm_y)

        object.__setattr__(self, "image_id", image_id)
        object.__setattr__(self, "source_group_id", source_group_id)
        object.__setattr__(self, "variant_name", variant_name)
        object.__setattr__(self, "pixels_x", pixels_x)
        object.__setattr__(self, "pixels_y", pixels_y)
        object.__setattr__(self, "size_nm_x", size_nm_x)
        object.__setattr__(self, "size_nm_y", size_nm_y)

    @property
    def path(self) -> Path:
        """Return the source path as a ``Path`` object."""

        return Path(self.source_path)

    @property
    def is_original(self) -> bool:
        """Return ``True`` when the image is the source/original variant."""

        return self.parent_image_id is None and self.variant_name == "original"

    @property
    def physical_calibration(self) -> Optional[PhysicalCalibration]:
        """Return the image-space physical calibration when scan metadata is valid."""

        if self.pixels_x <= 0 or self.pixels_y <= 0:
            return None
        if not np.isfinite(self.size_nm_x) or not np.isfinite(self.size_nm_y):
            return None
        if self.size_nm_x <= 0.0 or self.size_nm_y <= 0.0:
            return None
        return PhysicalCalibration(
            pixels_x=self.pixels_x,
            pixels_y=self.pixels_y,
            size_nm_x=self.size_nm_x,
            size_nm_y=self.size_nm_y,
            pixel_size_nm_x=self.size_nm_x / self.pixels_x,
            pixel_size_nm_y=self.size_nm_y / self.pixels_y,
        )

    @property
    def has_physical_calibration(self) -> bool:
        """Return ``True`` when valid scan-size metadata supports px->nm conversion."""

        return self.physical_calibration is not None

    @property
    def pixel_size_nm_x(self) -> Optional[float]:
        """Return the physical pixel size along X when available."""

        calibration = self.physical_calibration
        if calibration is None:
            return None
        return calibration.pixel_size_nm_x

    @property
    def pixel_size_nm_y(self) -> Optional[float]:
        """Return the physical pixel size along Y when available."""

        calibration = self.physical_calibration
        if calibration is None:
            return None
        return calibration.pixel_size_nm_y

    @property
    def calibration_summary(self) -> Optional[str]:
        """Return a short human-readable physical-calibration summary."""

        calibration = self.physical_calibration
        if calibration is None:
            return None
        return (
            f"{calibration.size_nm_x:.3f} x {calibration.size_nm_y:.3f} nm | "
            f"{calibration.pixel_size_nm_x:.4f} x {calibration.pixel_size_nm_y:.4f} nm/px"
        )

    def derive_variant(
        self,
        *,
        variant_name: str,
        image_data: np.ndarray,
        display_name: Optional[str] = None,
        metadata_updates: Optional[Dict[str, Any]] = None,
        raw_metadata_updates: Optional[Dict[str, Any]] = None,
        pixels_x: Optional[int] = None,
        pixels_y: Optional[int] = None,
        size_nm_x: Optional[float] = None,
        size_nm_y: Optional[float] = None,
    ) -> "LoadedImage":
        """Create a derived image variant that stays in the same source group."""

        updated_metadata = dict(self.metadata)
        if metadata_updates:
            updated_metadata.update(metadata_updates)

        updated_raw_metadata = dict(self.raw_metadata)
        if raw_metadata_updates:
            updated_raw_metadata.update(raw_metadata_updates)

        normalized_variant = str(variant_name).strip().lower()
        if not normalized_variant:
            raise ValueError("variant_name must be a non-empty string.")

        variant_display_name = display_name or f"{self.path.stem} [{normalized_variant}]{self.path.suffix}"

        return LoadedImage(
            image_id=uuid4().hex,
            source_group_id=self.source_group_id,
            parent_image_id=self.image_id,
            variant_name=normalized_variant,
            source_path=self.source_path,
            display_name=variant_display_name,
            file_extension=self.file_extension,
            image_data=np.asarray(image_data, dtype=float),
            pixels_x=self.pixels_x if pixels_x is None else int(pixels_x),
            pixels_y=self.pixels_y if pixels_y is None else int(pixels_y),
            size_nm_x=self.size_nm_x if size_nm_x is None else float(size_nm_x),
            size_nm_y=self.size_nm_y if size_nm_y is None else float(size_nm_y),
            metadata=updated_metadata,
            raw_metadata=updated_raw_metadata,
        )

    def to_dict(self) -> Dict[str, Any]:
        """Return a JSON-serializable dictionary representation of the image."""

        return {
            "source_path": self.source_path,
            "display_name": self.display_name,
            "file_extension": self.file_extension,
            "image_data": np.asarray(self.image_data, dtype=float).tolist(),
            "pixels_x": self.pixels_x,
            "pixels_y": self.pixels_y,
            "size_nm_x": self.size_nm_x,
            "size_nm_y": self.size_nm_y,
            "image_id": self.image_id,
            "source_group_id": self.source_group_id,
            "parent_image_id": self.parent_image_id,
            "variant_name": self.variant_name,
            "metadata": dict(self.metadata),
            "raw_metadata": dict(self.raw_metadata),
        }

    @classmethod
    def from_dict(cls, payload: Dict[str, Any]) -> "LoadedImage":
        """Rebuild a LoadedImage from a serialized payload."""

        image_data = np.asarray(payload["image_data"], dtype=float)
        if image_data.ndim != 2:
            raise ValueError("LoadedImage.image_data must be a 2D array.")
        return cls(
            source_path=payload["source_path"],
            display_name=payload["display_name"],
            file_extension=payload["file_extension"],
            image_data=image_data,
            pixels_x=payload["pixels_x"],
            pixels_y=payload["pixels_y"],
            size_nm_x=payload["size_nm_x"],
            size_nm_y=payload["size_nm_y"],
            image_id=payload.get("image_id", ""),
            source_group_id=payload.get("source_group_id", ""),
            parent_image_id=payload.get("parent_image_id"),
            variant_name=payload.get("variant_name", "original"),
            metadata=dict(payload.get("metadata", {})),
            raw_metadata=dict(payload.get("raw_metadata", {})),
        )


def _normalize_optional_float(value: Optional[float], field_name: str) -> Optional[float]:
    """Normalize optional float fields used in analytical point models."""

    if value is None:
        return None
    normalized = float(value)
    if not np.isfinite(normalized):
        raise ValueError(f"{field_name} must be finite when provided.")
    return normalized


@dataclass(frozen=True)
class AtomPoint:
    """Serialized analytical point stored from the current ROI/Gaussian fit."""

    row_id: str
    image_id: str
    source_group_id: str
    point_index: int
    x_px: float
    y_px: float
    point_id: str = ""
    x_nm: Optional[float] = None
    y_nm: Optional[float] = None
    amplitude: Optional[float] = None
    sigma_x_px: Optional[float] = None
    sigma_y_px: Optional[float] = None
    position_std_x_px: Optional[float] = None
    position_std_y_px: Optional[float] = None
    position_std_x_nm: Optional[float] = None
    position_std_y_nm: Optional[float] = None
    theta_deg: Optional[float] = None
    offset: Optional[float] = None
    fit_success: bool = True
    fit_error_message: Optional[str] = None
    manual_override: bool = False
    manual_override_source: Optional[str] = None
    original_x_px: Optional[float] = None
    original_y_px: Optional[float] = None
    metadata: Dict[str, Any] = field(default_factory=dict, repr=False)

    def __post_init__(self) -> None:
        point_id = str(self.point_id).strip() or uuid4().hex
        row_id = str(self.row_id).strip()
        image_id = str(self.image_id).strip()
        source_group_id = str(self.source_group_id).strip()
        if not row_id:
            raise ValueError("row_id must be a non-empty string.")
        if not image_id:
            raise ValueError("image_id must be a non-empty string.")
        if not source_group_id:
            raise ValueError("source_group_id must be a non-empty string.")

        point_index = int(self.point_index)
        if point_index < 0:
            raise ValueError("point_index must be >= 0.")

        x_px = float(self.x_px)
        y_px = float(self.y_px)
        if not np.isfinite(x_px) or not np.isfinite(y_px):
            raise ValueError("x_px and y_px must be finite.")

        fit_error_message = None
        if self.fit_error_message is not None:
            fit_error_message = str(self.fit_error_message).strip() or None

        manual_override_source = None
        if self.manual_override_source is not None:
            manual_override_source = str(self.manual_override_source).strip() or None

        manual_override = bool(self.manual_override)
        original_x_px = _normalize_optional_float(self.original_x_px, "original_x_px")
        original_y_px = _normalize_optional_float(self.original_y_px, "original_y_px")
        if manual_override:
            if original_x_px is None:
                original_x_px = x_px
            if original_y_px is None:
                original_y_px = y_px

        object.__setattr__(self, "point_id", point_id)
        object.__setattr__(self, "row_id", row_id)
        object.__setattr__(self, "image_id", image_id)
        object.__setattr__(self, "source_group_id", source_group_id)
        object.__setattr__(self, "point_index", point_index)
        object.__setattr__(self, "x_px", x_px)
        object.__setattr__(self, "y_px", y_px)
        object.__setattr__(self, "x_nm", _normalize_optional_float(self.x_nm, "x_nm"))
        object.__setattr__(self, "y_nm", _normalize_optional_float(self.y_nm, "y_nm"))
        object.__setattr__(self, "amplitude", _normalize_optional_float(self.amplitude, "amplitude"))
        object.__setattr__(self, "sigma_x_px", _normalize_optional_float(self.sigma_x_px, "sigma_x_px"))
        object.__setattr__(self, "sigma_y_px", _normalize_optional_float(self.sigma_y_px, "sigma_y_px"))
        object.__setattr__(
            self,
            "position_std_x_px",
            _normalize_optional_float(self.position_std_x_px, "position_std_x_px"),
        )
        object.__setattr__(
            self,
            "position_std_y_px",
            _normalize_optional_float(self.position_std_y_px, "position_std_y_px"),
        )
        object.__setattr__(
            self,
            "position_std_x_nm",
            _normalize_optional_float(self.position_std_x_nm, "position_std_x_nm"),
        )
        object.__setattr__(
            self,
            "position_std_y_nm",
            _normalize_optional_float(self.position_std_y_nm, "position_std_y_nm"),
        )
        object.__setattr__(self, "theta_deg", _normalize_optional_float(self.theta_deg, "theta_deg"))
        object.__setattr__(self, "offset", _normalize_optional_float(self.offset, "offset"))
        object.__setattr__(self, "fit_success", bool(self.fit_success))
        object.__setattr__(self, "fit_error_message", fit_error_message)
        object.__setattr__(self, "manual_override", manual_override)
        object.__setattr__(self, "manual_override_source", manual_override_source)
        object.__setattr__(self, "original_x_px", original_x_px)
        object.__setattr__(self, "original_y_px", original_y_px)
        object.__setattr__(self, "metadata", dict(self.metadata))

    @property
    def fit_x_px(self) -> float:
        """Return the original fit-derived X position preserved through manual edits."""

        return self.original_x_px if self.original_x_px is not None else self.x_px

    @property
    def fit_y_px(self) -> float:
        """Return the original fit-derived Y position preserved through manual edits."""

        return self.original_y_px if self.original_y_px is not None else self.y_px

    def with_manual_position(
        self,
        *,
        x_px: float,
        y_px: float,
        x_nm: Optional[float] = None,
        y_nm: Optional[float] = None,
        source: str = "manual",
    ) -> "AtomPoint":
        """Return a copy of the point with manually corrected coordinates."""

        metadata = dict(self.metadata)
        if self.position_std_x_px is not None or self.position_std_y_px is not None:
            metadata["position_uncertainty_reference"] = "original_fit_position"
        return replace(
            self,
            x_px=x_px,
            y_px=y_px,
            x_nm=self.x_nm if x_nm is None else x_nm,
            y_nm=self.y_nm if y_nm is None else y_nm,
            manual_override=True,
            manual_override_source=source,
            original_x_px=self.fit_x_px,
            original_y_px=self.fit_y_px,
            metadata=metadata,
        )

    def to_dict(self) -> Dict[str, Any]:
        """Return a JSON-serializable dictionary representation of the point."""

        return {
            "point_id": self.point_id,
            "row_id": self.row_id,
            "image_id": self.image_id,
            "source_group_id": self.source_group_id,
            "point_index": self.point_index,
            "x_px": self.x_px,
            "y_px": self.y_px,
            "x_nm": self.x_nm,
            "y_nm": self.y_nm,
            "amplitude": self.amplitude,
            "sigma_x_px": self.sigma_x_px,
            "sigma_y_px": self.sigma_y_px,
            "position_std_x_px": self.position_std_x_px,
            "position_std_y_px": self.position_std_y_px,
            "position_std_x_nm": self.position_std_x_nm,
            "position_std_y_nm": self.position_std_y_nm,
            "theta_deg": self.theta_deg,
            "offset": self.offset,
            "fit_success": self.fit_success,
            "fit_error_message": self.fit_error_message,
            "manual_override": self.manual_override,
            "manual_override_source": self.manual_override_source,
            "original_x_px": self.original_x_px,
            "original_y_px": self.original_y_px,
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_dict(cls, payload: Dict[str, Any]) -> "AtomPoint":
        """Rebuild an AtomPoint from a serialized payload."""

        return cls(
            point_id=payload.get("point_id", ""),
            row_id=payload["row_id"],
            image_id=payload["image_id"],
            source_group_id=payload["source_group_id"],
            point_index=payload["point_index"],
            x_px=payload["x_px"],
            y_px=payload["y_px"],
            x_nm=payload.get("x_nm"),
            y_nm=payload.get("y_nm"),
            amplitude=payload.get("amplitude"),
            sigma_x_px=payload.get("sigma_x_px"),
            sigma_y_px=payload.get("sigma_y_px"),
            position_std_x_px=payload.get("position_std_x_px"),
            position_std_y_px=payload.get("position_std_y_px"),
            position_std_x_nm=payload.get("position_std_x_nm"),
            position_std_y_nm=payload.get("position_std_y_nm"),
            theta_deg=payload.get("theta_deg"),
            offset=payload.get("offset"),
            fit_success=payload.get("fit_success", True),
            fit_error_message=payload.get("fit_error_message"),
            manual_override=payload.get("manual_override", False),
            manual_override_source=payload.get("manual_override_source"),
            original_x_px=payload.get("original_x_px"),
            original_y_px=payload.get("original_y_px"),
            metadata=dict(payload.get("metadata", {})),
        )


@dataclass(frozen=True)
class AtomRow:
    """Logical atom row bound to one source-group family of STM image variants."""

    source_group_id: str
    row_id: str = ""
    display_name: str = ""
    color_hex: Optional[str] = None
    points: Tuple[AtomPoint, ...] = field(default_factory=tuple, repr=False)
    metadata: Dict[str, Any] = field(default_factory=dict, repr=False)

    def __post_init__(self) -> None:
        row_id = str(self.row_id).strip() or uuid4().hex
        source_group_id = str(self.source_group_id).strip()
        if not source_group_id:
            raise ValueError("source_group_id must be a non-empty string.")

        display_name = str(self.display_name).strip() or f"Row {row_id[:8]}"
        color_hex = None if self.color_hex is None else str(self.color_hex).strip() or None
        normalized_points: list[AtomPoint] = []
        seen_point_ids: set[str] = set()
        for point in self.points:
            if point.row_id != row_id:
                raise ValueError("Every point in AtomRow.points must have the same row_id as the row.")
            if point.source_group_id != source_group_id:
                raise ValueError(
                    "Every point in AtomRow.points must have the same source_group_id as the row."
                )
            if point.point_id in seen_point_ids:
                raise ValueError("Duplicate point_id detected in AtomRow.points.")
            seen_point_ids.add(point.point_id)
            normalized_points.append(point)

        normalized_points.sort(key=lambda point: (point.point_index, point.point_id))

        object.__setattr__(self, "row_id", row_id)
        object.__setattr__(self, "source_group_id", source_group_id)
        object.__setattr__(self, "display_name", display_name)
        object.__setattr__(self, "color_hex", color_hex)
        object.__setattr__(self, "points", tuple(normalized_points))
        object.__setattr__(self, "metadata", dict(self.metadata))

    @property
    def point_count(self) -> int:
        """Return the number of stored points."""

        return len(self.points)

    @property
    def next_point_index(self) -> int:
        """Return the next index that should be used for appending a point."""

        if not self.points:
            return 0
        return max(point.point_index for point in self.points) + 1

    def with_point(self, point: AtomPoint) -> "AtomRow":
        """Return a new row with the provided point added or replaced."""

        if point.row_id != self.row_id:
            raise ValueError("point.row_id must match AtomRow.row_id.")
        if point.source_group_id != self.source_group_id:
            raise ValueError("point.source_group_id must match AtomRow.source_group_id.")

        updated_points = {existing.point_id: existing for existing in self.points}
        updated_points[point.point_id] = point
        return replace(self, points=tuple(updated_points.values()))

    def with_inserted_point(self, point: AtomPoint, *, insert_index: int | None = None) -> "AtomRow":
        """Return a new row with the point inserted at the requested order position."""

        if point.row_id != self.row_id:
            raise ValueError("point.row_id must match AtomRow.row_id.")
        if point.source_group_id != self.source_group_id:
            raise ValueError("point.source_group_id must match AtomRow.source_group_id.")

        ordered_points = list(self.points)
        ordered_points.sort(key=lambda existing: (existing.point_index, existing.point_id))
        ordered_points = [
            existing for existing in ordered_points if existing.point_id != point.point_id
        ]
        target_index = len(ordered_points) if insert_index is None else int(insert_index)
        target_index = max(0, min(target_index, len(ordered_points)))
        ordered_points.insert(target_index, point)
        return replace(self, points=self._reindex_points(ordered_points))

    def with_reordered_point(self, point_id: str, *, target_index: int) -> "AtomRow":
        """Return a new row with one point moved to a new order position."""

        normalized_point_id = str(point_id).strip()
        ordered_points = list(self.points)
        ordered_points.sort(key=lambda existing: (existing.point_index, existing.point_id))
        current_index = next(
            (index for index, point in enumerate(ordered_points) if point.point_id == normalized_point_id),
            None,
        )
        if current_index is None:
            raise ValueError(f"Point id '{point_id}' is not present in row '{self.row_id}'.")

        point = ordered_points.pop(current_index)
        clamped_target_index = max(0, min(int(target_index), len(ordered_points)))
        ordered_points.insert(clamped_target_index, point)
        return replace(self, points=self._reindex_points(ordered_points))

    def without_point(self, point_id: str) -> "AtomRow":
        """Return a new row with the selected point removed."""

        normalized_point_id = str(point_id).strip()
        filtered_points = tuple(
            point for point in self.points if point.point_id != normalized_point_id
        )
        if len(filtered_points) == len(self.points):
            return self
        return replace(self, points=self._reindex_points(filtered_points))

    def to_dict(self) -> Dict[str, Any]:
        """Return a JSON-serializable dictionary representation of the row."""

        return {
            "row_id": self.row_id,
            "source_group_id": self.source_group_id,
            "display_name": self.display_name,
            "color_hex": self.color_hex,
            "metadata": dict(self.metadata),
            "points": [point.to_dict() for point in self.points],
        }

    @classmethod
    def from_dict(cls, payload: Dict[str, Any]) -> "AtomRow":
        """Rebuild an AtomRow from a serialized payload."""

        points = tuple(AtomPoint.from_dict(item) for item in payload.get("points", []))
        return cls(
            row_id=payload.get("row_id", ""),
            source_group_id=payload["source_group_id"],
            display_name=payload.get("display_name", ""),
            color_hex=payload.get("color_hex"),
            metadata=dict(payload.get("metadata", {})),
            points=points,
        )

    @staticmethod
    def _reindex_points(points: Tuple[AtomPoint, ...] | list[AtomPoint]) -> tuple[AtomPoint, ...]:
        """Return points reindexed to a contiguous 0..N-1 sequence."""

        return tuple(
            replace(point, point_index=index)
            for index, point in enumerate(points)
        )
