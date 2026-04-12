"""Versioned AtomMapper session model and serializer."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .models import AtomRow, LoadedImage, ROIState
from .plots import PlotUnit, RowPlotMode

ATOMMAPPER_SESSION_VERSION = 1


@dataclass(frozen=True)
class SessionViewState:
    """Serializable view/UI state worth preserving across sessions."""

    show_gaussian_fit: bool = True
    row_plot_mode: RowPlotMode = RowPlotMode.X_PX
    row_plot_unit: PlotUnit = PlotUnit.PX
    row_metrics_unit: PlotUnit = PlotUnit.PX
    global_scatter_unit: PlotUnit = PlotUnit.PX

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable representation of the view state."""

        return {
            "show_gaussian_fit": bool(self.show_gaussian_fit),
            "row_plot_mode": self.row_plot_mode.value,
            "row_plot_unit": self.row_plot_unit.value,
            "row_metrics_unit": self.row_metrics_unit.value,
            "global_scatter_unit": self.global_scatter_unit.value,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any] | None) -> "SessionViewState":
        """Rebuild the view-state payload from serialized data."""

        data = payload or {}
        return cls(
            show_gaussian_fit=bool(data.get("show_gaussian_fit", True)),
            row_plot_mode=RowPlotMode(data.get("row_plot_mode", RowPlotMode.X_PX.value)),
            row_plot_unit=PlotUnit(data.get("row_plot_unit", PlotUnit.PX.value)),
            row_metrics_unit=PlotUnit(data.get("row_metrics_unit", PlotUnit.PX.value)),
            global_scatter_unit=PlotUnit(data.get("global_scatter_unit", PlotUnit.PX.value)),
        )


@dataclass(frozen=True)
class AtomMapperSession:
    """Versioned, JSON-serializable snapshot of AtomMapper application state."""

    loaded_images: tuple[LoadedImage, ...] = field(default_factory=tuple)
    active_image_id: str | None = None
    roi_states_by_image_id: dict[str, ROIState] = field(default_factory=dict)
    rows: tuple[AtomRow, ...] = field(default_factory=tuple)
    active_row_id_by_source_group: dict[str, str] = field(default_factory=dict)
    active_point_id_by_source_group: dict[str, str] = field(default_factory=dict)
    view_state: SessionViewState = field(default_factory=SessionViewState)
    version: int = ATOMMAPPER_SESSION_VERSION

    def __post_init__(self) -> None:
        if int(self.version) != ATOMMAPPER_SESSION_VERSION:
            raise ValueError(
                f"Unsupported AtomMapper session version: {self.version!r}."
            )

        image_ids = [image.image_id for image in self.loaded_images]
        if len(set(image_ids)) != len(image_ids):
            raise ValueError("AtomMapperSession.loaded_images must use unique image_id values.")

        source_groups = {image.source_group_id for image in self.loaded_images}
        if self.active_image_id is not None and self.active_image_id not in set(image_ids):
            raise ValueError("AtomMapperSession.active_image_id must reference a loaded image.")

        for image_id in self.roi_states_by_image_id:
            if image_id not in set(image_ids):
                raise ValueError("ROI session state must reference a loaded image_id.")

        row_ids = [row.row_id for row in self.rows]
        if len(set(row_ids)) != len(row_ids):
            raise ValueError("AtomMapperSession.rows must use unique row_id values.")

        point_ids_by_group: dict[str, set[str]] = {}
        row_id_to_group: dict[str, str] = {}
        for row in self.rows:
            if source_groups and row.source_group_id not in source_groups:
                raise ValueError("Session rows must belong to loaded image source groups.")
            row_id_to_group[row.row_id] = row.source_group_id
            point_ids_by_group.setdefault(row.source_group_id, set()).update(
                point.point_id for point in row.points
            )

        for source_group_id, row_id in self.active_row_id_by_source_group.items():
            expected_group = row_id_to_group.get(row_id)
            if expected_group is None:
                raise ValueError("Active row mapping must reference an existing row_id.")
            if expected_group != source_group_id:
                raise ValueError("Active row mapping must match the row source_group_id.")

        for source_group_id, point_id in self.active_point_id_by_source_group.items():
            if point_id not in point_ids_by_group.get(source_group_id, set()):
                raise ValueError("Active point mapping must reference an existing point_id.")

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable representation of the session."""

        return {
            "version": self.version,
            "loaded_images": [image.to_dict() for image in self.loaded_images],
            "active_image_id": self.active_image_id,
            "roi_states_by_image_id": {
                image_id: roi_state.to_dict()
                for image_id, roi_state in self.roi_states_by_image_id.items()
            },
            "rows": [row.to_dict() for row in self.rows],
            "active_row_id_by_source_group": dict(self.active_row_id_by_source_group),
            "active_point_id_by_source_group": dict(self.active_point_id_by_source_group),
            "view_state": self.view_state.to_dict(),
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "AtomMapperSession":
        """Rebuild the session model from a serialized payload."""

        version = int(payload.get("version", -1))
        if version != ATOMMAPPER_SESSION_VERSION:
            raise ValueError(f"Unsupported AtomMapper session version: {version!r}.")

        return cls(
            version=version,
            loaded_images=tuple(
                LoadedImage.from_dict(item) for item in payload.get("loaded_images", [])
            ),
            active_image_id=payload.get("active_image_id"),
            roi_states_by_image_id={
                str(image_id): ROIState.from_dict(roi_payload)
                for image_id, roi_payload in dict(payload.get("roi_states_by_image_id", {})).items()
            },
            rows=tuple(AtomRow.from_dict(item) for item in payload.get("rows", [])),
            active_row_id_by_source_group={
                str(group_id): str(row_id)
                for group_id, row_id in dict(payload.get("active_row_id_by_source_group", {})).items()
            },
            active_point_id_by_source_group={
                str(group_id): str(point_id)
                for group_id, point_id in dict(payload.get("active_point_id_by_source_group", {})).items()
            },
            view_state=SessionViewState.from_dict(payload.get("view_state")),
        )
