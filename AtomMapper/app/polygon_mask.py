"""Polygon-mask contracts and helpers for ROI-restricted local fitting."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable

import numpy as np

from .models import ROIState


@dataclass(frozen=True)
class PolygonMaskState:
    """Closed polygon described in image pixel coordinates."""

    vertices_xy: tuple[tuple[float, float], ...]

    def normalized(self) -> "PolygonMaskState":
        normalized_vertices: list[tuple[float, float]] = []
        for vertex in self.vertices_xy:
            if len(vertex) != 2:  # pragma: no cover - defensive
                continue
            x_value = float(vertex[0])
            y_value = float(vertex[1])
            if not np.isfinite(x_value) or not np.isfinite(y_value):
                continue
            normalized_vertices.append((x_value, y_value))
        return PolygonMaskState(vertices_xy=tuple(normalized_vertices))

    @property
    def is_valid(self) -> bool:
        return len(self.normalized().vertices_xy) >= 3

    def to_dict(self) -> dict[str, Any]:
        normalized = self.normalized()
        return {
            "vertices_xy": [
                {"x": float(x_value), "y": float(y_value)}
                for x_value, y_value in normalized.vertices_xy
            ]
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any] | None) -> "PolygonMaskState":
        data = payload or {}
        vertices = data.get("vertices_xy") or ()
        return cls(
            vertices_xy=tuple(
                (float(vertex.get("x", 0.0)), float(vertex.get("y", 0.0)))
                for vertex in vertices
                if isinstance(vertex, dict)
            )
        ).normalized()


def build_polygon_mask_for_roi(
    roi: ROIState,
    polygon_mask_state: PolygonMaskState | None,
) -> np.ndarray | None:
    """Rasterize a polygon defined in image coordinates into an ROI-local boolean mask."""

    if polygon_mask_state is None:
        return None

    normalized = polygon_mask_state.normalized()
    if not normalized.is_valid:
        return None

    vertices = np.asarray(normalized.vertices_xy, dtype=float)
    local_vertices = np.empty_like(vertices, dtype=float)
    local_vertices[:, 0] = vertices[:, 0] - float(roi.x)
    local_vertices[:, 1] = vertices[:, 1] - float(roi.y)
    return _rasterize_polygon_mask(
        height=int(roi.height),
        width=int(roi.width),
        vertices_xy=local_vertices,
    )


def _rasterize_polygon_mask(
    *,
    height: int,
    width: int,
    vertices_xy: Iterable[Iterable[float]],
) -> np.ndarray:
    """Rasterize a polygon into a boolean mask using ray casting on pixel centers."""

    if height <= 0 or width <= 0:
        return np.zeros((0, 0), dtype=bool)

    vertices = np.asarray(tuple(tuple(vertex) for vertex in vertices_xy), dtype=float)
    if vertices.ndim != 2 or vertices.shape[1] != 2 or vertices.shape[0] < 3:
        return np.zeros((height, width), dtype=bool)

    x_grid = np.arange(width, dtype=float)[None, :] + 0.5
    y_grid = np.arange(height, dtype=float)[:, None] + 0.5
    inside = np.zeros((height, width), dtype=bool)

    x_values = vertices[:, 0]
    y_values = vertices[:, 1]
    for index in range(len(vertices)):
        x1 = float(x_values[index])
        y1 = float(y_values[index])
        x2 = float(x_values[(index + 1) % len(vertices)])
        y2 = float(y_values[(index + 1) % len(vertices)])

        if y1 == y2:
            continue

        intersects = (y1 > y_grid) != (y2 > y_grid)
        x_intersection = ((x2 - x1) * (y_grid - y1) / (y2 - y1)) + x1
        inside ^= intersects & (x_grid < x_intersection)

    return inside
