"""CSV export helpers for AtomMapper point tables."""

from __future__ import annotations

import csv
import math
from pathlib import Path
from typing import Any, Sequence

from .models import AtomPoint, AtomRow, LoadedImage
from .plots import sorted_row_points

POINT_EXPORT_FIELDNAMES = (
    "image_name",
    "image_variant",
    "image_id",
    "source_group",
    "row_name",
    "row_id",
    "point_id",
    "point_index",
    "previous_point_id",
    "next_point_id",
    "x_px",
    "y_px",
    "x_nm",
    "y_nm",
    "distance_to_previous_px",
    "distance_to_next_px",
    "distance_to_previous_nm",
    "distance_to_next_nm",
    "amplitude",
    "sigma_x_px",
    "sigma_y_px",
    "position_std_x_px",
    "position_std_y_px",
    "position_std_x_nm",
    "position_std_y_nm",
    "position_uncertainty_status",
    "position_uncertainty_method",
    "position_uncertainty_reference",
    "theta_deg",
    "offset",
    "fit_success",
    "fit_error_message",
    "fit_model",
    "fit_method",
    "fit_mask_active",
    "fit_mask_pixel_count",
    "manual_override",
    "manual_override_source",
    "status",
)


def describe_point_status(point: AtomPoint) -> str:
    """Return the same status label used in GUI and CSV export."""

    if point.manual_override:
        source = point.manual_override_source or "manual"
        return f"manual ({source})"
    if point.fit_success:
        return "fit"
    if point.metadata.get("fallback_used"):
        return "fallback"
    return "stored"


def build_point_export_rows(
    rows: Sequence[AtomRow],
    images: Sequence[LoadedImage],
) -> list[dict[str, str]]:
    """Build CSV-ready rows for the provided AtomMapper rows."""

    image_by_id = {image.image_id: image for image in images}
    export_rows: list[dict[str, str]] = []

    sorted_rows = sorted(rows, key=lambda row: (row.display_name.lower(), row.row_id))
    for row in sorted_rows:
        ordered_points = sorted_row_points(row)
        for point_index, point in enumerate(ordered_points):
            image = image_by_id.get(point.image_id)
            fit_model = str(point.metadata.get("fit_model") or "").strip()
            fit_method = str(point.metadata.get("fit_method") or "").strip()
            if not fit_model:
                fit_model = _infer_fit_model_from_method(fit_method)
            previous_point = ordered_points[point_index - 1] if point_index > 0 else None
            next_point = (
                ordered_points[point_index + 1]
                if point_index + 1 < len(ordered_points)
                else None
            )
            export_rows.append(
                {
                    "image_name": "" if image is None else image.display_name,
                    "image_variant": "" if image is None else image.variant_name,
                    "image_id": point.image_id,
                    "source_group": point.source_group_id,
                    "row_name": row.display_name,
                    "row_id": row.row_id,
                    "point_id": point.point_id,
                    "point_index": str(point.point_index),
                    "previous_point_id": "" if previous_point is None else previous_point.point_id,
                    "next_point_id": "" if next_point is None else next_point.point_id,
                    "x_px": _format_optional_float(point.x_px),
                    "y_px": _format_optional_float(point.y_px),
                    "x_nm": _format_optional_float(point.x_nm),
                    "y_nm": _format_optional_float(point.y_nm),
                    "distance_to_previous_px": _format_optional_float(
                        _compute_point_distance_px(previous_point, point)
                    ),
                    "distance_to_next_px": _format_optional_float(
                        _compute_point_distance_px(point, next_point)
                    ),
                    "distance_to_previous_nm": _format_optional_float(
                        _compute_point_distance_nm(previous_point, point)
                    ),
                    "distance_to_next_nm": _format_optional_float(
                        _compute_point_distance_nm(point, next_point)
                    ),
                    "amplitude": _format_optional_float(point.amplitude),
                    "sigma_x_px": _format_optional_float(point.sigma_x_px),
                    "sigma_y_px": _format_optional_float(point.sigma_y_px),
                    "position_std_x_px": _format_optional_float(point.position_std_x_px),
                    "position_std_y_px": _format_optional_float(point.position_std_y_px),
                    "position_std_x_nm": _format_optional_float(point.position_std_x_nm),
                    "position_std_y_nm": _format_optional_float(point.position_std_y_nm),
                    "position_uncertainty_status": str(
                        point.metadata.get("position_uncertainty_status") or ""
                    ),
                    "position_uncertainty_method": str(
                        point.metadata.get("position_uncertainty_method") or ""
                    ),
                    "position_uncertainty_reference": str(
                        point.metadata.get("position_uncertainty_reference") or ""
                    ),
                    "theta_deg": _format_optional_float(point.theta_deg),
                    "offset": _format_optional_float(point.offset),
                    "fit_success": _format_bool(point.fit_success),
                    "fit_error_message": point.fit_error_message or "",
                    "fit_model": fit_model,
                    "fit_method": fit_method,
                    "fit_mask_active": _format_bool(point.metadata.get("fit_mask_active")),
                    "fit_mask_pixel_count": _format_optional_int(
                        point.metadata.get("fit_mask_pixel_count")
                    ),
                    "manual_override": _format_bool(point.manual_override),
                    "manual_override_source": point.manual_override_source or "",
                    "status": describe_point_status(point),
                }
            )

    return export_rows


def export_point_rows_to_csv(
    destination: str | Path,
    rows: Sequence[AtomRow],
    images: Sequence[LoadedImage],
) -> int:
    """Write point-export rows to CSV and return the exported point count."""

    export_rows = build_point_export_rows(rows, images)
    destination_path = Path(destination)
    destination_path.parent.mkdir(parents=True, exist_ok=True)
    with destination_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=POINT_EXPORT_FIELDNAMES)
        writer.writeheader()
        writer.writerows(export_rows)
    return len(export_rows)


def _format_optional_float(value: Any) -> str:
    if value is None:
        return ""
    return f"{float(value):.6f}"


def _format_bool(value: Any) -> str:
    return "true" if bool(value) else "false"


def _format_optional_int(value: Any) -> str:
    if value is None:
        return ""
    return str(int(value))


def _infer_fit_model_from_method(fit_method: str) -> str:
    method_lower = fit_method.strip().lower()
    if "gaussian" in method_lower:
        return "gaussian"
    if "lorentzian" in method_lower:
        return "lorentzian"
    if "voigt" in method_lower:
        return "voigt"
    return ""


def _compute_point_distance_px(
    first_point: AtomPoint | None,
    second_point: AtomPoint | None,
) -> float | None:
    if first_point is None or second_point is None:
        return None
    return math.hypot(second_point.x_px - first_point.x_px, second_point.y_px - first_point.y_px)


def _compute_point_distance_nm(
    first_point: AtomPoint | None,
    second_point: AtomPoint | None,
) -> float | None:
    if first_point is None or second_point is None:
        return None
    if (
        first_point.x_nm is None
        or first_point.y_nm is None
        or second_point.x_nm is None
        or second_point.y_nm is None
    ):
        return None
    return math.hypot(second_point.x_nm - first_point.x_nm, second_point.y_nm - first_point.y_nm)
