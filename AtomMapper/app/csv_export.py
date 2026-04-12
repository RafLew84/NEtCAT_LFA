"""CSV export helpers for AtomMapper point tables."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any, Iterable, Sequence

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
    "x_px",
    "y_px",
    "x_nm",
    "y_nm",
    "amplitude",
    "sigma_x_px",
    "sigma_y_px",
    "theta_deg",
    "offset",
    "fit_success",
    "fit_error_message",
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
        for point in sorted_row_points(row):
            image = image_by_id.get(point.image_id)
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
                    "x_px": _format_optional_float(point.x_px),
                    "y_px": _format_optional_float(point.y_px),
                    "x_nm": _format_optional_float(point.x_nm),
                    "y_nm": _format_optional_float(point.y_nm),
                    "amplitude": _format_optional_float(point.amplitude),
                    "sigma_x_px": _format_optional_float(point.sigma_x_px),
                    "sigma_y_px": _format_optional_float(point.sigma_y_px),
                    "theta_deg": _format_optional_float(point.theta_deg),
                    "offset": _format_optional_float(point.offset),
                    "fit_success": _format_bool(point.fit_success),
                    "fit_error_message": point.fit_error_message or "",
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
