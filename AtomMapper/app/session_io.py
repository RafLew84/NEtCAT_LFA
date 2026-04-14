"""Session snapshot builders and file I/O helpers for AtomMapper."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Mapping

from .controller import AtomMapperController
from .fit_settings import FitSettingsState
from .session_model import AtomMapperSession, SessionViewState


def build_session_from_runtime(
    controller: AtomMapperController,
    *,
    active_point_id_by_source_group: Mapping[str, str] | None = None,
    fit_settings: FitSettingsState | None = None,
    view_state: SessionViewState | None = None,
) -> AtomMapperSession:
    """Build a versioned session snapshot from the live application runtime."""

    active_point_mapping = _sanitize_active_point_mapping(
        controller,
        active_point_id_by_source_group or {},
    )
    return AtomMapperSession(
        loaded_images=controller.loaded_images,
        active_image_id=None if controller.active_image is None else controller.active_image.image_id,
        roi_states_by_image_id=controller.roi_states_by_image_id,
        rows=controller.atom_rows,
        active_row_id_by_source_group=controller.active_row_id_by_source_group,
        active_point_id_by_source_group=active_point_mapping,
        fit_settings=fit_settings or FitSettingsState(),
        view_state=view_state or SessionViewState(),
    )


def save_session_to_file(path: str | Path, session: AtomMapperSession) -> Path:
    """Write a session snapshot to disk as UTF-8 JSON and return the output path."""

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding="utf-8") as handle:
        json.dump(session.to_dict(), handle, indent=2, ensure_ascii=True)
        handle.write("\n")
    return destination


def load_session_from_file(path: str | Path) -> AtomMapperSession:
    """Read a session snapshot from disk and return the validated project model."""

    source = Path(path)
    with source.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)

    if not isinstance(payload, dict):
        raise ValueError("Session file must contain a top-level JSON object.")
    return AtomMapperSession.from_dict(payload)


def _sanitize_active_point_mapping(
    controller: AtomMapperController,
    active_point_id_by_source_group: Mapping[str, str],
) -> dict[str, str]:
    """Drop stale active-point references before building a session snapshot."""

    valid_points_by_group: dict[str, set[str]] = {}
    for row in controller.atom_rows:
        valid_points_by_group.setdefault(row.source_group_id, set()).update(
            point.point_id for point in row.points
        )

    sanitized: dict[str, str] = {}
    for source_group_id, point_id in active_point_id_by_source_group.items():
        if point_id in valid_points_by_group.get(source_group_id, set()):
            sanitized[str(source_group_id)] = str(point_id)
    return sanitized
