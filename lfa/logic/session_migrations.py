from __future__ import annotations

from copy import deepcopy
import math
from typing import Any, Dict

CURRENT_SESSION_VERSION = "2.0"


def migrate_payload(payload: Dict[str, Any]) -> Dict[str, Any]:
    """
    Return a deep-copied payload migrated to the latest session format.
    """
    migrated = deepcopy(payload)
    version = migrated.get("format_version")
    if version is None:
        version = "1.1"
        migrated["format_version"] = version

    if version == CURRENT_SESSION_VERSION:
        return migrated

    if version == "1.0":
        migrated = _migrate_1_0_to_1_1(migrated)
        version = migrated.get("format_version", "1.1")

    if version == "1.1":
        migrated = _migrate_1_1_to_2_0(migrated)
        migrated["format_version"] = CURRENT_SESSION_VERSION
        return migrated

    raise ValueError(f"Unsupported session format version: {version}")


def _migrate_1_1_to_2_0(payload: Dict[str, Any]) -> Dict[str, Any]:
    controller_state = dict(payload.get("controller_state", {}) or {})

    if "domain_wall_analysis_results" in controller_state and "superstructure_periodicity_results" not in controller_state:
        controller_state["superstructure_periodicity_results"] = controller_state.get(
            "domain_wall_analysis_results"
        )

    controller_state["substrate_visual_offset_nm"] = _normalise_offset(
        controller_state.get("substrate_visual_offset_nm")
    )

    raw_offsets = controller_state.get("adsorbate_visual_offsets_nm", {}) or {}
    if isinstance(raw_offsets, dict):
        normalised_offsets: Dict[int, tuple[float, float]] = {}
        for key, value in raw_offsets.items():
            try:
                idx = int(key)
            except (TypeError, ValueError):
                continue
            normalised_offsets[idx] = _normalise_offset(value)
        controller_state["adsorbate_visual_offsets_nm"] = normalised_offsets

    substrate_pairs = controller_state.get("substrate_spot_pairs") or []
    controller_state["substrate_spot_pairs"] = [
        {
            "raw": _normalise_point(entry.get("raw")),
            "transformed": _normalise_point(entry.get("transformed")),
        }
        if isinstance(entry, dict)
        else {
            "raw": _normalise_point(entry[0] if isinstance(entry, (list, tuple)) and len(entry) > 0 else None),
            "transformed": _normalise_point(
                entry[1] if isinstance(entry, (list, tuple)) and len(entry) > 1 else None
            ),
        }
        for entry in substrate_pairs
    ]

    adsorbate_pairs = controller_state.get("adsorbate_spot_pairs") or {}
    if isinstance(adsorbate_pairs, dict):
        converted: Dict[int, list] = {}
        for key, pairs in adsorbate_pairs.items():
            try:
                idx = int(key)
            except (TypeError, ValueError):
                continue
            normalised_pairs = []
            if isinstance(pairs, list):
                for pair in pairs:
                    if isinstance(pair, dict):
                        raw = _normalise_point(pair.get("raw"))
                        transformed = _normalise_point(pair.get("transformed"))
                    elif isinstance(pair, (list, tuple)) and len(pair) == 2:
                        raw = _normalise_point(pair[0])
                        transformed = _normalise_point(pair[1])
                    else:
                        raw = transformed = None
                    normalised_pairs.append({"raw": raw, "transformed": transformed})
            converted[idx] = normalised_pairs
        controller_state["adsorbate_spot_pairs"] = converted

    payload["controller_state"] = controller_state
    return payload


def _migrate_1_0_to_1_1(payload: Dict[str, Any]) -> Dict[str, Any]:
    payload.setdefault("controller_state", {})
    history_data = payload.setdefault("history_data", {})
    history_data.setdefault("original_images", [])
    history_data.setdefault("original_order", [])
    payload["format_version"] = "1.1"
    return payload


def _normalise_point(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, dict):
        return _normalise_point((value.get("dx") or value.get("x"), value.get("dy") or value.get("y")))
    if isinstance(value, (list, tuple)) and len(value) == 2:
        try:
            x = float(value[0])
            y = float(value[1])
        except (TypeError, ValueError):
            return None
        if not _is_finite(x) or not _is_finite(y):
            return None
        return (x, y)
    return None


def _normalise_offset(value: Any) -> tuple[float, float]:
    point = _normalise_point(value)
    if point is None:
        return (0.0, 0.0)
    return (point[0], point[1])


def _is_finite(value: float) -> bool:
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False
