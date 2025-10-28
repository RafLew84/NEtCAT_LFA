from __future__ import annotations

import logging
from typing import Dict, Tuple


class RealSpaceVisualizerState:
    """Encapsulates shared visual offset state for the real-space visualizer."""

    def __init__(self, app_controller: object, logger: logging.Logger | None = None) -> None:
        self._controller = app_controller
        self._logger = logger or logging.getLogger(__name__)

    # ---- Substrate offsets -------------------------------------------------
    def get_substrate_offset(self) -> Tuple[float, float]:
        """Return the currently stored substrate offset in nanometres."""
        value = getattr(self._controller, "substrate_visual_offset_nm", (0.0, 0.0))
        return _normalize_offset(value)

    def set_substrate_offset(self, offset: Tuple[float, float]) -> bool:
        """Persist a substrate offset if it has changed. Returns True on change."""
        normalized = _normalize_offset(offset)
        current = self.get_substrate_offset()
        if current == normalized:
            return False
        setattr(self._controller, "substrate_visual_offset_nm", normalized)
        self._logger.debug("RealSpaceVisualizerState: substrate offset set to %s.", normalized)
        return True

    # ---- Adsorbate offsets -------------------------------------------------
    def _get_adsorbate_offsets_dict(self) -> Dict[int, Tuple[float, float]]:
        offsets = getattr(self._controller, "adsorbate_visual_offsets_nm", None)
        if not isinstance(offsets, dict):
            offsets = {0: (0.0, 0.0)}
            setattr(self._controller, "adsorbate_visual_offsets_nm", offsets)
        return offsets

    def get_adsorbate_offset(self, set_index: int | None) -> Tuple[float, float]:
        """Return the offset for the given adsorbate set, creating defaults if needed."""
        if set_index is None or set_index < 0:
            return (0.0, 0.0)
        offsets = self._get_adsorbate_offsets_dict()
        if set_index not in offsets:
            offsets[set_index] = (0.0, 0.0)
        return _normalize_offset(offsets[set_index])

    def set_adsorbate_offset(self, set_index: int | None, offset: Tuple[float, float]) -> bool:
        """Persist an adsorbate offset if it has changed. Returns True on change."""
        if set_index is None or set_index < 0:
            return False
        normalized = _normalize_offset(offset)
        offsets = self._get_adsorbate_offsets_dict()
        if offsets.get(set_index, (0.0, 0.0)) == normalized:
            return False
        offsets[set_index] = normalized
        setattr(self._controller, "adsorbate_visual_offsets_nm", offsets)
        self._logger.debug(
            "RealSpaceVisualizerState: adsorbate offset for set %s set to %s.",
            set_index,
            normalized,
        )
        return True


def _normalize_offset(offset: Tuple[float, float] | object) -> Tuple[float, float]:
    """Convert different offset representations to a strict float tuple."""
    if isinstance(offset, tuple) and len(offset) == 2:
        return (float(offset[0]), float(offset[1]))
    if isinstance(offset, list) and len(offset) == 2:
        return (float(offset[0]), float(offset[1]))
    if hasattr(offset, "__iter__"):
        values = list(offset)[:2]
        if len(values) == 2:
            return (float(values[0]), float(values[1]))
    return (0.0, 0.0)
