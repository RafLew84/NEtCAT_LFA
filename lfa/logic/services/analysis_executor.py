"""Coordinates heavy analysis actions from AppController."""

from __future__ import annotations

from typing import Optional, Sequence

import numpy as np

from ...analysis.drift_correction import (
    analyze_affine_transform,
    apply_affine_transform,
    match_and_fit_transform,
)
from ...analysis.lattice import get_nearest_reciprocal_points
from ...core.constants import (
    LATTICE_TYPE_CUSTOM,
    LATTICE_TYPE_HEXAGONAL,
    LATTICE_TYPE_SQUARE,
)
from ...core.history import HistoryNode


class AnalysisExecutor:
    def __init__(self, controller) -> None:
        self._controller = controller

    # Legacy can_* helpers -------------------------------------------------
    def can_load_metadata(self, history_node: Optional[HistoryNode]) -> bool:
        history_manager = self._controller.history_manager
        if not history_node or not history_manager:
            return False
        root_node = history_manager.get_root_node_for_node(history_node.node_id)
        if not root_node:
            return False
        parameters = getattr(root_node, "parameters", {}) or {}
        return "raw_header" not in parameters

    def can_calculate_fft(self, history_node: Optional[HistoryNode]) -> bool:
        return bool(
            history_node
            and history_node.data_type == "STM"
            and getattr(history_node, "image_data", None) is not None
        )

    def can_select_spots(self, history_node: Optional[HistoryNode]) -> bool:
        return bool(history_node and history_node.data_type == "FFT")

    def can_analyze_superstructure(self, history_node: Optional[HistoryNode]) -> bool:
        return bool(
            history_node
            and history_node.data_type == "FFT"
            and self._controller.substrate_F_m2i is not None
        )

    def can_visualize_real_space(self, history_node: Optional[HistoryNode]) -> bool:
        return bool(
            history_node
            and history_node.data_type == "FFT"
            and self._controller.substrate_real_space_results
        )

    def can_open_real_space_reconstruction(self, history_node: Optional[HistoryNode]) -> bool:
        return bool(
            history_node
            and history_node.data_type == "FFT"
            and getattr(history_node, "complex_fft_data", None) is not None
        )

    def can_open_stm_transform(self, history_node: Optional[HistoryNode]) -> bool:
        return bool(
            history_node
            and history_node.data_type == "STM"
            and self._controller.substrate_F_m2i is not None
        )

    # FFT real-space enablement helpers ----------------------------------
    def can_calculate_substrate_real_space(self, history_node: Optional[HistoryNode]) -> bool:
        ctrl = self._controller
        if not (history_node and history_node.data_type == "FFT"):
            return False

        fitted_spots: Sequence[Sequence[float]] = getattr(
            ctrl, "displayable_fitted_substrate_spots_on_fft", []
        )
        if not fitted_spots:
            return False

        lattice_type = getattr(ctrl, "substrate_lattice_type", None)
        if lattice_type == LATTICE_TYPE_CUSTOM:
            if not getattr(ctrl, "custom_lattice_info", None):
                return False
        elif lattice_type in (LATTICE_TYPE_HEXAGONAL, LATTICE_TYPE_SQUARE):
            a_surf = getattr(ctrl, "substrate_a_surf", None)
            if not (a_surf and a_surf > 0):
                return False
        else:
            # Unknown lattice type -> cannot determine expectations
            return False

        fft_shape = self._get_fft_shape(history_node)
        if fft_shape is None:
            return False

        if not self._has_valid_calibration(history_node):
            return False

        expected_count = 0
        if lattice_type == LATTICE_TYPE_HEXAGONAL:
            expected_count = 6
        elif lattice_type == LATTICE_TYPE_SQUARE:
            expected_count = 4

        fitted_count = len(fitted_spots)
        if expected_count > 0 and fitted_count != expected_count:
            return False
        if expected_count == 0 and fitted_count < 2:
            return False

        return True

    def can_calculate_adsorbate_real_space(self, history_node: Optional[HistoryNode]) -> bool:
        ctrl = self._controller
        if not (history_node and history_node.data_type == "FFT"):
            return False

        corrected_sets = getattr(ctrl, "corrected_adsorbate_spot_sets", [])
        if not corrected_sets:
            return False

        set_index = getattr(ctrl, "current_adsorbate_set_index", 0)
        if not (0 <= set_index < len(corrected_sets)):
            return False

        corrected_spots = corrected_sets[set_index]
        if len(corrected_spots) < 2:
            return False

        if self._get_fft_shape(history_node) is None:
            return False

        if not self._has_valid_calibration(history_node):
            return False

        return True

    # ------------------------------------------------------------------
    def _get_fft_shape(self, history_node: Optional[HistoryNode]):
        if (
            history_node
            and history_node.data_type == "FFT"
            and getattr(history_node, "image_data", None) is not None
        ):
            return history_node.image_data.shape
        return getattr(self._controller, "current_fft_data_shape", None)

    def _has_valid_calibration(self, history_node: Optional[HistoryNode]) -> bool:
        history_manager = getattr(self._controller, "history_manager", None)
        if not (history_manager and history_node):
            return False

        root_node = history_manager.get_root_node_for_node(history_node.node_id)
        parameters = getattr(root_node, "parameters", {}) if root_node else {}
        if not parameters:
            return False

        Lx_nm = parameters.get("size_nm_x")
        Ly_nm = parameters.get("size_nm_y")
        return bool(Lx_nm and Ly_nm and Lx_nm > 0 and Ly_nm > 0)
