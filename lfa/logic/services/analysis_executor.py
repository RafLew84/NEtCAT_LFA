"""Coordinates heavy analysis actions from AppController."""

from __future__ import annotations

from typing import Optional

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
