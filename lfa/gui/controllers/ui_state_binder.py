from __future__ import annotations

import logging
from typing import Dict, Optional

from PyQt6.QtGui import QAction

from ...core.constants import (
    LATTICE_TYPE_CUSTOM,
    LATTICE_TYPE_HEXAGONAL,
    LATTICE_TYPE_SQUARE,
)

logger = logging.getLogger(__name__)


class UIStateBinder:
    """Centralises enable/disable logic for menu actions and panel buttons."""

    def __init__(
        self,
        app_controller,
        history_manager,
        fft_analysis_panel,
        fft_analysis_dock,
        actions: Dict[str, Optional[QAction]],
        availability: Dict[str, bool],
    ) -> None:
        self._app_controller = app_controller
        self._history_manager = history_manager
        self._panel = fft_analysis_panel
        self._dock = fft_analysis_dock
        self._actions = actions
        self._availability = availability

    # ------------------------------------------------------------------ Public API
    def refresh(self) -> None:
        """Recompute UI enabled state based on current controller + history context."""
        controller = self._app_controller
        history = self._history_manager
        panel = self._panel

        current_node = history.get_current_node() if history else None
        has_active_node = current_node is not None
        is_stm_data_active = bool(has_active_node and getattr(current_node, "data_type", "") == "STM")
        is_fft_data_active = bool(has_active_node and getattr(current_node, "data_type", "") == "FFT")

        load_metadata_action = self._actions.get("load_metadata")
        if load_metadata_action:
            can_load_metadata = False
            if current_node and history:
                root_node = history.get_root_node_for_node(current_node.node_id)
                if root_node and "raw_header" not in getattr(root_node, "parameters", {}):
                    can_load_metadata = True
            load_metadata_action.setEnabled(can_load_metadata)

        self._set_preprocessing_actions_enabled(
            enabled=has_active_node,
            fft_ready=is_stm_data_active,
        )
        self._set_analysis_actions_enabled(is_fft_data_active, is_stm_data_active)

        if self._dock:
            self._dock.setVisible(is_fft_data_active)

        if panel:
            self._update_fft_panel_state(panel, current_node, is_fft_data_active)

        logger.debug(
            "UIStateBinder.refresh -> node=%s stm=%s fft=%s",
            getattr(current_node, "node_id", None),
            is_stm_data_active,
            is_fft_data_active,
        )

    # ------------------------------------------------------------------ Helpers
    def _set_preprocessing_actions_enabled(self, enabled: bool, fft_ready: bool) -> None:
        dialogs_available = self._availability.get("preprocessing_dialogs", False)
        for key in ("gaussian_blur", "gaussian_sharpen", "plane_level", "median_filter", "nlmeans", "bm3d"):
            self._set_action_enabled(key, enabled and dialogs_available)
        self._set_action_enabled("fft", fft_ready and dialogs_available)

    def _set_analysis_actions_enabled(self, fft_active: bool, stm_active: bool) -> None:
        spot_dialogs = self._availability.get("spot_dialogs", False)
        superstructure_available = self._availability.get("superstructure_dialog", False)
        stm_transform_available = self._availability.get("stm_transform_dialog", False)

        can_open_spot_dialogs = fft_active and spot_dialogs
        self._set_action_enabled("select_substrate_spots", can_open_spot_dialogs)
        self._set_action_enabled("select_adsorbate_spots", can_open_spot_dialogs)

        can_analyse_superstructure = False
        if fft_active and superstructure_available and self._app_controller:
            can_analyse_superstructure = self._app_controller.substrate_F_m2i is not None

        self._set_action_enabled("superstructure_periodicity", can_analyse_superstructure)
        self._set_action_enabled("stm_transform", stm_active and stm_transform_available)

        # Current UX keeps these actions always enabled when present.
        self._set_action_enabled("visualize_real_space", True)
        self._set_action_enabled("real_space_reconstruction", True)

    def _update_fft_panel_state(self, panel, current_node, fft_active: bool) -> None:
        controller = self._app_controller
        if not controller or not panel:
            return

        if not fft_active:
            panel.set_edit_substrate_spots_button_enabled(False)
            panel.set_edit_adsorbate_spots_button_enabled(False)
            panel.set_reselect_adsorbate_set_button_enabled(False)
            panel.set_clear_all_adsorbate_sets_button_enabled(False)
            panel.set_calculate_substrate_rs_button_enabled(False)
            panel.set_calculate_adsorbate_rs_button_enabled(False)
            panel.update_substrate_real_space_display(None)
            panel.update_adsorbate_real_space_display(None)
            panel.update_transform_results_display(None)
            return

        panel.set_edit_substrate_spots_button_enabled(True)
        panel.set_edit_adsorbate_spots_button_enabled(True)

        current_ads_idx = controller.current_adsorbate_set_index
        ads_sets = controller.adsorbate_spot_sets

        can_clear_current_set = 0 <= current_ads_idx < len(ads_sets) and bool(ads_sets[current_ads_idx])
        panel.set_reselect_adsorbate_set_button_enabled(can_clear_current_set)

        can_clear_all_sets = any(ads_sets) if ads_sets else False
        panel.set_clear_all_adsorbate_sets_button_enabled(can_clear_all_sets)

        can_calc_substrate_rs = self._can_compute_substrate_real_space(controller)
        can_calc_adsorbate_rs = self._can_compute_adsorbate_real_space(controller, current_node)

        panel.set_calculate_substrate_rs_button_enabled(can_calc_substrate_rs)
        panel.set_calculate_adsorbate_rs_button_enabled(can_calc_adsorbate_rs)

    def _can_compute_substrate_real_space(self, controller) -> bool:
        if not controller or not self._availability.get("lattice_analysis", False):
            return False

        has_definition = False
        if controller.substrate_lattice_type == LATTICE_TYPE_CUSTOM:
            has_definition = isinstance(controller.custom_lattice_info, dict)
        else:
            has_definition = bool(
                controller.substrate_lattice_type
                and controller.substrate_a_surf
                and controller.substrate_a_surf > 0
            )

        if not (
            has_definition
            and controller.reference_ideal_substrate_spots_px
            and controller.current_fft_data_shape
        ):
            return False

        expected_count = 0
        if controller.substrate_lattice_type == LATTICE_TYPE_HEXAGONAL:
            expected_count = 6
        elif controller.substrate_lattice_type == LATTICE_TYPE_SQUARE:
            expected_count = 4

        if expected_count > 0:
            return len(controller.reference_ideal_substrate_spots_px) == expected_count
        return len(controller.reference_ideal_substrate_spots_px) >= 2

    def _can_compute_adsorbate_real_space(self, controller, current_node) -> bool:
        if not controller or not self._availability.get("lattice_analysis", False):
            return False

        idx = controller.current_adsorbate_set_index
        if not (
            0 <= idx < len(controller.corrected_adsorbate_spot_sets)
            and controller.corrected_adsorbate_spot_sets[idx]
            and controller.current_fft_data_shape
        ):
            return False

        if not current_node or not self._history_manager:
            return False

        root_node = self._history_manager.get_root_node_for_node(current_node.node_id)
        if not root_node:
            return False

        params = getattr(root_node, "parameters", {}) or {}
        if not (params.get("size_nm_x") and params.get("size_nm_y")):
            return False

        return len(controller.corrected_adsorbate_spot_sets[idx]) >= 2

    def _set_action_enabled(self, key: str, enabled: bool) -> None:
        action = self._actions.get(key)
        if action is not None:
            action.setEnabled(bool(enabled))
