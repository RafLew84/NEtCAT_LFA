from __future__ import annotations

import logging
from typing import Dict, Optional

from PyQt6.QtGui import QAction

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

        load_metadata_action = self._actions.get("load_metadata")
        if load_metadata_action:
            load_metadata_action.setEnabled(
                controller.can_load_metadata(current_node) if controller else False
            )

        self._set_preprocessing_actions_enabled(current_node)
        self._set_analysis_actions_enabled(current_node)

        lattice_analysis_enabled = self._availability.get("lattice_analysis", False)
        panel_state = (
            controller.evaluate_fft_panel_state(current_node, lattice_analysis_enabled)
            if controller
            else None
        )

        if self._dock and panel_state:
            self._dock.setVisible(panel_state.fft_active)

        if panel and panel_state:
            self._update_fft_panel_state(panel, panel_state, current_node)

        logger.debug(
            "UIStateBinder.refresh -> node=%s fft_active=%s",
            getattr(current_node, "node_id", None),
            panel_state.fft_active if panel_state else False,
        )

    # ------------------------------------------------------------------ Helpers
    def _set_preprocessing_actions_enabled(self, current_node) -> None:
        dialogs_available = self._availability.get("preprocessing_dialogs", False)
        has_active_node = current_node is not None
        for key in ("gaussian_blur", "gaussian_sharpen", "plane_level", "median_filter", "nlmeans", "bm3d"):
            self._set_action_enabled(key, has_active_node and dialogs_available)
        self._set_action_enabled(
            "fft",
            dialogs_available and self._app_controller.can_calculate_fft(current_node),
        )

    def _set_analysis_actions_enabled(self, current_node) -> None:
        spot_dialogs = self._availability.get("spot_dialogs", False)
        superstructure_available = self._availability.get("superstructure_dialog", False)
        stm_transform_available = self._availability.get("stm_transform_dialog", False)

        can_open_spot_dialogs = spot_dialogs and self._app_controller.can_select_spots(current_node)
        self._set_action_enabled("select_substrate_spots", can_open_spot_dialogs)
        self._set_action_enabled("select_adsorbate_spots", can_open_spot_dialogs)

        self._set_action_enabled(
            "superstructure_periodicity",
            superstructure_available and self._app_controller.can_analyze_superstructure(current_node),
        )
        self._set_action_enabled(
            "stm_transform",
            stm_transform_available and self._app_controller.can_open_stm_transform(current_node),
        )

        self._set_action_enabled(
            "visualize_real_space",
            self._app_controller.can_visualize_real_space(current_node),
        )
        self._set_action_enabled(
            "real_space_reconstruction",
            self._app_controller.can_open_real_space_reconstruction(current_node),
        )

    def _update_fft_panel_state(self, panel, panel_state, current_node) -> None:
        controller = self._app_controller
        if not controller or not panel or panel_state is None:
            return

        if not panel_state.fft_active:
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

        panel.set_edit_substrate_spots_button_enabled(panel_state.edit_substrate_enabled)
        panel.set_edit_adsorbate_spots_button_enabled(panel_state.edit_adsorbate_enabled)
        panel.set_reselect_adsorbate_set_button_enabled(panel_state.reselect_adsorbate_enabled)
        panel.set_clear_all_adsorbate_sets_button_enabled(panel_state.clear_all_adsorbate_sets_enabled)
        panel.set_calculate_substrate_rs_button_enabled(panel_state.can_calculate_substrate_rs)
        panel.set_calculate_adsorbate_rs_button_enabled(panel_state.can_calculate_adsorbate_rs)

    def _set_action_enabled(self, key: str, enabled: bool) -> None:
        action = self._actions.get(key)
        if action is not None:
            action.setEnabled(bool(enabled))




