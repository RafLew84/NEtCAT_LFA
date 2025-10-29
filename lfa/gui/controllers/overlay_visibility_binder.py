from __future__ import annotations

import logging
from typing import Dict, Optional

logger = logging.getLogger(__name__)


class OverlayVisibilityBinder:
    """Synchronises overlay visibility between panel widgets, controller, and visualiser."""

    def __init__(self, app_controller, fft_analysis_panel, visualization_manager) -> None:
        self._controller = app_controller
        self._panel = fft_analysis_panel
        self._visualization = visualization_manager

        self._adsorbate_raw_by_set: Dict[int, bool] = {}
        self._adsorbate_transformed_by_set: Dict[int, bool] = {}

        self.ensure_adsorbate_defaults()
        self._connect_signals()

    # ------------------------------------------------------------------ Public API
    def ensure_adsorbate_defaults(self) -> int:
        total_sets = 1
        if self._controller and getattr(self._controller, "adsorbate_spot_sets", None) is not None:
            total_sets = max(len(self._controller.adsorbate_spot_sets), 1)

        raw_defaults = {}
        trans_defaults = {}
        for idx in range(total_sets):
            raw_defaults[idx] = self._adsorbate_raw_by_set.get(idx, True)
            trans_defaults[idx] = self._adsorbate_transformed_by_set.get(idx, True)

        self._adsorbate_raw_by_set = raw_defaults
        self._adsorbate_transformed_by_set = trans_defaults
        return total_sets

    def apply_panel_state_to_controller(self) -> None:
        if not (self._panel and self._controller):
            return
        self._controller.set_substrate_raw_visibility(self._panel.is_show_substrate_raw_checked())
        self._controller.set_substrate_transformed_visibility(self._panel.is_show_substrate_transformed_checked())
        self._controller.set_adsorbate_raw_visibility(self._panel.is_show_adsorbate_raw_checked())
        self._controller.set_adsorbate_transformed_visibility(self._panel.is_show_adsorbate_transformed_checked())

    def handle_current_adsorbate_set_changed(self, set_index: int) -> None:
        self.ensure_adsorbate_defaults()
        self.apply_visibility_for_set(set_index)
        self.sync_checkboxes_for_set(set_index)

    def apply_visibility_for_set(self, set_index: int) -> None:
        if self._visualization:
            raw_visible = self._adsorbate_raw_by_set.get(set_index, True)
            trans_visible = self._adsorbate_transformed_by_set.get(set_index, True)
            self._visualization.set_adsorbate_raw_visible_for_set(set_index, raw_visible)
            self._visualization.set_adsorbate_transformed_visible_for_set(set_index, trans_visible)

    def sync_checkboxes_for_set(self, set_index: int) -> None:
        if not self._panel:
            return
        raw_checked = self._adsorbate_raw_by_set.get(set_index, True)
        transformed_checked = self._adsorbate_transformed_by_set.get(set_index, True)
        self._panel.set_show_adsorbate_raw_checked(raw_checked)
        self._panel.set_show_adsorbate_transformed_checked(transformed_checked)

    def refresh_all_sets_in_visualization(self) -> None:
        total = self.ensure_adsorbate_defaults()
        for idx in range(total):
            self.apply_visibility_for_set(idx)

    # ------------------------------------------------------------------ Internal signal wiring
    def _connect_signals(self) -> None:
        if self._panel:
            self._panel.substrate_raw_visibility_changed.connect(self._on_panel_substrate_raw_changed)
            self._panel.substrate_transformed_visibility_changed.connect(self._on_panel_substrate_transformed_changed)
            self._panel.adsorbate_raw_visibility_changed.connect(self._on_panel_adsorbate_raw_changed)
            self._panel.adsorbate_transformed_visibility_changed.connect(self._on_panel_adsorbate_transformed_changed)

        if self._controller:
            self._controller.substrate_raw_visibility_updated.connect(self._on_controller_substrate_raw_updated)
            self._controller.substrate_transformed_visibility_updated.connect(self._on_controller_substrate_transformed_updated)
            self._controller.adsorbate_raw_visibility_updated.connect(self._on_controller_adsorbate_raw_updated)
            self._controller.adsorbate_transformed_visibility_updated.connect(self._on_controller_adsorbate_transformed_updated)
            self._controller.adsorbate_set_updated.connect(self._on_adsorbate_set_updated)
            self._controller.adsorbate_sets_structure_changed.connect(self._on_adsorbate_sets_structure_changed)

    # ------------------------------------------------------------------ Panel -> Controller callbacks
    def _on_panel_substrate_raw_changed(self, is_visible: bool) -> None:
        if self._controller:
            self._controller.set_substrate_raw_visibility(is_visible)

    def _on_panel_substrate_transformed_changed(self, is_visible: bool) -> None:
        if self._controller:
            self._controller.set_substrate_transformed_visibility(is_visible)

    def _on_panel_adsorbate_raw_changed(self, is_visible: bool) -> None:
        if not self._controller:
            return
        current_set = self._controller.current_adsorbate_set_index
        if current_set < 0:
            return
        self._adsorbate_raw_by_set[current_set] = is_visible
        if self._visualization:
            self._visualization.set_adsorbate_raw_visible_for_set(current_set, is_visible)

    def _on_panel_adsorbate_transformed_changed(self, is_visible: bool) -> None:
        if not self._controller:
            return
        current_set = self._controller.current_adsorbate_set_index
        if current_set < 0:
            return
        self._adsorbate_transformed_by_set[current_set] = is_visible
        if self._visualization:
            self._visualization.set_adsorbate_transformed_visible_for_set(current_set, is_visible)

    # ------------------------------------------------------------------ Controller -> Panel/Viz callbacks
    def _on_controller_substrate_raw_updated(self, is_visible: bool) -> None:
        if self._visualization:
            self._visualization.set_substrate_raw_visible(is_visible)
        if self._panel:
            self._panel.set_show_substrate_raw_checked(is_visible)

    def _on_controller_substrate_transformed_updated(self, is_visible: bool) -> None:
        if self._visualization:
            self._visualization.set_substrate_transformed_visible(is_visible)
        if self._panel:
            self._panel.set_show_substrate_transformed_checked(is_visible)

    def _on_controller_adsorbate_raw_updated(self, is_visible: bool) -> None:
        total_sets = self.ensure_adsorbate_defaults()
        for idx in range(total_sets):
            self._adsorbate_raw_by_set[idx] = is_visible
        if self._visualization:
            self._visualization.set_adsorbate_raw_visible(is_visible)
        if self._panel:
            self._panel.set_show_adsorbate_raw_checked(is_visible)

    def _on_controller_adsorbate_transformed_updated(self, is_visible: bool) -> None:
        total_sets = self.ensure_adsorbate_defaults()
        for idx in range(total_sets):
            self._adsorbate_transformed_by_set[idx] = is_visible
        if self._visualization:
            self._visualization.set_adsorbate_transformed_visible(is_visible)
        if self._panel:
            self._panel.set_show_adsorbate_transformed_checked(is_visible)

    def _on_adsorbate_set_updated(self, set_index: int) -> None:
        self.ensure_adsorbate_defaults()
        self.apply_visibility_for_set(set_index)

    def _on_adsorbate_sets_structure_changed(self) -> None:
        self.ensure_adsorbate_defaults()
        self.refresh_all_sets_in_visualization()
        current_index = -1
        if self._controller:
            current_index = self._controller.current_adsorbate_set_index
        if current_index >= 0:
            self.sync_checkboxes_for_set(current_index)
