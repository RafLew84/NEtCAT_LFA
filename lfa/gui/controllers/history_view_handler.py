"""Helpers for keeping history-driven UI elements in sync."""

from __future__ import annotations

import logging
from typing import Optional

from PyQt6.QtCore import Qt

try:
    import pyqtgraph as pg
except Exception:  # pragma: no cover
    pg = None  # type: ignore

from ...core.constants import (
    SPOT_SELECTION_ADSORBATE,
    SPOT_SELECTION_SUBSTRATE,
    SPOT_SELECTION_SUPERSTRUCTURE,
)
from ...core.history import HistoryNode

logger = logging.getLogger(__name__)


class HistoryViewHandler:
    """Encapsulates history selection/display wiring for the main window."""

    def __init__(
        self,
        *,
        window,
        history_manager,
        app_controller,
        metadata_widget,
        fft_analysis_panel,
        visualization_manager,
    ) -> None:
        self._window = window
        self._history = history_manager
        self._controller = app_controller
        self._metadata = metadata_widget
        self._fft_panel = fft_analysis_panel
        self._visualization = visualization_manager

    # ------------------------------------------------------------------ selection ---------------------------------------------------------
    def handle_list_selection(self, current_item) -> None:
        """React to QListWidget selection changes."""
        if current_item is None or not self._history:
            return
        node_id = current_item.data(Qt.ItemDataRole.UserRole)
        if node_id != self._history.current_node_id:
            logger.info("HistoryViewHandler: list selection -> history node %s", node_id)
            self._history.set_current_node_by_id(node_id, emit_signal=True)

    # ------------------------------------------------------------------ node updates ------------------------------------------------------
    def on_current_node_changed(self, current_node: Optional[HistoryNode]) -> None:
        """Update UI components when HistoryManager switches nodes."""
        if self._controller:
            if current_node and current_node.data_type == "FFT" and current_node.image_data is not None:
                self._controller.current_fft_data_shape = current_node.image_data.shape
            else:
                self._controller.current_fft_data_shape = None

        self.display_image_data()

        if self._metadata:
            self._metadata.update_metadata(current_node, self._history)

        if (
            getattr(self._window, "fft_analysis_dock", None)
            and self._window.fft_analysis_dock.isVisible()
            and self._fft_panel
        ):
            self._refresh_fft_panel()

    def _refresh_fft_panel(self) -> None:
        controller = self._controller
        if not controller or not self._fft_panel:
            return

        self._fft_panel.update_transform_results_display(
            controller.substrate_transform_analysis_m2i
        )
        self._fft_panel.update_substrate_real_space_display(
            controller.substrate_real_space_results
        )

        current_ads_idx = controller.current_adsorbate_set_index
        expected_type = controller.adsorbate_expected_lattice_types.get(
            current_ads_idx, None
        )
        if expected_type is not None:
            self._fft_panel.set_expected_adsorbate_type(expected_type)

        ads_params = controller.adsorbate_real_space_results.get(current_ads_idx)
        self._fft_panel.update_adsorbate_real_space_display(ads_params)

        self.update_selected_spots_display()

    # ------------------------------------------------------------------ display helpers ---------------------------------------------------
    def display_image_data(self) -> None:
        """Push current node data into the visualization manager."""
        if not self._visualization:
            logger.error("HistoryViewHandler: VisualizationManager not available.")
            if getattr(self._window, "image_view", None):
                self._window.image_view.clear()  # pragma: no cover
            return
        if not self._history:
            logger.error("HistoryViewHandler: HistoryManager not available.")
            if getattr(self._window, "image_view", None):
                self._window.image_view.clear()
            return

        current_node = self._history.get_current_node()
        controller = self._controller
        show_ideal_lattice = False
        selected_substrate = "None"
        panel_custom_text = ""

        substrate_spots_to_draw = controller.displayable_fitted_substrate_spots_on_fft
        corrected_adsorbate_sets_ideal_sys = controller.corrected_adsorbate_spot_sets

        show_substrate_markers = controller.show_substrate_spots_markers
        show_adsorbate_markers = controller.show_adsorbate_spots_markers

        if self._fft_panel:
            show_ideal_lattice = self._fft_panel.is_show_ideal_lattice_checked()
            selected_substrate = self._fft_panel.get_current_substrate()
            panel_custom_text = self._fft_panel.custom_option_text

        self._visualization.update_view(
            current_node,
            show_ideal_lattice,
            selected_substrate,
            controller.custom_lattice_info,
            panel_custom_text,
            substrate_spots_to_draw,
            show_substrate_markers,
            corrected_adsorbate_sets_ideal_sys,
            show_adsorbate_markers,
        )

        if current_node and current_node.data_type == "FFT":
            substrate_pairs = controller.substrate_spot_pairs if show_substrate_markers else []
            self._visualization.update_substrate_spot_pairs(substrate_pairs)

            if show_adsorbate_markers:
                for set_index, pair_list in controller.adsorbate_spot_pairs.items():
                    self._visualization.update_adsorbate_spot_pairs(set_index, pair_list)
            else:
                self._visualization.clear_adsorbate_layers()
        else:
            self._visualization.update_substrate_spot_pairs([])
            self._visualization.clear_adsorbate_layers()

        if controller:
            self._visualization.update_superstructure_overlay(
                controller.superstructure_periodicity_results
            )

    def update_selected_spots_display(self) -> None:
        """Refresh FFT analysis panel text describing selected spots."""
        if not self._fft_panel or not hasattr(self._fft_panel, "selected_spots_display"):
            return

        text_output = []
        current_selection_status = ""
        controller = self._controller
        spot_selection_mode = controller.spot_selection_mode
        current_adsorbate_set_idx = controller.current_adsorbate_set_index
        substrate_spots = controller.substrate_spots
        adsorbate_spot_sets = controller.adsorbate_spot_sets

        if spot_selection_mode == SPOT_SELECTION_SUBSTRATE:
            current_selection_status = "Selecting: Substrate Spots"
            text_output.append("Substrate Spots:")
            if substrate_spots:
                for i, (kx, ky) in enumerate(substrate_spots):
                    text_output.append(f"  S{i+1}: (kx={kx}, ky={ky})")
            else:
                text_output.append("  None selected.")
        elif spot_selection_mode == SPOT_SELECTION_ADSORBATE:
            set_name = self._fft_panel.adsorbate_set_combo.itemText(current_adsorbate_set_idx) if current_adsorbate_set_idx < self._fft_panel.adsorbate_set_combo.count() else f"Set {current_adsorbate_set_idx + 1}"
            current_selection_status = f"Selecting: Adsorbate {set_name}"
            text_output.append(f"Adsorbate {set_name}:")

            if 0 <= current_adsorbate_set_idx < len(adsorbate_spot_sets):
                current_points_to_display = adsorbate_spot_sets[current_adsorbate_set_idx]
                if current_points_to_display:
                    for i, (kx, ky) in enumerate(current_points_to_display):
                        text_output.append(f"  A{i+1}: (kx={kx}, ky={ky})")
                else:
                    text_output.append("  No spots selected for this set.")
            else:
                text_output.append("  Invalid adsorbate set selected.")
        elif spot_selection_mode == SPOT_SELECTION_SUPERSTRUCTURE:
            current_selection_status = "Viewing: Superstructure Peaks"
            text_output.append("Superstructure Peaks:")
            results = controller.superstructure_periodicity_results or {}
            main_peak = results.get("main_peak_raw_px")
            satellite_peak = results.get("satellite_peak_raw_px")
            if main_peak:
                text_output.append(f"  Main: (kx={main_peak[0]}, ky={main_peak[1]})")
            if satellite_peak:
                text_output.append(f"  Satellite: (kx={satellite_peak[0]}, ky={satellite_peak[1]})")
            if not main_peak and not satellite_peak:
                text_output.append("  None selected.")

        if hasattr(self._fft_panel, "current_selection_label") and self._fft_panel.current_selection_label:
            self._fft_panel.current_selection_label.setText(current_selection_status)
        self._fft_panel.selected_spots_display.setPlainText("\n".join(text_output))

    # ------------------------------------------------------------------ misc ---------------------------------------------------------------
    def clear_spot_markers(self, view_box) -> None:
        """Remove markers from the provided view box."""
        if pg is None or not view_box:
            return
        logger.debug("HistoryViewHandler: request to clear spot markers (no-op placeholder).")
