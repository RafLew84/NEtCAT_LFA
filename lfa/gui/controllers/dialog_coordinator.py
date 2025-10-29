from __future__ import annotations

import logging
from typing import Optional

from PyQt6.QtWidgets import QDialog, QMessageBox

from ...core.constants import (
    ADSORBATE_LATTICE_TYPE_UNKNOWN,
    LATTICE_TYPE_CUSTOM,
    LATTICE_TYPE_HEXAGONAL,
    LATTICE_TYPE_SQUARE,
    PREDEFINED_SUBSTRATE_CUSTOM,
)

logger = logging.getLogger(__name__)


class DialogCoordinator:
    """Encapsulates dialog launching logic for the main window."""

    def __init__(
        self,
        main_window,
        app_controller,
        history_manager,
        visualization_manager,
        *,
        fft_dialog_class=None,
        substrate_dialog_class=None,
        adsorbate_dialog_class=None,
        superstructure_dialog_class=None,
        real_space_visualizer_class=None,
        real_space_reconstruction_class=None,
        stm_transform_dialog_class=None,
    ) -> None:
        self._window = main_window
        self._controller = app_controller
        self._history = history_manager
        self._visualization = visualization_manager

        self._fft_dialog_class = fft_dialog_class
        self._substrate_dialog_class = substrate_dialog_class
        self._adsorbate_dialog_class = adsorbate_dialog_class
        self._superstructure_dialog_class = superstructure_dialog_class
        self._real_space_visualizer_class = real_space_visualizer_class
        self._real_space_reconstruction_class = real_space_reconstruction_class
        self._stm_transform_dialog_class = stm_transform_dialog_class

        self._visualizer_dialog_instance: Optional[QDialog] = None

    # ------------------------------------------------------------------ FFT ------------------------------------------------------------------
    def open_fft_dialog(self) -> None:
        if not self._controller:
            return
        current_node_info = self._controller.get_current_node_info_for_dialogs()
        if not current_node_info:
            QMessageBox.warning(self._window, "No Image", "No data loaded or selected to calculate FFT.")
            return

        parent_id, parent_data_type, image_data_copy, source_image_id, source_label = current_node_info
        if parent_data_type != "STM":
            QMessageBox.warning(self._window, "Invalid Data Type", "FFT can only be calculated from STM data (not from an existing FFT).")
            return
        if not self._fft_dialog_class:
            QMessageBox.critical(self._window, "Error", "FFTDialog is not available.")
            return

        dialog = self._fft_dialog_class(image_data_copy, parent=self._window, source_label=source_label)
        dialog.source_image_id = source_image_id
        dialog.source_image_label = source_label
        results_added = {"count": 0}

        def _handle_fft_applied(params, processed_fft_data, complex_fft_data, roi_slice):
            self._controller.calculate_fft_operation(
                parent_node_id=parent_id,
                processed_fft_data=processed_fft_data,
                complex_fft_data=complex_fft_data,
                params=params,
                source_roi_slice=roi_slice,
            )
            results_added["count"] += 1
            self._window.statusBar().showMessage("FFT result added to history.", 3000)

        dialog.fftApplied.connect(_handle_fft_applied)
        dialog.exec()

        message = "FFT dialog closed without applying results." if results_added["count"] == 0 else "FFT dialog closed after applying results."
        self._window.statusBar().showMessage(message, 3000)

    # ------------------------------------------------------------------ Spot selection -------------------------------------------------------
    def open_substrate_spot_selection_dialog(self) -> None:
        if not self._controller or not self._history:
            return

        current_node_info = self._controller.get_current_node_info_for_dialogs()
        if not current_node_info:
            QMessageBox.warning(self._window, "Incorrect Data Type", "Substrate spots can only be selected on an FFT image.")
            return
        node_id, data_type, fft_image_data_copy, source_image_id, source_label = current_node_info
        if data_type != "FFT":
            QMessageBox.warning(self._window, "Incorrect Data Type", "Substrate spots can only be selected on an FFT image.")
            return
        if not self._substrate_dialog_class:
            QMessageBox.critical(self._window, "Dialog Error", "SubstrateSpotSelectionDialog is not available.")
            return

        controller = self._controller
        dialog = self._substrate_dialog_class(
            fft_image_data=fft_image_data_copy,
            history_manager=self._history,
            current_fft_node_id=node_id,
            current_spots=controller.user_selected_substrate_spots,
            initial_lattice_type=controller.substrate_lattice_type or LATTICE_TYPE_HEXAGONAL,
            initial_selected_substrate_name=controller.substrate_definition_name,
            initial_custom_a_surf=controller.substrate_a_surf
            if controller.substrate_definition_name == PREDEFINED_SUBSTRATE_CUSTOM
            and controller.substrate_lattice_type != LATTICE_TYPE_CUSTOM
            else None,
            initial_custom_definition=dict(controller.custom_lattice_info)
            if controller.substrate_definition_name == PREDEFINED_SUBSTRATE_CUSTOM
            and controller.substrate_lattice_type == LATTICE_TYPE_CUSTOM
            and isinstance(controller.custom_lattice_info, dict)
            else None,
            default_refinement_method=controller.spot_refinement_method,
            default_refinement_roi_size=controller.refinement_roi_size,
            initial_transform_F=controller.substrate_F_m2i,
            initial_transform_t=controller.substrate_t_m2i,
            initial_fitted_spots=controller.displayable_fitted_substrate_spots_on_fft,
            parent=self._window,
        )
        dialog.source_image_id = source_image_id
        dialog.source_image_label = source_label
        if source_label and hasattr(dialog, "setWindowTitle") and source_label not in dialog.windowTitle():
            dialog.setWindowTitle(f"{dialog.windowTitle()} [{source_label}]")

        if dialog.exec() == QDialog.DialogCode.Accepted:
            results = dialog.get_dialog_results()
            logger.info("Substrate spots dialog accepted. Results: %s", results)
            controller.update_substrate_analysis_results(results)
        else:
            logger.info("Substrate spots selection cancelled.")
            self._window.statusBar().showMessage("Substrate spots selection cancelled.", 3000)

    def open_adsorbate_spot_selection_dialog(self) -> None:
        if not self._controller:
            return

        current_node_info = self._controller.get_current_node_info_for_dialogs()
        if not current_node_info:
            QMessageBox.warning(self._window, "Incorrect Data Type", "Adsorbate spots can only be selected on an FFT image.")
            logger.warning("Attempted to open adsorbate spot selection on non-FFT data.")
            return
        _, data_type, fft_image_data_copy, source_image_id, source_label = current_node_info
        if data_type != "FFT":
            QMessageBox.warning(self._window, "Incorrect Data Type", "Adsorbate spots can only be selected on an FFT image.")
            logger.warning("Attempted to open adsorbate spot selection on non-FFT data.")
            return
        if not self._adsorbate_dialog_class:
            QMessageBox.critical(self._window, "Dialog Error", "AdsorbateSpotSelectionDialog is not available. Please check application setup.")
            logger.error("AdsorbateSpotSelectionDialog class is not available.")
            return

        controller = self._controller
        current_set_idx = controller.current_adsorbate_set_index
        initial_expected_type = controller.adsorbate_expected_lattice_types.get(current_set_idx, ADSORBATE_LATTICE_TYPE_UNKNOWN)
        current_adsorbate_spots_for_set = []
        if 0 <= current_set_idx < len(controller.adsorbate_spot_sets):
            current_adsorbate_spots_for_set = list(controller.adsorbate_spot_sets[current_set_idx])
        elif not controller.adsorbate_spot_sets:
            controller.adsorbate_spot_sets.append([])
            current_set_idx = 0
        else:
            current_set_idx = max(min(current_set_idx, len(controller.adsorbate_spot_sets) - 1), 0)
            current_adsorbate_spots_for_set = list(controller.adsorbate_spot_sets[current_set_idx])

        dialog = self._adsorbate_dialog_class(
            fft_image_data=fft_image_data_copy,
            current_adsorbate_spots=current_adsorbate_spots_for_set,
            current_adsorbate_set_index=current_set_idx,
            expected_adsorbate_type=initial_expected_type,
            parent=self._window,
        )
        dialog.source_image_id = source_image_id
        dialog.source_image_label = source_label
        if source_label and hasattr(dialog, "setWindowTitle") and source_label not in dialog.windowTitle():
            dialog.setWindowTitle(f"{dialog.windowTitle()} [{source_label}]")

        if dialog.exec() == QDialog.DialogCode.Accepted:
            results = dialog.get_dialog_results()
            raw_spots = results.get("raw_spots", [])
            corrected_spots = results.get("corrected_spots", [])
            set_idx_from_dialog = results.get("adsorbate_set_index", current_set_idx)
            logger.info(
                "Adsorbate spots dialog (set %s) accepted. Raw: %s, Corrected: %s",
                set_idx_from_dialog + 1,
                len(raw_spots),
                len(corrected_spots),
            )
            controller.set_expected_adsorbate_lattice_type(
                set_idx_from_dialog,
                results.get("expected_type", ADSORBATE_LATTICE_TYPE_UNKNOWN),
            )
            controller.update_adsorbate_set_results(
                set_index=set_idx_from_dialog,
                raw_spots=raw_spots,
                corrected_spots_ideal_system=corrected_spots,
            )
            self._window.statusBar().showMessage(f"Adsorbate spots (Set {set_idx_from_dialog + 1}) updated.", 3000)
        else:
            logger.info("Adsorbate spots selection for set %s cancelled.", current_set_idx + 1)
            self._window.statusBar().showMessage(f"Adsorbate spots (Set {current_set_idx + 1}) selection cancelled.", 3000)

    # ------------------------------------------------------------------ Superstructure --------------------------------------------------------
    def open_superstructure_periodicity_dialog(self) -> None:
        if not self._controller or not self._history:
            return
        if not self._superstructure_dialog_class:
            QMessageBox.critical(self._window, "Dialog Error", "SuperstructurePeriodicityDialog is not available.")
            return

        current_node_info = self._controller.get_current_node_info_for_dialogs()
        if not current_node_info:
            QMessageBox.warning(self._window, "Incorrect Data", "Superstructure periodicity analysis requires an active FFT image.")
            return
        node_id, data_type, fft_image_data_copy, source_image_id, source_label = current_node_info
        if data_type != "FFT":
            QMessageBox.warning(self._window, "Incorrect Data", "Superstructure periodicity analysis requires an active FFT image.")
            return
        if not (self._controller.substrate_F_m2i is not None and self._controller.substrate_t_m2i is not None):
            QMessageBox.warning(
                self._window,
                "Substrate Transform Missing",
                "Superstructure periodicity analysis requires a calculated substrate transform.",
            )
            return

        dialog = self._superstructure_dialog_class(
            fft_image_data=fft_image_data_copy,
            history_manager=self._history,
            current_fft_node_id=node_id,
            app_controller=self._controller,
            parent=self._window,
        )
        dialog.source_image_id = source_image_id
        dialog.source_image_label = source_label
        if source_label and hasattr(dialog, "setWindowTitle") and source_label not in dialog.windowTitle():
            dialog.setWindowTitle(f"{dialog.windowTitle()} [{source_label}]")

        if dialog.exec() == QDialog.DialogCode.Accepted:
            results = dialog.get_results()
            if results:
                logger.info("Superstructure periodicity analysis accepted with results: %s", results)
                self._controller.update_superstructure_periodicity_results(results)
            else:
                logger.info("Superstructure periodicity dialog closed without valid results.")
        else:
            logger.info("Superstructure periodicity dialog cancelled.")

    # ------------------------------------------------------------------ Real-space visualisation ----------------------------------------------
    def open_real_space_fft_visualizer(self) -> None:
        if not self._controller or not self._history:
            return
        if not self._real_space_visualizer_class:
            QMessageBox.critical(self._window, "Error", "RealSpaceFFTVisualizerDialog is not available.")
            return

        if self._visualizer_dialog_instance is not None and self._visualizer_dialog_instance.isVisible():
            logger.warning("RealSpaceFFTVisualizerDialog is already open.")
            self._visualizer_dialog_instance.raise_()
            self._visualizer_dialog_instance.activateWindow()
            return

        current_fft_node = self._history.get_current_node()
        if not (current_fft_node and current_fft_node.data_type == "FFT"):
            QMessageBox.warning(self._window, "No FFT Data", "Please calculate FFT first to use the visualizer.")
            return

        self._visualizer_dialog_instance = self._real_space_visualizer_class(
            app_controller=self._controller,
            history_manager=self._history,
            current_fft_node_id=current_fft_node.node_id,
            parent=self._window,
        )
        self._visualizer_dialog_instance.show()
        logger.info("RealSpaceFFTVisualizerDialog opened.")

    # ------------------------------------------------------------------ Real-space reconstruction ---------------------------------------------
    def open_real_space_reconstruction_dialog(self) -> None:
        if not self._history:
            return
        if not self._real_space_reconstruction_class:
            QMessageBox.critical(self._window, "Dialog Error", "RealSpaceReconstructionDialog is not available.")
            return

        current_node = self._history.get_current_node()
        if not (current_node and current_node.data_type == "FFT"):
            QMessageBox.warning(self._window, "Incorrect Data", "This feature requires an active FFT image.")
            return
        if current_node.complex_fft_data is None:
            QMessageBox.warning(
                self._window,
                "Phase Data Missing",
                "This history node does not contain the required phase information for a true reconstruction.",
            )
            return

        dialog = self._real_space_reconstruction_class(
            magnitude_fft_data=current_node.image_data,
            complex_fft_data=current_node.complex_fft_data,
            parent=self._window,
        )
        dialog.exec()
        logger.info("Real Space Reconstruction dialog closed.")

    # ------------------------------------------------------------------ STM transform --------------------------------------------------------
    def open_stm_transform_dialog(self) -> None:
        if not self._controller or not self._history:
            return
        if not self._stm_transform_dialog_class:
            QMessageBox.critical(self._window, "Dialog Error", "StmTransformDialog is unavailable.")
            return

        current_node = self._history.get_current_node()
        if not (current_node and current_node.data_type == "STM"):
            QMessageBox.warning(self._window, "Invalid Data", "This feature requires an active STM image.")
            return
        if self._controller.substrate_F_m2i is None:
            QMessageBox.warning(
                self._window,
                "Missing Data",
                "Please analyze the substrate and compute the transform in the \"Select Substrate Spots\" dialog first.",
            )
            return

        root_node = self._history.get_root_node_for_node(current_node.node_id)
        if not (root_node and root_node.operation_name == "Original"):
            QMessageBox.warning(self._window, "Missing Data", "Original image metadata could not be found in history.")
            return

        dialog = self._stm_transform_dialog_class(
            input_data=current_node.image_data,
            original_node=root_node,
            substrate_transform_F=self._controller.substrate_F_m2i,
            parent=self._window,
        )
        dialog.exec()
        logger.info("STM Transform dialog closed.")
