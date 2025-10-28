from __future__ import annotations

import logging
from typing import Optional


class SpotManager:
    """Encapsulates spot and lattice bookkeeping for the application controller."""

    def __init__(self, controller, default_adsorbate_type: str) -> None:
        self.controller = controller
        self.default_adsorbate_type = default_adsorbate_type
        self._logger = logging.getLogger(__name__)

    # ------------------------------------------------------------------ helpers
    def add_new_adsorbate_set(self) -> None:
        ctrl = self.controller
        ctrl.adsorbate_spot_sets.append([])
        ctrl.corrected_adsorbate_spot_sets.append([])
        ctrl.current_adsorbate_set_index = len(ctrl.adsorbate_spot_sets) - 1
        new_set_index = ctrl.current_adsorbate_set_index
        ctrl.adsorbate_spot_pairs.setdefault(new_set_index, [])
        ctrl.adsorbate_expected_lattice_types[new_set_index] = self.default_adsorbate_type
        ctrl.adsorbate_visual_offsets_nm[new_set_index] = (0.0, 0.0)
        self._logger.info("Added new adsorbate set. Index: %s", new_set_index)

        ctrl.spot_lists_updated.emit()
        ctrl.adsorbate_sets_structure_changed.emit()
        if hasattr(ctrl, "adsorbate_expected_type_updated"):
            ctrl.adsorbate_expected_type_updated.emit(new_set_index, self.default_adsorbate_type)

    def set_current_adsorbate_set_by_index(self, index: int) -> None:
        ctrl = self.controller
        if 0 <= index < len(ctrl.adsorbate_spot_sets):
            if ctrl.current_adsorbate_set_index != index:
                ctrl.current_adsorbate_set_index = index
                self._logger.info("Current adsorbate set changed to index: %s", index)
                ctrl.spot_selection_parameters_changed.emit()
        else:
            self._logger.warning("Attempted to set invalid adsorbate set index: %s", index)

    def clear_all_spot_data(self) -> None:
        ctrl = self.controller
        changed = False
        if ctrl.substrate_spots:
            ctrl.substrate_spots.clear()
            changed = True
        if ctrl.adsorbate_spot_sets != [[]] or ctrl.current_adsorbate_set_index != 0:
            ctrl.adsorbate_spot_sets = [[]]
            ctrl.corrected_adsorbate_spot_sets = [[]]
            ctrl.current_adsorbate_set_index = 0
            changed = True

        ctrl.user_selected_substrate_spots.clear()
        ctrl.adsorbate_real_space_results.clear()
        ctrl.substrate_F_m2i = None
        ctrl.substrate_t_m2i = None
        ctrl.substrate_transform_analysis_m2i = None
        ctrl.displayable_fitted_substrate_spots_on_fft.clear()
        ctrl.substrate_spot_pairs.clear()
        ctrl.substrate_real_space_results = None
        ctrl.adsorbate_expected_lattice_types = {0: self.default_adsorbate_type}
        ctrl.adsorbate_spot_pairs = {0: []}
        ctrl.substrate_visual_offset_nm = (0.0, 0.0)
        ctrl.adsorbate_visual_offsets_nm = {0: (0.0, 0.0)}
        ctrl.superstructure_periodicity_results = None
        ctrl.superstructure_periodicity_results_updated.emit(None)

        ctrl.set_substrate_raw_visibility(True)
        ctrl.set_substrate_transformed_visibility(True)
        ctrl.set_adsorbate_raw_visibility(True)
        ctrl.set_adsorbate_transformed_visibility(True)

        if hasattr(ctrl, "substrate_real_space_params_updated"):
            ctrl.substrate_real_space_params_updated.emit({})
        if hasattr(ctrl, "spot_lists_updated"):
            ctrl.spot_lists_updated.emit()
        if hasattr(ctrl, "adsorbate_sets_structure_changed"):
            ctrl.adsorbate_sets_structure_changed.emit()
        if hasattr(ctrl, "substrate_transform_results_updated"):
            ctrl.substrate_transform_results_updated.emit()
        if hasattr(ctrl, "adsorbate_real_space_params_updated"):
            ctrl.adsorbate_real_space_params_updated.emit(0, {})
        if hasattr(ctrl, "adsorbate_expected_type_updated"):
            ctrl.adsorbate_expected_type_updated.emit(0, self.default_adsorbate_type)
        if changed and hasattr(ctrl, "adsorbate_set_updated"):
            ctrl.adsorbate_set_updated.emit(0)

        if changed:
            self._logger.debug("Spot data cleared and reset to defaults.")
        else:
            self._logger.debug("Spot data already at default state; nothing cleared.")
