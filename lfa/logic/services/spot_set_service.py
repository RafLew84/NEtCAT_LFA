from __future__ import annotations

import logging
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np

from ...core.constants import (
    ADSORBATE_LATTICE_TYPE_HEXAGONAL,
    ADSORBATE_LATTICE_TYPE_SQUARE,
    ADSORBATE_LATTICE_TYPE_UNKNOWN,
    SPOT_SELECTION_ADSORBATE,
)


class SpotSetService:
    """Encapsulates adsorbate/substrate spot bookkeeping for the application controller."""

    def __init__(self, controller, default_adsorbate_type: str) -> None:
        self._controller = controller
        self._default_adsorbate_type = default_adsorbate_type
        self._logger = logging.getLogger(__name__)

    @staticmethod
    def _normalise_covariances(
        source_list: Optional[List[Optional[np.ndarray]]],
        expected_len: int,
    ) -> List[Optional[np.ndarray]]:
        covariances: List[Optional[np.ndarray]] = []
        if source_list:
            for cov in source_list:
                if cov is None:
                    covariances.append(None)
                else:
                    covariances.append(np.array(cov, dtype=float))
        if len(covariances) < expected_len:
            covariances.extend([None] * (expected_len - len(covariances)))
        elif len(covariances) > expected_len:
            covariances = covariances[:expected_len]
        return covariances

    # ------------------------------------------------------------------ Adsorbate set management (ported from SpotManager)
    def add_new_adsorbate_set(self) -> None:
        ctrl = self._controller
        ctrl.adsorbate_spot_sets.append([])
        ctrl.corrected_adsorbate_spot_sets.append([])
        ctrl.adsorbate_spot_covariance_sets.append([])
        ctrl.corrected_adsorbate_covariance_sets.append([])
        ctrl.current_adsorbate_set_index = len(ctrl.adsorbate_spot_sets) - 1
        new_set_index = ctrl.current_adsorbate_set_index
        ctrl.adsorbate_spot_pairs.setdefault(new_set_index, [])
        ctrl.adsorbate_expected_lattice_types[new_set_index] = self._default_adsorbate_type
        ctrl.adsorbate_visual_offsets_nm[new_set_index] = (0.0, 0.0)
        self._logger.info("Added new adsorbate set. Index: %s", new_set_index)

        ctrl.spot_lists_updated.emit()
        ctrl.adsorbate_sets_structure_changed.emit()
        if hasattr(ctrl, "adsorbate_expected_type_updated"):
            ctrl.adsorbate_expected_type_updated.emit(new_set_index, self._default_adsorbate_type)

    def set_current_adsorbate_set_by_index(self, index: int) -> None:
        ctrl = self._controller
        if 0 <= index < len(ctrl.adsorbate_spot_sets):
            if ctrl.current_adsorbate_set_index != index:
                ctrl.current_adsorbate_set_index = index
                self._logger.info("Current adsorbate set changed to index: %s", index)
                ctrl.spot_selection_parameters_changed.emit()
        else:
            self._logger.warning("Attempted to set invalid adsorbate set index: %s", index)

    def clear_all_spot_data(self) -> None:
        ctrl = self._controller
        changed = False
        if ctrl.substrate_spots:
            ctrl.substrate_spots.clear()
            changed = True

        if ctrl.adsorbate_spot_sets != [[]] or ctrl.current_adsorbate_set_index != 0:
            ctrl.adsorbate_spot_sets = [[]]
            ctrl.corrected_adsorbate_spot_sets = [[]]
            ctrl.adsorbate_spot_covariance_sets = [[]]
            ctrl.corrected_adsorbate_covariance_sets = [[]]
            ctrl.current_adsorbate_set_index = 0
            ctrl.adsorbate_spot_pairs = {0: []}
            ctrl.adsorbate_expected_lattice_types = {0: self._default_adsorbate_type}
            ctrl.adsorbate_visual_offsets_nm = {0: (0.0, 0.0)}
            changed = True

        ctrl.user_selected_substrate_spots.clear()
        ctrl.user_selected_substrate_covariances.clear()
        ctrl.substrate_real_space_results = None
        ctrl.adsorbate_real_space_results.clear()
        ctrl.substrate_F_m2i = None
        ctrl.substrate_t_m2i = None
        ctrl.substrate_transform_analysis_m2i = None
        ctrl.displayable_fitted_substrate_spots_on_fft.clear()
        ctrl.fitted_substrate_spot_covariances.clear()
        ctrl.substrate_spot_pairs.clear()
        ctrl.substrate_visual_offset_nm = (0.0, 0.0)
        ctrl.superstructure_periodicity_results = None

        ctrl.superstructure_periodicity_results_updated.emit(None)
        ctrl.set_substrate_raw_visibility(True)
        ctrl.set_substrate_transformed_visibility(True)
        ctrl.set_adsorbate_raw_visibility(True)
        ctrl.set_adsorbate_transformed_visibility(True)

        if hasattr(ctrl, "substrate_real_space_params_updated"):
            ctrl.substrate_real_space_params_updated.emit({})
        ctrl.spot_lists_updated.emit()
        ctrl.adsorbate_sets_structure_changed.emit()
        ctrl.substrate_transform_results_updated.emit()
        ctrl.adsorbate_real_space_params_updated.emit(0, {})
        ctrl.adsorbate_expected_type_updated.emit(0, self._default_adsorbate_type)

        if changed and hasattr(ctrl, "adsorbate_set_updated"):
            ctrl.adsorbate_set_updated.emit(0)

        if changed:
            self._logger.debug("Spot data cleared and reset to defaults.")
        else:
            self._logger.debug("Spot data already at default state; nothing cleared.")

    # ------------------------------------------------------------------ Adsorbate spot manipulation
    def clear_last_adsorbate_spot(self) -> None:
        ctrl = self._controller
        idx = ctrl.current_adsorbate_set_index
        if ctrl.spot_selection_mode == SPOT_SELECTION_ADSORBATE and 0 <= idx < len(ctrl.adsorbate_spot_sets):
            current_set = ctrl.adsorbate_spot_sets[idx]
            if current_set:
                removed_point = current_set.pop()
                if idx < len(ctrl.corrected_adsorbate_spot_sets) and ctrl.corrected_adsorbate_spot_sets[idx]:
                    ctrl.corrected_adsorbate_spot_sets[idx].pop()
                if idx < len(ctrl.adsorbate_spot_covariance_sets) and ctrl.adsorbate_spot_covariance_sets[idx]:
                    ctrl.adsorbate_spot_covariance_sets[idx].pop()
                if idx < len(ctrl.corrected_adsorbate_covariance_sets) and ctrl.corrected_adsorbate_covariance_sets[idx]:
                    ctrl.corrected_adsorbate_covariance_sets[idx].pop()
                if idx in ctrl.adsorbate_spot_pairs and ctrl.adsorbate_spot_pairs[idx]:
                    ctrl.adsorbate_spot_pairs[idx].pop()
                self._logger.info("Removed last adsorbate spot %s from set %s.", removed_point, idx)
                ctrl.spot_lists_updated.emit()
                ctrl.adsorbate_set_updated.emit(idx)
        else:
            self._logger.debug("clear_last_adsorbate_spot ignored (not in adsorbate mode or invalid index).")

    def clear_current_adsorbate_set(self) -> None:
        ctrl = self._controller
        idx = ctrl.current_adsorbate_set_index
        if ctrl.spot_selection_mode == SPOT_SELECTION_ADSORBATE and 0 <= idx < len(ctrl.adsorbate_spot_sets):
            if ctrl.adsorbate_spot_sets[idx]:
                ctrl.adsorbate_spot_sets[idx].clear()
                if 0 <= idx < len(ctrl.corrected_adsorbate_spot_sets):
                    ctrl.corrected_adsorbate_spot_sets[idx].clear()
                if 0 <= idx < len(ctrl.adsorbate_spot_covariance_sets):
                    ctrl.adsorbate_spot_covariance_sets[idx].clear()
                if 0 <= idx < len(ctrl.corrected_adsorbate_covariance_sets):
                    ctrl.corrected_adsorbate_covariance_sets[idx].clear()
                ctrl.adsorbate_spot_pairs[idx] = []
                self._logger.info("Cleared all spots from adsorbate set %s.", idx)
                ctrl.spot_lists_updated.emit()
                ctrl.adsorbate_set_updated.emit(idx)
        else:
            self._logger.debug("clear_current_adsorbate_set ignored (not in adsorbate mode or invalid index).")

    def set_expected_adsorbate_lattice_type(self, set_index: int, lattice_type: str) -> None:
        ctrl = self._controller
        valid_types = [
            ADSORBATE_LATTICE_TYPE_UNKNOWN,
            ADSORBATE_LATTICE_TYPE_HEXAGONAL,
            ADSORBATE_LATTICE_TYPE_SQUARE,
        ]
        if not (0 <= set_index < len(ctrl.adsorbate_spot_sets)) or lattice_type not in valid_types:
            self._logger.warning(
                "Invalid set_index %s or lattice_type '%s' for adsorbate expected type.",
                set_index,
                lattice_type,
            )
            return

        if ctrl.adsorbate_expected_lattice_types.get(set_index) != lattice_type:
            ctrl.adsorbate_expected_lattice_types[set_index] = lattice_type
            self._logger.info("Expected lattice type for adsorbate set %s set to '%s'.", set_index, lattice_type)
            if set_index in ctrl.adsorbate_real_space_results:
                del ctrl.adsorbate_real_space_results[set_index]
                ctrl.adsorbate_real_space_params_updated.emit(set_index, {})

            ctrl.adsorbate_expected_type_updated.emit(set_index, lattice_type)

    def clear_all_adsorbate_sets(self) -> None:
        ctrl = self._controller
        if ctrl.adsorbate_spot_sets != [[]] or ctrl.current_adsorbate_set_index != 0:
            ctrl.adsorbate_spot_sets = [[]]
            ctrl.corrected_adsorbate_spot_sets = [[]]
            ctrl.adsorbate_spot_covariance_sets = [[]]
            ctrl.corrected_adsorbate_covariance_sets = [[]]
            ctrl.current_adsorbate_set_index = 0
            ctrl.adsorbate_expected_lattice_types = {0: self._default_adsorbate_type}
            ctrl.adsorbate_visual_offsets_nm = {0: (0.0, 0.0)}
            ctrl.adsorbate_spot_pairs = {0: []}
            self._logger.info("All adsorbate spot sets cleared. Reset to one empty set.")
            ctrl.adsorbate_sets_structure_changed.emit()
            ctrl.spot_lists_updated.emit()
            ctrl.adsorbate_set_updated.emit(0)
        else:
            self._logger.debug("clear_all_adsorbate_sets: nothing to clear.")

    def update_adsorbate_set_results(
        self,
        set_index: int,
        raw_spots: List[Tuple[float, float]],
        corrected_spots_ideal_system: List[Tuple[float, float]],
        raw_covariances: Optional[List[Optional[np.ndarray]]] = None,
        corrected_covariances: Optional[List[Optional[np.ndarray]]] = None,
    ) -> None:
        ctrl = self._controller
        if not (0 <= set_index < len(ctrl.adsorbate_spot_sets)):
            self._logger.error("Invalid set_index %s for updating adsorbate spots.", set_index)
            return

        while len(ctrl.corrected_adsorbate_spot_sets) <= set_index:
            ctrl.corrected_adsorbate_spot_sets.append([])
        while len(ctrl.adsorbate_spot_covariance_sets) <= set_index:
            ctrl.adsorbate_spot_covariance_sets.append([])
        while len(ctrl.corrected_adsorbate_covariance_sets) <= set_index:
            ctrl.corrected_adsorbate_covariance_sets.append([])

        ctrl.adsorbate_spot_pairs.setdefault(set_index, [])

        raw_changed = ctrl.adsorbate_spot_sets[set_index] != raw_spots
        corrected_changed = ctrl.corrected_adsorbate_spot_sets[set_index] != corrected_spots_ideal_system

        normalised_raw_covs = self._normalise_covariances(raw_covariances, len(raw_spots))
        normalised_corrected_covs = self._normalise_covariances(
            corrected_covariances,
            len(corrected_spots_ideal_system),
        )

        if raw_changed:
            ctrl.adsorbate_spot_sets[set_index] = list(raw_spots)
            self._logger.info(
                "Updated raw adsorbate spots for set %s. Count: %s",
                set_index,
                len(raw_spots),
            )

        if corrected_changed:
            ctrl.corrected_adsorbate_spot_sets[set_index] = list(corrected_spots_ideal_system)
            self._logger.info(
                "Updated corrected adsorbate spots (ideal sys) for set %s. Count: %s",
                set_index,
                len(corrected_spots_ideal_system),
            )

        ctrl.adsorbate_spot_covariance_sets[set_index] = normalised_raw_covs
        ctrl.corrected_adsorbate_covariance_sets[set_index] = normalised_corrected_covs

        pair_count = min(len(ctrl.adsorbate_spot_sets[set_index]), len(ctrl.corrected_adsorbate_spot_sets[set_index]))
        if pair_count > 0:
            ctrl.adsorbate_spot_pairs[set_index] = [
                (tuple(ctrl.adsorbate_spot_sets[set_index][i]), tuple(ctrl.corrected_adsorbate_spot_sets[set_index][i]))
                for i in range(pair_count)
            ]
        else:
            ctrl.adsorbate_spot_pairs[set_index] = []

        if raw_changed or corrected_changed:
            ctrl.adsorbate_set_updated.emit(set_index)
            ctrl.spot_lists_updated.emit()

        if set_index in ctrl.adsorbate_real_space_results:
            del ctrl.adsorbate_real_space_results[set_index]
            ctrl.adsorbate_real_space_params_updated.emit(set_index, {})

    # ------------------------------------------------------------------ View-state helpers
    def evaluate_fft_panel_state(
        self,
        history_node,
        lattice_analysis_enabled: bool,
        analysis_functions_available: bool,
    ) -> Dict[str, bool]:
        ctrl = self._controller

        if not history_node or history_node.data_type != "FFT":
            return {
                "fft_active": False,
                "edit_substrate_enabled": False,
                "edit_adsorbate_enabled": False,
                "reselect_adsorbate_enabled": False,
                "clear_all_adsorbate_sets_enabled": False,
                "can_calculate_substrate_rs": False,
                "can_calculate_adsorbate_rs": False,
            }

        ads_sets = ctrl.adsorbate_spot_sets
        current_idx = ctrl.current_adsorbate_set_index
        reselect_enabled = 0 <= current_idx < len(ads_sets) and bool(ads_sets[current_idx])
        clear_all_enabled = any(ads_sets) if ads_sets else False

        can_calc_sub = (
            lattice_analysis_enabled
            and analysis_functions_available
            and ctrl.analysis_executor.can_calculate_substrate_real_space(history_node)
        )
        can_calc_ads = (
            lattice_analysis_enabled
            and analysis_functions_available
            and ctrl.analysis_executor.can_calculate_adsorbate_real_space(history_node)
        )

        return {
            "fft_active": True,
            "edit_substrate_enabled": True,
            "edit_adsorbate_enabled": True,
            "reselect_adsorbate_enabled": reselect_enabled,
            "clear_all_adsorbate_sets_enabled": clear_all_enabled,
            "can_calculate_substrate_rs": can_calc_sub,
            "can_calculate_adsorbate_rs": can_calc_ads,
        }
