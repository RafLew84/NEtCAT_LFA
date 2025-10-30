from pathlib import Path
from types import SimpleNamespace
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

import pytest
import numpy as np

from lfa.core.constants import (
    ADSORBATE_LATTICE_TYPE_HEXAGONAL,
    ADSORBATE_LATTICE_TYPE_SQUARE,
    ADSORBATE_LATTICE_TYPE_UNKNOWN,
    SPOT_SELECTION_ADSORBATE,
)
from lfa.logic.services import SpotSetService


class FakeSignal:
    def __init__(self) -> None:
        self.calls = []

    def emit(self, *args, **kwargs) -> None:
        self.calls.append((args, kwargs))


class StubAnalysisExecutor:
    def __init__(self) -> None:
        self.allow_sub = False
        self.allow_ads = False
        self.received_nodes = []

    def can_calculate_substrate_real_space(self, node) -> bool:
        self.received_nodes.append(("sub", node))
        return self.allow_sub

    def can_calculate_adsorbate_real_space(self, node) -> bool:
        self.received_nodes.append(("ads", node))
        return self.allow_ads


class DummyController:
    def __init__(self) -> None:
        self.adsorbate_spot_sets = [[]]
        self.corrected_adsorbate_spot_sets = [[]]
        self.current_adsorbate_set_index = 0
        self.adsorbate_spot_pairs = {0: []}
        self.adsorbate_expected_lattice_types = {0: ADSORBATE_LATTICE_TYPE_UNKNOWN}
        self.adsorbate_visual_offsets_nm = {0: (0.0, 0.0)}

        self.substrate_spots = []
        self.user_selected_substrate_spots = []
        self.substrate_real_space_results = None
        self.adsorbate_real_space_results = {}
        self.substrate_F_m2i = None
        self.substrate_t_m2i = None
        self.substrate_transform_analysis_m2i = None
        self.displayable_fitted_substrate_spots_on_fft = []
        self.substrate_spot_pairs = []
        self.substrate_visual_offset_nm = (0.0, 0.0)
        self.superstructure_periodicity_results = None

        self.spot_lists_updated = FakeSignal()
        self.adsorbate_sets_structure_changed = FakeSignal()
        self.adsorbate_expected_type_updated = FakeSignal()
        self.spot_selection_parameters_changed = FakeSignal()
        self.superstructure_periodicity_results_updated = FakeSignal()
        self.substrate_real_space_params_updated = FakeSignal()
        self.substrate_transform_results_updated = FakeSignal()
        self.adsorbate_real_space_params_updated = FakeSignal()
        self.adsorbate_set_updated = FakeSignal()

        self.analysis_executor = StubAnalysisExecutor()
        self.spot_selection_mode = SPOT_SELECTION_ADSORBATE

        self.visibility_calls = []

    def set_substrate_raw_visibility(self, visible: bool) -> None:
        self.visibility_calls.append(("substrate_raw", visible))

    def set_substrate_transformed_visibility(self, visible: bool) -> None:
        self.visibility_calls.append(("substrate_transformed", visible))

    def set_adsorbate_raw_visibility(self, visible: bool) -> None:
        self.visibility_calls.append(("adsorbate_raw", visible))

    def set_adsorbate_transformed_visibility(self, visible: bool) -> None:
        self.visibility_calls.append(("adsorbate_transformed", visible))


@pytest.fixture()
def controller() -> DummyController:
    return DummyController()


@pytest.fixture()
def service(controller: DummyController) -> SpotSetService:
    return SpotSetService(controller, ADSORBATE_LATTICE_TYPE_UNKNOWN)


def test_add_new_adsorbate_set_sets_defaults_and_emits(controller, service):
    service.add_new_adsorbate_set()

    assert controller.current_adsorbate_set_index == 1
    assert controller.adsorbate_spot_sets == [[], []]
    assert controller.corrected_adsorbate_spot_sets == [[], []]
    assert controller.adsorbate_expected_lattice_types[1] == ADSORBATE_LATTICE_TYPE_UNKNOWN
    assert controller.adsorbate_visual_offsets_nm[1] == (0.0, 0.0)

    assert controller.spot_lists_updated.calls
    assert controller.adsorbate_sets_structure_changed.calls
    assert controller.adsorbate_expected_type_updated.calls[-1] == ((1, ADSORBATE_LATTICE_TYPE_UNKNOWN), {})


def test_clear_all_spot_data_resets_state_and_emits(controller, service):
    controller.substrate_spots = [(0.1, 0.2)]
    controller.adsorbate_spot_sets = [[(1.0, 1.0)], [(2.0, 2.0)]]
    controller.corrected_adsorbate_spot_sets = [[(0.9, 1.1)], [(1.9, 2.1)]]
    controller.current_adsorbate_set_index = 1
    controller.adsorbate_spot_pairs = {
        0: [((1.0, 1.0), (0.9, 1.1))],
        1: [((2.0, 2.0), (1.9, 2.1))],
    }
    controller.adsorbate_expected_lattice_types = {
        0: ADSORBATE_LATTICE_TYPE_SQUARE,
        1: ADSORBATE_LATTICE_TYPE_HEXAGONAL,
    }
    controller.adsorbate_visual_offsets_nm = {
        0: (1.0, 1.0),
        1: (2.0, 2.0),
    }
    controller.user_selected_substrate_spots = [(3.0, 3.0)]
    controller.substrate_real_space_results = {"foo": "bar"}
    controller.adsorbate_real_space_results = {1: {"baz": 1}}
    controller.substrate_F_m2i = np.eye(3)
    controller.substrate_t_m2i = np.zeros(2)
    controller.substrate_transform_analysis_m2i = np.eye(3)
    controller.displayable_fitted_substrate_spots_on_fft = [(0.0, 0.0)]
    controller.substrate_spot_pairs = [((0.0, 0.0), (1.0, 1.0))]
    controller.substrate_visual_offset_nm = (5.0, 6.0)
    controller.superstructure_periodicity_results = {"existing": True}

    service.clear_all_spot_data()

    assert controller.substrate_spots == []
    assert controller.adsorbate_spot_sets == [[]]
    assert controller.corrected_adsorbate_spot_sets == [[]]
    assert controller.current_adsorbate_set_index == 0
    assert controller.adsorbate_spot_pairs == {0: []}
    assert controller.adsorbate_expected_lattice_types == {0: ADSORBATE_LATTICE_TYPE_UNKNOWN}
    assert controller.adsorbate_visual_offsets_nm == {0: (0.0, 0.0)}
    assert controller.user_selected_substrate_spots == []
    assert controller.substrate_real_space_results is None
    assert controller.adsorbate_real_space_results == {}
    assert controller.substrate_F_m2i is None
    assert controller.substrate_t_m2i is None
    assert controller.substrate_transform_analysis_m2i is None
    assert controller.displayable_fitted_substrate_spots_on_fft == []
    assert controller.substrate_spot_pairs == []
    assert controller.substrate_visual_offset_nm == (0.0, 0.0)
    assert controller.superstructure_periodicity_results is None

    assert controller.superstructure_periodicity_results_updated.calls[-1] == ((None,), {})
    assert controller.adsorbate_expected_type_updated.calls[-1] == ((0, ADSORBATE_LATTICE_TYPE_UNKNOWN), {})
    assert controller.spot_lists_updated.calls
    assert controller.adsorbate_sets_structure_changed.calls
    assert controller.substrate_transform_results_updated.calls
    assert controller.adsorbate_real_space_params_updated.calls[-1] == ((0, {}), {})
    assert controller.visibility_calls.count(("substrate_raw", True)) == 1
    assert controller.visibility_calls.count(("substrate_transformed", True)) == 1
    assert controller.visibility_calls.count(("adsorbate_raw", True)) == 1
    assert controller.visibility_calls.count(("adsorbate_transformed", True)) == 1


def test_clear_last_adsorbate_spot_removes_entries_and_emits(controller, service):
    controller.adsorbate_spot_sets = [[(0.0, 0.0), (1.0, 1.0)]]
    controller.corrected_adsorbate_spot_sets = [[(0.0, 0.0), (1.1, 1.1)]]
    controller.adsorbate_spot_pairs = {0: [((0.0, 0.0), (0.0, 0.0)), ((1.0, 1.0), (1.1, 1.1))]}
    controller.current_adsorbate_set_index = 0

    service.clear_last_adsorbate_spot()

    assert controller.adsorbate_spot_sets[0] == [(0.0, 0.0)]
    assert controller.corrected_adsorbate_spot_sets[0] == [(0.0, 0.0)]
    assert controller.adsorbate_spot_pairs[0] == [((0.0, 0.0), (0.0, 0.0))]
    assert controller.spot_lists_updated.calls
    assert controller.adsorbate_set_updated.calls[-1] == ((0,), {})


def test_evaluate_fft_panel_state_uses_executor_flags(controller, service):
    node = SimpleNamespace(data_type="FFT")
    controller.adsorbate_spot_sets = [[(0.0, 0.0)]]
    controller.current_adsorbate_set_index = 0

    controller.analysis_executor.allow_sub = True
    controller.analysis_executor.allow_ads = False

    state = service.evaluate_fft_panel_state(node, lattice_analysis_enabled=True, analysis_functions_available=True)

    assert state["fft_active"] is True
    assert state["reselect_adsorbate_enabled"] is True
    assert state["clear_all_adsorbate_sets_enabled"] is True
    assert state["can_calculate_substrate_rs"] is True
    assert state["can_calculate_adsorbate_rs"] is False

    state_unavailable = service.evaluate_fft_panel_state(None, True, True)
    assert state_unavailable["fft_active"] is False
