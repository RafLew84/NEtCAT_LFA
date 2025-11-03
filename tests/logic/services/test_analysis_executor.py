import sys
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

import numpy as np

from lfa.core.constants import (
    ADSORBATE_LATTICE_TYPE_UNKNOWN,
    LATTICE_TYPE_HEXAGONAL,
    LATTICE_TYPE_SQUARE,
)
from lfa.core.history import HistoryNode
from lfa.logic.services import AnalysisExecutor


class DummyHistoryManager:
    def __init__(self, root_map):
        self._root_map = root_map

    def get_root_node_for_node(self, node_id):
        return self._root_map.get(node_id)


class DummyController:
    def __init__(self):
        self.history_manager = None
        self.substrate_F_m2i = None
        self.substrate_real_space_results = None
        self.displayable_fitted_substrate_spots_on_fft = []
        self.substrate_lattice_type = None
        self.substrate_a_surf = None
        self.custom_lattice_info = None
        self.current_fft_data_shape = None
        self.corrected_adsorbate_spot_sets = [[]]
        self.current_adsorbate_set_index = 0
        self.adsorbate_expected_lattice_types = {0: ADSORBATE_LATTICE_TYPE_UNKNOWN}


def make_history_node(node_id=1, data_type="STM", image_data=None, complex_fft_data=None):
    return HistoryNode(
        operation_name="test",
        data_type=data_type,
        image_data=image_data,
        complex_fft_data=complex_fft_data,
        node_id=node_id,
    )


def test_can_load_metadata_true_when_raw_header_missing():
    controller = DummyController()
    node = make_history_node(node_id=5)
    root = SimpleNamespace(parameters={"other": "value"})
    controller.history_manager = DummyHistoryManager({node.node_id: root})

    executor = AnalysisExecutor(controller)
    assert executor.can_load_metadata(node) is True


def test_can_load_metadata_false_with_raw_header_present():
    controller = DummyController()
    node = make_history_node(node_id=7)
    root = SimpleNamespace(parameters={"raw_header": "exists"})
    controller.history_manager = DummyHistoryManager({node.node_id: root})

    executor = AnalysisExecutor(controller)
    assert executor.can_load_metadata(node) is False


def test_can_load_metadata_handles_missing_history_manager():
    controller = DummyController()
    node = make_history_node(node_id=3)
    executor = AnalysisExecutor(controller)
    assert executor.can_load_metadata(node) is False
    assert executor.can_load_metadata(None) is False


def test_can_calculate_fft_requires_stm_with_image():
    controller = DummyController()
    executor = AnalysisExecutor(controller)

    node = make_history_node(data_type="STM", image_data=np.zeros((2, 2)))
    assert executor.can_calculate_fft(node) is True
    assert executor.can_calculate_fft(make_history_node(data_type="STM", image_data=None)) is False
    assert executor.can_calculate_fft(make_history_node(data_type="FFT", image_data=np.zeros((2, 2)))) is False


def test_can_select_spots_only_for_fft_nodes():
    controller = DummyController()
    executor = AnalysisExecutor(controller)
    assert executor.can_select_spots(make_history_node(data_type="FFT")) is True
    assert executor.can_select_spots(make_history_node(data_type="STM")) is False


def test_can_analyze_superstructure_depends_on_substrate_transform():
    controller = DummyController()
    executor = AnalysisExecutor(controller)
    node = make_history_node(data_type="FFT")
    assert executor.can_analyze_superstructure(node) is False

    controller.substrate_F_m2i = np.eye(3)
    assert executor.can_analyze_superstructure(node) is True


def test_can_visualize_real_space_requires_results():
    controller = DummyController()
    executor = AnalysisExecutor(controller)
    node = make_history_node(data_type="FFT")
    assert executor.can_visualize_real_space(node) is False

    controller.substrate_real_space_results = {"a": 1}
    assert executor.can_visualize_real_space(node) is True


def test_can_open_real_space_reconstruction_checks_fft_data():
    controller = DummyController()
    executor = AnalysisExecutor(controller)
    fft_node = make_history_node(data_type="FFT", complex_fft_data=np.zeros((2, 2)))
    assert executor.can_open_real_space_reconstruction(fft_node) is True

    fft_node_no_data = make_history_node(data_type="FFT")
    assert executor.can_open_real_space_reconstruction(fft_node_no_data) is False


def test_can_open_stm_transform_requires_substrate_transform():
    controller = DummyController()
    executor = AnalysisExecutor(controller)

    stm_node = make_history_node(data_type="STM", image_data=np.zeros((2, 2)))
    assert executor.can_open_stm_transform(stm_node) is False

    controller.substrate_F_m2i = np.eye(3)
    assert executor.can_open_stm_transform(stm_node) is True


def test_can_calculate_substrate_real_space_true_when_requirements_met():
    node = make_history_node(node_id=42, data_type="FFT", image_data=np.zeros((8, 8)))
    root = SimpleNamespace(parameters={"size_nm_x": 10.0, "size_nm_y": 10.0})

    controller = DummyController()
    controller.history_manager = DummyHistoryManager({node.node_id: root})
    controller.displayable_fitted_substrate_spots_on_fft = [(1.0, 1.0)] * 6
    controller.substrate_lattice_type = LATTICE_TYPE_HEXAGONAL
    controller.substrate_a_surf = 0.288

    executor = AnalysisExecutor(controller)
    assert executor.can_calculate_substrate_real_space(node) is True


def test_can_calculate_substrate_real_space_false_with_missing_spots():
    node = make_history_node(node_id=43, data_type="FFT", image_data=np.zeros((8, 8)))
    root = SimpleNamespace(parameters={"size_nm_x": 10.0, "size_nm_y": 10.0})

    controller = DummyController()
    controller.history_manager = DummyHistoryManager({node.node_id: root})
    controller.displayable_fitted_substrate_spots_on_fft = [(1.0, 1.0)] * 2
    controller.substrate_lattice_type = LATTICE_TYPE_SQUARE
    controller.substrate_a_surf = 0.25

    executor = AnalysisExecutor(controller)
    assert executor.can_calculate_substrate_real_space(node) is False


def test_can_calculate_adsorbate_real_space_true_with_corrected_spots():
    node = make_history_node(node_id=55, data_type="FFT", image_data=np.zeros((8, 8)))
    root = SimpleNamespace(parameters={"size_nm_x": 10.0, "size_nm_y": 10.0})

    controller = DummyController()
    controller.history_manager = DummyHistoryManager({node.node_id: root})
    controller.corrected_adsorbate_spot_sets = [[(1.0, 2.0), (3.0, 4.0)]]
    controller.current_adsorbate_set_index = 0

    executor = AnalysisExecutor(controller)
    assert executor.can_calculate_adsorbate_real_space(node) is True


def test_can_calculate_adsorbate_real_space_false_when_insufficient_spots():
    node = make_history_node(node_id=56, data_type="FFT", image_data=np.zeros((8, 8)))
    root = SimpleNamespace(parameters={"size_nm_x": 10.0, "size_nm_y": 10.0})

    controller = DummyController()
    controller.history_manager = DummyHistoryManager({node.node_id: root})
    controller.corrected_adsorbate_spot_sets = [[(1.0, 2.0)]]

    executor = AnalysisExecutor(controller)
    assert executor.can_calculate_adsorbate_real_space(node) is False
