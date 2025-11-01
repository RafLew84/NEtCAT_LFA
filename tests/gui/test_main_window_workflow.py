import numpy as np
import pytest

pytest.importorskip("PyQt6", reason="PyQt6 is required for GUI workflow tests")
pytest.importorskip("pytestqt", reason="pytest-qt is required for qtbot fixture")

from lfa.gui.main_window import MainWindow
from lfa.core.data_models import STMImage
from lfa.core.history import HistoryNode
from lfa.logic.services.session_service import OriginalImageRecord
from lfa.core.constants import LATTICE_TYPE_HEXAGONAL


def _register_original_image(window: MainWindow, data: np.ndarray) -> HistoryNode:
    stm_image = STMImage(
        file_name="synthetic.stp",
        raw_header={},
        data=data,
        pixels_x=data.shape[1],
        pixels_y=data.shape[0],
        size_nm_x=float(data.shape[1]),
        size_nm_y=float(data.shape[0]),
    )
    record = OriginalImageRecord(
        display_name="Original Image 1",
        stm_image=stm_image,
        source_path="synthetic.stp",
    )
    record.extra_metadata["source_image_id"] = record.image_id

    root_params = {
        "filename": "synthetic.stp",
        "source_path": "synthetic.stp",
        "source_image_id": record.image_id,
        "source_image_label": record.display_name,
    }

    root_node = HistoryNode(
        operation_name="Original",
        image_data=stm_image.data.copy(),
        parameters=root_params,
        data_type="STM",
        original_image_id=record.image_id,
    )

    window.app_controller.session_service.register_new_original(record, root_node)
    return root_node


def test_main_window_fft_history_flow(qtbot):
    window = MainWindow()
    qtbot.addWidget(window)

    data = np.ones((4, 4), dtype=np.float32)
    root_node = _register_original_image(window, data)

    assert window.history_list_widget.count() == 1
    assert window.app_controller.current_fft_data_shape is None

    fft_processed = np.full((4, 4), 2.0, dtype=np.float32)
    fft_complex = np.ones((4, 4), dtype=np.complex128)

    window.app_controller.calculate_fft_operation(
        parent_node_id=root_node.node_id,
        processed_fft_data=fft_processed,
        complex_fft_data=fft_complex,
        params={"apply_window": True},
    )

    qtbot.waitUntil(lambda: window.app_controller.current_fft_data_shape == (4, 4))

    assert window.history_list_widget.count() == 2
    current_node = window.history_manager.get_current_node()
    assert current_node is not None and current_node.data_type == "FFT"
    assert window.app_controller.current_fft_data_shape == (4, 4)


def test_main_window_substrate_and_adsorbate_updates(qtbot):
    window = MainWindow()
    qtbot.addWidget(window)

    data = np.ones((4, 4), dtype=np.float32)
    root_node = _register_original_image(window, data)

    fft_processed = np.full((4, 4), 3.0, dtype=np.float32)
    fft_complex = np.ones((4, 4), dtype=np.complex128)
    window.app_controller.calculate_fft_operation(
        parent_node_id=root_node.node_id,
        processed_fft_data=fft_processed,
        complex_fft_data=fft_complex,
        params={"apply_window": True},
    )
    qtbot.waitUntil(lambda: window.history_manager.get_current_node().data_type == "FFT")  # type: ignore

    substrate_results = {
        "spots": [(1.0, 2.0), (3.0, 4.0)],
        "spot_covariances": [np.eye(2), np.eye(2)],
        "lattice_type": LATTICE_TYPE_HEXAGONAL,
        "substrate_definition": "Au(111)",
        "a_surf": 0.288,
        "transformation_F_m2i": np.eye(2),
        "translation_t_m2i": np.zeros(2),
        "transform_analysis_m2i": {
            "rotation_deg": 0.0,
            "stretch_x": 1.0,
            "stretch_y": 1.0,
            "rmse": 0.0,
        },
        "displayable_fitted_spots": [(1.1, 2.1), (3.1, 4.1)],
        "fitted_spot_covariances": [np.eye(2), np.eye(2)],
        "ideal_substrate_spots_px_for_reference": [(0.5, 0.5)],
    }
    window.app_controller.update_substrate_analysis_results(substrate_results)
    qtbot.wait(10)

    text = window.fft_analysis_panel_widget.selected_spots_display.toPlainText()
    assert "Substrate Spots:" in text
    assert "S1: (kx=1.0, ky=2.0)" in text

    window.fft_analysis_panel_widget.update_adsorbate_set_combo(["Set 1"], "Set 1")
    window.app_controller.spot_selection_mode = "Adsorbate"
    adsorbate_raw = [(5.0, 6.0), (7.0, 8.0)]
    adsorbate_corrected = [(5.5, 6.5), (7.5, 8.5)]
    window.app_controller.update_adsorbate_set_results(
        0,
        raw_spots=adsorbate_raw,
        corrected_spots_ideal_system=adsorbate_corrected,
    )
    qtbot.wait(10)

    text = window.fft_analysis_panel_widget.selected_spots_display.toPlainText()
    assert "Adsorbate Set 1" in text
    assert "A1: (kx=5.0, ky=6.0)" in text
