import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

pytest.importorskip("PyQt6", reason="PyQt6 is required for spot-selection dialog tests")
pytest.importorskip("pytestqt", reason="pytest-qt is required for qtbot fixture")

from PyQt6.QtWidgets import QListWidget

from lfa.gui.dialogs.adsorbate_spot_dialog import REFINEMENT_MAX_PIXEL as ADS_REFINEMENT_MAX_PIXEL
from lfa.gui.dialogs.adsorbate_spot_dialog import AdsorbateSpotSelectionDialog
from lfa.gui.dialogs.presenters.substrate_spot_presenter import TransformComputation
from lfa.gui.dialogs.substrate_spot_dialog import REFINEMENT_MAX_PIXEL, SubstrateSpotSelectionDialog
from lfa.logic.history_manager import HistoryManager


@pytest.fixture
def substrate_dialog(qtbot, monkeypatch):
    fft_data = np.zeros((16, 16), dtype=np.float32)
    list_widget = QListWidget()
    qtbot.addWidget(list_widget)
    history_manager = HistoryManager(list_widget)
    dialog = SubstrateSpotSelectionDialog(
        fft_image_data=fft_data,
        history_manager=history_manager,
        current_fft_node_id="node",
        default_refinement_method=REFINEMENT_MAX_PIXEL,
    )
    qtbot.addWidget(dialog)
    dialog.selection_roi.setVisible(True)
    dialog.selection_roi.setPos((2, 2))
    dialog.selection_roi.setSize((6, 6))
    return dialog


@pytest.fixture
def adsorbate_dialog(qtbot, monkeypatch):
    fft_data = np.zeros((16, 16), dtype=np.float32)
    list_widget = QListWidget()
    qtbot.addWidget(list_widget)
    history_manager = HistoryManager(list_widget)
    dialog = AdsorbateSpotSelectionDialog(
        fft_image_data=fft_data,
        history_manager=history_manager,
        current_fft_node_id="node",
        current_adsorbate_spots=None,
        adsorbate_set_index=0,
        default_refinement_method=ADS_REFINEMENT_MAX_PIXEL,
    )
    qtbot.addWidget(dialog)
    dialog.selection_roi.setVisible(True)
    dialog.selection_roi.setPos((4, 4))
    dialog.selection_roi.setSize((8, 8))
    return dialog


def test_substrate_max_pixel_centers_marker(substrate_dialog, monkeypatch):
    monkeypatch.setattr(
        "lfa.gui.dialogs.substrate_spot_dialog.find_max_pixel_in_roi",
        lambda data, center, radius: (5, 10)
    )
    substrate_dialog.current_refinement_method = REFINEMENT_MAX_PIXEL
    substrate_dialog.selected_spots.clear()

    substrate_dialog._add_current_roi_spot()

    assert substrate_dialog.selected_spots
    x, y = substrate_dialog.selected_spots[-1]
    assert x == pytest.approx(10.5)
    assert y == pytest.approx(5.5)


def test_adsorbate_max_pixel_centers_marker(adsorbate_dialog, monkeypatch):
    monkeypatch.setattr(
        "lfa.gui.dialogs.adsorbate_spot_dialog.find_max_pixel_in_roi",
        lambda data, center, radius: (6, 12)
    )
    adsorbate_dialog.current_refinement_method = ADS_REFINEMENT_MAX_PIXEL
    adsorbate_dialog.state.raw_spots.clear()

    adsorbate_dialog._add_current_adsorbate_spot_from_roi()

    assert adsorbate_dialog.state.raw_spots
    x, y = adsorbate_dialog.state.raw_spots[-1]
    assert x == pytest.approx(12.5)
    assert y == pytest.approx(6.5)


def test_substrate_dialog_transform_labels_include_uncertainty(substrate_dialog, monkeypatch):
    analysis = {
        "rotation_angle_deg": 2.34,
        "rotation_angle_deg_sigma": 0.12,
        "principal_stretches": (1.01, 0.98),
        "principal_stretches_sigma": (0.01, 0.02),
        "rmse": 0.045,
    }

    substrate_dialog.selected_spots = [(0.0, 0.0), (1.0, 0.0), (0.0, 1.0)]
    substrate_dialog.presenter.state.selected_spots = list(substrate_dialog.selected_spots)
    substrate_dialog.selected_spot_covariances = [np.eye(2)] * 3
    substrate_dialog.presenter.state.selected_spot_covariances = list(substrate_dialog.selected_spot_covariances)

    def fake_calculate_transform(*, preferred_point_count):
        matrix_F = np.eye(2, dtype=float)
        translation = np.array([0.0, 0.0], dtype=float)
        measured = np.array(substrate_dialog.presenter.state.selected_spots, dtype=float)
        ideal = measured.copy()

        substrate_dialog.presenter.state.transform_matrix_F = matrix_F
        substrate_dialog.presenter.state.transform_translation_t = translation
        substrate_dialog.presenter.state.transform_analysis = analysis.copy()
        substrate_dialog.presenter.state.fitted_spots_px = [tuple(pt) for pt in measured]
        substrate_dialog.presenter.state.fitted_spot_covariances = [np.eye(2)] * len(measured)
        substrate_dialog.presenter.state.ideal_spots_px_for_reference = [tuple(pt) for pt in ideal]

        return TransformComputation(
            matrix_F=matrix_F,
            translation_t=translation,
            analysis=analysis.copy(),
            fitted_spots_px=substrate_dialog.presenter.state.fitted_spots_px,
            measured_spots_px=measured,
            ideal_spots_px=ideal,
            matched_pairs=[(idx, idx) for idx in range(len(measured))],
        )

    monkeypatch.setattr(substrate_dialog.presenter, "calculate_transform", fake_calculate_transform)

    substrate_dialog._on_calculate_transform_clicked()

    rotation_text = substrate_dialog.rotation_angle_label.text()
    assert "+/- 0.12" in rotation_text
    assert rotation_text.endswith("deg") or "deg" in rotation_text

    stretch_text = substrate_dialog.scale_factor_label.text()
    assert "+/- 0.010" in stretch_text
    assert "+/- 0.020" in stretch_text
    assert "Stretches" in stretch_text

    assert substrate_dialog.substrate_transform_analysis["rotation_angle_deg_sigma"] == pytest.approx(0.12)
    assert substrate_dialog.transform_status_label.text() == "Transformation calculated."


def test_adsorbate_dialog_displays_substrate_transform_sigma(adsorbate_dialog):
    analysis = {
        "rotation_angle_deg": -1.75,
        "rotation_angle_deg_sigma": 0.08,
        "principal_stretches": (1.05, 0.97),
        "principal_stretches_sigma": (0.012, 0.018),
        "rmse": 0.03,
    }

    adsorbate_dialog.state.substrate_analysis = analysis
    adsorbate_dialog._display_substrate_transform_info()

    rotation_label = adsorbate_dialog.sub_transform_info_label_rot.text()
    assert "Sub. Rotation (M->I):" in rotation_label
    assert "+/- 0.08" in rotation_label
    assert "deg" in rotation_label

    stretch_label = adsorbate_dialog.sub_transform_info_label_scale.text()
    assert "+/- 0.012" in stretch_label
    assert "+/- 0.018" in stretch_label
    assert "Stretches" in stretch_label

    rmse_label = adsorbate_dialog.sub_transform_info_label_rmse.text()
    assert "0.030" in rmse_label
