from pathlib import Path
import sys
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

pytest.importorskip("PyQt6", reason="PyQt6 is required for real-space visualizer tests")
pytest.importorskip("pyqtgraph", reason="pyqtgraph is required for real-space visualizer tests")
pytest.importorskip("pytestqt", reason="pytest-qt provides the qtbot fixture")

from lfa.gui.dialogs.real_space_visualizer_dialog import RealSpaceFFTVisualizerDialog


class _DummyController:
    def __init__(self) -> None:
        self.substrate_real_space_results = {}
        self.adsorbate_real_space_results = {}
        self.substrate_transform_analysis_m2i = {}
        self.adsorbate_spot_sets = []
        self.corrected_adsorbate_spot_sets = []
        self.current_adsorbate_set_index = 0
        self.substrate_visual_offset_nm = (0.0, 0.0)
        self.adsorbate_visual_offsets_nm = {}


class _DummyHistory:
    def get_node_by_id(self, _):
        return None


def test_calibration_sigma_label_prefers_substrate_then_adsorbate(qtbot):
    controller = _DummyController()
    history = _DummyHistory()

    dialog = RealSpaceFFTVisualizerDialog(controller, history, current_fft_node_id=None)
    qtbot.addWidget(dialog)

    dialog.ads_set_combo_vis.clear()
    dialog.ads_set_combo_vis.addItem("Set 1", 0)
    dialog.ads_set_combo_vis.setCurrentIndex(0)

    controller.substrate_real_space_results = {
        "a1_nm": 0.25,
        "a1_nm_sigma": 0.001,
        "a2_nm": 0.28,
        "a2_nm_sigma": 0.002,
        "alpha_deg": 60.0,
        "alpha_deg_sigma": 0.2,
        "pixel_calibration_sigma_nm": (0.001, 0.002),
    }
    controller.adsorbate_real_space_results = {
        0: {
            "a1_nm": 0.45,
            "a1_nm_sigma": 0.003,
            "a2_nm": 0.47,
            "a2_nm_sigma": 0.004,
            "alpha_deg": 58.0,
            "alpha_deg_sigma": 0.25,
            "pixel_calibration_sigma_nm": (0.003, 0.004),
        }
    }

    dialog._update_real_space_param_labels()
    assert dialog.calibration_sigma_label.text() == "(0.0010, 0.0020) nm"
    assert dialog.sub_real_a1_label.text() == "0.250 +/- 0.001 nm"
    assert dialog.sub_real_a2_label.text() == "0.280 +/- 0.002 nm"
    assert dialog.sub_real_alpha_label.text() == "60.00 +/- 0.20 deg"

    controller.substrate_real_space_results = {}
    controller.adsorbate_real_space_results[0]["pixel_calibration_sigma_nm"] = (0.005, 0.006)

    dialog._update_real_space_param_labels()
    assert dialog.calibration_sigma_label.text() == "(0.0050, 0.0060) nm"
    assert dialog.sub_real_a1_label.text() == "- nm"
    assert dialog.ads_real_a1_label.text() == "0.450 +/- 0.003 nm"


def test_calculate_sub_ads_angle_includes_sigma(qtbot):
    controller = _DummyController()
    history = _DummyHistory()

    dialog = RealSpaceFFTVisualizerDialog(controller, history, current_fft_node_id=None)
    qtbot.addWidget(dialog)

    dialog.ads_set_combo_vis.clear()
    dialog.ads_set_combo_vis.addItem("Set 1", 0)
    dialog.ads_set_combo_vis.setCurrentIndex(0)

    controller.substrate_real_space_results = {
        "a1_vec_nm": (1.0, 0.0),
        "a2_vec_nm": (0.0, 1.0),
    }
    controller.adsorbate_real_space_results = {
        0: {
            "a1_vec_nm": (0.8660254, 0.5),
            "a2_vec_nm": (-0.5, 0.8660254),
        }
    }

    dialog._estimate_sub_ads_angle_sigma_deg = lambda *args, **kwargs: 0.123  # type: ignore[attr-defined]
    dialog._on_calculate_sub_ads_angle_clicked()

    label_text = dialog.angle_sub_ads_label.text()
    assert "+/- 0.123" in label_text
