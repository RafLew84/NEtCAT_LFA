from pathlib import Path
import sys
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

pytest.importorskip("PyQt6", reason="PyQt6 is required for FFTAnalysisPanel tests")
pytest.importorskip("pytestqt", reason="pytest-qt provides the qtbot fixture")

from lfa.gui.panels.fft_analysis_panel import FFTAnalysisPanel


def test_fft_panel_updates_sigma_labels(qtbot):
    panel = FFTAnalysisPanel()
    qtbot.addWidget(panel)

    substrate_params = {
        "a1_nm": 0.25,
        "a1_nm_sigma": 0.005,
        "a2_nm": 0.28,
        "a2_nm_sigma": 0.006,
        "alpha_deg": 60.0,
        "alpha_deg_sigma": 0.2,
        "pixel_calibration_sigma_nm": (0.001, 0.002),
    }

    panel.update_substrate_real_space_display(substrate_params)

    text = panel.sub_rs_a1_label.text()
    assert "0.25" in text
    assert "0.005" in text
    assert "nm" in text

    alpha_text = panel.sub_rs_alpha_label.text()
    assert "60.0" in alpha_text or "60.00" in alpha_text
    assert "0.2" in alpha_text
    assert "+/-" in alpha_text
    assert "deg" in alpha_text
    assert chr(176) not in alpha_text

    calibration_text = panel.calibration_sigma_label.text()
    assert "(0.0010, 0.0020) nm" == calibration_text
    assert chr(176) not in calibration_text

    adsorbate_params = {
        "a1_nm": 0.42,
        "a1_nm_sigma": 0.01,
        "a2_nm": 0.45,
        "a2_nm_sigma": 0.012,
        "alpha_deg": 58.5,
        "alpha_deg_sigma": 0.3,
        "pixel_calibration_sigma_nm": (0.003, 0.004),
    }

    panel.update_adsorbate_real_space_display(adsorbate_params)

    ads_alpha_text = panel.ads_rs_alpha_label.text()
    assert "58.5" in ads_alpha_text
    assert "+/- 0.3" in ads_alpha_text or "+/- 0.30" in ads_alpha_text
    assert "deg" in ads_alpha_text

    calibration_text_after_ads = panel.calibration_sigma_label.text()
    assert "(0.0030, 0.0040) nm" == calibration_text_after_ads


def test_fft_panel_transform_results_include_uncertainty(qtbot):
    panel = FFTAnalysisPanel()
    qtbot.addWidget(panel)

    analysis = {
        "rotation_angle_deg": 3.21,
        "rotation_angle_deg_sigma": 0.11,
        "principal_stretches": (0.99, 1.04),
        "principal_stretches_sigma": (0.015, 0.017),
        "rmse": 0.062,
    }

    panel.update_transform_results_display(analysis)

    rotation_text = panel.rotation_angle_label.text()
    assert "Rotation (M->I):" in rotation_text
    assert "+/- 0.11" in rotation_text
    assert "deg" in rotation_text

    stretch_text = panel.scale_factor_label.text()
    assert "+/- 0.015" in stretch_text
    assert "+/- 0.017" in stretch_text
    assert "Stretches" in stretch_text

    rmse_text = panel.rmse_label.text()
    assert "0.062" in rmse_text
