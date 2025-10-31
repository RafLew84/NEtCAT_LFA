import pytest

pytest.importorskip("PyQt6", reason="PyQt6 is required for FFTAnalysisPanel tests")
pytest.importorskip("pytestqt", reason="pytest-qt provides the qtbot fixture")

from lfa.gui.panels.fft_analysis_panel import FFTAnalysisPanel


def test_fft_panel_updates_sigma_labels(qtbot):
    panel = FFTAnalysisPanel()
    qtbot.addWidget(panel)

    params = {
        "a1_nm": 0.25,
        "a1_nm_sigma": 0.005,
        "a2_nm": 0.28,
        "a2_nm_sigma": 0.006,
        "alpha_deg": 60.0,
        "alpha_deg_sigma": 0.2,
    }

    panel.update_substrate_real_space_display(params)

    text = panel.sub_rs_a1_label.text()
    assert "0.25" in text
    assert "0.005" in text
    assert "nm" in text

    alpha_text = panel.sub_rs_alpha_label.text()
    assert "60.0" in alpha_text or "60.00" in alpha_text
    assert "0.2" in alpha_text
