import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from lfa.gui.dialogs.presenters import (
    AngleCalculationResult,
    RealSpaceVisualizerPresenter,
)


class DummyController:
    def __init__(self) -> None:
        self.adsorbate_spot_sets = [[(1.0, 2.0)], [(3.0, 4.0)]]
        self.corrected_adsorbate_spot_sets = [[(1.1, 2.1)], [(3.1, 4.1)]]
        self.current_adsorbate_set_index = 1
        self.adsorbate_visual_offsets_nm = {0: (0.0, 0.0), 1: (0.0, 0.0)}

        self.substrate_real_space_results = {
            "a1_nm": 0.25,
            "a1_nm_sigma": 0.001,
            "a2_nm": 0.28,
            "a2_nm_sigma": 0.002,
            "alpha_deg": 60.0,
            "alpha_deg_sigma": 0.2,
            "pixel_calibration_sigma_nm": (0.001, 0.002),
            "a1_vec_nm": (1.0, 0.0),
            "a2_vec_nm": (0.0, 1.0),
        }
        self.adsorbate_real_space_results = {
            0: {
                "a1_nm": 0.44,
                "a1_nm_sigma": 0.003,
                "a2_nm": 0.46,
                "a2_nm_sigma": 0.004,
                "alpha_deg": 58.0,
                "alpha_deg_sigma": 0.25,
                "pixel_calibration_sigma_nm": (0.004, 0.005),
                "a1_vec_nm": (0.9, 0.2),
                "a2_vec_nm": (-0.3, 0.85),
                "g1_vec_nm_inv": (1.0, 0.0),
                "g2_vec_nm_inv": (0.0, 1.0),
            },
            1: {
                "a1_nm": 0.54,
                "a1_nm_sigma": 0.003,
                "a2_nm": 0.56,
                "a2_nm_sigma": 0.004,
                "alpha_deg": 59.0,
                "alpha_deg_sigma": 0.2,
                "pixel_calibration_sigma_nm": (0.006, 0.007),
                "a1_vec_nm": (1.0, 0.0),
                "a2_vec_nm": (0.0, 1.0),
                "g1_vec_nm_inv": (1.0, 0.0),
                "g2_vec_nm_inv": (0.0, 1.0),
            },
        }


def test_presenter_adsorbate_summary_uses_controller_index():
    presenter = RealSpaceVisualizerPresenter(DummyController())
    summary = presenter.get_adsorbate_sets_summary()
    assert len(summary.sets) == 2
    assert summary.selected_index == 1
    assert summary.sets[0].label == "Set 1"
    assert summary.sets[1].has_real_space_result is True


def test_presenter_build_real_space_bundle_prefers_substrate_sigma():
    presenter = RealSpaceVisualizerPresenter(DummyController())
    bundle = presenter.build_real_space_label_bundle(active_adsorbate_index=0)

    assert bundle.substrate_a1.text.startswith("0.250")
    assert bundle.adsorbate_a1.text.startswith("0.440")
    assert bundle.calibration_text == "(0.0010, 0.0020) nm"
    assert bundle.angle_button_enabled is True


def test_presenter_calculate_angle_returns_alignment():
    presenter = RealSpaceVisualizerPresenter(DummyController())
    result = presenter.calculate_sub_ads_angle(adsorbate_index=1)

    assert isinstance(result, AngleCalculationResult)
    assert "deg" in result.display_text
    assert result.alignment_angle_rad is not None
    assert np.isfinite(result.alignment_angle_rad)
