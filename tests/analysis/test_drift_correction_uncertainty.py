import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from lfa.analysis.drift_correction import match_and_fit_transform  # noqa: E402


def test_match_and_fit_transform_produces_rotation_and_stretch_uncertainty():
    measured = np.array(
        [
            [10.0, 5.0],
            [20.0, 5.0],
            [10.0, 15.0],
            [20.0, 15.0],
        ],
        dtype=float,
    )

    F, t, analysis, pairs, transformed_covariances = match_and_fit_transform(
        measured_pts_px=measured,
        ideal_pts_pool_px=measured.copy(),
        num_expected_matches=4,
        measured_covariances_px=None,
    )

    assert F is not None
    assert t is not None
    assert analysis is not None
    assert pairs is not None
    assert transformed_covariances is None or isinstance(transformed_covariances, list)

    assert "rotation_angle_deg_sigma" in analysis
    assert "principal_stretches_sigma" in analysis
    assert "principal_stretches_covariance" in analysis

    rotation_sigma = analysis["rotation_angle_deg_sigma"]
    stretch_sigma = analysis["principal_stretches_sigma"]

    assert pytest.approx(rotation_sigma, abs=1e-6) == 0.0
    assert isinstance(stretch_sigma, (tuple, list, np.ndarray))
    assert pytest.approx(stretch_sigma[0], abs=1e-6) == 0.0
    assert pytest.approx(stretch_sigma[1], abs=1e-6) == 0.0
