import numpy as np
import pytest

from lfa.gui.dialogs.presenters.adsorbate_spot_presenter import (
    AdsorbateSpotPresenter,
    AdsorbateSpotState,
)


def test_adsorbate_presenter_build_results_dict_includes_covariances():
    state = AdsorbateSpotState(set_index=2)
    presenter = AdsorbateSpotPresenter(state=state)

    cov = np.array([[0.05, 0.0], [0.0, 0.07]], dtype=float)
    presenter.add_raw_spot((1.0, 2.0), covariance=cov)
    presenter.state.corrected_spots = [(3.0, 4.0)]
    presenter.state.corrected_spot_covariances = [cov]

    payload = presenter.build_results_dict()

    assert len(payload["raw_adsorbate_spot_covariances"]) == 1
    assert np.allclose(payload["raw_adsorbate_spot_covariances"][0], cov)
    assert len(payload["corrected_adsorbate_spot_covariances"]) == 1
    assert np.allclose(payload["corrected_adsorbate_spot_covariances"][0], cov)
    assert payload["adsorbate_set_index"] == 2
