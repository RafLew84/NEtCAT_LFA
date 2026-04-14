"""Tests for the AtomMapper Gaussian-fit adapter."""

from __future__ import annotations

import numpy as np
import pytest

from lfa.analysis.peak_fitting import PeakRefinementResult, SCIPY_AVAILABLE

from AtomMapper.app.fit_models import LocalFitModelType, LocalFitRequest
from AtomMapper.app.fit_settings import (
    CommonFitSettings,
    FitSettingsState,
    GaussianFitSettings,
    LorentzianFitSettings,
    ParameterBounds,
    VoigtFitSettings,
)
from AtomMapper.app.gaussian_fit import fit_gaussian_to_roi_patch, fit_local_peak


def _make_lorentzian_patch(
    rows: int = 15,
    cols: int = 17,
    *,
    center_y: float = 7.2,
    center_x: float = 9.1,
    amplitude: float = 14.0,
    gamma_y: float = 1.8,
    gamma_x: float = 2.4,
    theta_rad: float = 0.12,
    offset: float = 0.9,
) -> np.ndarray:
    y_grid, x_grid = np.mgrid[0:rows, 0:cols]
    dy = y_grid - center_y
    dx = x_grid - center_x
    y_rot = np.cos(theta_rad) * dy + np.sin(theta_rad) * dx
    x_rot = -np.sin(theta_rad) * dy + np.cos(theta_rad) * dx
    q = (y_rot / gamma_y) ** 2 + (x_rot / gamma_x) ** 2
    return offset + amplitude / (1.0 + q)


def _make_voigt_patch(
    rows: int = 15,
    cols: int = 17,
    *,
    center_y: float = 7.1,
    center_x: float = 9.3,
    amplitude: float = 16.0,
    sigma_y: float = 1.3,
    sigma_x: float = 2.0,
    gamma_y: float = 0.8,
    gamma_x: float = 1.1,
    theta_rad: float = 0.1,
    offset: float = 0.7,
) -> np.ndarray:
    from scipy.special import voigt_profile

    y_grid, x_grid = np.mgrid[0:rows, 0:cols]
    dy = y_grid - center_y
    dx = x_grid - center_x
    y_rot = np.cos(theta_rad) * dy + np.sin(theta_rad) * dx
    x_rot = -np.sin(theta_rad) * dy + np.cos(theta_rad) * dx
    profile_y = voigt_profile(y_rot, sigma_y, gamma_y)
    profile_x = voigt_profile(x_rot, sigma_x, gamma_x)
    profile_y /= float(voigt_profile(np.array([0.0]), sigma_y, gamma_y)[0])
    profile_x /= float(voigt_profile(np.array([0.0]), sigma_x, gamma_x)[0])
    return offset + amplitude * profile_y * profile_x


@pytest.mark.skipif(not SCIPY_AVAILABLE, reason="SciPy not available, skipping Gaussian fit tests")
def test_atommapper_gaussian_adapter_returns_patch_and_image_coordinates():
    rows, cols = 15, 17
    y_grid, x_grid = np.mgrid[0:rows, 0:cols]
    y0_true, x0_true = 7.3, 9.2
    amplitude = 12.0
    sigma_y, sigma_x = 2.0, 1.6
    offset = 1.25
    patch = offset + amplitude * np.exp(
        -(((y_grid - y0_true) ** 2) / (2.0 * sigma_y**2) + ((x_grid - x0_true) ** 2) / (2.0 * sigma_x**2))
    )

    result = fit_gaussian_to_roi_patch(patch, roi_origin_yx=(40, 60))

    assert result.model is LocalFitModelType.GAUSSIAN
    assert result.success is True
    assert result.error_message is None
    assert result.method == "gaussian_fit"
    assert result.center_patch_yx is not None
    assert result.center_image_yx is not None
    assert np.isclose(result.center_patch_yx[0], y0_true, atol=0.1)
    assert np.isclose(result.center_patch_yx[1], x0_true, atol=0.1)
    assert np.isclose(result.center_image_yx[0], 40 + y0_true, atol=0.1)
    assert np.isclose(result.center_image_yx[1], 60 + x0_true, atol=0.1)
    assert result.center_std_yx is not None
    assert result.amplitude is not None
    assert result.sigma_y is not None
    assert result.sigma_x is not None
    assert result.model_patch is not None
    assert result.model_patch.shape == patch.shape


@pytest.mark.skipif(not SCIPY_AVAILABLE, reason="SciPy not available, skipping Gaussian fit tests")
def test_atommapper_gaussian_adapter_handles_flat_patch_without_crashing():
    patch = np.ones((9, 9), dtype=float)

    result = fit_gaussian_to_roi_patch(patch)

    assert result.model is LocalFitModelType.GAUSSIAN
    assert result.success is False
    assert result.center_patch_yx is None
    assert result.center_image_yx is None
    assert result.error_message is not None
    assert result.model_patch is None


def test_atommapper_gaussian_adapter_rejects_non_2d_patch():
    patch = np.arange(10, dtype=float)

    result = fit_gaussian_to_roi_patch(patch)

    assert result.model is LocalFitModelType.GAUSSIAN
    assert result.success is False
    assert result.error_message is not None
    assert "Expected a 2D ROI patch" in result.error_message
    assert result.model_patch is None


def test_common_fit_interface_returns_structured_failure_for_unknown_models():
    patch = np.ones((9, 9), dtype=float)

    result = fit_local_peak(
        LocalFitRequest(
            model="mystery-model",  # type: ignore[arg-type]
            roi_patch=patch,
        )
    )

    assert result.model == "mystery-model"
    assert result.success is False
    assert result.error_message == "Unsupported local fit model: 'mystery-model'."
    assert result.model_patch is None


def test_gaussian_adapter_builds_curve_fit_options_from_fit_settings(monkeypatch):
    captured: dict[str, object] = {}

    def fake_fit_2d_gaussian_on_patch(
        roi_patch,
        *,
        roi_origin_yx=(0, 0),
        fit_mask=None,
        initial_params=None,
        parameter_bounds=None,
        max_nfev=0,
        compute_uncertainty=True,
    ):
        captured["roi_origin_yx"] = roi_origin_yx
        captured["fit_mask"] = None if fit_mask is None else np.asarray(fit_mask, dtype=bool).copy()
        captured["initial_params"] = list(initial_params)
        captured["parameter_bounds"] = parameter_bounds
        captured["max_nfev"] = max_nfev
        captured["compute_uncertainty"] = compute_uncertainty
        return PeakRefinementResult(
            center=(13.5, 24.5),
            center_std=(0.2, 0.3),
            method="gaussian_fit",
            success=True,
            roi_patch=np.asarray(roi_patch, dtype=float).copy(),
            noise_sigma=0.1,
            popt=np.array([11.0, 3.5, 4.5, 1.7, 2.2, 0.15, 0.9], dtype=float),
            pcov=np.eye(7, dtype=float),
            metadata={},
        )

    monkeypatch.setattr("AtomMapper.app.gaussian_fit.fit_2d_gaussian_on_patch", fake_fit_2d_gaussian_on_patch)

    fit_settings = FitSettingsState(
        common=CommonFitSettings(
            compute_uncertainty=False,
            use_custom_initial_guess=True,
            use_custom_bounds=True,
            max_nfev=3200,
        ),
        gaussian=GaussianFitSettings(
            amplitude_init=50.0,
            center_y_init=3.0,
            center_x_init=4.0,
            sigma_y_init=1.5,
            sigma_x_init=2.5,
            theta_init_rad=0.2,
            offset_init=0.7,
            amplitude_bounds=ParameterBounds(0.0, 100.0),
            center_y_bounds=ParameterBounds(1.0, 7.0),
            center_x_bounds=ParameterBounds(2.0, 8.0),
            sigma_y_bounds=ParameterBounds(0.5, 4.0),
            sigma_x_bounds=ParameterBounds(0.75, 5.0),
            theta_bounds_rad=ParameterBounds(-0.5, 0.5),
            offset_bounds=ParameterBounds(-1.0, 2.0),
        ),
    )

    result = fit_local_peak(
        LocalFitRequest(
            model=LocalFitModelType.GAUSSIAN,
            roi_patch=np.arange(81, dtype=float).reshape((9, 9)),
            roi_origin_yx=(10, 20),
            compute_uncertainty=fit_settings.common.compute_uncertainty,
            fit_settings_state=fit_settings,
        )
    )

    assert captured["roi_origin_yx"] == (10, 20)
    assert captured["fit_mask"] is None
    assert captured["initial_params"] == pytest.approx([50.0, 3.0, 4.0, 1.5, 2.5, 0.2, 0.7])
    lower, upper = captured["parameter_bounds"]
    assert lower == pytest.approx([0.0, 1.0, 2.0, 0.5, 0.75, -0.5, -1.0])
    assert upper == pytest.approx([100.0, 7.0, 8.0, 4.0, 5.0, 0.5, 2.0])
    assert captured["max_nfev"] == 3200
    assert captured["compute_uncertainty"] is False
    assert result.success is True
    assert result.center_patch_yx == pytest.approx((3.5, 4.5))


def test_gaussian_adapter_passes_fit_mask_to_backend(monkeypatch):
    captured: dict[str, object] = {}

    def fake_fit_2d_gaussian_on_patch(
        roi_patch,
        *,
        roi_origin_yx=(0, 0),
        fit_mask=None,
        initial_params=None,
        parameter_bounds=None,
        max_nfev=0,
        compute_uncertainty=True,
    ):
        captured["fit_mask"] = None if fit_mask is None else np.asarray(fit_mask, dtype=bool).copy()
        return PeakRefinementResult(
            center=(11.5, 22.5),
            center_std=(0.2, 0.3),
            method="gaussian_fit",
            success=True,
            roi_patch=np.asarray(roi_patch, dtype=float).copy(),
            noise_sigma=0.1,
            popt=np.array([10.0, 3.5, 4.5, 1.5, 1.7, 0.1, 0.8], dtype=float),
            pcov=np.eye(7, dtype=float),
            metadata={},
        )

    monkeypatch.setattr("AtomMapper.app.gaussian_fit.fit_2d_gaussian_on_patch", fake_fit_2d_gaussian_on_patch)

    fit_mask = np.array(
        [
            [1, 0, 1],
            [0, 1, 0],
            [1, 1, 0],
        ],
        dtype=bool,
    )
    result = fit_local_peak(
        LocalFitRequest(
            model=LocalFitModelType.GAUSSIAN,
            roi_patch=np.arange(9, dtype=float).reshape((3, 3)),
            fit_mask=fit_mask,
        )
    )

    assert captured["fit_mask"] is not None
    assert np.array_equal(captured["fit_mask"], fit_mask)
    assert result.fit_mask is not None
    assert np.array_equal(result.fit_mask, fit_mask)


def test_gaussian_default_initial_guess_uses_masked_peak(monkeypatch):
    captured: dict[str, object] = {}

    def fake_fit_2d_gaussian_on_patch(
        roi_patch,
        *,
        roi_origin_yx=(0, 0),
        fit_mask=None,
        initial_params=None,
        parameter_bounds=None,
        max_nfev=0,
        compute_uncertainty=True,
    ):
        captured["initial_params"] = list(initial_params)
        return PeakRefinementResult(
            center=(10.0, 20.0),
            center_std=(0.1, 0.1),
            method="gaussian_fit",
            success=True,
            roi_patch=np.asarray(roi_patch, dtype=float).copy(),
            noise_sigma=0.1,
            popt=np.array([8.0, 7.0, 7.0, 1.0, 1.0, 0.0, 0.0], dtype=float),
            pcov=np.eye(7, dtype=float),
            metadata={},
        )

    monkeypatch.setattr("AtomMapper.app.gaussian_fit.fit_2d_gaussian_on_patch", fake_fit_2d_gaussian_on_patch)

    patch = np.zeros((9, 9), dtype=float)
    patch[1, 1] = 50.0
    patch[7, 7] = 20.0
    fit_mask = np.zeros((9, 9), dtype=bool)
    fit_mask[5:, 5:] = True

    fit_local_peak(
        LocalFitRequest(
            model=LocalFitModelType.GAUSSIAN,
            roi_patch=patch,
            fit_mask=fit_mask,
            fit_settings_state=FitSettingsState(
                common=CommonFitSettings(
                    compute_uncertainty=False,
                    use_custom_initial_guess=False,
                )
            ),
        )
    )

    assert captured["initial_params"] == pytest.approx([20.0, 7.0, 7.0, 1.0, 1.0, 0.0, 0.0])


@pytest.mark.skipif(not SCIPY_AVAILABLE, reason="SciPy not available, skipping Lorentzian fit tests")
def test_atommapper_lorentzian_adapter_returns_patch_and_image_coordinates():
    patch = _make_lorentzian_patch()

    result = fit_local_peak(
        LocalFitRequest(
            model=LocalFitModelType.LORENTZIAN,
            roi_patch=patch,
            roi_origin_yx=(30, 50),
        )
    )

    assert result.model is LocalFitModelType.LORENTZIAN
    assert result.success is True
    assert result.error_message is None
    assert result.method == "lorentzian_fit"
    assert result.center_patch_yx is not None
    assert result.center_image_yx is not None
    assert np.isclose(result.center_patch_yx[0], 7.2, atol=0.15)
    assert np.isclose(result.center_patch_yx[1], 9.1, atol=0.15)
    assert np.isclose(result.center_image_yx[0], 37.2, atol=0.15)
    assert np.isclose(result.center_image_yx[1], 59.1, atol=0.15)
    assert result.center_std_yx is not None
    assert result.amplitude is not None
    assert result.sigma_y is not None
    assert result.sigma_x is not None
    assert result.model_patch is not None
    assert result.model_patch.shape == patch.shape


@pytest.mark.skipif(not SCIPY_AVAILABLE, reason="SciPy not available, skipping Lorentzian fit tests")
def test_lorentzian_fit_prefers_peak_inside_mask_for_overlapping_peaks():
    rows = cols = 25
    y_grid, x_grid = np.mgrid[0:rows, 0:cols]

    def _lorentzian(center_y: float, center_x: float, amplitude: float) -> np.ndarray:
        return amplitude / (
            1.0
            + ((y_grid - center_y) / 1.5) ** 2
            + ((x_grid - center_x) / 1.8) ** 2
        )

    patch = 0.8 + _lorentzian(12.0, 9.0, 14.0) + _lorentzian(12.0, 16.0, 26.0)
    fit_mask = np.zeros((rows, cols), dtype=bool)
    fit_mask[:, :13] = True

    result = fit_local_peak(
        LocalFitRequest(
            model=LocalFitModelType.LORENTZIAN,
            roi_patch=patch,
            fit_mask=fit_mask,
            fit_settings_state=FitSettingsState(
                model=LocalFitModelType.LORENTZIAN,
                common=CommonFitSettings(compute_uncertainty=False),
            ),
        )
    )

    assert result.success is True
    assert result.center_patch_yx is not None
    assert result.center_patch_yx[0] == pytest.approx(12.0, abs=0.4)
    assert result.center_patch_yx[1] == pytest.approx(9.0, abs=0.6)


def test_lorentzian_adapter_builds_curve_fit_options_from_fit_settings(monkeypatch):
    captured: dict[str, object] = {}

    def fake_fit_2d_lorentzian_on_patch(
        roi_patch,
        *,
        roi_origin_yx=(0, 0),
        fit_mask=None,
        initial_params=None,
        parameter_bounds=None,
        max_nfev=0,
        compute_uncertainty=True,
    ):
        captured["roi_origin_yx"] = roi_origin_yx
        captured["fit_mask"] = None if fit_mask is None else np.asarray(fit_mask, dtype=bool).copy()
        captured["initial_params"] = list(initial_params)
        captured["parameter_bounds"] = parameter_bounds
        captured["max_nfev"] = max_nfev
        captured["compute_uncertainty"] = compute_uncertainty
        return PeakRefinementResult(
            center=(18.5, 29.5),
            center_std=(0.25, 0.35),
            method="lorentzian_fit",
            success=True,
            roi_patch=np.asarray(roi_patch, dtype=float).copy(),
            noise_sigma=0.1,
            popt=np.array([9.0, 4.5, 5.5, 1.4, 2.1, 0.18, 0.8], dtype=float),
            pcov=np.eye(7, dtype=float),
            metadata={},
        )

    monkeypatch.setattr("AtomMapper.app.gaussian_fit.fit_2d_lorentzian_on_patch", fake_fit_2d_lorentzian_on_patch)

    fit_settings = FitSettingsState(
        model=LocalFitModelType.LORENTZIAN,
        common=CommonFitSettings(
            compute_uncertainty=False,
            use_custom_initial_guess=True,
            use_custom_bounds=True,
            max_nfev=4100,
        ),
        lorentzian=LorentzianFitSettings(
            amplitude_init=22.0,
            center_y_init=4.0,
            center_x_init=5.0,
            gamma_y_init=1.25,
            gamma_x_init=2.5,
            theta_init_rad=0.15,
            offset_init=0.4,
            amplitude_bounds=ParameterBounds(0.0, 30.0),
            center_y_bounds=ParameterBounds(2.0, 7.0),
            center_x_bounds=ParameterBounds(3.0, 8.0),
            gamma_y_bounds=ParameterBounds(0.5, 4.0),
            gamma_x_bounds=ParameterBounds(0.75, 5.0),
            theta_bounds_rad=ParameterBounds(-0.4, 0.4),
            offset_bounds=ParameterBounds(-1.0, 2.0),
        ),
    )

    result = fit_local_peak(
        LocalFitRequest(
            model=LocalFitModelType.LORENTZIAN,
            roi_patch=np.arange(81, dtype=float).reshape((9, 9)),
            roi_origin_yx=(14, 24),
            compute_uncertainty=fit_settings.common.compute_uncertainty,
            fit_settings_state=fit_settings,
        )
    )

    assert captured["roi_origin_yx"] == (14, 24)
    assert captured["fit_mask"] is None
    assert captured["initial_params"] == pytest.approx([22.0, 4.0, 5.0, 1.25, 2.5, 0.15, 0.4])
    lower, upper = captured["parameter_bounds"]
    assert lower == pytest.approx([0.0, 2.0, 3.0, 0.5, 0.75, -0.4, -1.0])
    assert upper == pytest.approx([30.0, 7.0, 8.0, 4.0, 5.0, 0.4, 2.0])
    assert captured["max_nfev"] == 4100
    assert captured["compute_uncertainty"] is False
    assert result.success is True
    assert result.center_patch_yx == pytest.approx((4.5, 5.5))


def test_lorentzian_default_initial_guess_uses_masked_peak(monkeypatch):
    captured: dict[str, object] = {}

    def fake_fit_2d_lorentzian_on_patch(
        roi_patch,
        *,
        roi_origin_yx=(0, 0),
        fit_mask=None,
        initial_params=None,
        parameter_bounds=None,
        max_nfev=0,
        compute_uncertainty=True,
    ):
        captured["initial_params"] = list(initial_params)
        return PeakRefinementResult(
            center=(10.0, 20.0),
            center_std=(0.1, 0.1),
            method="lorentzian_fit",
            success=True,
            roi_patch=np.asarray(roi_patch, dtype=float).copy(),
            noise_sigma=0.1,
            popt=np.array([8.0, 7.0, 7.0, 1.0, 1.0, 0.0, 0.0], dtype=float),
            pcov=np.eye(7, dtype=float),
            metadata={},
        )

    monkeypatch.setattr(
        "AtomMapper.app.gaussian_fit.fit_2d_lorentzian_on_patch",
        fake_fit_2d_lorentzian_on_patch,
    )

    patch = np.zeros((9, 9), dtype=float)
    patch[1, 1] = 60.0
    patch[7, 7] = 18.0
    fit_mask = np.zeros((9, 9), dtype=bool)
    fit_mask[5:, 5:] = True

    fit_local_peak(
        LocalFitRequest(
            model=LocalFitModelType.LORENTZIAN,
            roi_patch=patch,
            fit_mask=fit_mask,
            fit_settings_state=FitSettingsState(
                model=LocalFitModelType.LORENTZIAN,
                common=CommonFitSettings(
                    compute_uncertainty=False,
                    use_custom_initial_guess=False,
                ),
            ),
        )
    )

    assert captured["initial_params"] == pytest.approx([18.0, 7.0, 7.0, 1.0, 1.0, 0.0, 0.0])


@pytest.mark.skipif(not SCIPY_AVAILABLE, reason="SciPy not available, skipping Voigt fit tests")
def test_atommapper_voigt_adapter_returns_patch_and_image_coordinates():
    patch = _make_voigt_patch()

    result = fit_local_peak(
        LocalFitRequest(
            model=LocalFitModelType.VOIGT,
            roi_patch=patch,
            roi_origin_yx=(25, 45),
        )
    )

    assert result.model is LocalFitModelType.VOIGT
    assert result.success is True
    assert result.error_message is None
    assert result.method == "voigt_fit"
    assert result.center_patch_yx is not None
    assert result.center_image_yx is not None
    assert np.isclose(result.center_patch_yx[0], 7.1, atol=0.2)
    assert np.isclose(result.center_patch_yx[1], 9.3, atol=0.2)
    assert np.isclose(result.center_image_yx[0], 32.1, atol=0.2)
    assert np.isclose(result.center_image_yx[1], 54.3, atol=0.2)
    assert result.center_std_yx is not None
    assert result.amplitude is not None
    assert result.sigma_y is not None
    assert result.sigma_x is not None
    assert result.model_patch is not None
    assert result.model_patch.shape == patch.shape
    assert result.shape_parameters["gamma_y"] is not None
    assert result.shape_parameters["gamma_x"] is not None


def test_voigt_adapter_builds_curve_fit_options_from_fit_settings(monkeypatch):
    captured: dict[str, object] = {}

    def fake_fit_2d_voigt_on_patch(
        roi_patch,
        *,
        roi_origin_yx=(0, 0),
        fit_mask=None,
        initial_params=None,
        parameter_bounds=None,
        max_nfev=0,
        compute_uncertainty=True,
    ):
        captured["roi_origin_yx"] = roi_origin_yx
        captured["fit_mask"] = None if fit_mask is None else np.asarray(fit_mask, dtype=bool).copy()
        captured["initial_params"] = list(initial_params)
        captured["parameter_bounds"] = parameter_bounds
        captured["max_nfev"] = max_nfev
        captured["compute_uncertainty"] = compute_uncertainty
        return PeakRefinementResult(
            center=(17.5, 28.5),
            center_std=(0.22, 0.31),
            method="voigt_fit",
            success=True,
            roi_patch=np.asarray(roi_patch, dtype=float).copy(),
            noise_sigma=0.1,
            popt=np.array([8.0, 4.5, 5.5, 1.2, 2.0, 0.6, 1.0, 0.12, 0.7], dtype=float),
            pcov=np.eye(9, dtype=float),
            metadata={},
        )

    monkeypatch.setattr("AtomMapper.app.gaussian_fit.fit_2d_voigt_on_patch", fake_fit_2d_voigt_on_patch)

    fit_settings = FitSettingsState(
        model=LocalFitModelType.VOIGT,
        common=CommonFitSettings(
            compute_uncertainty=False,
            use_custom_initial_guess=True,
            use_custom_bounds=True,
            max_nfev=5100,
        ),
        voigt=VoigtFitSettings(
            amplitude_init=21.0,
            center_y_init=4.0,
            center_x_init=5.0,
            sigma_y_init=1.1,
            sigma_x_init=2.2,
            gamma_y_init=0.7,
            gamma_x_init=1.3,
            theta_init_rad=0.14,
            offset_init=0.35,
            amplitude_bounds=ParameterBounds(0.0, 40.0),
            center_y_bounds=ParameterBounds(2.0, 7.0),
            center_x_bounds=ParameterBounds(3.0, 8.0),
            sigma_y_bounds=ParameterBounds(0.3, 3.5),
            sigma_x_bounds=ParameterBounds(0.5, 4.5),
            gamma_y_bounds=ParameterBounds(0.2, 2.0),
            gamma_x_bounds=ParameterBounds(0.2, 3.0),
            theta_bounds_rad=ParameterBounds(-0.4, 0.4),
            offset_bounds=ParameterBounds(-1.0, 2.0),
        ),
    )

    result = fit_local_peak(
        LocalFitRequest(
            model=LocalFitModelType.VOIGT,
            roi_patch=np.arange(81, dtype=float).reshape((9, 9)),
            roi_origin_yx=(13, 23),
            compute_uncertainty=fit_settings.common.compute_uncertainty,
            fit_settings_state=fit_settings,
        )
    )

    assert captured["roi_origin_yx"] == (13, 23)
    assert captured["fit_mask"] is None
    assert captured["initial_params"] == pytest.approx([21.0, 4.0, 5.0, 1.1, 2.2, 0.7, 1.3, 0.14, 0.35])
    lower, upper = captured["parameter_bounds"]
    assert lower == pytest.approx([0.0, 2.0, 3.0, 0.3, 0.5, 0.2, 0.2, -0.4, -1.0])
    assert upper == pytest.approx([40.0, 7.0, 8.0, 3.5, 4.5, 2.0, 3.0, 0.4, 2.0])
    assert captured["max_nfev"] == 5100
    assert captured["compute_uncertainty"] is False
    assert result.success is True
    assert result.center_patch_yx == pytest.approx((4.5, 5.5))
    assert result.shape_parameters["gamma_y"] == pytest.approx(0.6)
    assert result.shape_parameters["gamma_x"] == pytest.approx(1.0)


def test_voigt_default_initial_guess_uses_masked_peak(monkeypatch):
    captured: dict[str, object] = {}

    def fake_fit_2d_voigt_on_patch(
        roi_patch,
        *,
        roi_origin_yx=(0, 0),
        fit_mask=None,
        initial_params=None,
        parameter_bounds=None,
        max_nfev=0,
        compute_uncertainty=True,
    ):
        captured["initial_params"] = list(initial_params)
        return PeakRefinementResult(
            center=(10.0, 20.0),
            center_std=(0.1, 0.1),
            method="voigt_fit",
            success=True,
            roi_patch=np.asarray(roi_patch, dtype=float).copy(),
            noise_sigma=0.1,
            popt=np.array([8.0, 7.0, 7.0, 1.0, 1.0, 0.6, 0.6, 0.0, 0.0], dtype=float),
            pcov=np.eye(9, dtype=float),
            metadata={},
        )

    monkeypatch.setattr("AtomMapper.app.gaussian_fit.fit_2d_voigt_on_patch", fake_fit_2d_voigt_on_patch)

    patch = np.zeros((10, 10), dtype=float)
    patch[1, 1] = 55.0
    patch[8, 8] = 16.0
    fit_mask = np.zeros((10, 10), dtype=bool)
    fit_mask[6:, 6:] = True

    fit_local_peak(
        LocalFitRequest(
            model=LocalFitModelType.VOIGT,
            roi_patch=patch,
            fit_mask=fit_mask,
            fit_settings_state=FitSettingsState(
                model=LocalFitModelType.VOIGT,
                common=CommonFitSettings(
                    compute_uncertainty=False,
                    use_custom_initial_guess=False,
                ),
            ),
        )
    )

    assert captured["initial_params"] == pytest.approx([16.0, 8.0, 8.0, 0.8, 0.8, 0.6666666667, 0.6666666667, 0.0, 0.0])
