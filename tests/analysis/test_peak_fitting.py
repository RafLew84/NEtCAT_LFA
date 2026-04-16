# tests/analysis/test_peak_fitting.py
"""
Unit tests for peak fitting functions in lfa.analysis.peak_fitting.
"""
import sys
import warnings
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import logging

# Importuj funkcje do testowania
try:
    from lfa.analysis.peak_fitting import (
        OptimizeWarning,
        SCIPY_AVAILABLE,
        find_max_pixel_in_roi,
        fit_2d_gaussian_on_patch,
        fit_2d_gaussian_in_roi,
        fit_2d_lorentzian_on_patch,
        fit_2d_voigt_on_patch,
    )
    import lfa.analysis.peak_fitting as peak_fitting_module
except ImportError:
    pytest.fail("Could not import from lfa.analysis.peak_fitting", pytrace=False)

logger = logging.getLogger(__name__)

# --- Fixtures ---

@pytest.fixture
def simple_peak_image() -> np.ndarray:
    """A 10x10 image with a single distinct peak."""
    img = np.zeros((10, 10), dtype=np.float32)
    img[5, 5] = 100.0  # Peak at (5, 5)
    img[5, 6] = 50.0
    img[6, 5] = 50.0
    return img

@pytest.fixture
def noisy_peak_image() -> np.ndarray:
    """A 10x10 image with a peak and some noise."""
    img = np.zeros((10, 10), dtype=np.float32)
    img[5, 5] = 100.0
    img[4:7, 4:7] += 20.0 # Broaden the peak slightly
    # Add some random noise
    rng = np.random.default_rng(seed=42) # for reproducibility
    img += rng.normal(0, 5, img.shape).astype(np.float32)
    img = np.clip(img, 0, None) # Ensure non-negative values
    return img

@pytest.fixture
def gaussian_peak_image() -> tuple[np.ndarray, tuple[float, float], tuple[float, float]]:
    """A 20x20 image with a 2D Gaussian peak."""
    size = 20
    Y, X = np.mgrid[0:size, 0:size]
    amplitude = 100.0
    y0, x0 = 9.5, 9.5  # Center of the Gaussian (sub-pixel)
    sigma_y, sigma_x = 2.0, 2.0
    offset = 10.0
    theta = 0 # No rotation

    # Re-use the _gaussian_2d definition for consistency (or define a simpler one here for testing)
    a = (np.cos(theta)**2)/(2*sigma_y**2) + (np.sin(theta)**2)/(2*sigma_x**2)
    b = -(np.sin(2*theta))/(4*sigma_y**2) + (np.sin(2*theta))/(4*sigma_x**2)
    c = (np.sin(theta)**2)/(2*sigma_y**2) + (np.cos(theta)**2)/(2*sigma_x**2)
    g = offset + amplitude * np.exp( - (a*((Y-y0)**2) + 2*b*(Y-y0)*(X-x0) + c*((X-x0)**2)))
    return g.astype(np.float32), (y0, x0), (sigma_y, sigma_x)


@pytest.fixture
def lorentzian_peak_image() -> tuple[np.ndarray, tuple[float, float], tuple[float, float]]:
    """A 20x20 image with a rotated 2D Lorentzian peak."""

    size = 20
    y_grid, x_grid = np.mgrid[0:size, 0:size]
    amplitude = 90.0
    y0, x0 = 9.2, 10.4
    gamma_y, gamma_x = 1.8, 2.4
    offset = 4.0
    theta = 0.15

    dy = y_grid - y0
    dx = x_grid - x0
    y_rot = np.cos(theta) * dy + np.sin(theta) * dx
    x_rot = -np.sin(theta) * dy + np.cos(theta) * dx
    q = (y_rot / gamma_y) ** 2 + (x_rot / gamma_x) ** 2
    image = offset + amplitude / (1.0 + q)
    return image.astype(np.float32), (y0, x0), (gamma_y, gamma_x)


@pytest.fixture
def voigt_peak_image() -> tuple[np.ndarray, tuple[float, float], tuple[float, float, float, float]]:
    """A 20x20 image with a rotated separable 2D Voigt peak."""

    from scipy.special import voigt_profile

    size = 20
    y_grid, x_grid = np.mgrid[0:size, 0:size]
    amplitude = 85.0
    y0, x0 = 9.1, 10.2
    sigma_y, sigma_x = 1.3, 1.9
    gamma_y, gamma_x = 0.7, 1.1
    offset = 3.0
    theta = 0.12

    dy = y_grid - y0
    dx = x_grid - x0
    y_rot = np.cos(theta) * dy + np.sin(theta) * dx
    x_rot = -np.sin(theta) * dy + np.cos(theta) * dx
    profile_y = voigt_profile(y_rot, sigma_y, gamma_y)
    profile_x = voigt_profile(x_rot, sigma_x, gamma_x)
    profile_y /= float(voigt_profile(np.array([0.0]), sigma_y, gamma_y)[0])
    profile_x /= float(voigt_profile(np.array([0.0]), sigma_x, gamma_x)[0])
    image = offset + amplitude * profile_y * profile_x
    return image.astype(np.float32), (y0, x0), (sigma_y, sigma_x, gamma_y, gamma_x)


# --- Tests for find_max_pixel_in_roi ---

def test_find_max_pixel_center_exact(simple_peak_image):
    """Test when the click is exactly on the peak."""
    img = simple_peak_image
    center_yx = (5, 5)
    roi_radius = 2
    refined_y, refined_x = find_max_pixel_in_roi(img, center_yx, roi_radius)
    assert (refined_y, refined_x) == (5, 5)

def test_find_max_pixel_center_offset(simple_peak_image):
    """Test when the click is near the peak."""
    img = simple_peak_image
    center_yx = (4, 4) # Click offset from true peak
    roi_radius = 2     # ROI (3x3 to 7x7) should include the true peak (5,5)
    refined_y, refined_x = find_max_pixel_in_roi(img, center_yx, roi_radius)
    assert (refined_y, refined_x) == (5, 5)

def test_find_max_pixel_roi_edge(simple_peak_image):
    """Test when the ROI is at the edge of the image."""
    img = simple_peak_image # Peak at (5,5)
    center_yx = (0, 0)    # Click at corner
    roi_radius = 2        # ROI will be 3x3 or 5x5, clipped
    refined_y, refined_x = find_max_pixel_in_roi(img, center_yx, roi_radius)
    # In this case, (0,0) itself is the max within the small clipped ROI if no other peaks are there
    # Check values within the ROI:
    # img[0:3, 0:3] are all zeros if radius is 1 (3x3 roi)
    # img[0:5, 0:5] are all zeros if radius is 2 (5x5 roi)
    # So refined should be the first pixel of the ROI in case of all zeros.
    # If roi_radius = 2, y_start=0, y_end=3, x_start=0, x_end=3. Patch is img[0:3, 0:3]
    # argmax on zeros gives (0,0) within patch. refined_y = 0+0, refined_x = 0+0
    assert (refined_y, refined_x) == (0,0) # Max in the top-left corner of the small ROI

    center_yx_near_peak_edge = (5,3) # Click near peak (5,5) but ROI might clip
    refined_y_edge, refined_x_edge = find_max_pixel_in_roi(img, center_yx_near_peak_edge, roi_radius=1) # 3x3 ROI: [(4,2)-(6,4)]
    # ROI: img[4:7, 2:5]. Peak is at img[5,5] which is outside this ROI. Max within this ROI will be at the edge.
    # img[5,4] = 0, img[5,3]=0, img[5,2]=0. Max is 0.
    # img[4,3] etc.
    # Expected refined position depends on noise/actual values. For simple_peak_image, it will be a corner of ROI.
    # Let's test on noisy image where this makes more sense.
    pass


def test_find_max_pixel_noisy(noisy_peak_image):
    """Test on a noisy image."""
    img = noisy_peak_image
    # True peak is around (5, 5)
    center_yx = (5, 5)
    roi_radius = 2
    refined_y, refined_x = find_max_pixel_in_roi(img, center_yx, roi_radius)
    # The refined peak should be within the broader peak area (4:7, 4:7) due to noise
    assert 4 <= refined_y <= 6 and 4 <= refined_x <= 6

def test_find_max_pixel_invalid_input(simple_peak_image):
    """Test invalid inputs."""
    assert find_max_pixel_in_roi(None, (1,1), 1) == (1,1) # Should return original click
    assert find_max_pixel_in_roi(np.zeros(5), (1,1), 1) == (1,1) # 1D image
    # radius=0 produces a 1×1 ROI centred on the click
    refined_y, refined_x = find_max_pixel_in_roi(simple_peak_image, (0,0), roi_radius=0)
    assert (refined_y, refined_x) == (0,0)


# --- Tests for fit_2d_gaussian_in_roi ---

@pytest.mark.skipif(not SCIPY_AVAILABLE, reason="SciPy not available, skipping Gaussian fit tests")
def test_fit_gaussian_perfect_peak(gaussian_peak_image):
    """Test fitting on a synthetic Gaussian peak."""
    img, (y0_true, x0_true), _ = gaussian_peak_image
    # Click exactly at the integer coordinates nearest to the true sub-pixel peak
    center_yx_click = (int(round(y0_true)), int(round(x0_true)))
    roi_radius = 5 # ROI radius for fitting

    fit_result = fit_2d_gaussian_in_roi(img, center_yx_click, roi_radius)
    assert fit_result is not None, "Gaussian fit failed to return a result."
    y0_fit, x0_fit = fit_result.center
    assert fit_result.center_std is not None and all(sigma >= 0 for sigma in fit_result.center_std)

    # Check if the fitted center is close to the true sub-pixel center
    assert np.isclose(y0_fit, y0_true, atol=0.1), f"Fitted y0 ({y0_fit:.2f}) far from true y0 ({y0_true:.2f})"
    assert np.isclose(x0_fit, x0_true, atol=0.1), f"Fitted x0 ({x0_fit:.2f}) far from true x0 ({x0_true:.2f})"

@pytest.mark.skipif(not SCIPY_AVAILABLE, reason="SciPy not available, skipping Gaussian fit tests")
def test_fit_gaussian_noisy_peak(noisy_peak_image, gaussian_peak_image):
    """Test fitting on a peak with some noise."""
    # We use noisy_peak_image for the noise characteristics, but for checking accuracy,
    # it's better to know the true underlying peak center.
    # Let's create a noisy version of gaussian_peak_image
    clean_gauss_img, (y0_true, x0_true), _ = gaussian_peak_image
    rng = np.random.default_rng(seed=12)
    noisy_gauss_img = clean_gauss_img + rng.normal(0, clean_gauss_img.max()*0.05, clean_gauss_img.shape).astype(np.float32)

    center_yx_click = (int(round(y0_true)), int(round(x0_true)))
    roi_radius = 5

    fit_result = fit_2d_gaussian_in_roi(noisy_gauss_img, center_yx_click, roi_radius)
    assert fit_result is not None
    y0_fit, x0_fit = fit_result.center

    # Expect slightly larger tolerance due to noise
    assert np.isclose(y0_fit, y0_true, atol=0.5), f"Fitted y0 ({y0_fit:.2f}) too far from true y0 ({y0_true:.2f}) on noisy data"
    assert np.isclose(x0_fit, x0_true, atol=0.5), f"Fitted x0 ({x0_fit:.2f}) too far from true x0 ({x0_true:.2f}) on noisy data"


@pytest.mark.skipif(not SCIPY_AVAILABLE, reason="SciPy not available, skipping Gaussian fit tests")
def test_fit_gaussian_flat_roi(simple_peak_image):
    """Test fitting on an ROI with no peak (flat data)."""
    img = simple_peak_image # Has a peak at (5,5)
    # Click in a flat area, ROI should also be flat
    center_yx_click = (0, 0)
    roi_radius = 1 # Small ROI (3x3) in the flat corner
    # img[0:3, 0:3] are all zeros for simple_peak_image
    fit_result = fit_2d_gaussian_in_roi(img, center_yx_click, roi_radius)
    # curve_fit might fail (RuntimeError) or return nonsensical results.
    # The function is designed to return None on RuntimeError.
    assert fit_result is None, "Gaussian fit should fail or return None for flat ROI"


@pytest.mark.skipif(not SCIPY_AVAILABLE, reason="SciPy not available, skipping Gaussian fit tests")
def test_fit_gaussian_on_patch_suppresses_optimize_warning(monkeypatch, gaussian_peak_image):
    patch_source, _, _ = gaussian_peak_image
    roi_patch = np.asarray(patch_source[4:15, 4:15], dtype=float)

    def _fake_curve_fit(*args, **kwargs):
        warnings.warn(
            "Covariance of the parameters could not be estimated",
            OptimizeWarning,
        )
        return (
            np.array([100.0, 5.5, 5.5, 1.8, 1.8, 0.0, 10.0], dtype=float),
            np.full((7, 7), np.inf, dtype=float),
        )

    monkeypatch.setattr(peak_fitting_module, "curve_fit", _fake_curve_fit)
    monkeypatch.setattr(
        peak_fitting_module,
        "_monte_carlo_uncertainty",
        lambda *args, **kwargs: (0.2, 0.3),
    )

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        fit_result = fit_2d_gaussian_on_patch(roi_patch, roi_origin_yx=(4, 4))

    assert fit_result is not None
    assert fit_result.success is True
    assert not any(issubclass(item.category, OptimizeWarning) for item in caught)


@pytest.mark.skipif(not SCIPY_AVAILABLE, reason="SciPy not available, skipping Gaussian fit tests")
def test_gaussian_fit_estimator_suppresses_optimize_warning(monkeypatch, gaussian_peak_image):
    patch_source, _, _ = gaussian_peak_image
    roi_patch = np.asarray(patch_source[4:15, 4:15], dtype=float)

    def _fake_curve_fit(*args, **kwargs):
        warnings.warn(
            "Covariance of the parameters could not be estimated",
            OptimizeWarning,
        )
        return (
            np.array([100.0, 5.5, 5.5, 1.8, 1.8, 0.0, 10.0], dtype=float),
            np.full((7, 7), np.inf, dtype=float),
        )

    monkeypatch.setattr(peak_fitting_module, "curve_fit", _fake_curve_fit)

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        center = peak_fitting_module._gaussian_fit_estimator(
            roi_patch,
            4,
            4,
            [100.0, 5.5, 5.5, 1.8, 1.8, 0.0, 10.0],
        )

    assert center is not None
    assert not any(issubclass(item.category, OptimizeWarning) for item in caught)

@pytest.mark.skipif(not SCIPY_AVAILABLE, reason="SciPy not available, skipping Gaussian fit tests")
def test_fit_gaussian_invalid_input(simple_peak_image):
    """Test invalid inputs for Gaussian fitting."""
    assert fit_2d_gaussian_in_roi(None, (1,1), 1) is None
    assert fit_2d_gaussian_in_roi(np.zeros(5), (1,1), 1) is None # 1D image
    # ROI completely outside image data.
    # y_start = max(0, 100 - 1) = 99. y_end = min(10, 100 + 1 + 1) = 10. y_start > y_end.
    assert fit_2d_gaussian_in_roi(simple_peak_image, (100,100), 1) is None

@pytest.mark.skipif(not SCIPY_AVAILABLE, reason="SciPy not available, skipping Gaussian fit tests")
def test_fit_gaussian_small_roi_on_peak(gaussian_peak_image):
    """Test fitting with a very small ROI centered on a broad peak."""
    img, (y0_true, x0_true), (sigma_y, sigma_x) = gaussian_peak_image
    center_yx_click = (int(round(y0_true)), int(round(x0_true)))
    # ROI radius 1 gives a 3x3 patch. If sigma is larger (e.g. 2.0),
    # this patch might not capture enough of the Gaussian shape for a good fit.
    roi_radius = 1 # Very small ROI (3x3)

    # Ensure sigmas are larger than roi_radius for this test to be meaningful
    assert sigma_y > roi_radius and sigma_x > roi_radius, "For this test, peak sigma should be larger than ROI radius"

    fit_result = fit_2d_gaussian_in_roi(img, center_yx_click, roi_radius)
    # The fit might still succeed but could be less accurate or even unstable.
    # It's hard to assert specific failure without knowing curve_fit's exact behavior.
    # For now, check it returns something, or None if it fails robustly.
    if fit_result is not None:
        logger.info(f"Small ROI fit result: {fit_result.center}. True: ({y0_true}, {x0_true})")
        # Accuracy expectations are intentionally loose for this exploratory case.
    else:
        # This is also an acceptable outcome if the fit fails due to insufficient data
        logger.info("Small ROI fit returned None, indicating robust failure or poor fit quality.")
    # No strict assertion on outcome, more of an exploratory test.


@pytest.mark.skipif(not SCIPY_AVAILABLE, reason="SciPy not available, skipping Gaussian fit tests")
def test_fit_gaussian_on_patch_perfect_peak(gaussian_peak_image):
    img, (y0_true, x0_true), _ = gaussian_peak_image

    fit_result = fit_2d_gaussian_on_patch(img, roi_origin_yx=(20, 30))

    assert fit_result is not None
    assert np.isclose(fit_result.center[0], 20 + y0_true, atol=0.1)
    assert np.isclose(fit_result.center[1], 30 + x0_true, atol=0.1)
    assert fit_result.center_std is not None
    assert fit_result.method == "gaussian_fit"


@pytest.mark.skipif(not SCIPY_AVAILABLE, reason="SciPy not available, skipping Gaussian fit tests")
def test_fit_gaussian_on_patch_flat_roi_returns_none():
    img = np.zeros((8, 8), dtype=float)
    assert fit_2d_gaussian_on_patch(img) is None


@pytest.mark.skipif(not SCIPY_AVAILABLE, reason="SciPy not available, skipping Gaussian fit tests")
def test_fit_gaussian_on_patch_accepts_custom_initial_params_bounds_and_maxfev(gaussian_peak_image):
    img, (y0_true, x0_true), _ = gaussian_peak_image

    fit_result = fit_2d_gaussian_on_patch(
        img,
        roi_origin_yx=(10, 20),
        initial_params=[95.0, 9.0, 10.0, 2.3, 1.9, 0.05, 8.0],
        parameter_bounds=(
            [0.0, 6.0, 6.0, 0.5, 0.5, -np.pi / 2.0, 0.0],
            [200.0, 13.0, 13.0, 5.0, 5.0, np.pi / 2.0, 30.0],
        ),
        max_nfev=3200,
        compute_uncertainty=False,
    )

    assert fit_result is not None
    assert np.isclose(fit_result.center[0], 10 + y0_true, atol=0.1)
    assert np.isclose(fit_result.center[1], 20 + x0_true, atol=0.1)
    assert fit_result.metadata["initial_params"] == pytest.approx((95.0, 9.0, 10.0, 2.3, 1.9, 0.05, 8.0))
    assert fit_result.metadata["parameter_bounds"] is not None
    assert fit_result.metadata["parameter_bounds"]["lower"] == pytest.approx(
        (0.0, 6.0, 6.0, 0.5, 0.5, -np.pi / 2.0, 0.0)
    )
    assert fit_result.metadata["parameter_bounds"]["upper"] == pytest.approx(
        (200.0, 13.0, 13.0, 5.0, 5.0, np.pi / 2.0, 30.0)
    )
    assert fit_result.metadata["max_nfev"] == 3200


@pytest.mark.skipif(not SCIPY_AVAILABLE, reason="SciPy not available, skipping Gaussian fit tests")
def test_fit_gaussian_on_patch_accepts_fit_mask(gaussian_peak_image):
    img, _, _ = gaussian_peak_image
    fit_mask = np.zeros_like(img, dtype=bool)
    fit_mask[4:16, 4:16] = True

    fit_result = fit_2d_gaussian_on_patch(img, fit_mask=fit_mask)

    assert fit_result is not None
    assert fit_result.metadata["fit_mask_pixel_count"] == int(fit_mask.sum())
    assert fit_result.metadata["fit_mask_fraction"] == pytest.approx(float(fit_mask.mean()))


@pytest.mark.skipif(not SCIPY_AVAILABLE, reason="SciPy not available, skipping Gaussian fit tests")
def test_fit_gaussian_on_patch_uses_fast_max_pixel_fallback_without_monte_carlo(
    monkeypatch,
    gaussian_peak_image,
):
    img, _, _ = gaussian_peak_image

    monkeypatch.setattr(
        peak_fitting_module,
        "_curve_fit_safely",
        lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("fit failed")),
    )
    monkeypatch.setattr(
        peak_fitting_module,
        "_monte_carlo_uncertainty",
        lambda *args, **kwargs: pytest.fail("Monte Carlo fallback should not run here."),
    )

    fit_result = fit_2d_gaussian_on_patch(
        img,
        roi_origin_yx=(0, 0),
        compute_uncertainty=False,
    )

    assert fit_result is not None
    assert fit_result.success is False
    assert fit_result.method == "max_pixel_fallback"
    assert fit_result.center_std is None
    peak_y, peak_x = np.unravel_index(int(np.argmax(img)), img.shape)
    assert fit_result.center == pytest.approx((float(peak_y), float(peak_x)))


@pytest.mark.skipif(not SCIPY_AVAILABLE, reason="SciPy not available, skipping Gaussian fit tests")
def test_fit_gaussian_on_patch_with_too_small_mask_uses_fast_max_pixel_fallback(
    monkeypatch,
    gaussian_peak_image,
):
    img, _, _ = gaussian_peak_image
    fit_mask = np.zeros_like(img, dtype=bool)
    fit_mask[9, 9] = True

    monkeypatch.setattr(
        peak_fitting_module,
        "_monte_carlo_uncertainty",
        lambda *args, **kwargs: pytest.fail("Monte Carlo fallback should not run here."),
    )

    fit_result = fit_2d_gaussian_on_patch(
        img,
        roi_origin_yx=(0, 0),
        fit_mask=fit_mask,
        compute_uncertainty=False,
    )

    assert fit_result is not None
    assert fit_result.success is False
    assert fit_result.method == "max_pixel_fallback"
    assert fit_result.center_std is None
    assert fit_result.center == pytest.approx((9.0, 9.0))


@pytest.mark.skipif(not SCIPY_AVAILABLE, reason="SciPy not available, skipping Lorentzian fit tests")
def test_fit_lorentzian_on_patch_perfect_peak(lorentzian_peak_image):
    img, (y0_true, x0_true), _ = lorentzian_peak_image

    fit_result = fit_2d_lorentzian_on_patch(img, roi_origin_yx=(12, 28))

    assert fit_result is not None
    assert np.isclose(fit_result.center[0], 12 + y0_true, atol=0.15)
    assert np.isclose(fit_result.center[1], 28 + x0_true, atol=0.15)
    assert fit_result.center_std is not None
    assert fit_result.method == "lorentzian_fit"


@pytest.mark.skipif(not SCIPY_AVAILABLE, reason="SciPy not available, skipping Lorentzian fit tests")
def test_fit_lorentzian_on_patch_accepts_custom_initial_params_bounds_and_maxfev(lorentzian_peak_image):
    img, (y0_true, x0_true), _ = lorentzian_peak_image

    fit_result = fit_2d_lorentzian_on_patch(
        img,
        roi_origin_yx=(7, 9),
        initial_params=[85.0, 8.8, 10.1, 1.5, 2.1, 0.1, 3.5],
        parameter_bounds=(
            [0.0, 6.0, 7.0, 0.5, 0.5, -np.pi / 2.0, 0.0],
            [150.0, 13.0, 14.0, 4.0, 5.0, np.pi / 2.0, 20.0],
        ),
        max_nfev=2600,
        compute_uncertainty=False,
    )

    assert fit_result is not None
    assert np.isclose(fit_result.center[0], 7 + y0_true, atol=0.15)
    assert np.isclose(fit_result.center[1], 9 + x0_true, atol=0.15)
    assert fit_result.metadata["initial_params"] == pytest.approx((85.0, 8.8, 10.1, 1.5, 2.1, 0.1, 3.5))
    assert fit_result.metadata["parameter_bounds"] is not None
    assert fit_result.metadata["parameter_bounds"]["lower"] == pytest.approx(
        (0.0, 6.0, 7.0, 0.5, 0.5, -np.pi / 2.0, 0.0)
    )
    assert fit_result.metadata["parameter_bounds"]["upper"] == pytest.approx(
        (150.0, 13.0, 14.0, 4.0, 5.0, np.pi / 2.0, 20.0)
    )
    assert fit_result.metadata["max_nfev"] == 2600


@pytest.mark.skipif(not SCIPY_AVAILABLE, reason="SciPy not available, skipping Lorentzian fit tests")
def test_fit_lorentzian_on_patch_uses_fast_max_pixel_fallback_without_monte_carlo(
    monkeypatch,
    lorentzian_peak_image,
):
    img, _, _ = lorentzian_peak_image

    monkeypatch.setattr(
        peak_fitting_module,
        "_curve_fit_safely",
        lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("fit failed")),
    )
    monkeypatch.setattr(
        peak_fitting_module,
        "_monte_carlo_uncertainty",
        lambda *args, **kwargs: pytest.fail("Monte Carlo fallback should not run here."),
    )

    fit_result = fit_2d_lorentzian_on_patch(
        img,
        roi_origin_yx=(0, 0),
        compute_uncertainty=False,
    )

    assert fit_result is not None
    assert fit_result.success is False
    assert fit_result.method == "max_pixel_fallback"
    assert fit_result.center_std is None
    peak_y, peak_x = np.unravel_index(int(np.argmax(img)), img.shape)
    assert fit_result.center == pytest.approx((float(peak_y), float(peak_x)))


@pytest.mark.skipif(not SCIPY_AVAILABLE, reason="SciPy not available, skipping Voigt fit tests")
def test_fit_voigt_on_patch_perfect_peak(voigt_peak_image):
    img, (y0_true, x0_true), _ = voigt_peak_image

    fit_result = fit_2d_voigt_on_patch(img, roi_origin_yx=(11, 27))

    assert fit_result is not None
    assert np.isclose(fit_result.center[0], 11 + y0_true, atol=0.2)
    assert np.isclose(fit_result.center[1], 27 + x0_true, atol=0.2)
    assert fit_result.center_std is not None
    assert fit_result.method == "voigt_fit"


@pytest.mark.skipif(not SCIPY_AVAILABLE, reason="SciPy not available, skipping Voigt fit tests")
def test_fit_voigt_on_patch_accepts_custom_initial_params_bounds_and_maxfev(voigt_peak_image):
    img, (y0_true, x0_true), _ = voigt_peak_image

    fit_result = fit_2d_voigt_on_patch(
        img,
        roi_origin_yx=(5, 8),
        initial_params=[80.0, 8.8, 10.0, 1.1, 1.8, 0.6, 1.0, 0.1, 2.5],
        parameter_bounds=(
            [0.0, 6.0, 7.0, 0.3, 0.3, 0.1, 0.1, -np.pi / 2.0, 0.0],
            [140.0, 13.0, 14.0, 4.0, 4.0, 3.0, 3.0, np.pi / 2.0, 20.0],
        ),
        max_nfev=3600,
        compute_uncertainty=False,
    )

    assert fit_result is not None
    assert np.isclose(fit_result.center[0], 5 + y0_true, atol=0.2)
    assert np.isclose(fit_result.center[1], 8 + x0_true, atol=0.2)
    assert fit_result.metadata["initial_params"] == pytest.approx((80.0, 8.8, 10.0, 1.1, 1.8, 0.6, 1.0, 0.1, 2.5))
    assert fit_result.metadata["parameter_bounds"] is not None
    assert fit_result.metadata["parameter_bounds"]["lower"] == pytest.approx(
        (0.0, 6.0, 7.0, 0.3, 0.3, 0.1, 0.1, -np.pi / 2.0, 0.0)
    )
    assert fit_result.metadata["parameter_bounds"]["upper"] == pytest.approx(
        (140.0, 13.0, 14.0, 4.0, 4.0, 3.0, 3.0, np.pi / 2.0, 20.0)
    )
    assert fit_result.metadata["max_nfev"] == 3600


@pytest.mark.skipif(not SCIPY_AVAILABLE, reason="SciPy not available, skipping Voigt fit tests")
def test_fit_voigt_on_patch_uses_fast_max_pixel_fallback_without_monte_carlo(
    monkeypatch,
    voigt_peak_image,
):
    img, _, _ = voigt_peak_image

    monkeypatch.setattr(
        peak_fitting_module,
        "_curve_fit_safely",
        lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("fit failed")),
    )
    monkeypatch.setattr(
        peak_fitting_module,
        "_monte_carlo_uncertainty",
        lambda *args, **kwargs: pytest.fail("Monte Carlo fallback should not run here."),
    )

    fit_result = fit_2d_voigt_on_patch(
        img,
        roi_origin_yx=(0, 0),
        compute_uncertainty=False,
    )

    assert fit_result is not None
    assert fit_result.success is False
    assert fit_result.method == "max_pixel_fallback"
    assert fit_result.center_std is None
    peak_y, peak_x = np.unravel_index(int(np.argmax(img)), img.shape)
    assert fit_result.center == pytest.approx((float(peak_y), float(peak_x)))

def test_internal_gaussian_2d_function():
    """Test the _gaussian_2d helper function directly (optional)."""
    # This is usually not necessary if `fit_2d_gaussian_in_roi` is well-tested,
    # but can be useful for debugging the model function itself.
    from lfa.analysis.peak_fitting import _gaussian_2d  # Import locally for this test

    y, x = np.mgrid[0:5, 0:5]
    xy_tuple = (y, x)
    amplitude, y0, x0, sigma_y, sigma_x, theta, offset = 10.0, 2.0, 2.0, 1.0, 1.0, 0.0, 1.0
    params = [amplitude, y0, x0, sigma_y, sigma_x, theta, offset]

    result_flat = _gaussian_2d(xy_tuple, *params)
    assert result_flat.shape == (25,), "Raveling in _gaussian_2d failed"

    # Check peak value
    # Create a single point tuple for the peak
    peak_val = _gaussian_2d((np.array([y0]), np.array([x0])), *params)
    assert np.isclose(peak_val[0], offset + amplitude)

    # Check value far from peak (should be close to offset)
    far_val = _gaussian_2d((np.array([0]), np.array([0])), *params)
    assert far_val[0] < offset + amplitude # Should be less than peak
    assert np.isclose(far_val[0], offset + amplitude * np.exp(-((y0**2)/(2*sigma_y**2) + (x0**2)/(2*sigma_x**2))) )
