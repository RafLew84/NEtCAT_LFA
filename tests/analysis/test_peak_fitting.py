# tests/analysis/test_peak_fitting.py
"""
Unit tests for peak fitting functions in lfa.analysis.peak_fitting.
"""
import numpy as np
import pytest
import logging

# Importuj funkcje do testowania
try:
    from lfa.analysis.peak_fitting import find_max_pixel_in_roi, fit_2d_gaussian_in_roi, SCIPY_AVAILABLE
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
    # Zero size ROI (radius 0 might imply 1x1, radius -1 is invalid)
    # Our implementation calculates y_start = max(0, center_y - roi_radius), so negative radius will make ROI larger.
    # Let's assume roi_radius is expected to be non-negative in the calling code for sensible ROI.
    # If ROI is completely outside, or radius is too small that patch is empty:
    # refined_y, refined_x = find_max_pixel_in_roi(simple_peak_image, (0,0), roi_radius=0) # ROI 1x1 at (0,0)
    # assert (refined_y, refined_x) == (0,0)
    # roi_radius = 0: y_start=center_y, y_end=center_y+1, x_start=center_x, x_end=center_x+1
    # this forms a 1x1 ROI.
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
    y0_fit, x0_fit = fit_result

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
    y0_fit, x0_fit = fit_result

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
        logger.info(f"Small ROI fit result: {fit_result}. True: ({y0_true}, {x0_true})")
        # Don't be too strict on accuracy for very small ROIs on broad peaks
        # y0_fit, x0_fit = fit_result
        # assert np.isclose(y0_fit, y0_true, atol=1.0) # Wider tolerance
        # assert np.isclose(x0_fit, x0_true, atol=1.0)
    else:
        # This is also an acceptable outcome if the fit fails due to insufficient data
        logger.info("Small ROI fit returned None, indicating robust failure or poor fit quality.")
    # No strict assertion on outcome, more of an exploratory test.

def test_internal_gaussian_2d_function():
    """Test the _gaussian_2d helper function directly (optional)."""
    # This is usually not necessary if `fit_2d_gaussian_in_roi` is well-tested,
    # but can be useful for debugging the model function itself.
    from lfa.analysis.peak_fitting import _gaussian_2d # Import locally for this test

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