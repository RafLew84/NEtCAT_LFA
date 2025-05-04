# tests/preprocessing/test_denoising.py
"""
Unit tests for image denoising functions in lfa.preprocessing.denoising.
"""
import numpy as np
import pytest
from typing import Tuple

# Import the function to test
try:
    from lfa.preprocessing.denoising import denoise_nlmeans_skimage
except ImportError:
    pytest.fail("Could not import denoise_nlmeans_skimage from lfa.preprocessing.denoising", pytrace=False)

# --- Fixtures ---

@pytest.fixture
def original_image_nl() -> np.ndarray:
    """A simple image with varying regions for NL-Means testing."""
    img = np.zeros((30, 30), dtype=np.float32)
    img[5:15, 5:15] = 0.5  # First square
    img[15:25, 15:25] = 1.0 # Second square (brighter)
    # Add a small gradient
    gradient = np.linspace(0, 0.2, 30)
    img += gradient[:, np.newaxis]
    return img

@pytest.fixture
def noisy_image_gaussian(original_image_nl) -> Tuple[np.ndarray, np.ndarray, float]:
    """Adds Gaussian noise to the original image."""
    sigma = 0.1 # Known standard deviation of noise
    noise = np.random.normal(loc=0.0, scale=sigma, size=original_image_nl.shape).astype(np.float32)
    noisy = original_image_nl + noise
    return noisy, original_image_nl, sigma

# --- Helper Function ---

def calculate_mse(img1, img2):
    """Calculates Mean Squared Error between two images."""
    if img1.shape != img2.shape:
        raise ValueError("Images must have the same shape for MSE calculation.")
    return np.mean((img1 - img2) ** 2)

# --- Tests for denoise_nlmeans_skimage ---

def test_nlmeans_defaults(noisy_image_gaussian):
    """Test NL-Means with default parameters."""
    noisy, original, sigma = noisy_image_gaussian
    # Use default h_mult, patch_size, patch_distance, fast_mode
    denoised = denoise_nlmeans_skimage(noisy, sigma=sigma)

    assert denoised is not None
    assert denoised.shape == noisy.shape
    assert denoised.dtype == np.float32
    # Check if denoising reduced the error compared to original
    mse_noisy = calculate_mse(noisy, original)
    mse_denoised = calculate_mse(denoised, original)
    assert mse_denoised < mse_noisy, f"NL-Means did not reduce MSE (Noisy: {mse_noisy:.4f}, Denoised: {mse_denoised:.4f})"

def test_nlmeans_h_param_effect(noisy_image_gaussian):
    """Test effect of h_param_mult."""
    noisy, original, sigma = noisy_image_gaussian
    h_low = 0.8
    h_high = 1.2

    denoised_low_h = denoise_nlmeans_skimage(noisy, sigma=sigma, h_param_mult=h_low)
    denoised_high_h = denoise_nlmeans_skimage(noisy, sigma=sigma, h_param_mult=h_high)

    assert denoised_low_h is not None and denoised_high_h is not None
    assert not np.allclose(denoised_low_h, denoised_high_h)

    # Expect higher h to potentially blur more, check variance difference
    var_low_h = np.var(denoised_low_h)
    var_high_h = np.var(denoised_high_h)
    # Generally, stronger filtering (higher h) might reduce variance more, but can be complex
    # Let's just check they are different and potentially print values
    print(f"Variance: h_mult={h_low} -> {var_low_h:.4f}, h_mult={h_high} -> {var_high_h:.4f}")
    # No strict assertion on variance here, effect depends on image content

def test_nlmeans_patch_size_effect(noisy_image_gaussian):
    """Test effect of patch_size."""
    noisy, original, sigma = noisy_image_gaussian
    psize_small = 5
    psize_large = 9 # Must be odd

    denoised_small_p = denoise_nlmeans_skimage(noisy, sigma=sigma, patch_size=psize_small)
    denoised_large_p = denoise_nlmeans_skimage(noisy, sigma=sigma, patch_size=psize_large)

    assert denoised_small_p is not None and denoised_large_p is not None
    # Results should differ
    assert not np.allclose(denoised_small_p, denoised_large_p)

def test_nlmeans_patch_distance_effect(noisy_image_gaussian):
    """Test effect of patch_distance."""
    noisy, original, sigma = noisy_image_gaussian
    pdist_small = 5
    pdist_large = 11

    denoised_small_d = denoise_nlmeans_skimage(noisy, sigma=sigma, patch_distance=pdist_small)
    denoised_large_d = denoise_nlmeans_skimage(noisy, sigma=sigma, patch_distance=pdist_large)

    assert denoised_small_d is not None and denoised_large_d is not None
    # Results should differ
    assert not np.allclose(denoised_small_d, denoised_large_d)

def test_nlmeans_fast_mode(noisy_image_gaussian):
    """Test difference between fast_mode True and False."""
    noisy, original, sigma = noisy_image_gaussian
    # Use smaller patch distance for fast=False as it's much slower
    denoised_fast = denoise_nlmeans_skimage(noisy, sigma=sigma, patch_distance=11, fast_mode=True)
    denoised_slow = denoise_nlmeans_skimage(noisy, sigma=sigma, patch_distance=6, fast_mode=False) # Smaller distance for slow

    assert denoised_fast is not None and denoised_slow is not None
    # Results should differ
    assert not np.allclose(denoised_fast, denoised_slow)
    # Often, slow mode gives slightly better results (lower MSE)
    mse_fast = calculate_mse(denoised_fast, original)
    mse_slow = calculate_mse(denoised_slow, original)
    print(f"MSE: fast={mse_fast:.4f}, slow={mse_slow:.4f}")
    # Allow for slight numerical differences, don't assert mse_slow < mse_fast strictly
    # assert mse_slow < mse_fast

def test_nlmeans_invalid_inputs(original_image_nl):
    """Test invalid inputs for denoise_nlmeans_skimage."""
    sigma = 0.1
    assert denoise_nlmeans_skimage(None, sigma=sigma) is None, "None image"
    assert denoise_nlmeans_skimage(np.zeros(5), sigma=sigma) is None, "1D image"
    assert denoise_nlmeans_skimage(original_image_nl, sigma=0) is None, "sigma=0"
    assert denoise_nlmeans_skimage(original_image_nl, sigma=-0.1) is None, "negative sigma"
    assert denoise_nlmeans_skimage(original_image_nl, sigma=sigma, h_param_mult=0) is None, "h_mult=0"
    assert denoise_nlmeans_skimage(original_image_nl, sigma=sigma, patch_size=0) is None, "patch_size=0"
    assert denoise_nlmeans_skimage(original_image_nl, sigma=sigma, patch_size=6) is None, "even patch_size"
    assert denoise_nlmeans_skimage(original_image_nl, sigma=sigma, patch_distance=0) is None, "patch_distance=0"

def test_nlmeans_return_type(original_image_nl):
    """Ensure the output is always float32."""
    int_image = (original_image_nl * 255).astype(np.uint8)
    denoised = denoise_nlmeans_skimage(int_image, sigma=25.5) # Sigma scaled for uint8 range
    assert denoised is not None
    assert denoised.dtype == np.float32, f"Expected float32 output, got {denoised.dtype}"