# tests/preprocessing/test_filtering.py
"""
Unit tests for image filtering functions in lfa.preprocessing.filtering.
"""

import numpy as np
import pytest # Import pytest
import logging
from typing import Tuple
from scipy.ndimage import gaussian_filter as scipy_gaussian_filter # For comparison if needed

logger = logging.getLogger(__name__)

# Import the function to be tested
try:
    from lfa.preprocessing.filtering import gaussian_blur, median_filter_lfa, gaussian_sharpen_unsharp_mask
except ImportError:
    pytest.fail("Could not import gaussian_blur from lfa.preprocessing.filtering", pytrace=False)
    gaussian_sharpen_unsharp_mask = None
    median_filter_lfa = None
    gaussian_blur = None

# --- Fixtures (optional reusable test data) ---

@pytest.fixture
def sample_image() -> np.ndarray:
    """Provides a simple 5x5 sample image for testing."""
    img = np.zeros((5, 5), dtype=np.float32)
    img[2, 2] = 1.0 # Single bright pixel in the center
    return img

@pytest.fixture
def noisy_image() -> Tuple[np.ndarray, np.ndarray]:
    """Provides a 10x10 image with salt & pepper noise and the original."""
    original = np.zeros((10, 10), dtype=np.float32)
    original[3:7, 3:7] = 1.0 # A square in the middle
    noisy = original.copy()
    # Add salt (white) noise
    salt_coords = (np.random.randint(0, 10, 5), np.random.randint(0, 10, 5))
    noisy[salt_coords] = 2.0 # Use a value higher than max
    # Add pepper (black) noise
    pepper_coords = (np.random.randint(0, 10, 5), np.random.randint(0, 10, 5))
    noisy[pepper_coords] = -1.0 # Use a negative value
    return noisy, original

# --- Test Functions (must start with 'test_') ---

def test_median_filter_defaults(noisy_image):
    """Test median filter with default parameters on noisy image."""
    noisy, original = noisy_image
    filtered = median_filter_lfa(noisy) # size=3, mode='reflect'

    assert filtered is not None
    assert filtered.shape == noisy.shape
    assert filtered.dtype == np.float32
    # Check if some noise pixels were removed (difficult to be exact without knowing coords)
    # Let's check if the number of extreme pixels decreased
    original_extreme_count = np.sum((noisy > 1.5) | (noisy < -0.5))
    filtered_extreme_count = np.sum((filtered > 1.5) | (filtered < -0.5))
    assert filtered_extreme_count < original_extreme_count, "Median filter did not reduce extreme noise pixels"


def test_median_filter_invalid_inputs(sample_image):
    """Test invalid inputs for median_filter_lfa."""
    assert median_filter_lfa(None, size=3) is None, "Did not return None for None image"
    assert median_filter_lfa(np.zeros(5), size=3) is None, "Did not return None for 1D image"
    assert median_filter_lfa(sample_image, size=0) is None, "Did not return None for size=0"
    assert median_filter_lfa(sample_image, size=-3) is None, "Did not return None for negative size"
    assert median_filter_lfa(sample_image, size=3, mode='invalid_mode') is None, "Did not return None for invalid mode"

def test_median_filter_return_type(sample_image):
    """Ensure the output is always float32."""
    int_image = sample_image.astype(np.uint8)
    filtered = median_filter_lfa(int_image, size=3)
    assert filtered is not None
    assert filtered.dtype == np.float32, f"Expected float32 output, got {filtered.dtype}"


def test_gaussian_blur_zero_sigma(sample_image):
    """Test if gaussian_blur with sigma=0 returns the original image."""
    sigma = 0.0
    original = sample_image
    processed = gaussian_blur(original, sigma)

    assert processed is not None, "Function returned None unexpectedly"
    assert processed.shape == original.shape, "Output shape mismatch"
    assert processed.dtype == np.float32, f"Expected dtype float32, got {processed.dtype}"
    # Check if arrays are numerically close (handles potential minor float inaccuracies)
    assert np.allclose(processed, original), "Output data differs from input with sigma=0"

def test_gaussian_blur_negative_sigma(sample_image):
    """Test if gaussian_blur handles negative sigma by clamping to 0."""
    sigma = -1.0
    original = sample_image
    processed = gaussian_blur(original, sigma)

    assert processed is not None
    assert processed.shape == original.shape
    assert processed.dtype == np.float32
    # Should behave exactly like sigma=0
    assert np.allclose(processed, original), "Negative sigma did not produce same result as sigma=0"

def test_gaussian_blur_basic_blur(sample_image):
    """Test basic blurring effect with a positive sigma."""
    sigma = 1.0
    original = sample_image
    processed = gaussian_blur(original, sigma)

    assert processed is not None
    assert processed.shape == original.shape, "Output shape mismatch"
    assert processed.dtype == np.float32, f"Expected dtype float32, got {processed.dtype}"

    # Check if the peak value decreased due to blurring
    assert processed[2, 2] < original[2, 2], "Peak value did not decrease after blurring"
    assert processed[2, 2] > 0, "Peak value became non-positive after blurring"

    # Check if surrounding pixels increased value
    assert processed[1, 2] > original[1, 2], "Neighbor pixel value did not increase"
    assert processed[2, 1] > original[2, 1], "Neighbor pixel value did not increase"

    # Check if the sum (mass) is conserved (approximately)
    assert np.allclose(np.sum(processed), np.sum(original)), "Sum of elements changed significantly after blur"

def test_gaussian_blur_different_sigma(sample_image):
    """Test if blurring with a larger sigma blurs more."""
    sigma1 = 1.0
    sigma2 = 2.0
    original = sample_image

    processed1 = gaussian_blur(original, sigma1)
    processed2 = gaussian_blur(original, sigma2)

    assert processed1 is not None and processed2 is not None
    # The peak should be lower for the larger sigma
    assert processed2[2, 2] < processed1[2, 2], "Blurring with larger sigma did not decrease peak value further"

def test_gaussian_blur_return_type(sample_image):
    """Ensure the output is always float32."""
    int_image = sample_image.astype(np.uint8) # Start with int image
    processed = gaussian_blur(int_image, sigma=1.0)
    assert processed.dtype == np.float32, f"Expected float32 output, got {processed.dtype}"

def test_sharpen_defaults(original_image_nl):
    """Test sharpening with default parameters."""
    original = original_image_nl
    # Domyślne: radius=1.0, amount=1.0
    sharpened = gaussian_sharpen_unsharp_mask(original)

    assert sharpened is not None
    assert sharpened.shape == original.shape
    assert sharpened.dtype == np.float32
    # Sharpening should change the image
    assert not np.allclose(sharpened, original)
    # Sharpening usually increases variance due to enhanced edges/noise
    assert np.var(sharpened) > np.var(original), "Variance did not increase after sharpening"

def test_sharpen_radius_effect(original_image_nl):
    """Test effect of radius parameter."""
    original = original_image_nl
    amount = 1.0
    radius_small = 0.5
    radius_large = 3.0

    sharpened_small_r = gaussian_sharpen_unsharp_mask(original, radius=radius_small, amount=amount)
    sharpened_large_r = gaussian_sharpen_unsharp_mask(original, radius=radius_large, amount=amount)

    assert sharpened_small_r is not None and sharpened_large_r is not None
    # Different radius should produce different results
    assert not np.allclose(sharpened_small_r, sharpened_large_r)
    # Variance might change differently depending on features sharpened
    print(f"Variance: R={radius_small} -> {np.var(sharpened_small_r):.4f}, R={radius_large} -> {np.var(sharpened_large_r):.4f}")

def test_sharpen_amount_effect(original_image_nl):
    """Test effect of amount parameter."""
    original = original_image_nl
    radius = 1.0
    amount_low = 0.5
    amount_high = 2.5

    sharpened_low_a = gaussian_sharpen_unsharp_mask(original, radius=radius, amount=amount_low)
    sharpened_high_a = gaussian_sharpen_unsharp_mask(original, radius=radius, amount=amount_high)

    assert sharpened_low_a is not None and sharpened_high_a is not None
    # Different amount should produce different results
    assert not np.allclose(sharpened_low_a, sharpened_high_a)
    # Higher amount should generally increase variance more
    var_low = np.var(sharpened_low_a)
    var_high = np.var(sharpened_high_a)
    print(f"Variance: A={amount_low} -> {var_low:.4f}, A={amount_high} -> {var_high:.4f}")
    assert var_high > var_low, "Variance did not increase with higher amount"

def test_sharpen_zero_params(original_image_nl):
    """Test sharpening with zero radius or amount."""
    original = original_image_nl

    # Amount = 0 should return original image
    sharpened_zero_amount = gaussian_sharpen_unsharp_mask(original, radius=1.0, amount=0.0)
    assert sharpened_zero_amount is not None
    assert np.allclose(sharpened_zero_amount, original), "amount=0 did not return original image"

    # Radius = 0 means no blurring, so difference is zero, should return original
    sharpened_zero_radius = gaussian_sharpen_unsharp_mask(original, radius=0.0, amount=1.0)
    assert sharpened_zero_radius is not None
    # Note: skimage unsharp_mask with radius=0 might have tiny float precision diffs
    assert np.allclose(sharpened_zero_radius, original, atol=1e-6), "radius=0 did not return original image"

def test_sharpen_invalid_inputs(original_image_nl):
    """Test invalid inputs for gaussian_sharpen_unsharp_mask."""
    assert gaussian_sharpen_unsharp_mask(None, radius=1.0, amount=1.0) is None, "None image"
    assert gaussian_sharpen_unsharp_mask(np.zeros(5), radius=1.0, amount=1.0) is None, "1D image"
    # Function clamps negative values, so these should not return None but run with 0
    # assert gaussian_sharpen_unsharp_mask(original_image_nl, radius=-1.0, amount=1.0) is None, "negative radius"
    # assert gaussian_sharpen_unsharp_mask(original_image_nl, radius=1.0, amount=-1.0) is None, "negative amount"
    sharpened_neg_r = gaussian_sharpen_unsharp_mask(original_image_nl, radius=-1.0, amount=1.0)
    assert sharpened_neg_r is not None # Should run with radius=0
    assert np.allclose(sharpened_neg_r, original_image_nl, atol=1e-6) # Should be like radius=0

    sharpened_neg_a = gaussian_sharpen_unsharp_mask(original_image_nl, radius=1.0, amount=-1.0)
    assert sharpened_neg_a is not None # Should run with amount=0
    assert np.allclose(sharpened_neg_a, original_image_nl) # Should be like amount=0


def test_sharpen_return_type(original_image_nl):
    """Ensure the output is always float32."""
    int_image = (original_image_nl * 100).astype(np.int16) # Create int version
    sharpened = gaussian_sharpen_unsharp_mask(int_image, radius=1.0, amount=1.0)
    assert sharpened is not None
    assert sharpened.dtype == np.float32, f"Expected float32 output, got {sharpened.dtype}"
