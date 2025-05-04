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
    from lfa.preprocessing.filtering import gaussian_blur, median_filter_lfa
except ImportError:
    pytest.fail("Could not import gaussian_blur from lfa.preprocessing.filtering", pytrace=False)

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