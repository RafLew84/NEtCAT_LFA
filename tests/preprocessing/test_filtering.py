# tests/preprocessing/test_filtering.py
"""
Unit tests for image filtering functions in lfa.preprocessing.filtering.
"""

import numpy as np
import pytest # Import pytest
from scipy.ndimage import gaussian_filter as scipy_gaussian_filter # For comparison if needed

# Import the function to be tested
try:
    from lfa.preprocessing.filtering import gaussian_blur
except ImportError:
    pytest.fail("Could not import gaussian_blur from lfa.preprocessing.filtering", pytrace=False)

# --- Fixtures (optional reusable test data) ---

@pytest.fixture
def sample_image() -> np.ndarray:
    """Provides a simple 5x5 sample image for testing."""
    img = np.zeros((5, 5), dtype=np.float32)
    img[2, 2] = 1.0 # Single bright pixel in the center
    return img

# --- Test Functions (must start with 'test_') ---

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