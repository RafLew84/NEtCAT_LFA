# tests/preprocessing/test_leveling.py
"""
Unit tests for plane leveling functions in lfa.preprocessing.leveling.
"""
import numpy as np
import pytest
from typing import Optional, Tuple, List # For type hints if needed

# Import functions to test
try:
    from lfa.preprocessing.leveling import fit_plane, fit_plane_3pts, level_by_plane
except ImportError:
    pytest.fail("Could not import leveling functions from lfa.preprocessing.leveling", pytrace=False)

# --- Fixtures (Reusable Test Data) ---

@pytest.fixture
def flat_image() -> np.ndarray:
    """A perfectly flat 10x10 image."""
    return np.full((10, 10), 5.0, dtype=np.float32)

@pytest.fixture
def tilted_image() -> np.ndarray:
    """A 10x10 image representing a simple tilted plane: z = 0.1x + 0.2y + 3."""
    rows, cols = 10, 10
    X, Y = np.meshgrid(np.arange(cols), np.arange(rows))
    a, b, c = 0.1, 0.2, 3.0
    return (a * X + b * Y + c).astype(np.float32)

@pytest.fixture
def expected_tilted_plane() -> np.ndarray:
     """The exact plane used to create tilted_image."""
     rows, cols = 10, 10
     X, Y = np.meshgrid(np.arange(cols), np.arange(rows))
     a, b, c = 0.1, 0.2, 3.0
     return (a * X + b * Y + c).astype(np.float32)

# --- Tests for fit_plane ---

def test_fit_plane_whole_tilted(tilted_image, expected_tilted_plane):
    """Test fitting a plane to the whole tilted image."""
    fitted = fit_plane(tilted_image, roi_slice=None)
    assert fitted is not None, "fit_plane returned None"
    assert fitted.shape == tilted_image.shape
    # Check if the fitted plane is numerically close to the expected one
    assert np.allclose(fitted, expected_tilted_plane, atol=1e-6), "Fitted plane differs significantly from expected"

def test_fit_plane_whole_flat(flat_image):
    """Test fitting a plane to a flat image."""
    expected_plane = np.full_like(flat_image, np.mean(flat_image))
    fitted = fit_plane(flat_image, roi_slice=None)
    assert fitted is not None
    assert fitted.shape == flat_image.shape
    assert np.allclose(fitted, expected_plane, atol=1e-6), "Fitted plane for flat image is incorrect"

def test_fit_plane_roi_tilted(tilted_image, expected_tilted_plane):
    """Test fitting a plane to an ROI of the tilted image."""
    rows, cols = tilted_image.shape
    # Define an ROI slice (e.g., inner part)
    roi = (slice(2, 8), slice(2, 8)) # Rows 2-7, Cols 2-7
    fitted = fit_plane(tilted_image, roi_slice=roi)
    assert fitted is not None
    assert fitted.shape == tilted_image.shape # Should return plane for whole image
    # The fitted plane should still be very close to the original plane
    assert np.allclose(fitted, expected_tilted_plane, atol=1e-6), "Plane fitted to ROI differs significantly"

def test_fit_plane_roi_invalid(tilted_image):
    """Test fit_plane with an invalid ROI slice."""
    roi_invalid_format = (2, 8, 2, 8) # Incorrect tuple format
    roi_out_of_bounds = (slice(20, 30), slice(0, 5))
    roi_empty = (slice(5, 5), slice(0, 5))

    assert fit_plane(tilted_image, roi_slice=roi_invalid_format) is None
    assert fit_plane(tilted_image, roi_slice=roi_out_of_bounds) is None
    assert fit_plane(tilted_image, roi_slice=roi_empty) is None

def test_fit_plane_invalid_input():
    """Test fit_plane with invalid image input."""
    assert fit_plane(None) is None
    assert fit_plane(np.zeros(5)) is None # 1D array
    assert fit_plane(np.zeros((5, 5, 3))) is None # 3D array

# --- Tests for fit_plane_3pts ---

def test_fit_plane_3pts_noncollinear(tilted_image, expected_tilted_plane):
    """Test fitting with 3 non-collinear points on the known plane."""
    # Points selected from the tilted_image: (x, y)
    # P1=(1, 2), P2=(5, 4), P3=(3, 7)
    points = [(1, 2), (5, 4), (3, 7)]
    fitted = fit_plane_3pts(tilted_image, points)
    assert fitted is not None
    assert fitted.shape == tilted_image.shape
    assert np.allclose(fitted, expected_tilted_plane, atol=1e-6), "3pt fitted plane differs from expected"

def test_fit_plane_3pts_collinear(tilted_image):
    """Test fitting with 3 collinear points."""
    # Points: (1, 1), (2, 2), (3, 3) - These lie on the line y=x
    points_collinear = [(1, 1), (2, 2), (3, 3)]
    fitted = fit_plane_3pts(tilted_image, points_collinear)
    # Expect None because np.linalg.solve should fail for singular matrix
    assert fitted is None, "fit_plane_3pts did not return None for collinear points"

def test_fit_plane_3pts_not_enough_points(tilted_image):
    """Test fitting with fewer than 3 points."""
    assert fit_plane_3pts(tilted_image, [(1, 1)]) is None
    assert fit_plane_3pts(tilted_image, [(1, 1), (2, 2)]) is None

def test_fit_plane_3pts_oob(tilted_image):
    """Test fitting with points outside image bounds."""
    rows, cols = tilted_image.shape
    points_oob = [(1, 1), (5, 5), (cols + 5, rows + 5)] # Third point is out
    assert fit_plane_3pts(tilted_image, points_oob) is None

def test_fit_plane_3pts_invalid_input():
    """Test fit_plane_3pts with invalid image input."""
    assert fit_plane_3pts(None, [(1,1), (2,2), (3,3)]) is None
    assert fit_plane_3pts(np.zeros(5), [(1,1), (2,2), (3,3)]) is None

# --- Tests for level_by_plane ---

def test_level_by_plane_simple(tilted_image, expected_tilted_plane):
    """Test subtracting the known plane from the tilted image."""
    leveled = level_by_plane(tilted_image, expected_tilted_plane)
    assert leveled is not None
    assert leveled.shape == tilted_image.shape
    # Result should be close to zero everywhere
    assert np.allclose(leveled, np.zeros_like(tilted_image), atol=1e-6), "Leveling did not result in near-zero image"
    assert leveled.dtype == np.float32, f"Expected float32 output, got {leveled.dtype}"

def test_level_by_plane_different_plane(tilted_image):
    """Test subtracting a different plane."""
    rows, cols = tilted_image.shape
    X, Y = np.meshgrid(np.arange(cols), np.arange(rows))
    # Create a slightly different plane, e.g., just the constant term
    mean_plane = np.full_like(tilted_image, np.mean(tilted_image))
    leveled = level_by_plane(tilted_image, mean_plane)
    assert leveled is not None
    assert leveled.shape == tilted_image.shape
    # Check that the result is the original image minus the mean
    assert np.allclose(leveled, tilted_image - np.mean(tilted_image), atol=1e-6)

def test_level_by_plane_invalid_input(tilted_image, expected_tilted_plane):
    """Test level_by_plane with None or shape mismatch."""
    assert level_by_plane(None, expected_tilted_plane) is None
    assert level_by_plane(tilted_image, None) is None
    wrong_shape_plane = np.zeros((5, 5)) # Different shape
    assert level_by_plane(tilted_image, wrong_shape_plane) is None