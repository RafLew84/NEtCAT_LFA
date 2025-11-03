# tests/analysis/test_fft_engine.py
"""
Unit tests for FFT engine functions in lfa.analysis.fft_engine.
"""
import logging
from importlib import util
from typing import Tuple

import numpy as np
import pytest

logger = logging.getLogger(__name__)

# Sprawdź, czy SciPy jest dostępne dla funkcji okien
scipy_available = util.find_spec("scipy.signal.windows") is not None

# Importuj funkcję do testowania
try:
    from lfa.analysis.fft_engine import calculate_fft
except ImportError:
    pytest.fail("Could not import calculate_fft from lfa.analysis.fft_engine", pytrace=False)

# Oznacz testy wymagające SciPy
pytestmark_scipy = pytest.mark.skipif(not scipy_available, reason="scipy not installed, windowing tests skipped")

# --- Fixtures ---

@pytest.fixture
def square_image() -> np.ndarray:
    """A simple 16x16 square image."""
    return np.zeros((16, 16), dtype=np.float32)

@pytest.fixture
def rect_image() -> np.ndarray:
    """A simple 16x32 rectangular image."""
    return np.zeros((16, 32), dtype=np.float32)

@pytest.fixture
def sine_grating_image() -> Tuple[np.ndarray, int, int]:
    """A 32x32 image with a 2D sine grating."""
    size = 32
    # Frequency components (number of cycles across the image)
    freq_y = 4
    freq_x = 2
    Y, X = np.mgrid[:size, :size]
    # Simple grating: sin(2*pi*fx*x/N) + sin(2*pi*fy*y/N)
    grating = (np.sin(2 * np.pi * freq_x * X / size) +
               np.sin(2 * np.pi * freq_y * Y / size))
    return grating.astype(np.float32), freq_x, freq_y


# --- Tests for calculate_fft ---

def test_fft_return_type_and_shape(square_image):
    """Test return type and shape for standard FFT."""
    fft_result = calculate_fft(square_image, apply_window=False)
    assert fft_result is not None
    # Output should be complex
    assert np.iscomplexobj(fft_result), f"Expected complex output, got {fft_result.dtype}"
    # Output shape should match input shape when no padding
    assert fft_result.shape == square_image.shape, "Output shape mismatch"

def test_fft_shift(square_image):
    """Test if fftshift correctly centers the DC component."""
    # Create image with non-zero mean (DC component)
    test_img = square_image + 5.0
    fft_result = calculate_fft(test_img, apply_window=False)
    assert fft_result is not None
    # Find index of maximum absolute value (should be DC component at center)
    max_idx = np.unravel_index(np.argmax(np.abs(fft_result)), fft_result.shape)
    center_idx = (test_img.shape[0] // 2, test_img.shape[1] // 2)
    assert max_idx == center_idx, f"DC component not centered. Max at {max_idx}, expected {center_idx}"

@pytestmark_scipy
def test_fft_no_window_vs_window(sine_grating_image): # Zmieniono fixture na sine_grating_image
    """Test if applying a window changes the result for non-zero image."""
    img, _, _ = sine_grating_image # Użyj obrazu z sinusem
    fft_no_win = calculate_fft(img, apply_window=False)
    fft_hann_win = calculate_fft(img, apply_window=True, window_type='hann')
    assert fft_no_win is not None and fft_hann_win is not None
    # Results should be different for a non-zero image
    assert not np.allclose(fft_no_win, fft_hann_win), "Hann window did not change FFT result"


@pytestmark_scipy
def test_fft_different_windows(sine_grating_image): # Zmieniono fixture na sine_grating_image
    """Test if different window types produce different results for non-zero image."""
    img, _, _ = sine_grating_image # Użyj obrazu z sinusem
    fft_hann = calculate_fft(img, apply_window=True, window_type='hann')
    fft_hamming = calculate_fft(img, apply_window=True, window_type='hamming')
    assert fft_hann is not None and fft_hamming is not None
    # Results should be different
    assert not np.allclose(fft_hann, fft_hamming), "Hann and Hamming windows produced identical FFT results"

def test_fft_padding(rect_image):
    """Test zero-padding effect on output shape."""
    original_shape = rect_image.shape
    target_shape = (original_shape[0] * 2, original_shape[1] * 2) # Example larger shape

    fft_no_pad = calculate_fft(rect_image, apply_window=False, pad_to_shape=None)
    fft_pad = calculate_fft(rect_image, apply_window=False, pad_to_shape=target_shape)

    assert fft_no_pad is not None and fft_pad is not None
    assert fft_no_pad.shape == original_shape, "FFT without padding has wrong shape"
    assert fft_pad.shape == target_shape, "FFT with padding has wrong shape"
    # Also check they are not the same result (padding changes things)
    # Need to compare carefully, maybe resize smaller one? For now, just check shape difference.
    assert fft_no_pad.shape != fft_pad.shape

def test_fft_roi_padding(sine_grating_image): # Zmieniono fixture na sine_grating_image
    """Test applying FFT to an ROI with padding back to original size."""
    img, _, _ = sine_grating_image # Użyj obrazu z sinusem
    original_shape = img.shape
    # Weź ROI, które na pewno nie jest zerowe
    roi_slice = (slice(img.shape[0]//4, 3*img.shape[0]//4), slice(img.shape[1]//4, 3*img.shape[1]//4))
    roi_data = img[roi_slice]
    assert np.any(roi_data != 0), "ROI data should not be all zeros for this test" # sanity check

    fft_roi_padded = calculate_fft(roi_data, apply_window=False, pad_to_shape=original_shape)
    assert fft_roi_padded is not None
    assert fft_roi_padded.shape == original_shape, "FFT of padded ROI has wrong shape"

    fft_whole = calculate_fft(img, apply_window=False)
    assert fft_whole is not None
    # Result from padded ROI should differ significantly from whole image FFT
    assert not np.allclose(fft_roi_padded, fft_whole), "FFT of padded ROI was identical to whole image FFT"


def test_fft_known_frequency(sine_grating_image):
    """Test if known input frequencies appear at correct FFT locations."""
    img, fx, fy = sine_grating_image
    rows, cols = img.shape
    center_y, center_x = rows // 2, cols // 2

    fft_result = calculate_fft(img, apply_window=False)
    assert fft_result is not None
    magnitude = np.abs(fft_result)

    # --- POPRAWIONE Oczekiwane pozycje pików (tylko wzdłuż osi) ---
    expected_peaks_pos = [
        (center_y, center_x + fx), # +fx
        (center_y, center_x - fx), # -fx
        (center_y + fy, center_x), # +fy
        (center_y - fy, center_x)  # -fy
    ]
    # ----------------------------------------------------------

    # Find indices of the largest magnitudes, excluding the center DC peak
    magnitude[center_y, center_x] = 0 # Zero out DC component
    # Szukamy 4 najsilniejszych pików
    peak_indices = np.unravel_index(np.argsort(magnitude, axis=None)[-4:], magnitude.shape)
    found_peaks = list(zip(peak_indices[0], peak_indices[1]))

    logger.debug(f"Expected peaks near (y, x): {expected_peaks_pos}")
    logger.debug(f"Found top 4 peak indices (y, x): {found_peaks}")

    # Check if the expected peak locations are among the found peaks (allow 1 pixel tolerance)
    def check_peak(expected_pos, found_list, tolerance=1):
        ey, ex = expected_pos
        for fy, fx in found_list:
            if abs(fy - ey) <= tolerance and abs(fx - ex) <= tolerance:
                return True
        return False

    # --- POPRAWIONE Asercje ---
    for expected_pos in expected_peaks_pos:
         assert check_peak(expected_pos, found_peaks), f"Did not find peak near {expected_pos}"
    # -------------------------


def test_fft_invalid_inputs():
    """Test invalid inputs for calculate_fft."""
    assert calculate_fft(None) is None, "None image"
    assert calculate_fft(np.zeros(5)) is None, "1D image"
    assert calculate_fft(np.array([[]])) is None, "Empty image"
    # --- POPRAWIONA ASERCJA dla invalid_window ---
    # Funkcja loguje ostrzeżenie i kontynuuje bez okna, zwracając wynik
    result_invalid_win = calculate_fft(np.zeros((5,5)), window_type='invalid_window')
    assert result_invalid_win is not None, "Invalid window type should not return None"
    assert np.allclose(result_invalid_win, np.zeros((5,5), dtype=complex)), "Invalid window should return FFT of original"
    # ---------------------------------------------
