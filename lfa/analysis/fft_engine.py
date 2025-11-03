# lfa/analysis/fft_engine.py
"""
Fast Fourier Transform (FFT) analysis engine for LFA.
Provides functions for calculating 2D FFT with optional windowing and zero-padding.
"""
import logging
import numpy as np
from typing import Optional, Tuple, Dict, Callable
try:
    from scipy.signal import windows
except ImportError:
    logging.warning("SciPy not found. Window functions will not be available.")
    windows = None

logger = logging.getLogger(__name__)

AVAILABLE_WINDOWS = {
    'hann': windows.hann if windows else None,
    'hamming': windows.hamming if windows else None,
    'blackman': windows.blackman if windows else None,
}

def calculate_fft(image_data: np.ndarray, apply_window: bool = True,
                  window_type: str = 'hann',
                  pad_to_shape: Optional[Tuple[int, int]] = None) -> Optional[np.ndarray]:
    """
    Calculates the 2D FFT, optionally padding ROI data and applying a window.

    Args:
        image_data (np.ndarray): Input 2D image data (should be float).
        apply_window (bool, optional): Whether to apply a window function. Defaults to True.
        window_type (str, optional): Window type ('hann', 'hamming', etc.). Defaults to 'hann'.
        pad_to_shape (Optional[Tuple[int, int]], optional): Target shape (rows, cols)
            to pad the input data with zeros before FFT. Defaults to None.

    Returns:
        Optional[np.ndarray]: The complex, shifted FFT result (dtype complex),
                              or None if an error occurs.
    """
    if image_data is None or image_data.ndim != 2:
        logger.error(f"FFT: Invalid input image (None or shape {getattr(image_data, 'shape', 'N/A')}).")
        return None
    if image_data.size == 0:
        logger.error("FFT: Input image is empty.")
        return None

    rows, cols = image_data.shape
    processed_data = image_data.astype(np.float32, copy=True) # Work on a float32 copy

    # Zero-padding section
    if pad_to_shape is not None and pad_to_shape != (rows, cols):
        target_rows, target_cols = pad_to_shape
        if target_rows < rows or target_cols < cols:
            logger.warning(f"FFT: pad_to_shape {pad_to_shape} is smaller than image shape {(rows, cols)}. Padding skipped.")
        else:
            logger.debug(f"Padding image from {(rows, cols)} to {pad_to_shape} with zeros.")
            padded_data = np.zeros(pad_to_shape, dtype=np.float32)
            r_offset = (target_rows - rows) // 2
            c_offset = (target_cols - cols) // 2
            padded_data[r_offset:r_offset+rows, c_offset:c_offset+cols] = processed_data
            processed_data = padded_data
            rows, cols = target_rows, target_cols
            logger.debug(f"Data padded. New shape: {processed_data.shape}")

    # Windowing section
    if apply_window:
        window_func = AVAILABLE_WINDOWS.get(window_type.lower())
        if window_func is None:
            if window_type.lower() == 'none':
                logger.debug("FFT: Windowing explicitly disabled ('none').")
            elif windows is None:
                 logger.error("FFT: Cannot apply window. SciPy is not available.")
                 return None
            else:
                 logger.warning(f"FFT: Unsupported window type '{window_type}'. No window applied.")
        else:
            try:
                logger.debug(f"Applying 2D '{window_type}' window to shape {(rows, cols)}...")
                win_y = windows.get_window(window_type, rows)
                win_x = windows.get_window(window_type, cols)
                win_2d = np.outer(win_y, win_x)
                processed_data *= win_2d.astype(np.float32)
            except Exception as e:
                logger.exception(f"Error applying window function '{window_type}': {e}")
                return None
    else:
         logger.debug("FFT: Skipping window function.")

    # FFT calculation section
    try:
        logger.debug("Calculating 2D FFT...")
        fft_result = np.fft.fft2(processed_data)

        logger.debug("Applying fftshift...")
        shifted_fft = np.fft.fftshift(fft_result)

        logger.info("Complex FFT calculation successful.")
        return shifted_fft # Type will be complex64 or complex128

    except Exception as e:
        logger.exception(f"Error during FFT calculation: {e}")
        return None