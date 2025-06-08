# lfa/analysis/peak_fitting.py
"""
Functions for refining peak positions in FFT data.
"""
import logging
import numpy as np
from typing import Optional, Tuple

try:
    from scipy.optimize import curve_fit
    SCIPY_AVAILABLE = True
except ImportError:
    logging.error("SciPy not found. 2D Gaussian fitting will not be available.")
    SCIPY_AVAILABLE = False

logger = logging.getLogger(__name__)

def find_max_pixel_in_roi(
    fft_magnitude_data: np.ndarray,
    center_yx: Tuple[int, int],
    roi_radius: int
) -> Tuple[int, int]:
    """
    Finds the pixel with the maximum intensity within a square ROI
    centered around center_yx.

    Args:
        fft_magnitude_data (np.ndarray): The 2D FFT magnitude data.
        center_yx (Tuple[int, int]): The (row_idx, col_idx) of the user click (kx, ky).
                                     Note: For this function, input (y,x) means (ky_idx, kx_idx).
        roi_radius (int): Radius of the square ROI (e.g., radius 2 means a 5x5 area).

    Returns:
        Tuple[int, int]: Refined (row_idx, col_idx) of the maximum intensity pixel
                         within the ROI, in the coordinate system of fft_magnitude_data.
    """
    if fft_magnitude_data is None or fft_magnitude_data.ndim != 2:
        logger.warning("find_max_pixel_in_roi: Invalid input image data.")
        return center_yx # Return original click if data is invalid

    center_y, center_x = center_yx # These are (ky_idx, kx_idx)

    # Define ROI boundaries, ensuring they are within the image
    img_rows, img_cols = fft_magnitude_data.shape
    y_start = max(0, center_y - roi_radius)
    y_end = min(img_rows, center_y + roi_radius + 1)
    x_start = max(0, center_x - roi_radius)
    x_end = min(img_cols, center_x + roi_radius + 1)
    print(f"y_start: {y_start}, y_end: {y_end}, x_start: {x_start}, x_end: {x_end}")

    if y_start >= y_end or x_start >= x_end:
        logger.warning("find_max_pixel_in_roi: Invalid ROI calculated (zero size). Returning original click.")
        return center_yx

    roi_patch = fft_magnitude_data[y_start:y_end, x_start:x_end]

    if roi_patch.size == 0:
        logger.warning("find_max_pixel_in_roi: Extracted ROI patch is empty. Returning original click.")
        return center_yx

    # Find indices of max value within the patch (relative to patch origin)
    dy_patch, dx_patch = np.unravel_index(np.argmax(roi_patch), roi_patch.shape)

    # Convert back to original image coordinates
    refined_y = y_start + dy_patch
    refined_x = x_start + dx_patch

    logger.debug(f"Max pixel refinement: Click ({center_y},{center_x}), ROI [{y_start}:{y_end},{x_start}:{x_end}], Refined ({refined_y},{refined_x})")
    return int(refined_y), int(refined_x)


def _gaussian_2d(xy_tuple, amplitude, y0, x0, sigma_y, sigma_x, theta, offset):
    """2D Gaussian function for fitting."""
    (y, x) = xy_tuple
    y0 = float(y0)
    x0 = float(x0)
    a = (np.cos(theta)**2)/(2*sigma_y**2) + (np.sin(theta)**2)/(2*sigma_x**2)
    b = -(np.sin(2*theta))/(4*sigma_y**2) + (np.sin(2*theta))/(4*sigma_x**2)
    c = (np.sin(theta)**2)/(2*sigma_y**2) + (np.cos(theta)**2)/(2*sigma_x**2)
    g = offset + amplitude * np.exp( - (a*((y-y0)**2) + 2*b*(y-y0)*(x-x0) + c*((x-x0)**2)))
    return g.ravel()


def fit_2d_gaussian_in_roi(
    fft_magnitude_data: np.ndarray,
    center_yx: Tuple[int, int],
    roi_radius: int
) -> Optional[Tuple[float, float]]:
    """
    Fits a 2D Gaussian to a square ROI in the FFT magnitude data.

    Args:
        fft_magnitude_data (np.ndarray): The 2D FFT magnitude data.
        center_yx (Tuple[int, int]): The (row_idx, col_idx) of the user click (kx, ky).
                                     Used as the center of the ROI for fitting.
        roi_radius (int): Radius of the square ROI.

    Returns:
        Optional[Tuple[float, float]]: Refined (row_idx_float, col_idx_float) of the
                                       Gaussian center (sub-pixel accuracy), or None if fit fails.
                                       Coordinates are in the system of fft_magnitude_data.
    """
    print(f"2d gaussian fit in roi")
    if not SCIPY_AVAILABLE:
        logger.error("fit_2d_gaussian_in_roi: SciPy is not available for curve_fit.")
        return None
    if fft_magnitude_data is None or fft_magnitude_data.ndim != 2:
        logger.warning("fit_2d_gaussian_in_roi: Invalid input image data.")
        return None
    print("fft_magnitude_data is valid")

    center_y, center_x = center_yx # These are (ky_idx, kx_idx)
    img_rows, img_cols = fft_magnitude_data.shape

    y_start = max(0, center_y - roi_radius)
    y_end = min(img_rows, center_y + roi_radius + 1)
    x_start = max(0, center_x - roi_radius)
    x_end = min(img_cols, center_x + roi_radius + 1)
    print(f"y_start: {y_start}, y_end: {y_end}, x_start: {x_start}, x_end: {x_end}")

    if y_start >= y_end or x_start >= x_end:
        logger.warning("fit_2d_gaussian_in_roi: Invalid ROI for fitting (zero size).")
        return None

    roi_patch = fft_magnitude_data[y_start:y_end, x_start:x_end]
    if roi_patch.size == 0:
        logger.warning("fit_2d_gaussian_in_roi: Extracted ROI patch for fitting is empty.")
        return None

    # Create coordinate grid for the patch
    y_roi_coords = np.arange(roi_patch.shape[0])
    x_roi_coords = np.arange(roi_patch.shape[1])
    X_roi, Y_roi = np.meshgrid(x_roi_coords, y_roi_coords)
    xy_roi_flat = (Y_roi.flatten(), X_roi.flatten()) # Tuple for _gaussian_2d function
    print(f"xy_roi_flat: {xy_roi_flat}")
    data_roi_flat = roi_patch.flatten()

    # Initial guess for parameters: amplitude, y0, x0, sigma_y, sigma_x, theta, offset
    # y0, x0 are relative to the patch's top-left corner
    initial_y0_patch = roi_patch.shape[0] / 2.0
    initial_x0_patch = roi_patch.shape[1] / 2.0
    initial_amplitude = np.max(roi_patch) - np.min(roi_patch)
    initial_offset = np.min(roi_patch)

    p0 = [
        initial_amplitude,  # amplitude
        initial_y0_patch,   # y0 (center of patch)
        initial_x0_patch,   # x0 (center of patch)
        float(roi_radius),  # sigma_y
        float(roi_radius),  # sigma_x
        0.0,                # theta (no rotation initially)
        initial_offset      # offset
    ]

    try:
        logger.debug(f"Fitting 2D Gaussian to ROI of size {roi_patch.shape} centered near ({center_y},{center_x}). Initial guess: {p0}")
        popt, pcov = curve_fit(_gaussian_2d, xy_roi_flat, data_roi_flat, p0=p0) #, bounds=bounds_tuple)

        amplitude_fit, y0_fit_patch, x0_fit_patch, sigma_y_fit, sigma_x_fit, theta_fit, offset_fit = popt

        # Convert fitted center (relative to patch) back to original image coordinates
        refined_y_float = y_start + y0_fit_patch
        refined_x_float = x_start + x0_fit_patch

        logger.info(f"2D Gaussian fit successful: Center_abs=({refined_y_float:.2f}, {refined_x_float:.2f})")
        return refined_y_float, refined_x_float

    except RuntimeError:
        logger.warning("2D Gaussian fit: Optimal parameters not found. Using click position.")
        return None # Or return center_yx as int if preferred fallback
    except Exception as e:
        logger.exception(f"Error during 2D Gaussian fitting: {e}")
        return None
    
def fit_2d_gaussian_in_roi_with_all_data(
    fft_magnitude_data: np.ndarray,
    center_yx: Tuple[int, int],
    roi_radius: int
) -> Optional[Tuple[np.ndarray, Tuple[float, float], np.ndarray]]:
    """
    Fits a 2D Gaussian to a square ROI in the FFT magnitude data.

    Args:
        fft_magnitude_data (np.ndarray): The 2D FFT magnitude data.
        center_yx (Tuple[int, int]): The (row_idx, col_idx) of the user click (ky, kx).
        roi_radius (int): Radius of the square ROI.

    Returns:
        Optional[Tuple[np.ndarray, Tuple[float, float], np.ndarray]]: 
            A tuple containing:
            - popt (np.ndarray): Optimal parameters for the Gaussian function.
            - (refined_y_float, refined_x_float): Refined center coordinates.
            - roi_patch (np.ndarray): The actual data patch used for fitting.
            Returns None if the fit fails.
    """
    if not SCIPY_AVAILABLE:
        logger.error("fit_2d_gaussian_in_roi: SciPy is not available for curve_fit.")
        return None
    if fft_magnitude_data is None or fft_magnitude_data.ndim != 2:
        logger.warning("fit_2d_gaussian_in_roi: Invalid input image data.")
        return None

    center_y, center_x = center_yx
    img_rows, img_cols = fft_magnitude_data.shape

    y_start = max(0, center_y - roi_radius)
    y_end = min(img_rows, center_y + roi_radius + 1)
    x_start = max(0, center_x - roi_radius)
    x_end = min(img_cols, center_x + roi_radius + 1)

    if y_start >= y_end or x_start >= x_end:
        logger.warning("fit_2d_gaussian_in_roi: Invalid ROI for fitting (zero size).")
        return None

    roi_patch = fft_magnitude_data[y_start:y_end, x_start:x_end]
    if roi_patch.size == 0:
        logger.warning("fit_2d_gaussian_in_roi: Extracted ROI patch for fitting is empty.")
        return None

    y_roi_coords = np.arange(roi_patch.shape[0])
    x_roi_coords = np.arange(roi_patch.shape[1])
    X_roi, Y_roi = np.meshgrid(x_roi_coords, y_roi_coords)
    xy_roi_flat = (Y_roi.flatten(), X_roi.flatten())
    data_roi_flat = roi_patch.flatten()

    initial_y0_patch = roi_patch.shape[0] / 2.0
    initial_x0_patch = roi_patch.shape[1] / 2.0
    initial_amplitude = np.max(roi_patch) - np.min(roi_patch)
    initial_offset = np.min(roi_patch)

    p0 = [initial_amplitude, initial_y0_patch, initial_x0_patch, float(roi_radius), float(roi_radius), 0.0, initial_offset]

    try:
        logger.debug(f"Fitting 2D Gaussian to ROI of size {roi_patch.shape} centered near ({center_y},{center_x}).")
        popt, pcov = curve_fit(_gaussian_2d, xy_roi_flat, data_roi_flat, p0=p0)
        
        y0_fit_patch, x0_fit_patch = popt[1], popt[2]

        refined_y_float = y_start + y0_fit_patch
        refined_x_float = x_start + x0_fit_patch

        logger.info(f"2D Gaussian fit successful: Center_abs=({refined_y_float:.2f}, {refined_x_float:.2f})")
        
        return popt, (refined_y_float, refined_x_float), roi_patch

    except RuntimeError:
        logger.warning("2D Gaussian fit: Optimal parameters not found. Using click position.")
        return None
    except Exception as e:
        logger.exception(f"Error during 2D Gaussian fitting: {e}")
        return None