# lfa/preprocessing/leveling.py
"""
Functions for plane fitting and subtraction for image leveling.
Adapted from user-provided code, using SciPy for fitting and NumPy.
"""
import logging
import numpy as np
from typing import Optional, Tuple, List

# Make sure SciPy is installed (add 'scipy' to requirements.txt)
try:
    from scipy.optimize import least_squares
except ImportError:
    logging.critical("SciPy not found. Please install it: pip install scipy")
    # Define a dummy function if scipy is not available to avoid hard crash on import
    def least_squares(*args, **kwargs):
        raise NotImplementedError("SciPy is required for least_squares fitting.")

logger = logging.getLogger(__name__)

def fit_plane(image: np.ndarray, roi_slice: Optional[Tuple[slice, slice]] = None) -> Optional[np.ndarray]:
    """
    Fits a plane (z = ax + by + c) to the image or a specified ROI using least squares.

    Args:
        image (np.ndarray): Input 2D image data (should be float).
        roi_slice (Optional[Tuple[slice, slice]]): A tuple of slices (row_slice, col_slice)
            specifying the region for fitting. If None, the whole image is used.

    Returns:
        Optional[np.ndarray]: A 2D numpy array of the same shape as the input image,
                              representing the fitted plane values across the whole image.
                              Returns None if fitting fails.
    """
    if image is None or image.ndim != 2:
        logger.error(f"fit_plane: Invalid input image (None or shape {image.shape}).")
        return None

    rows, cols = image.shape

    # --- Prepare coordinates and Z data for fitting ---
    if roi_slice is not None:
        row_slice, col_slice = roi_slice
        if not (isinstance(row_slice, slice) and isinstance(col_slice, slice)):
             logger.error(f"fit_plane: Invalid roi_slice format: {roi_slice}")
             return None

        # Define region bounds, handling None in slice start/stop
        r_start = row_slice.start if row_slice.start is not None else 0
        r_stop = row_slice.stop if row_slice.stop is not None else rows
        c_start = col_slice.start if col_slice.start is not None else 0
        c_stop = col_slice.stop if col_slice.stop is not None else cols

        # Check for valid ranges
        if not (0 <= r_start < r_stop <= rows and 0 <= c_start < c_stop <= cols):
             logger.error(f"fit_plane: ROI slice [{r_start}:{r_stop}, {c_start}:{c_stop}] is out of bounds for image shape {(rows, cols)}.")
             return None

        image_region = image[r_start:r_stop, c_start:c_stop]
        if image_region.size == 0:
             logger.error("fit_plane: ROI slice resulted in an empty region.")
             return None

        # Create coordinate grid for the *region*
        y_coords_region = np.arange(r_start, r_stop)
        x_coords_region = np.arange(c_start, c_stop)
        # Double check shape consistency after creating ranges
        if len(y_coords_region) != image_region.shape[0] or len(x_coords_region) != image_region.shape[1]:
             logger.error(f"fit_plane: Mismatch between calculated ranges ({len(y_coords_region)}x{len(x_coords_region)}) and region shape {image_region.shape}. Slice: {roi_slice}")
             return None

        X_region, Y_region = np.meshgrid(x_coords_region, y_coords_region)
        X_flat = X_region.flatten()
        Y_flat = Y_region.flatten()
        Z_flat = image_region.flatten()
        logger.debug(f"Fitting plane to ROI: Rows {row_slice}, Cols {col_slice}. Points: {len(Z_flat)}")

    else: # Use the whole image
        X, Y = np.meshgrid(np.arange(cols), np.arange(rows))
        X_flat = X.flatten()
        Y_flat = Y.flatten()
        Z_flat = image.flatten()
        logger.debug(f"Fitting plane to whole image. Points: {len(Z_flat)}")

    # Avoid fitting if data is constant
    if np.allclose(Z_flat, Z_flat[0]):
         logger.warning("fit_plane: Data within the fitting region is constant. Returning a flat plane.")
         a, b = 0.0, 0.0
         c = Z_flat[0]
    else:
        # Define the error function for least_squares: params = [a, b, c]
        def plane_residuals(params, x, y, z):
            a, b, c = params
            return a * x + b * y + c - z

        initial_guess = [0, 0, np.mean(Z_flat)] # Initial guess: flat plane at mean height

        try:
            result = least_squares(plane_residuals, initial_guess, args=(X_flat, Y_flat, Z_flat), method='lm')
            if not result.success or result.status < 1: # Check for positive status indicating convergence
                logger.warning(f"fit_plane: Least squares fitting did not converge successfully. Status: {result.status}, Message: {result.message}")
                return None
            a, b, c = result.x
            logger.info(f'Fitted plane parameters: a={a:.4e}, b={b:.4e}, c={c:.4f}')

        except NotImplementedError: # Handle case where SciPy is missing
             logger.critical("SciPy optimize is required for fit_plane but not found or dummy function used.")
             return None
        except Exception as e:
            logger.exception(f"Error during least squares fitting in fit_plane: {e}")
            return None

    # Create the fitted plane covering the *whole image* dimensions
    X_full, Y_full = np.meshgrid(np.arange(cols), np.arange(rows))
    fitted_plane_full = a * X_full + b * Y_full + c

    return fitted_plane_full.astype(np.float32)


def fit_plane_3pts(image: np.ndarray, points: List[Tuple[int, int]]) -> Optional[np.ndarray]:
    """
    Fits a plane (z = ax + by + c) defined by three points on the image.

    Args:
        image (np.ndarray): Input 2D image data (should be float).
        points (List[Tuple[int, int]]): A list containing three tuples (x, y).

    Returns:
        Optional[np.ndarray]: The fitted plane covering the whole image, or None on failure.
    """
    if image is None or image.ndim != 2: logger.error("fit_plane_3pts: Invalid input image."); return None
    if not isinstance(points, list) or len(points) != 3: logger.error(f"fit_plane_3pts: Requires 3 points, got {len(points)}."); return None

    rows, cols = image.shape
    logger.info(f"Fitting plane to 3 points: {points}")

    try:
        (x1, y1), (x2, y2), (x3, y3) = points
        # Boundary checks
        if not (0<=y1<rows and 0<=x1<cols and 0<=y2<rows and 0<=x2<cols and 0<=y3<rows and 0<=x3<cols):
             logger.error("fit_plane_3pts: Points out of bounds."); return None

        z1, z2, z3 = image[y1, x1], image[y2, x2], image[y3, x3]

        # Solve Ax = B for x = [a, b, c]
        A = np.array([[x1, y1, 1], [x2, y2, 1], [x3, y3, 1]], dtype=float)
        B = np.array([z1, z2, z3], dtype=float)

        try:
            plane_params = np.linalg.solve(A, B)
            a, b, c = plane_params
            logger.info(f'3pt plane parameters: a={a:.4e}, b={b:.4e}, c={c:.4f}')
        except np.linalg.LinAlgError:
            logger.error("fit_plane_3pts: Cannot solve system. Points might be collinear."); return None

        # Create the full plane
        X_full, Y_full = np.meshgrid(np.arange(cols), np.arange(rows))
        fitted_plane_full = a * X_full + b * Y_full + c

        return fitted_plane_full.astype(np.float32)

    except Exception as e:
        logger.exception(f"Error during 3-point plane fitting: {e}")
        return None


def level_by_plane(image: np.ndarray, fitted_plane: np.ndarray) -> Optional[np.ndarray]:
    """Subtracts the fitted plane from the image."""
    if image is None or fitted_plane is None: logger.error("level_by_plane: Input image or plane is None."); return None
    if image.shape != fitted_plane.shape: logger.error(f"level_by_plane: Shape mismatch {image.shape} vs {fitted_plane.shape}."); return None

    try:
        # Ensure both are float before subtraction
        leveled_image = image.astype(np.float32) - fitted_plane.astype(np.float32)
        return leveled_image
    except Exception as e:
         logger.exception(f"Error subtracting plane: {e}")
         return None