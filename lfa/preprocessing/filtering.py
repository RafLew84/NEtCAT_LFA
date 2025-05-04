# lfa/preprocessing/filtering.py
"""
Image filtering operations for LFA.
"""
import logging
import numpy as np
from typing import Optional
# Use SciPy for Gaussian filter, it's often faster for this specific task
# than scikit-image's version
try:
    from scipy.ndimage import gaussian_filter, median_filter
except ImportError:
    logging.critical("SciPy not found. pip install scipy")
    def gaussian_filter(image, sigma, **kwargs): return image
    def median_filter(image, size, **kwargs): return image 

logger = logging.getLogger(__name__)

def gaussian_blur(image: np.ndarray, sigma: float) -> np.ndarray:
    """
    Applies a Gaussian blur filter to the image.

    Args:
        image (np.ndarray): The input 2D image data.
        sigma (float): Standard deviation for Gaussian kernel. Controls the blur intensity.
                       Should be non-negative.

    Returns:
        np.ndarray: The blurred image data as a float32 array.
                    Returns the original image if sigma is effectively zero.
    """
    if image is None:
        logger.error("gaussian_blur: Input image is None.")
        return None # Or raise error
    if sigma < 0:
        logger.warning(f"gaussian_blur: Sigma value {sigma} is negative. Clamping to 0.")
        sigma = 0

    # Avoid processing if sigma is negligible
    if np.isclose(sigma, 0):
        logger.debug("gaussian_blur: Sigma is close to zero, returning original image.")
        # Ensure output is float32 for consistency
        return image.astype(np.float32, copy=False) # Avoid copy if already float32

    try:
        # Ensure input is float for filtering, output is float32
        blurred_image = gaussian_filter(image.astype(float, copy=False), sigma=sigma)
        logger.debug(f"Applied Gaussian blur with sigma={sigma:.2f}")
        return blurred_image.astype(np.float32)
    except Exception as e:
        logger.exception(f"Error during Gaussian filtering with sigma={sigma}: {e}")
        # Return original image on error to prevent crash downstream
        return image.astype(np.float32, copy=False)
    
def median_filter_lfa(image: np.ndarray, size: int = 3, mode: str = 'reflect', cval: float = 0.0) -> Optional[np.ndarray]:
    """
    Applies a median filter to the image using scipy.ndimage.median_filter.

    Effective for reducing salt-and-pepper noise while preserving edges.

    Args:
        image (np.ndarray): Input 2D image data (should be float or integer type).
        size (int, optional): The size of the filter neighborhood.
                               Should ideally be an odd integer. Defaults to 3.
        mode (str, optional): The mode parameter determines how the input array
                              is extended beyond its boundaries.
                              Options: 'reflect', 'constant', 'nearest', 'mirror', 'wrap'.
                              Defaults to 'reflect'.
        cval (float, optional): Value to fill past edges of input if mode is 'constant'.
                                Defaults to 0.0.

    Returns:
        Optional[np.ndarray]: The filtered image data as float32, or None if an error occurs.
    """
    if image is None or image.ndim != 2:
        logger.error(f"median_filter_lfa: Invalid input image (None or shape {getattr(image, 'shape', 'N/A')}).")
        return None
    if not isinstance(size, int) or size <= 0:
        logger.error(f"median_filter_lfa: Size must be a positive integer, got {size}.")
        return None

    if size % 2 == 0:
         logger.warning(f"median_filter_lfa: Filter size {size} is even. Odd sizes are generally preferred.")

    valid_modes = {'reflect', 'constant', 'nearest', 'mirror', 'wrap'}
    if mode not in valid_modes:
        logger.error(f"median_filter_lfa: Invalid mode '{mode}'. Valid modes are: {valid_modes}")
        return None

    try:
        logger.debug(f"Applying median filter with size={size}, mode='{mode}', cval={cval}")
        # median_filter zachowuje typ danych, konwertujemy na float32 na końcu dla spójności
        filtered_image = median_filter(image, size=size, mode=mode, cval=cval)
        return filtered_image.astype(np.float32)
    except Exception as e:
        logger.exception(f"Error during median filtering: {e}")
        return None