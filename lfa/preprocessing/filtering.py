# lfa/preprocessing/filtering.py
"""
Image filtering operations for LFA.
"""
import logging
import numpy as np
# Use SciPy for Gaussian filter, it's often faster for this specific task
# than scikit-image's version
from scipy.ndimage import gaussian_filter

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