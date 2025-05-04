# lfa/preprocessing/denoising.py
"""
Functions for image denoising operations.
"""
import logging
import numpy as np
from typing import Optional

# Upewnij się, że scikit-image jest zainstalowane
try:
    from skimage.restoration import denoise_nl_means
    # from skimage.util import img_as_float # denoise_nl_means robi to wewnętrznie
except ImportError:
    logging.critical("Scikit-image not found. Please install it: pip install scikit-image")
    # Dummy function if skimage is not available
    def denoise_nl_means(*args, **kwargs):
        logging.error("denoise_nl_means called, but scikit-image is not available.")
        return args[0] if args else None


logger = logging.getLogger(__name__)

def denoise_nlmeans_skimage(image: np.ndarray, sigma: float, h_param_mult: float = 1.0,
                            patch_size: int = 7, patch_distance: int = 11,
                            fast_mode: bool = True) -> Optional[np.ndarray]:
    """
    Applies Non-Local Means denoising using scikit-image.

    Args:
        image (np.ndarray): Input 2D image data (float or integer types).
        sigma (float): Estimated noise standard deviation. Crucial parameter.
        h_param_mult (float, optional): Multiplier for the 'h' parameter (filtering strength).
                                        h = h_param_mult * sigma. Defaults to 1.0.
        patch_size (int, optional): Size of patches used for comparison (odd integer). Defaults to 7.
        patch_distance (int, optional): Maximum distance (in pixels) to search for similar patches.
                                        Defaults to 11 (skimage default for fast_mode=True).
        fast_mode (bool, optional): Whether to use the fast algorithm approximation. Defaults to True.

    Returns:
        Optional[np.ndarray]: The denoised image data as float32, or None if an error occurs.
    """
    if image is None or image.ndim != 2:
        logger.error(f"NL-Means: Invalid input image (None or shape {getattr(image, 'shape', 'N/A')}).")
        return None
    if sigma <= 0:
        logger.error(f"NL-Means: Noise standard deviation sigma must be positive, got {sigma}.")
        return None
    if not isinstance(patch_size, int) or patch_size <= 0 or patch_size % 2 == 0:
        logger.error(f"NL-Means: patch_size must be a positive odd integer, got {patch_size}.")
        return None
    if not isinstance(patch_distance, int) or patch_distance <= 0:
        logger.error(f"NL-Means: patch_distance must be a positive integer, got {patch_distance}.")
        return None
    if h_param_mult <= 0:
         logger.warning(f"NL-Means: h_param_mult is non-positive ({h_param_mult}). Setting h directly to sigma.")
         h = sigma # Fallback if multiplier is invalid
    else:
         h = h_param_mult * sigma

    try:
        logger.debug(f"Applying NL-Means: sigma={sigma:.3f}, h={h:.3f} (mult={h_param_mult:.2f}), "
                     f"patch_size={patch_size}, patch_distance={patch_distance}, fast_mode={fast_mode}")

        # denoise_nl_means handles float conversion internally, returns float64 by default
        # preserve_range=True ensures output scale matches input scale
        denoised_image = denoise_nl_means(
            image,
            h=h,
            sigma=sigma,
            patch_size=patch_size,
            patch_distance=patch_distance,
            fast_mode=fast_mode,
            preserve_range=True,
            channel_axis=None # Indicate 2D grayscale image
        )
        logger.info("NL-Means denoising completed.")
        # Convert to float32 for consistency
        return denoised_image.astype(np.float32)

    except Exception as e:
        logger.exception(f"Error during NL-Means denoising: {e}")
        return None