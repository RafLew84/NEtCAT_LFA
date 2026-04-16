"""Image preprocessing helpers for AtomMapper."""

from __future__ import annotations

from typing import Any

import numpy as np
from scipy.ndimage import gaussian_filter
from lfa.preprocessing.denoising import denoise_bm3d_lfa

try:
    from skimage.restoration import denoise_nl_means
except ImportError:  # pragma: no cover - dependency availability is env-specific
    denoise_nl_means = None

try:
    import bm3d as _bm3d_module  # noqa: F401
except ImportError:  # pragma: no cover - dependency availability is env-specific
    _bm3d_module = None

_BM3D_AVAILABLE = _bm3d_module is not None


def apply_blur(image_data: np.ndarray, *, sigma_px: float = 1.0, mode: str = "nearest") -> np.ndarray:
    """Return a blurred copy of a 2D STM image."""

    image_array = np.asarray(image_data, dtype=float)
    if image_array.ndim != 2:
        raise ValueError(f"Expected 2D image data, got shape {image_array.shape!r}.")

    sigma = float(sigma_px)
    if sigma <= 0.0:
        raise ValueError("sigma_px must be greater than 0.")

    return np.asarray(gaussian_filter(image_array, sigma=sigma, mode=mode), dtype=float)


def build_blur_metadata(*, sigma_px: float) -> dict[str, Any]:
    """Return metadata fields describing the blur preprocessing step."""

    return {
        "preprocess": "blur",
        "blur_sigma_px": float(sigma_px),
    }


def apply_rotation(image_data: np.ndarray, *, quarter_turns: int = 1) -> np.ndarray:
    """Return a copy of a 2D STM image rotated counter-clockwise in 90-degree steps."""

    image_array = np.asarray(image_data, dtype=float)
    if image_array.ndim != 2:
        raise ValueError(f"Expected 2D image data, got shape {image_array.shape!r}.")

    turns = int(quarter_turns) % 4
    if turns == 0:
        raise ValueError("quarter_turns must not be a multiple of 4.")

    return np.array(np.rot90(image_array, k=turns), dtype=float, copy=True)


def build_rotation_metadata(*, quarter_turns: int) -> dict[str, Any]:
    """Return metadata fields describing a 90-degree rotation preprocessing step."""

    turns = int(quarter_turns) % 4
    if turns == 0:
        raise ValueError("quarter_turns must not be a multiple of 4.")

    return {
        "preprocess": "rotate",
        "rotate_quarter_turns": turns,
        "rotate_angle_deg": 90 * turns,
        "rotate_direction": "ccw",
    }


def apply_flip(
    image_data: np.ndarray,
    *,
    flip_x: bool = True,
    flip_y: bool = False,
) -> np.ndarray:
    """Return a flipped copy of a 2D STM image."""

    image_array = np.asarray(image_data, dtype=float)
    if image_array.ndim != 2:
        raise ValueError(f"Expected 2D image data, got shape {image_array.shape!r}.")

    flip_x_value = bool(flip_x)
    flip_y_value = bool(flip_y)
    if not flip_x_value and not flip_y_value:
        raise ValueError("At least one of flip_x or flip_y must be enabled.")

    transformed = np.array(image_array, dtype=float, copy=True)
    if flip_x_value:
        transformed = np.fliplr(transformed)
    if flip_y_value:
        transformed = np.flipud(transformed)
    return np.array(transformed, dtype=float, copy=True)


def build_flip_metadata(*, flip_x: bool, flip_y: bool) -> dict[str, Any]:
    """Return metadata fields describing a flip preprocessing step."""

    flip_x_value = bool(flip_x)
    flip_y_value = bool(flip_y)
    if not flip_x_value and not flip_y_value:
        raise ValueError("At least one of flip_x or flip_y must be enabled.")

    return {
        "preprocess": "flip",
        "flip_x": flip_x_value,
        "flip_y": flip_y_value,
    }


def apply_non_local_means(
    image_data: np.ndarray,
    *,
    h: float = 0.1,
    patch_size: int = 5,
    patch_distance: int = 6,
    fast_mode: bool = True,
) -> np.ndarray:
    """Return an NLM-denoised copy of a 2D STM image."""

    if denoise_nl_means is None:
        raise RuntimeError("scikit-image is required for non-local means denoising.")

    image_array = np.asarray(image_data, dtype=float)
    if image_array.ndim != 2:
        raise ValueError(f"Expected 2D image data, got shape {image_array.shape!r}.")

    h_value = float(h)
    if h_value <= 0.0:
        raise ValueError("h must be greater than 0.")

    patch_size_value = int(patch_size)
    if patch_size_value <= 0 or patch_size_value % 2 == 0:
        raise ValueError("patch_size must be a positive odd integer.")

    patch_distance_value = int(patch_distance)
    if patch_distance_value <= 0:
        raise ValueError("patch_distance must be greater than 0.")

    denoised = denoise_nl_means(
        image_array,
        h=h_value,
        patch_size=patch_size_value,
        patch_distance=patch_distance_value,
        fast_mode=bool(fast_mode),
        preserve_range=True,
        channel_axis=None,
    )
    return np.asarray(denoised, dtype=float)


def build_nlm_metadata(
    *,
    h: float,
    patch_size: int,
    patch_distance: int,
    fast_mode: bool,
) -> dict[str, Any]:
    """Return metadata fields describing the NLM preprocessing step."""

    return {
        "preprocess": "nlm",
        "nlm_h": float(h),
        "nlm_patch_size": int(patch_size),
        "nlm_patch_distance": int(patch_distance),
        "nlm_fast_mode": bool(fast_mode),
    }


def is_bm3d_available() -> bool:
    """Return ``True`` when the optional BM3D dependency is available."""

    return bool(_BM3D_AVAILABLE)


def apply_bm3d(image_data: np.ndarray, *, sigma_psd: float = 0.1) -> np.ndarray:
    """Return a BM3D-denoised copy of a 2D STM image."""

    if not is_bm3d_available():
        raise RuntimeError("BM3D package is not available.")

    image_array = np.asarray(image_data, dtype=float)
    if image_array.ndim != 2:
        raise ValueError(f"Expected 2D image data, got shape {image_array.shape!r}.")

    sigma_value = float(sigma_psd)
    if sigma_value <= 0.0:
        raise ValueError("sigma_psd must be greater than 0.")

    denoised = denoise_bm3d_lfa(image_array, sigma_psd=sigma_value)
    if denoised is None:
        raise RuntimeError("BM3D backend failed to produce an output image.")

    return np.asarray(denoised, dtype=float)


def build_bm3d_metadata(*, sigma_psd: float, stage: str) -> dict[str, Any]:
    """Return metadata fields describing the BM3D preprocessing step."""

    return {
        "preprocess": "bm3d",
        "bm3d_sigma_psd": float(sigma_psd),
        "bm3d_stage": str(stage).strip().lower() or "all_stages",
    }
