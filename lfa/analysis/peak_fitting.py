from __future__ import annotations

"""
Peak refinement helpers used by FFT dialogs.

The module provides multiple sub-pixel refinement strategies together with
uncertainty estimates based on either analytical covariance (Gaussian fit) or
Monte-Carlo sampling of the ROI patch noise.
"""

import logging
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Optional, Tuple

import numpy as np

try:  # pragma: no cover - SciPy may be absent in limited environments
    from scipy.optimize import curve_fit

    SCIPY_AVAILABLE = True
except ImportError:  # pragma: no cover
    curve_fit = None  # type: ignore
    SCIPY_AVAILABLE = False
    logging.error("SciPy not found. 2D Gaussian fitting will not be available.")

logger = logging.getLogger(__name__)

DEFAULT_CURVE_FIT_MAXFEV = 5000
MC_SAMPLE_COUNT = 256
MC_SAMPLE_COUNT_FAST = 128
_MIN_NOISE_SIGMA = 1e-8


@dataclass
class PeakRefinementResult:
    """Container describing the outcome of a peak refinement step."""

    center: Tuple[float, float]  # (y_abs, x_abs)
    center_std: Optional[Tuple[float, float]]
    method: str
    success: bool
    roi_patch: np.ndarray
    noise_sigma: float
    residual_rms: float = float("nan")
    popt: Optional[np.ndarray] = None
    pcov: Optional[np.ndarray] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


def _estimate_noise_sigma(patch: np.ndarray) -> float:
    """Robustly estimate noise sigma for a given ROI patch."""
    if patch.size == 0:
        return 0.0
    patch = np.asarray(patch, dtype=float)
    median = float(np.median(patch))
    mad = float(np.median(np.abs(patch - median)))
    if mad > 0.0:
        return 1.4826 * mad
    std = float(np.std(patch, ddof=1)) if patch.size > 1 else 0.0
    return std


def _monte_carlo_uncertainty(
    base_patch: np.ndarray,
    noise_sigma: float,
    estimator: Callable[[np.ndarray], Optional[Tuple[float, float]]],
    runs: int = MC_SAMPLE_COUNT,
) -> Optional[Tuple[float, float]]:
    """Estimate coordinate uncertainty via Monte Carlo sampling."""
    if noise_sigma < _MIN_NOISE_SIGMA:
        return None
    rng = np.random.default_rng()
    coords = np.empty((runs, 2), dtype=float)
    count = 0
    for _ in range(runs):
        noisy_patch = base_patch + rng.normal(0.0, noise_sigma, base_patch.shape)
        estimate = estimator(noisy_patch)
        if estimate is None or not np.all(np.isfinite(estimate)):
            continue
        coords[count] = estimate
        count += 1
    if count == 0:
        return None
    coords = coords[:count]
    std = coords.std(axis=0, ddof=1) if count > 1 else np.zeros(2, dtype=float)
    return float(std[0]), float(std[1])


def _max_pixel_from_patch(patch: np.ndarray, y_start: int, x_start: int) -> Tuple[float, float]:
    dy, dx = np.unravel_index(np.argmax(patch), patch.shape)
    return float(y_start + dy), float(x_start + dx)


def _run_monte_carlo_max_pixel(
    roi_patch: np.ndarray,
    y_start: int,
    x_start: int,
    noise_sigma: float,
    runs: int = MC_SAMPLE_COUNT,
) -> Optional[PeakRefinementResult]:
    if noise_sigma < _MIN_NOISE_SIGMA or roi_patch.size == 0:
        return None

    center = _max_pixel_from_patch(roi_patch, y_start, x_start)
    std = _monte_carlo_uncertainty(
        roi_patch,
        noise_sigma,
        lambda noisy: _max_pixel_from_patch(noisy, y_start, x_start),
        runs=runs,
    )

    return PeakRefinementResult(
        center=center,
        center_std=std,
        method="monte_carlo_max",
        success=False,
        roi_patch=roi_patch.copy(),
        noise_sigma=float(noise_sigma),
        metadata={"mc_samples": runs},
    )


def find_max_pixel_in_roi(
    fft_magnitude_data: np.ndarray,
    center_yx: Tuple[int, int],
    roi_radius: int,
) -> Tuple[int, int]:
    """
    Finds the pixel with the maximum intensity within a square ROI centered on ``center_yx``.

    Args:
        fft_magnitude_data: 2D FFT magnitude data.
        center_yx: (row_idx, col_idx) pair of the initial click location.
        roi_radius: Radius of the square ROI (radius 2 => 5x5 region).
    """
    if fft_magnitude_data is None or fft_magnitude_data.ndim != 2:
        logger.warning("find_max_pixel_in_roi: Invalid input image data.")
        return center_yx

    center_y, center_x = center_yx
    img_rows, img_cols = fft_magnitude_data.shape
    y_start = max(0, center_y - roi_radius)
    y_end = min(img_rows, center_y + roi_radius + 1)
    x_start = max(0, center_x - roi_radius)
    x_end = min(img_cols, center_x + roi_radius + 1)

    if y_start >= y_end or x_start >= x_end:
        logger.warning("find_max_pixel_in_roi: Invalid ROI calculated (zero size). Returning original click.")
        return center_yx

    roi_patch = fft_magnitude_data[y_start:y_end, x_start:x_end]
    if roi_patch.size == 0:
        logger.warning("find_max_pixel_in_roi: Extracted ROI patch is empty. Returning original click.")
        return center_yx

    dy_patch, dx_patch = np.unravel_index(np.argmax(roi_patch), roi_patch.shape)
    refined_y = y_start + dy_patch
    refined_x = x_start + dx_patch

    logger.debug(
        "Max pixel refinement: Click (%s,%s), ROI [%s:%s,%s:%s], Refined (%s,%s)",
        center_y,
        center_x,
        y_start,
        y_end,
        x_start,
        x_end,
        refined_y,
        refined_x,
    )
    return int(refined_y), int(refined_x)


def _gaussian_2d(xy_tuple, amplitude, y0, x0, sigma_y, sigma_x, theta, offset):
    """2D Gaussian model for curve fitting."""
    (y, x) = xy_tuple
    y0 = float(y0)
    x0 = float(x0)
    a = (np.cos(theta) ** 2) / (2 * sigma_y**2) + (np.sin(theta) ** 2) / (2 * sigma_x**2)
    b = -(np.sin(2 * theta)) / (4 * sigma_y**2) + (np.sin(2 * theta)) / (4 * sigma_x**2)
    c = (np.sin(theta) ** 2) / (2 * sigma_y**2) + (np.cos(theta) ** 2) / (2 * sigma_x**2)
    g = offset + amplitude * np.exp(-(a * ((y - y0) ** 2) + 2 * b * (y - y0) * (x - x0) + c * ((x - x0) ** 2)))
    return g.ravel()


def fit_2d_gaussian_in_roi(
    fft_magnitude_data: np.ndarray,
    center_yx: Tuple[int, int],
    roi_radius: int,
) -> Optional[PeakRefinementResult]:
    """
    Fit a rotated 2D Gaussian to a square ROI in the FFT magnitude data.

    Returns ``PeakRefinementResult`` on success, or ``None`` when the fit cannot be
    performed (invalid ROI, missing SciPy, etc.).
    """
    if not SCIPY_AVAILABLE or curve_fit is None:
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

    roi_patch = np.asarray(roi_patch, dtype=float)
    if not np.isfinite(roi_patch).all():
        logger.warning("fit_2d_gaussian_in_roi: ROI patch contains non-finite values.")
        return None

    peak_to_peak = float(np.ptp(roi_patch))
    if peak_to_peak <= 0.0:
        logger.debug("fit_2d_gaussian_in_roi: ROI patch has no variation; skipping fit.")
        return None

    noise_sigma = _estimate_noise_sigma(roi_patch)
    noise_sigma_eff = max(noise_sigma, _MIN_NOISE_SIGMA)

    y_roi_coords = np.arange(roi_patch.shape[0], dtype=float)
    x_roi_coords = np.arange(roi_patch.shape[1], dtype=float)
    X_roi, Y_roi = np.meshgrid(x_roi_coords, y_roi_coords)
    xy_roi_flat = (Y_roi.ravel(), X_roi.ravel())
    data_roi_flat = roi_patch.ravel()

    initial_y0_patch = roi_patch.shape[0] / 2.0
    initial_x0_patch = roi_patch.shape[1] / 2.0
    initial_amplitude = peak_to_peak
    initial_offset = float(np.min(roi_patch))

    p0 = [
        initial_amplitude,
        initial_y0_patch,
        initial_x0_patch,
        max(float(roi_radius), 1.0),
        max(float(roi_radius), 1.0),
        0.0,
        initial_offset,
    ]

    try:
        popt, pcov = curve_fit(
            _gaussian_2d,
            xy_roi_flat,
            data_roi_flat,
            p0=p0,
            sigma=np.full_like(data_roi_flat, noise_sigma_eff, dtype=float),
            absolute_sigma=True,
            maxfev=DEFAULT_CURVE_FIT_MAXFEV,
        )
    except RuntimeError:
        logger.warning("fit_2d_gaussian_in_roi: curve_fit failed, attempting Monte Carlo fallback.")
        return _run_monte_carlo_max_pixel(roi_patch, y_start, x_start, noise_sigma)
    except Exception as exc:  # pragma: no cover - defensive path
        logger.exception("fit_2d_gaussian_in_roi: Unexpected error during fitting: %s", exc)
        return None

    refined_y_float = y_start + popt[1]
    refined_x_float = x_start + popt[2]

    model_flat = _gaussian_2d(xy_roi_flat, *popt)
    residuals = data_roi_flat - model_flat
    residual_rms = float(np.sqrt(np.mean(residuals**2)))

    center_std: Optional[Tuple[float, float]] = None
    if pcov is not None and pcov.shape[0] >= 3:
        diag = np.diag(pcov)
        if np.all(np.isfinite(diag[1:3])) and np.all(diag[1:3] >= 0.0):
            center_std = (float(np.sqrt(diag[1])), float(np.sqrt(diag[2])))

    if center_std is None:
        center_std = _monte_carlo_uncertainty(
            roi_patch,
            noise_sigma_eff,
            lambda noisy: _gaussian_fit_estimator(noisy, y_start, x_start, p0),
            runs=MC_SAMPLE_COUNT_FAST,
        )

    metadata: Dict[str, Any] = {
        "sigma_y_fit": float(popt[3]),
        "sigma_x_fit": float(popt[4]),
        "theta_fit": float(popt[5]),
        "noise_sigma": float(noise_sigma),
        "residual_rms": residual_rms,
    }

    return PeakRefinementResult(
        center=(float(refined_y_float), float(refined_x_float)),
        center_std=center_std,
        method="gaussian_fit",
        success=True,
        roi_patch=roi_patch.copy(),
        noise_sigma=float(noise_sigma),
        residual_rms=residual_rms,
        popt=popt,
        pcov=pcov,
        metadata=metadata,
    )


def _gaussian_fit_estimator(
    noisy_patch: np.ndarray,
    y_start: int,
    x_start: int,
    p0: list[float],
) -> Optional[Tuple[float, float]]:
    """Helper used during Monte Carlo when covariance is not available."""
    if not SCIPY_AVAILABLE or curve_fit is None:
        return None

    patch = np.asarray(noisy_patch, dtype=float)
    y_coords = np.arange(patch.shape[0], dtype=float)
    x_coords = np.arange(patch.shape[1], dtype=float)
    X, Y = np.meshgrid(x_coords, y_coords)
    xy_flat = (Y.ravel(), X.ravel())
    data_flat = patch.ravel()
    noise_sigma = max(_estimate_noise_sigma(patch), _MIN_NOISE_SIGMA)

    try:
        popt, _ = curve_fit(
            _gaussian_2d,
            xy_flat,
            data_flat,
            p0=p0,
            sigma=np.full_like(data_flat, noise_sigma, dtype=float),
            absolute_sigma=True,
            maxfev=DEFAULT_CURVE_FIT_MAXFEV,
        )
    except Exception:
        return None

    return float(y_start + popt[1]), float(x_start + popt[2])


def refine_peak_parabola_3x3(
    fft_magnitude_data: np.ndarray,
    center_yx: Tuple[int, int],
) -> Optional[PeakRefinementResult]:
    """
    Refine a peak using separable 1D parabolic interpolation on a 3x3 neighbourhood.
    """
    if fft_magnitude_data is None or fft_magnitude_data.ndim != 2:
        logger.warning("refine_peak_parabola_3x3: Invalid input image data.")
        return None

    center_y, center_x = center_yx
    img_rows, img_cols = fft_magnitude_data.shape
    y_start = center_y - 1
    y_end = center_y + 2
    x_start = center_x - 1
    x_end = center_x + 2

    if y_start < 0 or x_start < 0 or y_end > img_rows or x_end > img_cols:
        logger.debug("refine_peak_parabola_3x3: ROI touches image boundary; skipping.")
        return None

    patch = fft_magnitude_data[y_start:y_end, x_start:x_end].astype(float, copy=False)
    if patch.shape != (3, 3):
        logger.debug("refine_peak_parabola_3x3: ROI is not 3x3; skipping.")
        return None

    noise_sigma = _estimate_noise_sigma(patch)

    row = patch[1, :]
    col = patch[:, 1]
    denom_x = row[0] - 2 * row[1] + row[2]
    denom_y = col[0] - 2 * col[1] + col[2]
    if denom_x == 0 or denom_y == 0:
        logger.debug("refine_peak_parabola_3x3: Degenerate parabola; skipping.")
        return None

    dx = 0.5 * (row[0] - row[2]) / denom_x
    dy = 0.5 * (col[0] - col[2]) / denom_y
    dx = float(np.clip(dx, -1.0, 1.0))
    dy = float(np.clip(dy, -1.0, 1.0))

    refined_y = (y_start + 1) + dy
    refined_x = (x_start + 1) + dx

    std = _monte_carlo_uncertainty(
        patch,
        noise_sigma,
        lambda noisy: _parabola_estimator(noisy, y_start, x_start),
        runs=MC_SAMPLE_COUNT_FAST,
    )

    return PeakRefinementResult(
        center=(float(refined_y), float(refined_x)),
        center_std=std,
        method="parabola_3x3",
        success=True,
        roi_patch=patch.copy(),
        noise_sigma=float(noise_sigma),
        metadata={"roi_shape": (3, 3)},
    )


def _parabola_estimator(patch: np.ndarray, y_start: int, x_start: int) -> Optional[Tuple[float, float]]:
    if patch.shape != (3, 3):
        return None
    row = patch[1, :]
    col = patch[:, 1]
    denom_x = row[0] - 2 * row[1] + row[2]
    denom_y = col[0] - 2 * col[1] + col[2]
    if denom_x == 0 or denom_y == 0:
        return None
    dx = 0.5 * (row[0] - row[2]) / denom_x
    dy = 0.5 * (col[0] - col[2]) / denom_y
    dx = float(np.clip(dx, -1.0, 1.0))
    dy = float(np.clip(dy, -1.0, 1.0))
    return float((y_start + 1) + dy), float((x_start + 1) + dx)


def refine_peak_local_dft(
    fft_magnitude_data: np.ndarray,
    center_yx: Tuple[int, int],
    roi_radius: int,
    upsample_factor: int = 8,
) -> Optional[PeakRefinementResult]:
    """
    Refine a peak by locally upsampling the ROI using zero-padded DFT interpolation.
    """
    if fft_magnitude_data is None or fft_magnitude_data.ndim != 2:
        logger.warning("refine_peak_local_dft: Invalid input image data.")
        return None
    if upsample_factor < 2:
        logger.warning("refine_peak_local_dft: Upsample factor must be >= 2.")
        return None

    center_y, center_x = center_yx
    img_rows, img_cols = fft_magnitude_data.shape
    y_start = max(0, center_y - roi_radius)
    y_end = min(img_rows, center_y + roi_radius + 1)
    x_start = max(0, center_x - roi_radius)
    x_end = min(img_cols, center_x + roi_radius + 1)

    if y_start >= y_end or x_start >= x_end:
        logger.warning("refine_peak_local_dft: Invalid ROI for refinement (zero size).")
        return None

    patch = fft_magnitude_data[y_start:y_end, x_start:x_end].astype(float, copy=False)
    if patch.size == 0:
        logger.warning("refine_peak_local_dft: Extracted ROI patch is empty.")
        return None

    noise_sigma = _estimate_noise_sigma(patch)
    refined = _local_dft_peak(patch, y_start, x_start, upsample_factor)
    if refined is None:
        return None
    refined_y, refined_x = refined

    std = _monte_carlo_uncertainty(
        patch,
        noise_sigma,
        lambda noisy: _local_dft_peak(noisy, y_start, x_start, upsample_factor),
        runs=MC_SAMPLE_COUNT_FAST,
    )

    return PeakRefinementResult(
        center=(float(refined_y), float(refined_x)),
        center_std=std,
        method="local_dft",
        success=True,
        roi_patch=patch.copy(),
        noise_sigma=float(noise_sigma),
        metadata={"roi_shape": patch.shape, "upsample_factor": upsample_factor},
    )


def _local_dft_peak(
    patch: np.ndarray,
    y_start: int,
    x_start: int,
    upsample_factor: int,
) -> Optional[Tuple[float, float]]:
    """Upsample the ROI patch using zero-padded Fourier interpolation and locate the maximum."""
    rows, cols = patch.shape
    if rows == 0 or cols == 0:
        return None

    F = np.fft.fftshift(np.fft.fft2(patch))
    pad_rows = rows * upsample_factor
    pad_cols = cols * upsample_factor
    pad = np.zeros((pad_rows, pad_cols), dtype=complex)

    r_start = pad_rows // 2 - rows // 2
    c_start = pad_cols // 2 - cols // 2
    pad[r_start : r_start + rows, c_start : c_start + cols] = F

    upsampled = np.fft.ifft2(np.fft.ifftshift(pad))
    upsampled = np.abs(upsampled) * (upsample_factor**2)

    max_index = np.unravel_index(np.argmax(upsampled), upsampled.shape)
    refined_y = y_start + max_index[0] / upsample_factor
    refined_x = x_start + max_index[1] / upsample_factor
    return float(refined_y), float(refined_x)


def fit_2d_gaussian_in_roi_with_all_data(
    fft_magnitude_data: np.ndarray,
    center_yx: Tuple[int, int],
    roi_radius: int,
) -> Optional[PeakRefinementResult]:
    """
    Legacy compatibility wrapper returning the rich ``PeakRefinementResult``.
    """
    return fit_2d_gaussian_in_roi(fft_magnitude_data, center_yx, roi_radius)


__all__ = [
    "PeakRefinementResult",
    "find_max_pixel_in_roi",
    "fit_2d_gaussian_in_roi",
    "fit_2d_gaussian_in_roi_with_all_data",
    "refine_peak_parabola_3x3",
    "refine_peak_local_dft",
    "SCIPY_AVAILABLE",
]
