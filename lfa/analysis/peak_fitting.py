from __future__ import annotations

"""
Peak refinement helpers used by FFT dialogs.

The module provides multiple sub-pixel refinement strategies together with
uncertainty estimates based on either analytical covariance (Gaussian fit) or
Monte-Carlo sampling of the ROI patch noise.
"""

import logging
import warnings
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Optional, Tuple

import numpy as np

try:  # pragma: no cover
    from scipy.optimize import OptimizeWarning, curve_fit
    from scipy.special import voigt_profile

    SCIPY_AVAILABLE = True
except ImportError:  # pragma: no cover
    curve_fit = None 
    OptimizeWarning = Warning
    voigt_profile = None
    SCIPY_AVAILABLE = False
    logging.error("SciPy not found. 2D Gaussian fitting will not be available.")

logger = logging.getLogger(__name__)

DEFAULT_CURVE_FIT_MAXFEV = 5000
MC_SAMPLE_COUNT = 256
MC_SAMPLE_COUNT_FAST = 128
_MIN_NOISE_SIGMA = 1e-8
POSITION_SIGMA_FLOOR_PX = float(1.0 / np.sqrt(12.0))


def _curve_fit_safely(*args, **kwargs):
    """Run ``scipy.optimize.curve_fit`` while suppressing non-fatal covariance warnings."""

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=OptimizeWarning)
        return curve_fit(*args, **kwargs)


def _apply_position_sigma_floor(
    center_std: Optional[Tuple[float, float]],
    floor_px: float = POSITION_SIGMA_FLOOR_PX,
) -> Optional[Tuple[float, float]]:
    if center_std is None:
        return None
    if floor_px <= 0.0:
        return center_std
    return (
        float(np.hypot(center_std[0], floor_px)),
        float(np.hypot(center_std[1], floor_px)),
    )


@dataclass
class PeakRefinementResult:
    """Container describing the outcome of a peak refinement step."""

    center: Tuple[float, float] 
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


def _normalize_fit_mask(roi_patch: np.ndarray, fit_mask: Optional[np.ndarray]) -> Optional[np.ndarray]:
    """Validate and normalize an optional ROI-local boolean mask."""

    if fit_mask is None:
        return None
    mask = np.asarray(fit_mask, dtype=bool)
    if mask.shape != roi_patch.shape:
        logger.warning(
            "Local fit mask shape %r does not match ROI patch shape %r.",
            mask.shape,
            roi_patch.shape,
        )
        return None
    return mask


def _prepare_fit_sample_arrays(
    roi_patch: np.ndarray,
    fit_mask: Optional[np.ndarray],
    *,
    minimum_sample_count: int,
) -> Optional[Tuple[Tuple[np.ndarray, np.ndarray], np.ndarray, np.ndarray]]:
    """Build masked coordinate/data arrays for curve fitting."""

    mask = _normalize_fit_mask(roi_patch, fit_mask)
    if fit_mask is None:
        mask = np.ones(roi_patch.shape, dtype=bool)
    if mask is None:
        return None

    selected_count = int(mask.sum())
    if selected_count < max(int(minimum_sample_count), 1):
        logger.warning(
            "Local fit requires at least %s selected pixels, got %s.",
            max(int(minimum_sample_count), 1),
            selected_count,
        )
        return None

    y_roi_coords = np.arange(roi_patch.shape[0], dtype=float)
    x_roi_coords = np.arange(roi_patch.shape[1], dtype=float)
    X_roi, Y_roi = np.meshgrid(x_roi_coords, y_roi_coords)
    xy_roi_flat = (Y_roi[mask].ravel(), X_roi[mask].ravel())
    data_roi_flat = roi_patch[mask].ravel()
    return xy_roi_flat, data_roi_flat, mask


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


def _max_pixel_from_patch(
    patch: np.ndarray,
    y_start: int,
    x_start: int,
    fit_mask: Optional[np.ndarray] = None,
) -> Optional[Tuple[float, float]]:
    mask = _normalize_fit_mask(patch, fit_mask)
    if fit_mask is not None and mask is None:
        return None
    if mask is not None:
        if not mask.any():
            return None
        patch = np.where(mask, patch, -np.inf)
    dy, dx = np.unravel_index(np.argmax(patch), patch.shape)
    return float(y_start + dy), float(x_start + dx)


def _run_monte_carlo_max_pixel(
    roi_patch: np.ndarray,
    y_start: int,
    x_start: int,
    noise_sigma: float,
    fit_mask: Optional[np.ndarray] = None,
    runs: int = MC_SAMPLE_COUNT,
) -> Optional[PeakRefinementResult]:
    if noise_sigma < _MIN_NOISE_SIGMA or roi_patch.size == 0:
        return None

    center = _max_pixel_from_patch(roi_patch, y_start, x_start, fit_mask)
    if center is None:
        return None
    std = _monte_carlo_uncertainty(
        roi_patch,
        noise_sigma,
        lambda noisy: _max_pixel_from_patch(noisy, y_start, x_start, fit_mask),
        runs=runs,
    )
    std = _apply_position_sigma_floor(std)

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


def _safe_positive_scale(value: float, floor: float = 1e-12) -> float:
    """Clamp a scale-like parameter away from zero for stable model evaluation."""

    return max(abs(float(value)), floor)


def _lorentzian_2d(xy_tuple, amplitude, y0, x0, gamma_y, gamma_x, theta, offset):
    """Rotated 2D Lorentzian model for curve fitting."""

    (y, x) = xy_tuple
    y0 = float(y0)
    x0 = float(x0)
    gamma_y = _safe_positive_scale(gamma_y)
    gamma_x = _safe_positive_scale(gamma_x)

    dy = y - y0
    dx = x - x0
    y_rot = np.cos(theta) * dy + np.sin(theta) * dx
    x_rot = -np.sin(theta) * dy + np.cos(theta) * dx
    q = (y_rot / gamma_y) ** 2 + (x_rot / gamma_x) ** 2
    model = offset + amplitude / (1.0 + q)
    return model.ravel()


def _normalized_voigt_1d(axis_delta: np.ndarray, sigma: float, gamma: float) -> np.ndarray:
    """Return a Voigt profile normalized to 1 at the line center."""

    sigma = _safe_positive_scale(sigma)
    gamma = _safe_positive_scale(gamma)
    if voigt_profile is None:  # pragma: no cover
        raise RuntimeError("SciPy voigt_profile is not available.")
    profile = voigt_profile(axis_delta, sigma, gamma)
    center_value = float(voigt_profile(np.array([0.0]), sigma, gamma)[0])
    if not np.isfinite(center_value) or center_value <= 0.0:
        center_value = 1.0
    return np.asarray(profile, dtype=float) / center_value


def _voigt_2d(xy_tuple, amplitude, y0, x0, sigma_y, sigma_x, gamma_y, gamma_x, theta, offset):
    """Rotated separable 2D Voigt model for curve fitting."""

    (y, x) = xy_tuple
    y0 = float(y0)
    x0 = float(x0)
    sigma_y = _safe_positive_scale(sigma_y)
    sigma_x = _safe_positive_scale(sigma_x)
    gamma_y = _safe_positive_scale(gamma_y)
    gamma_x = _safe_positive_scale(gamma_x)

    dy = y - y0
    dx = x - x0
    y_rot = np.cos(theta) * dy + np.sin(theta) * dx
    x_rot = -np.sin(theta) * dy + np.cos(theta) * dx
    profile_y = _normalized_voigt_1d(y_rot, sigma_y, gamma_y)
    profile_x = _normalized_voigt_1d(x_rot, sigma_x, gamma_x)
    model = offset + amplitude * profile_y * profile_x
    return model.ravel()


def _fit_2d_gaussian_patch_with_initial_guess(
    roi_patch: np.ndarray,
    y_start: int,
    x_start: int,
    p0: list[float],
    noise_sigma: float,
    noise_sigma_eff: float,
    *,
    fit_mask: Optional[np.ndarray] = None,
    parameter_bounds: Optional[Tuple[list[float], list[float]]] = None,
    max_nfev: int = DEFAULT_CURVE_FIT_MAXFEV,
    compute_uncertainty: bool = True,
) -> Optional[PeakRefinementResult]:
    """Fit a 2D Gaussian on an already extracted ROI patch."""

    prepared = _prepare_fit_sample_arrays(
        roi_patch,
        fit_mask,
        minimum_sample_count=max(len(p0), 7),
    )
    if prepared is None:
        return _run_monte_carlo_max_pixel(roi_patch, y_start, x_start, noise_sigma, fit_mask=fit_mask)
    xy_roi_flat, data_roi_flat, normalized_mask = prepared

    try:
        bounds = parameter_bounds if parameter_bounds is not None else (-np.inf, np.inf)
        popt, pcov = _curve_fit_safely(
            _gaussian_2d,
            xy_roi_flat,
            data_roi_flat,
            p0=p0,
            bounds=bounds,
            sigma=np.full_like(data_roi_flat, noise_sigma_eff, dtype=float),
            absolute_sigma=True,
            maxfev=max(int(max_nfev), 10),
        )
    except RuntimeError:
        logger.warning("fit_2d_gaussian_in_roi: curve_fit failed, attempting Monte Carlo fallback.")
        return _run_monte_carlo_max_pixel(roi_patch, y_start, x_start, noise_sigma, fit_mask=fit_mask)
    except Exception as exc:  # pragma: no cover
        logger.exception("fit_2d_gaussian_in_roi: Unexpected error during fitting: %s", exc)
        return None

    refined_y_float = y_start + popt[1]
    refined_x_float = x_start + popt[2]

    model_flat = _gaussian_2d(xy_roi_flat, *popt)
    residuals = data_roi_flat - model_flat
    residual_rms = float(np.sqrt(np.mean(residuals**2)))
    chi2_red = None
    if noise_sigma_eff > 0.0:
        dof = max(int(data_roi_flat.size) - int(len(popt)), 1)
        chi2 = float(np.sum((residuals / noise_sigma_eff) ** 2))
        chi2_red = chi2 / float(dof)
        if pcov is not None and np.isfinite(chi2_red) and chi2_red > 1.0:
            pcov = pcov * float(chi2_red)

    center_std: Optional[Tuple[float, float]] = None
    if compute_uncertainty:
        if pcov is not None and pcov.shape[0] >= 3:
            diag = np.diag(pcov)
            if np.all(np.isfinite(diag[1:3])) and np.all(diag[1:3] >= 0.0):
                center_std = (float(np.sqrt(diag[1])), float(np.sqrt(diag[2])))

        if center_std is None:
            center_std = _monte_carlo_uncertainty(
                roi_patch,
                noise_sigma_eff,
                lambda noisy: _gaussian_fit_estimator(
                    noisy,
                    y_start,
                    x_start,
                    p0,
                    fit_mask=normalized_mask,
                    parameter_bounds=parameter_bounds,
                    max_nfev=max_nfev,
                ),
                runs=MC_SAMPLE_COUNT_FAST,
            )
        center_std = _apply_position_sigma_floor(center_std)

    metadata: Dict[str, Any] = {
        "sigma_y_fit": float(popt[3]),
        "sigma_x_fit": float(popt[4]),
        "theta_fit": float(popt[5]),
        "noise_sigma": float(noise_sigma),
        "residual_rms": residual_rms,
        "chi2_reduced": float(chi2_red) if chi2_red is not None else None,
        "position_sigma_floor_px": float(POSITION_SIGMA_FLOOR_PX),
        "roi_origin": (int(y_start), int(x_start)),
        "fit_mask_pixel_count": int(normalized_mask.sum()),
        "fit_mask_fraction": float(normalized_mask.mean()),
        "initial_params": tuple(float(param) for param in p0),
        "parameter_bounds": None
        if parameter_bounds is None
        else {
            "lower": tuple(float(param) for param in parameter_bounds[0]),
            "upper": tuple(float(param) for param in parameter_bounds[1]),
        },
        "max_nfev": int(max(max_nfev, 10)),
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


def _fit_2d_lorentzian_patch_with_initial_guess(
    roi_patch: np.ndarray,
    y_start: int,
    x_start: int,
    p0: list[float],
    noise_sigma: float,
    noise_sigma_eff: float,
    *,
    fit_mask: Optional[np.ndarray] = None,
    parameter_bounds: Optional[Tuple[list[float], list[float]]] = None,
    max_nfev: int = DEFAULT_CURVE_FIT_MAXFEV,
    compute_uncertainty: bool = True,
) -> Optional[PeakRefinementResult]:
    """Fit a rotated 2D Lorentzian on an already extracted ROI patch."""

    prepared = _prepare_fit_sample_arrays(
        roi_patch,
        fit_mask,
        minimum_sample_count=max(len(p0), 7),
    )
    if prepared is None:
        return _run_monte_carlo_max_pixel(roi_patch, y_start, x_start, noise_sigma, fit_mask=fit_mask)
    xy_roi_flat, data_roi_flat, normalized_mask = prepared

    try:
        bounds = parameter_bounds if parameter_bounds is not None else (-np.inf, np.inf)
        popt, pcov = _curve_fit_safely(
            _lorentzian_2d,
            xy_roi_flat,
            data_roi_flat,
            p0=p0,
            bounds=bounds,
            sigma=np.full_like(data_roi_flat, noise_sigma_eff, dtype=float),
            absolute_sigma=True,
            maxfev=max(int(max_nfev), 10),
        )
    except RuntimeError:
        logger.warning("fit_2d_lorentzian_on_patch: curve_fit failed, attempting Monte Carlo fallback.")
        return _run_monte_carlo_max_pixel(roi_patch, y_start, x_start, noise_sigma, fit_mask=fit_mask)
    except Exception as exc:  # pragma: no cover
        logger.exception("fit_2d_lorentzian_on_patch: Unexpected error during fitting: %s", exc)
        return None

    refined_y_float = y_start + popt[1]
    refined_x_float = x_start + popt[2]

    model_flat = _lorentzian_2d(xy_roi_flat, *popt)
    residuals = data_roi_flat - model_flat
    residual_rms = float(np.sqrt(np.mean(residuals**2)))
    chi2_red = None
    if noise_sigma_eff > 0.0:
        dof = max(int(data_roi_flat.size) - int(len(popt)), 1)
        chi2 = float(np.sum((residuals / noise_sigma_eff) ** 2))
        chi2_red = chi2 / float(dof)
        if pcov is not None and np.isfinite(chi2_red) and chi2_red > 1.0:
            pcov = pcov * float(chi2_red)

    center_std: Optional[Tuple[float, float]] = None
    if compute_uncertainty:
        if pcov is not None and pcov.shape[0] >= 3:
            diag = np.diag(pcov)
            if np.all(np.isfinite(diag[1:3])) and np.all(diag[1:3] >= 0.0):
                center_std = (float(np.sqrt(diag[1])), float(np.sqrt(diag[2])))

        if center_std is None:
            center_std = _monte_carlo_uncertainty(
                roi_patch,
                noise_sigma_eff,
                lambda noisy: _lorentzian_fit_estimator(
                    noisy,
                    y_start,
                    x_start,
                    p0,
                    fit_mask=normalized_mask,
                    parameter_bounds=parameter_bounds,
                    max_nfev=max_nfev,
                ),
                runs=MC_SAMPLE_COUNT_FAST,
            )
        center_std = _apply_position_sigma_floor(center_std)

    metadata: Dict[str, Any] = {
        "gamma_y_fit": float(popt[3]),
        "gamma_x_fit": float(popt[4]),
        "theta_fit": float(popt[5]),
        "noise_sigma": float(noise_sigma),
        "residual_rms": residual_rms,
        "chi2_reduced": float(chi2_red) if chi2_red is not None else None,
        "position_sigma_floor_px": float(POSITION_SIGMA_FLOOR_PX),
        "roi_origin": (int(y_start), int(x_start)),
        "fit_mask_pixel_count": int(normalized_mask.sum()),
        "fit_mask_fraction": float(normalized_mask.mean()),
        "initial_params": tuple(float(param) for param in p0),
        "parameter_bounds": None
        if parameter_bounds is None
        else {
            "lower": tuple(float(param) for param in parameter_bounds[0]),
            "upper": tuple(float(param) for param in parameter_bounds[1]),
        },
        "max_nfev": int(max(max_nfev, 10)),
    }

    return PeakRefinementResult(
        center=(float(refined_y_float), float(refined_x_float)),
        center_std=center_std,
        method="lorentzian_fit",
        success=True,
        roi_patch=roi_patch.copy(),
        noise_sigma=float(noise_sigma),
        residual_rms=residual_rms,
        popt=popt,
        pcov=pcov,
        metadata=metadata,
    )


def _fit_2d_voigt_patch_with_initial_guess(
    roi_patch: np.ndarray,
    y_start: int,
    x_start: int,
    p0: list[float],
    noise_sigma: float,
    noise_sigma_eff: float,
    *,
    fit_mask: Optional[np.ndarray] = None,
    parameter_bounds: Optional[Tuple[list[float], list[float]]] = None,
    max_nfev: int = DEFAULT_CURVE_FIT_MAXFEV,
    compute_uncertainty: bool = True,
) -> Optional[PeakRefinementResult]:
    """Fit a rotated 2D Voigt on an already extracted ROI patch."""

    prepared = _prepare_fit_sample_arrays(
        roi_patch,
        fit_mask,
        minimum_sample_count=max(len(p0), 9),
    )
    if prepared is None:
        return _run_monte_carlo_max_pixel(roi_patch, y_start, x_start, noise_sigma, fit_mask=fit_mask)
    xy_roi_flat, data_roi_flat, normalized_mask = prepared

    try:
        bounds = parameter_bounds if parameter_bounds is not None else (-np.inf, np.inf)
        popt, pcov = _curve_fit_safely(
            _voigt_2d,
            xy_roi_flat,
            data_roi_flat,
            p0=p0,
            bounds=bounds,
            sigma=np.full_like(data_roi_flat, noise_sigma_eff, dtype=float),
            absolute_sigma=True,
            maxfev=max(int(max_nfev), 10),
        )
    except RuntimeError:
        logger.warning("fit_2d_voigt_on_patch: curve_fit failed, attempting Monte Carlo fallback.")
        return _run_monte_carlo_max_pixel(roi_patch, y_start, x_start, noise_sigma, fit_mask=fit_mask)
    except Exception as exc:  # pragma: no cover
        logger.exception("fit_2d_voigt_on_patch: Unexpected error during fitting: %s", exc)
        return None

    refined_y_float = y_start + popt[1]
    refined_x_float = x_start + popt[2]

    model_flat = _voigt_2d(xy_roi_flat, *popt)
    residuals = data_roi_flat - model_flat
    residual_rms = float(np.sqrt(np.mean(residuals**2)))
    chi2_red = None
    if noise_sigma_eff > 0.0:
        dof = max(int(data_roi_flat.size) - int(len(popt)), 1)
        chi2 = float(np.sum((residuals / noise_sigma_eff) ** 2))
        chi2_red = chi2 / float(dof)
        if pcov is not None and np.isfinite(chi2_red) and chi2_red > 1.0:
            pcov = pcov * float(chi2_red)

    center_std: Optional[Tuple[float, float]] = None
    if compute_uncertainty:
        if pcov is not None and pcov.shape[0] >= 3:
            diag = np.diag(pcov)
            if np.all(np.isfinite(diag[1:3])) and np.all(diag[1:3] >= 0.0):
                center_std = (float(np.sqrt(diag[1])), float(np.sqrt(diag[2])))

        if center_std is None:
            center_std = _monte_carlo_uncertainty(
                roi_patch,
                noise_sigma_eff,
                lambda noisy: _voigt_fit_estimator(
                    noisy,
                    y_start,
                    x_start,
                    p0,
                    fit_mask=normalized_mask,
                    parameter_bounds=parameter_bounds,
                    max_nfev=max_nfev,
                ),
                runs=MC_SAMPLE_COUNT_FAST,
            )
        center_std = _apply_position_sigma_floor(center_std)

    metadata: Dict[str, Any] = {
        "sigma_y_fit": float(popt[3]),
        "sigma_x_fit": float(popt[4]),
        "gamma_y_fit": float(popt[5]),
        "gamma_x_fit": float(popt[6]),
        "theta_fit": float(popt[7]),
        "noise_sigma": float(noise_sigma),
        "residual_rms": residual_rms,
        "chi2_reduced": float(chi2_red) if chi2_red is not None else None,
        "position_sigma_floor_px": float(POSITION_SIGMA_FLOOR_PX),
        "roi_origin": (int(y_start), int(x_start)),
        "fit_mask_pixel_count": int(normalized_mask.sum()),
        "fit_mask_fraction": float(normalized_mask.mean()),
        "initial_params": tuple(float(param) for param in p0),
        "parameter_bounds": None
        if parameter_bounds is None
        else {
            "lower": tuple(float(param) for param in parameter_bounds[0]),
            "upper": tuple(float(param) for param in parameter_bounds[1]),
        },
        "max_nfev": int(max(max_nfev, 10)),
    }

    return PeakRefinementResult(
        center=(float(refined_y_float), float(refined_x_float)),
        center_std=center_std,
        method="voigt_fit",
        success=True,
        roi_patch=roi_patch.copy(),
        noise_sigma=float(noise_sigma),
        residual_rms=residual_rms,
        popt=popt,
        pcov=pcov,
        metadata=metadata,
    )


def fit_2d_gaussian_in_roi(
    fft_magnitude_data: np.ndarray,
    center_yx: Tuple[int, int],
    roi_radius: int,
    *,
    initial_params: Optional[list[float]] = None,
    parameter_bounds: Optional[Tuple[list[float], list[float]]] = None,
    max_nfev: int = DEFAULT_CURVE_FIT_MAXFEV,
    compute_uncertainty: bool = True,
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

    initial_y0_patch = roi_patch.shape[0] / 2.0
    initial_x0_patch = roi_patch.shape[1] / 2.0
    initial_amplitude = peak_to_peak
    initial_offset = float(np.min(roi_patch))

    default_p0 = [
        initial_amplitude,
        initial_y0_patch,
        initial_x0_patch,
        max(float(roi_radius), 1.0),
        max(float(roi_radius), 1.0),
        0.0,
        initial_offset,
    ]
    p0 = default_p0 if initial_params is None else [float(param) for param in initial_params]

    return _fit_2d_gaussian_patch_with_initial_guess(
        roi_patch,
        y_start,
        x_start,
        p0,
        noise_sigma,
        noise_sigma_eff,
        parameter_bounds=parameter_bounds,
        max_nfev=max_nfev,
        compute_uncertainty=compute_uncertainty,
    )


def fit_2d_gaussian_on_patch(
    roi_patch: np.ndarray,
    *,
    roi_origin_yx: Tuple[int, int] = (0, 0),
    fit_mask: Optional[np.ndarray] = None,
    initial_params: Optional[list[float]] = None,
    parameter_bounds: Optional[Tuple[list[float], list[float]]] = None,
    max_nfev: int = DEFAULT_CURVE_FIT_MAXFEV,
    compute_uncertainty: bool = True,
) -> Optional[PeakRefinementResult]:
    """
    Fit a rotated 2D Gaussian directly on an already extracted ROI patch.

    The returned center coordinates are expressed in the image coordinate system
    defined by ``roi_origin_yx``.
    """
    if not SCIPY_AVAILABLE or curve_fit is None:
        logger.error("fit_2d_gaussian_on_patch: SciPy is not available for curve_fit.")
        return None

    patch = np.asarray(roi_patch, dtype=float)
    if patch.ndim != 2:
        logger.warning("fit_2d_gaussian_on_patch: Invalid ROI patch shape %r.", patch.shape)
        return None
    if patch.size == 0:
        logger.warning("fit_2d_gaussian_on_patch: Empty ROI patch.")
        return None
    if not np.isfinite(patch).all():
        logger.warning("fit_2d_gaussian_on_patch: ROI patch contains non-finite values.")
        return None

    peak_to_peak = float(np.ptp(patch))
    if peak_to_peak <= 0.0:
        logger.debug("fit_2d_gaussian_on_patch: ROI patch has no variation; skipping fit.")
        return None

    normalized_mask = _normalize_fit_mask(patch, fit_mask)
    if fit_mask is not None and normalized_mask is None:
        return None
    noise_source = patch if normalized_mask is None else patch[normalized_mask]
    if noise_source.size == 0:
        logger.debug("fit_2d_gaussian_on_patch: Fit mask selects no pixels.")
        return None
    noise_sigma = _estimate_noise_sigma(noise_source)
    noise_sigma_eff = max(noise_sigma, _MIN_NOISE_SIGMA)
    peak_y, peak_x = np.unravel_index(np.argmax(patch), patch.shape)
    sigma_y_guess = max(float(patch.shape[0]) / 4.0, 1.0)
    sigma_x_guess = max(float(patch.shape[1]) / 4.0, 1.0)
    default_p0 = [
        peak_to_peak,
        float(peak_y),
        float(peak_x),
        sigma_y_guess,
        sigma_x_guess,
        0.0,
        float(np.min(patch)),
    ]
    p0 = default_p0 if initial_params is None else [float(param) for param in initial_params]

    return _fit_2d_gaussian_patch_with_initial_guess(
        patch,
        int(roi_origin_yx[0]),
        int(roi_origin_yx[1]),
        p0,
        noise_sigma,
        noise_sigma_eff,
        fit_mask=normalized_mask,
        parameter_bounds=parameter_bounds,
        max_nfev=max_nfev,
        compute_uncertainty=compute_uncertainty,
    )


def fit_2d_lorentzian_on_patch(
    roi_patch: np.ndarray,
    *,
    roi_origin_yx: Tuple[int, int] = (0, 0),
    fit_mask: Optional[np.ndarray] = None,
    initial_params: Optional[list[float]] = None,
    parameter_bounds: Optional[Tuple[list[float], list[float]]] = None,
    max_nfev: int = DEFAULT_CURVE_FIT_MAXFEV,
    compute_uncertainty: bool = True,
) -> Optional[PeakRefinementResult]:
    """
    Fit a rotated 2D Lorentzian directly on an already extracted ROI patch.

    The returned center coordinates are expressed in the image coordinate system
    defined by ``roi_origin_yx``.
    """

    if not SCIPY_AVAILABLE or curve_fit is None:
        logger.error("fit_2d_lorentzian_on_patch: SciPy is not available for curve_fit.")
        return None

    patch = np.asarray(roi_patch, dtype=float)
    if patch.ndim != 2:
        logger.warning("fit_2d_lorentzian_on_patch: Invalid ROI patch shape %r.", patch.shape)
        return None
    if patch.size == 0:
        logger.warning("fit_2d_lorentzian_on_patch: Empty ROI patch.")
        return None
    if not np.isfinite(patch).all():
        logger.warning("fit_2d_lorentzian_on_patch: ROI patch contains non-finite values.")
        return None

    peak_to_peak = float(np.ptp(patch))
    if peak_to_peak <= 0.0:
        logger.debug("fit_2d_lorentzian_on_patch: ROI patch has no variation; skipping fit.")
        return None

    normalized_mask = _normalize_fit_mask(patch, fit_mask)
    if fit_mask is not None and normalized_mask is None:
        return None
    noise_source = patch if normalized_mask is None else patch[normalized_mask]
    if noise_source.size == 0:
        logger.debug("fit_2d_lorentzian_on_patch: Fit mask selects no pixels.")
        return None
    noise_sigma = _estimate_noise_sigma(noise_source)
    noise_sigma_eff = max(noise_sigma, _MIN_NOISE_SIGMA)
    peak_y, peak_x = np.unravel_index(np.argmax(patch), patch.shape)
    gamma_y_guess = max(float(patch.shape[0]) / 4.0, 1.0)
    gamma_x_guess = max(float(patch.shape[1]) / 4.0, 1.0)
    default_p0 = [
        peak_to_peak,
        float(peak_y),
        float(peak_x),
        gamma_y_guess,
        gamma_x_guess,
        0.0,
        float(np.min(patch)),
    ]
    p0 = default_p0 if initial_params is None else [float(param) for param in initial_params]

    return _fit_2d_lorentzian_patch_with_initial_guess(
        patch,
        int(roi_origin_yx[0]),
        int(roi_origin_yx[1]),
        p0,
        noise_sigma,
        noise_sigma_eff,
        fit_mask=normalized_mask,
        parameter_bounds=parameter_bounds,
        max_nfev=max_nfev,
        compute_uncertainty=compute_uncertainty,
    )


def fit_2d_voigt_on_patch(
    roi_patch: np.ndarray,
    *,
    roi_origin_yx: Tuple[int, int] = (0, 0),
    fit_mask: Optional[np.ndarray] = None,
    initial_params: Optional[list[float]] = None,
    parameter_bounds: Optional[Tuple[list[float], list[float]]] = None,
    max_nfev: int = DEFAULT_CURVE_FIT_MAXFEV,
    compute_uncertainty: bool = True,
) -> Optional[PeakRefinementResult]:
    """
    Fit a rotated 2D Voigt directly on an already extracted ROI patch.

    The returned center coordinates are expressed in the image coordinate system
    defined by ``roi_origin_yx``.
    """

    if not SCIPY_AVAILABLE or curve_fit is None or voigt_profile is None:
        logger.error("fit_2d_voigt_on_patch: SciPy is not available for curve_fit/voigt_profile.")
        return None

    patch = np.asarray(roi_patch, dtype=float)
    if patch.ndim != 2:
        logger.warning("fit_2d_voigt_on_patch: Invalid ROI patch shape %r.", patch.shape)
        return None
    if patch.size == 0:
        logger.warning("fit_2d_voigt_on_patch: Empty ROI patch.")
        return None
    if not np.isfinite(patch).all():
        logger.warning("fit_2d_voigt_on_patch: ROI patch contains non-finite values.")
        return None

    peak_to_peak = float(np.ptp(patch))
    if peak_to_peak <= 0.0:
        logger.debug("fit_2d_voigt_on_patch: ROI patch has no variation; skipping fit.")
        return None

    normalized_mask = _normalize_fit_mask(patch, fit_mask)
    if fit_mask is not None and normalized_mask is None:
        return None
    noise_source = patch if normalized_mask is None else patch[normalized_mask]
    if noise_source.size == 0:
        logger.debug("fit_2d_voigt_on_patch: Fit mask selects no pixels.")
        return None
    noise_sigma = _estimate_noise_sigma(noise_source)
    noise_sigma_eff = max(noise_sigma, _MIN_NOISE_SIGMA)
    peak_y, peak_x = np.unravel_index(np.argmax(patch), patch.shape)
    sigma_y_guess = max(float(patch.shape[0]) / 5.0, 0.75)
    sigma_x_guess = max(float(patch.shape[1]) / 5.0, 0.75)
    gamma_y_guess = max(float(patch.shape[0]) / 6.0, 0.5)
    gamma_x_guess = max(float(patch.shape[1]) / 6.0, 0.5)
    default_p0 = [
        peak_to_peak,
        float(peak_y),
        float(peak_x),
        sigma_y_guess,
        sigma_x_guess,
        gamma_y_guess,
        gamma_x_guess,
        0.0,
        float(np.min(patch)),
    ]
    p0 = default_p0 if initial_params is None else [float(param) for param in initial_params]

    return _fit_2d_voigt_patch_with_initial_guess(
        patch,
        int(roi_origin_yx[0]),
        int(roi_origin_yx[1]),
        p0,
        noise_sigma,
        noise_sigma_eff,
        fit_mask=normalized_mask,
        parameter_bounds=parameter_bounds,
        max_nfev=max_nfev,
        compute_uncertainty=compute_uncertainty,
    )


def _gaussian_fit_estimator(
    noisy_patch: np.ndarray,
    y_start: int,
    x_start: int,
    p0: list[float],
    *,
    fit_mask: Optional[np.ndarray] = None,
    parameter_bounds: Optional[Tuple[list[float], list[float]]] = None,
    max_nfev: int = DEFAULT_CURVE_FIT_MAXFEV,
) -> Optional[Tuple[float, float]]:
    """Helper used during Monte Carlo when covariance is not available."""
    if not SCIPY_AVAILABLE or curve_fit is None:
        return None

    patch = np.asarray(noisy_patch, dtype=float)
    prepared = _prepare_fit_sample_arrays(
        patch,
        fit_mask,
        minimum_sample_count=max(len(p0), 7),
    )
    if prepared is None:
        return None
    xy_flat, data_flat, normalized_mask = prepared
    noise_sigma = max(
        _estimate_noise_sigma(patch if normalized_mask is None else patch[normalized_mask]),
        _MIN_NOISE_SIGMA,
    )

    try:
        bounds = parameter_bounds if parameter_bounds is not None else (-np.inf, np.inf)
        popt, _ = _curve_fit_safely(
            _gaussian_2d,
            xy_flat,
            data_flat,
            p0=p0,
            bounds=bounds,
            sigma=np.full_like(data_flat, noise_sigma, dtype=float),
            absolute_sigma=True,
            maxfev=max(int(max_nfev), 10),
        )
    except Exception:
        return None

    return float(y_start + popt[1]), float(x_start + popt[2])


def _lorentzian_fit_estimator(
    noisy_patch: np.ndarray,
    y_start: int,
    x_start: int,
    p0: list[float],
    *,
    fit_mask: Optional[np.ndarray] = None,
    parameter_bounds: Optional[Tuple[list[float], list[float]]] = None,
    max_nfev: int = DEFAULT_CURVE_FIT_MAXFEV,
) -> Optional[Tuple[float, float]]:
    """Helper used during Monte Carlo when Lorentzian covariance is not available."""

    if not SCIPY_AVAILABLE or curve_fit is None:
        return None

    patch = np.asarray(noisy_patch, dtype=float)
    prepared = _prepare_fit_sample_arrays(
        patch,
        fit_mask,
        minimum_sample_count=max(len(p0), 7),
    )
    if prepared is None:
        return None
    xy_flat, data_flat, normalized_mask = prepared
    noise_sigma = max(
        _estimate_noise_sigma(patch if normalized_mask is None else patch[normalized_mask]),
        _MIN_NOISE_SIGMA,
    )

    try:
        bounds = parameter_bounds if parameter_bounds is not None else (-np.inf, np.inf)
        popt, _ = _curve_fit_safely(
            _lorentzian_2d,
            xy_flat,
            data_flat,
            p0=p0,
            bounds=bounds,
            sigma=np.full_like(data_flat, noise_sigma, dtype=float),
            absolute_sigma=True,
            maxfev=max(int(max_nfev), 10),
        )
    except Exception:
        return None

    return float(y_start + popt[1]), float(x_start + popt[2])


def _voigt_fit_estimator(
    noisy_patch: np.ndarray,
    y_start: int,
    x_start: int,
    p0: list[float],
    *,
    fit_mask: Optional[np.ndarray] = None,
    parameter_bounds: Optional[Tuple[list[float], list[float]]] = None,
    max_nfev: int = DEFAULT_CURVE_FIT_MAXFEV,
) -> Optional[Tuple[float, float]]:
    """Helper used during Monte Carlo when Voigt covariance is not available."""

    if not SCIPY_AVAILABLE or curve_fit is None or voigt_profile is None:
        return None

    patch = np.asarray(noisy_patch, dtype=float)
    prepared = _prepare_fit_sample_arrays(
        patch,
        fit_mask,
        minimum_sample_count=max(len(p0), 7),
    )
    if prepared is None:
        return None
    xy_flat, data_flat, normalized_mask = prepared
    noise_sigma = max(
        _estimate_noise_sigma(patch if normalized_mask is None else patch[normalized_mask]),
        _MIN_NOISE_SIGMA,
    )

    try:
        bounds = parameter_bounds if parameter_bounds is not None else (-np.inf, np.inf)
        popt, _ = _curve_fit_safely(
            _voigt_2d,
            xy_flat,
            data_flat,
            p0=p0,
            bounds=bounds,
            sigma=np.full_like(data_flat, noise_sigma, dtype=float),
            absolute_sigma=True,
            maxfev=max(int(max_nfev), 10),
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
    std = _apply_position_sigma_floor(std)

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
    std = _apply_position_sigma_floor(std)

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
    *,
    compute_uncertainty: bool = True,
) -> Optional[PeakRefinementResult]:
    """
    Legacy compatibility wrapper returning the rich ``PeakRefinementResult``.
    """
    return fit_2d_gaussian_in_roi(
        fft_magnitude_data,
        center_yx,
        roi_radius,
        compute_uncertainty=compute_uncertainty,
    )


def derive_gaussian_center_std(
    result: PeakRefinementResult,
) -> Optional[Tuple[float, float]]:
    """
    Post-process an existing fit result to obtain ``center_std``.

    Used by dialogs that capture lightweight preview fits (no covariance/MC yet)
    and only need the uncertainties once the user confirms a spot.
    """
    pcov = result.pcov
    if pcov is not None and pcov.shape[0] >= 3:
        diag = np.diag(pcov)
        if np.all(np.isfinite(diag[1:3])) and np.all(diag[1:3] >= 0.0):
            return _apply_position_sigma_floor(
                (float(np.sqrt(diag[1])), float(np.sqrt(diag[2]))),
            )

    roi_origin = (result.metadata or {}).get("roi_origin")
    initial_params = (result.metadata or {}).get("initial_params")
    roi_patch = getattr(result, "roi_patch", None)
    if roi_origin is None or initial_params is None or roi_patch is None:
        return None

    noise_sigma = max(float(result.noise_sigma), _MIN_NOISE_SIGMA)
    y_start, x_start = roi_origin
    p0 = list(initial_params)
    center_std = _monte_carlo_uncertainty(
        roi_patch,
        noise_sigma,
        lambda noisy: _gaussian_fit_estimator(noisy, y_start, x_start, p0),
        runs=MC_SAMPLE_COUNT_FAST,
    )
    return _apply_position_sigma_floor(center_std)


__all__ = [
    "PeakRefinementResult",
    "find_max_pixel_in_roi",
    "fit_2d_gaussian_in_roi",
    "fit_2d_gaussian_on_patch",
    "fit_2d_gaussian_in_roi_with_all_data",
    "derive_gaussian_center_std",
    "refine_peak_parabola_3x3",
    "refine_peak_local_dft",
    "SCIPY_AVAILABLE",
]
