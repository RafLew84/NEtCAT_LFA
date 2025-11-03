"""
Utility helpers for propagating measurement uncertainties.

This module offers two complementary approaches:

* Linear (Jacobian-based) propagation of a covariance matrix.
* Monte Carlo sampling of an arbitrary (possibly non-linear) function.

Both operate on numpy arrays and return the transformed value together with the
estimated covariance matrix in the output space.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional

import numpy as np

ArrayLike = np.ndarray
Function = Callable[[ArrayLike], ArrayLike]


@dataclass(frozen=True)
class PropagationResult:
    """Container for propagated values and their covariance."""

    mean: ArrayLike
    covariance: ArrayLike


def _ensure_array(x: ArrayLike) -> ArrayLike:
    arr = np.atleast_1d(np.asarray(x, dtype=float))
    if arr.ndim != 1:
        raise ValueError("Input vector must be one-dimensional.")
    return arr


def _ensure_covariance(cov: ArrayLike, size: int) -> ArrayLike:
    cov_arr = np.asarray(cov, dtype=float)
    if cov_arr.ndim == 1:
        if cov_arr.size != size:
            raise ValueError(f"Diagonal covariance length {cov_arr.size} does not match vector size {size}.")
        cov_arr = np.diag(cov_arr)
    if cov_arr.shape != (size, size):
        raise ValueError(f"Covariance matrix must be square with shape ({size}, {size}).")
    return cov_arr


def numeric_jacobian(
    func: Function,
    x: ArrayLike,
    *,
    eps: float = 1e-6,
    method: str = "central",
) -> ArrayLike:
    """
    Estimate the Jacobian of ``func`` at ``x`` using finite differences.

    Parameters
    ----------
    func:
        Callable mapping a 1-D vector to a 1-D vector.
    x:
        Point at which the Jacobian is to be evaluated.
    eps:
        Relative step size used for perturbation.
    method:
        One of ``"central"`` or ``"forward"``.
    """
    x = _ensure_array(x)
    f0 = np.atleast_1d(np.asarray(func(x), dtype=float))
    if f0.ndim != 1:
        raise ValueError("Function output must be a one-dimensional vector.")

    n_out, n_vars = f0.size, x.size
    jac = np.zeros((n_out, n_vars), dtype=float)

    for idx in range(n_vars):
        step = eps * max(1.0, abs(x[idx]))
        delta = np.zeros_like(x)
        delta[idx] = step

        if method == "central":
            f_plus = np.asarray(func(x + delta), dtype=float)
            f_minus = np.asarray(func(x - delta), dtype=float)
            jac[:, idx] = (f_plus - f_minus) / (2.0 * step)
        elif method == "forward":
            f_plus = np.asarray(func(x + delta), dtype=float)
            jac[:, idx] = (f_plus - f0) / step
        else:
            raise ValueError(f"Unsupported differentiation method: {method}")

    return jac


def propagate_linear(
    func: Function,
    x: ArrayLike,
    cov_x: ArrayLike,
    *,
    eps: float = 1e-6,
    method: str = "central",
) -> PropagationResult:
    """
    Propagate a covariance matrix through ``func`` using linear approximation.

    Returns the function value ``func(x)`` and the corresponding output covariance.
    """
    x = _ensure_array(x)
    cov_x = _ensure_covariance(cov_x, x.size)

    y = np.atleast_1d(np.asarray(func(x), dtype=float))
    if y.ndim != 1:
        raise ValueError("Function output must be a one-dimensional vector.")

    jac = numeric_jacobian(func, x, eps=eps, method=method)
    cov_y = jac @ cov_x @ jac.T
    return PropagationResult(mean=y, covariance=cov_y)


def propagate_monte_carlo(
    func: Function,
    x: ArrayLike,
    cov_x: ArrayLike,
    *,
    samples: int = 512,
    rng: Optional[np.random.Generator | int] = None,
) -> PropagationResult:
    """
    Propagate uncertainty by Monte Carlo sampling.

    Parameters
    ----------
    func:
        Callable mapping a 1-D vector to a 1-D vector.
    x:
        Nominal input vector.
    cov_x:
        Covariance matrix (or diagonal) of the input vector.
    samples:
        Number of Monte Carlo draws.
    rng:
        Optional numpy Generator or seed.
    """
    if samples <= 0:
        raise ValueError("Number of samples must be positive.")

    x = _ensure_array(x)
    cov_x = _ensure_covariance(cov_x, x.size)
    rng = np.random.default_rng(rng)

    try:
        chol = np.linalg.cholesky(cov_x)
    except np.linalg.LinAlgError:
        vals, vecs = np.linalg.eigh(cov_x)
        vals = np.clip(vals, 0.0, None)
        chol = vecs @ np.diag(np.sqrt(vals))

    draws = rng.normal(size=(samples, x.size))
    perturbed = x + draws @ chol.T

    outputs = np.asarray([func(sample) for sample in perturbed], dtype=float)
    if outputs.ndim != 2:
        raise ValueError("Function must return 1-D vectors for Monte Carlo propagation.")

    mean = outputs.mean(axis=0)
    centered = outputs - mean
    cov_y = (centered.T @ centered) / max(samples - 1, 1)
    return PropagationResult(mean=mean, covariance=cov_y)


__all__ = [
    "PropagationResult",
    "numeric_jacobian",
    "propagate_linear",
    "propagate_monte_carlo",
]
