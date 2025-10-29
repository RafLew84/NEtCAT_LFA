"""Shared formatting and sanitisation helpers for GUI labels and overlays."""

from __future__ import annotations

import math
from typing import Iterable, Optional, Sequence, Tuple

import numpy as np

__all__ = [
    "format_float",
    "format_pair",
    "format_ratio",
    "sanitize_numeric_array",
]


def _is_finite_number(value: float) -> bool:
    """Return True when value is a finite float."""
    return math.isfinite(value)


def format_float(value: Optional[float], precision: int = 2, fallback: str = "-") -> str:
    """Return a formatted float or a fallback when value is missing/invalid."""
    try:
        if value is None:
            return fallback
        numeric = float(value)
    except (TypeError, ValueError):
        return fallback
    if not _is_finite_number(numeric):
        return fallback
    return f"{numeric:.{precision}f}"


def _coerce_pair(values: Optional[Iterable]) -> Optional[Tuple[float, float]]:
    if values is None:
        return None
    if isinstance(values, np.ndarray):
        seq = values.flatten().tolist()
    elif isinstance(values, Sequence):
        seq = list(values)
    else:
        return None
    if len(seq) != 2:
        return None
    first, second = seq
    try:
        first_f = float(first)
        second_f = float(second)
    except (TypeError, ValueError):
        return None
    if not (_is_finite_number(first_f) and _is_finite_number(second_f)):
        return None
    return first_f, second_f


def format_pair(
    values: Optional[Iterable],
    precision: int = 3,
    fallback: str = "-",
) -> str:
    """Format an iterable of two floats as `(x, y)` with given precision."""
    pair = _coerce_pair(values)
    if pair is None:
        return fallback
    first = format_float(pair[0], precision, fallback)
    second = format_float(pair[1], precision, fallback)
    if fallback in (first, second):
        return fallback
    return f"({first}, {second})"


def format_ratio(value: Optional[float], precision: int = 3) -> str:
    """Format ratio-style values with a fixed precision."""
    return format_float(value, precision)


def sanitize_numeric_array(values: Optional[Iterable], allow_empty: bool = False):
    """
    Convert `values` to a NumPy float array and drop non-finite entries.

    Returns:
        np.ndarray | None: Array containing only finite values or None if conversion fails.
    """
    if values is None:
        return None
    try:
        array = np.asarray(values, dtype=float)
    except (TypeError, ValueError):
        rows: list[np.ndarray] = []
        for item in values:
            try:
                row = np.asarray(item, dtype=float)
            except (TypeError, ValueError):
                continue
            if row.ndim == 0:
                if _is_finite_number(float(row)):
                    rows.append(np.asarray([float(row)], dtype=float))
            else:
                mask = np.isfinite(row)
                if mask.all():
                    rows.append(row.astype(float))
        if not rows:
            return None
        array = np.asarray(rows, dtype=float)
        if array.ndim > 2:
            array = array.reshape(array.shape[0], -1)

    if array.size == 0:
        return array if allow_empty else None

    finite_mask = np.isfinite(array)
    if not finite_mask.all():
        if array.ndim == 1:
            array = array[finite_mask]
        else:
            axis = tuple(range(1, array.ndim))
            row_mask = finite_mask.all(axis=axis)
            array = array[row_mask]

    if array.size == 0 and not allow_empty:
        return None
    return array
