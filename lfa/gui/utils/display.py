"""Shared formatting helpers for GUI labels and overlays."""

from __future__ import annotations

from typing import Optional, Tuple


def format_float(value: Optional[float], precision: int = 2, fallback: str = "-") -> str:
    """Return a formatted float or a fallback when value is missing/invalid."""
    try:
        if value is None:
            return fallback
        return f"{float(value):.{precision}f}"
    except (TypeError, ValueError):
        return fallback


def format_pair(
    values: Optional[Tuple[Optional[float], Optional[float]]],
    precision: int = 3,
    fallback: str = "-",
) -> str:
    """Format a tuple of two floats as `(x, y)` with given precision."""
    if not isinstance(values, (tuple, list)) or len(values) != 2:
        return fallback
    first = format_float(values[0], precision, fallback)
    second = format_float(values[1], precision, fallback)
    if fallback in (first, second):
        return fallback
    return f"({first}, {second})"


def format_ratio(value: Optional[float], precision: int = 3) -> str:
    """Format ratio-style values with a fixed precision."""
    return format_float(value, precision)
