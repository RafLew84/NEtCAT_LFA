from __future__ import annotations

from collections.abc import Sequence
from typing import Optional, SupportsFloat, Tuple, Union

import numpy as np

NumericInput = Optional[Union[SupportsFloat, np.generic]]


def _coerce_to_float(value: NumericInput) -> Optional[float]:
    """Best-effort conversion to a finite float; returns None when not possible."""
    if value is None:
        return None

    if isinstance(value, np.ndarray):
        if value.size != 1:
            return None
        value = value.item()

    try:
        numeric = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None

    if not np.isfinite(numeric):
        return None

    return numeric


def _normalise_sequence(values: object) -> Tuple[NumericInput, ...]:
    """Convert arbitrary iterables (including numpy arrays) into a flat tuple."""
    if isinstance(values, np.ndarray):
        if values.ndim == 0:
            return (values.item(),)
        return tuple(values.flatten().tolist())
    if isinstance(values, Sequence) and not isinstance(values, (str, bytes)):
        return tuple(values)
    return tuple()


def format_fft_scalar(
    value: NumericInput,
    *,
    precision: int = 4,
    fallback: str = "-",
) -> str:
    """Format a single FFT scalar value."""
    numeric = _coerce_to_float(value)
    if numeric is None:
        return fallback
    return f"{numeric:.{precision}f}"


def format_fft_pair(
    pair: Optional[Sequence[NumericInput]],
    *,
    precision: int = 4,
    fallback: str = "-",
) -> str:
    """Format a 2-tuple representing an FFT coordinate pair."""
    if pair is None or len(pair) != 2:
        return fallback

    first = format_fft_scalar(pair[0], precision=precision, fallback=fallback)
    second = format_fft_scalar(pair[1], precision=precision, fallback=fallback)
    if fallback in {first, second}:
        return fallback
    return f"({first}, {second})"


def summarise_fft_metrics(
    value: Optional[object],
    *,
    sigma: Optional[object] = None,
    precision: int = 3,
    sigma_precision: Optional[int] = None,
    unit: str = "",
    fallback: str = "-",
) -> str:
    """
    Format FFT-derived scalars or pairs with optional uncertainty.

    Parameters
    ----------
    value:
        Scalar or (kx, ky) pair to display.
    sigma:
        Optional uncertainty. When a sequence is provided for pair values, each
        component is formatted independently (e.g. ``+/- 0.012, +/- 0.018``).
    precision:
        Decimal precision for the primary values.
    sigma_precision:
        Optional override for the uncertainty precision. Falls back to ``precision``
        when ``None``.
    unit:
        Unit suffix appended to scalar values.
    fallback:
        Replacement text when the value cannot be formatted.
    """
    if value is None:
        return fallback

    sigma_precision = sigma_precision if sigma_precision is not None else precision

    # Sequence handling (kx, ky) pairs
    sequence_value = _normalise_sequence(value)
    if sequence_value:
        if len(sequence_value) >= 2:
            base = format_fft_pair(
                sequence_value[:2], precision=precision, fallback=fallback
            )
            if base == fallback:
                return fallback

            formatted_sigmas = []
            if sigma is not None:
                for component in _normalise_sequence(sigma):
                    formatted = format_fft_scalar(
                        component, precision=sigma_precision, fallback=fallback
                    )
                    if formatted != fallback:
                        formatted_sigmas.append(f"+/- {formatted}")

            if formatted_sigmas:
                return f"{base} {', '.join(formatted_sigmas)}"
            return base
        if len(sequence_value) == 1:
            value = sequence_value[0]
        else:
            return fallback

    # Scalar handling
    base = format_fft_scalar(value, precision=precision, fallback=fallback)
    if base == fallback:
        return fallback

    if sigma is not None:
        sigma_components = list(_normalise_sequence(sigma))
        sigma_value: NumericInput
        if sigma_components:
            sigma_value = sigma_components[0]
        else:
            sigma_value = sigma  # type: ignore[assignment]

        sigma_text = format_fft_scalar(
            sigma_value, precision=sigma_precision, fallback=fallback
        )
        if sigma_text != fallback:
            return f"{base} +/- {sigma_text} {unit}".strip()

    return f"{base} {unit}".strip()
