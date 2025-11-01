import math

import numpy as np

from lfa.gui.utils.display import (
    format_float,
    format_ratio,
    sanitize_numeric_array,
)
from lfa.gui.utils.formatters import (
    format_fft_pair,
    format_fft_scalar,
    summarise_fft_metrics,
)


def test_format_float_valid():
    assert format_float(3.14159, precision=2) == "3.14"


def test_format_float_handles_none_and_nan():
    assert format_float(None) == "-"
    assert format_float(float("nan")) == "-"
    assert format_float(float("inf")) == "-"


def test_format_fft_pair_valid():
    assert format_fft_pair((1.2345, 6.789), precision=2) == "(1.23, 6.79)"


def test_format_fft_pair_invalid_returns_fallback():
    assert format_fft_pair(("a", 1.0)) == "-"
    assert format_fft_pair((float("nan"), 1.0)) == "-"


def test_format_fft_scalar_handles_invalid_input():
    assert format_fft_scalar(None) == "-"
    assert format_fft_scalar("not-a-number") == "-"
    assert format_fft_scalar(float("nan")) == "-"


def test_summarise_fft_metrics_pair_with_numpy_sigma():
    value = np.array([1.05, 0.97], dtype=float)
    sigma = np.array([0.012, 0.018], dtype=float)
    result = summarise_fft_metrics(value, sigma=sigma, precision=3, sigma_precision=3)
    assert "(1.050, 0.970)" in result
    assert "+/- 0.012" in result
    assert "+/- 0.018" in result


def test_summarise_fft_metrics_scalar_with_sequence_sigma():
    sigma = (0.004, 0.006)
    result = summarise_fft_metrics(0.75, sigma=sigma, precision=2, sigma_precision=3, unit="nm")
    assert result.startswith("0.75 +/- 0.004")
    assert result.endswith("nm")


def test_summarise_fft_metrics_invalid_value_returns_fallback():
    result = summarise_fft_metrics(("bad", 1.0))
    assert result == "-"


def test_format_ratio_delegates_to_float():
    assert format_ratio(0.3333, precision=2) == "0.33"
    assert format_ratio(None) == "-"


def test_sanitize_numeric_array_with_valid_input():
    result = sanitize_numeric_array([(1, 2), (3, 4)])
    assert isinstance(result, np.ndarray)
    assert result.shape == (2, 2)


def test_sanitize_numeric_array_filters_invalid_entries():
    result = sanitize_numeric_array([(1, 2), ("x", 4), (math.inf, 5)])
    assert result.shape == (1, 2)
    assert np.all(result == np.array([[1.0, 2.0]]))


def test_sanitize_numeric_array_returns_none_when_all_invalid():
    assert sanitize_numeric_array([("a", "b"), (None, None)]) is None
