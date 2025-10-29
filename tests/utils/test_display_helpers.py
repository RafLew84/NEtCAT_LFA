import math

import numpy as np

from lfa.gui.utils.display import (
    format_float,
    format_pair,
    format_ratio,
    sanitize_numeric_array,
)


def test_format_float_valid():
    assert format_float(3.14159, precision=2) == "3.14"


def test_format_float_handles_none_and_nan():
    assert format_float(None) == "-"
    assert format_float(float("nan")) == "-"
    assert format_float(float("inf")) == "-"


def test_format_pair_valid():
    assert format_pair((1.2345, 6.789), precision=2) == "(1.23, 6.79)"


def test_format_pair_invalid_returns_fallback():
    assert format_pair(("a", 1.0)) == "-"
    assert format_pair((float("nan"), 1.0)) == "-"


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
