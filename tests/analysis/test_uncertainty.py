from pathlib import Path
import sys
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from lfa.analysis.uncertainty import (  # noqa: E402
    numeric_jacobian,
    propagate_linear,
    propagate_monte_carlo,
)


def test_numeric_jacobian_linear_map():
    def linear_fn(v):
        a = np.array([[2.0, -1.0], [0.0, 3.0]])
        return a @ v

    x = np.array([1.0, 2.0])
    jac = numeric_jacobian(linear_fn, x)
    expected = np.array([[2.0, -1.0], [0.0, 3.0]])
    np.testing.assert_allclose(jac, expected, atol=1e-8)


def test_propagate_linear_matches_manual():
    def fn(v):
        return np.array([v[0] + 2 * v[1]])

    x = np.array([1.0, -1.0])
    cov_x = np.array([[0.04, 0.0], [0.0, 0.09]])

    result = propagate_linear(fn, x, cov_x)
    expected_mean = np.array([fn(x)[0]])
    expected_cov = np.array([[0.04 + 4 * 0.09]])

    np.testing.assert_allclose(result.mean, expected_mean, atol=1e-9)
    np.testing.assert_allclose(result.covariance, expected_cov, atol=1e-9)


def test_propagate_monte_carlo_reproduces_statistics():
    rng = np.random.default_rng(1234)

    def fn(v):
        return np.array([v[0] ** 2 + v[1]])

    x = np.array([0.2, -0.5])
    cov_x = np.diag([0.01, 0.04])

    lin_result = propagate_linear(fn, x, cov_x)
    mc_result = propagate_monte_carlo(fn, x, cov_x, samples=2000, rng=rng)

    np.testing.assert_allclose(mc_result.mean, lin_result.mean, atol=5e-3)
    np.testing.assert_allclose(mc_result.covariance, lin_result.covariance, atol=2e-2)
