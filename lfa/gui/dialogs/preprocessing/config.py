"""Central configuration for preprocessing dialog parameters."""

from __future__ import annotations

PREPROCESSING_CONFIG = {
    "gaussian_blur": {
        "sigma": {
            "slider_min": 0,
            "slider_max": 100,
            "tick_interval": 10,
            "default": 0,
            "scale": 0.1,
        },
    },
    "gaussian_sharpen": {
        "radius": {
            "slider_min": 0,
            "slider_max": 100,
            "tick_interval": 10,
            "default": 10,
            "scale": 0.1,
        },
        "amount": {
            "slider_min": 0,
            "slider_max": 50,
            "tick_interval": 5,
            "default": 10,
            "scale": 0.1,
        },
    },
    "nlmeans": {
        "sigma": {
            "min": 0.0001,
            "max": 1_000_000.0,
            "decimals": 4,
            "single_step": 0.01,
            "estimate_divisor": 10.0,
            "fallback": 0.001,
        },
        "h_multiplier": {
            "min": 0.1,
            "max": 5.0,
            "decimals": 2,
            "single_step": 0.05,
            "default": 1.0,
        },
        "patch_size": {
            "min": 3,
            "max": 21,
            "single_step": 2,
            "default": 7,
        },
        "patch_distance": {
            "min": 1,
            "max": 100,
            "single_step": 1,
            "default": 11,
        },
        "fast_mode": {
            "default": True,
        },
    },
    "bm3d": {
        "sigma": {
            "min": 0.0001,
            "max": 1.0,
            "decimals": 4,
            "single_step": 0.005,
            "default": 0.05,
        },
    },
    "median": {
        "kernel": {
            "min": 1,
            "max": 31,
            "single_step": 2,
            "default": 3,
        },
        "mode": {
            "options": ["reflect", "constant", "nearest", "mirror", "wrap"],
            "default": "reflect",
        },
        "constant_value": {
            "min": -1_000_000.0,
            "max": 1_000_000.0,
            "decimals": 3,
            "single_step": 0.1,
            "default": 0.0,
        },
    },
}

__all__ = ["PREPROCESSING_CONFIG"]
