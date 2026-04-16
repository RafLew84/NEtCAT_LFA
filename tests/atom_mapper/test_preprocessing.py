"""Tests for AtomMapper preprocessing helpers."""

from __future__ import annotations

import numpy as np
import pytest

from AtomMapper.app.preprocessing import (
    apply_bm3d,
    apply_blur,
    apply_flip,
    apply_non_local_means,
    apply_rotation,
    build_bm3d_metadata,
    build_blur_metadata,
    build_flip_metadata,
    build_nlm_metadata,
    build_rotation_metadata,
    is_bm3d_available,
)


def test_apply_blur_returns_new_smoothed_array():
    image_data = np.zeros((9, 9), dtype=float)
    image_data[4, 4] = 10.0

    blurred = apply_blur(image_data, sigma_px=1.0)

    assert blurred.shape == image_data.shape
    assert blurred.dtype == float
    assert not np.shares_memory(blurred, image_data)
    assert blurred[4, 4] < image_data[4, 4]
    assert blurred[4, 4] > 0.0
    assert np.isclose(float(blurred.sum()), float(image_data.sum()), atol=1e-6)


def test_apply_blur_rejects_invalid_inputs():
    with pytest.raises(ValueError, match="2D"):
        apply_blur(np.zeros((3, 3, 3), dtype=float))

    with pytest.raises(ValueError, match="sigma_px"):
        apply_blur(np.zeros((3, 3), dtype=float), sigma_px=0.0)


def test_build_blur_metadata_returns_serializable_fields():
    metadata = build_blur_metadata(sigma_px=1.5)

    assert metadata == {
        "preprocess": "blur",
        "blur_sigma_px": 1.5,
    }


def test_apply_rotation_returns_rotated_array_and_rejects_invalid_turns():
    image_data = np.arange(12, dtype=float).reshape((3, 4))

    rotated = apply_rotation(image_data, quarter_turns=1)

    assert rotated.shape == (4, 3)
    assert rotated.dtype == float
    assert np.array_equal(rotated, np.rot90(image_data, k=1))

    with pytest.raises(ValueError, match="quarter_turns"):
        apply_rotation(image_data, quarter_turns=4)


def test_build_rotation_metadata_returns_serializable_fields():
    metadata = build_rotation_metadata(quarter_turns=3)

    assert metadata == {
        "preprocess": "rotate",
        "rotate_quarter_turns": 3,
        "rotate_angle_deg": 270,
        "rotate_direction": "ccw",
    }


def test_apply_flip_returns_flipped_array_and_rejects_empty_transform():
    image_data = np.arange(9, dtype=float).reshape((3, 3))

    flipped = apply_flip(image_data, flip_x=True, flip_y=True)

    assert flipped.shape == image_data.shape
    assert flipped.dtype == float
    assert np.array_equal(flipped, np.flipud(np.fliplr(image_data)))

    with pytest.raises(ValueError, match="At least one"):
        apply_flip(image_data, flip_x=False, flip_y=False)


def test_build_flip_metadata_returns_serializable_fields():
    metadata = build_flip_metadata(flip_x=True, flip_y=False)

    assert metadata == {
        "preprocess": "flip",
        "flip_x": True,
        "flip_y": False,
    }


def test_apply_non_local_means_returns_new_denoised_array():
    rng = np.random.default_rng(12345)
    image_data = np.zeros((21, 21), dtype=float)
    image_data[10, 10] = 5.0
    noisy = image_data + rng.normal(0.0, 0.1, size=image_data.shape)

    denoised = apply_non_local_means(
        noisy,
        h=0.12,
        patch_size=5,
        patch_distance=6,
        fast_mode=True,
    )

    assert denoised.shape == noisy.shape
    assert denoised.dtype == float
    assert not np.shares_memory(denoised, noisy)
    assert not np.allclose(denoised, noisy)


def test_apply_non_local_means_rejects_invalid_inputs():
    with pytest.raises(ValueError, match="2D"):
        apply_non_local_means(np.zeros((3, 3, 3), dtype=float))

    with pytest.raises(ValueError, match="h must"):
        apply_non_local_means(np.zeros((5, 5), dtype=float), h=0.0)

    with pytest.raises(ValueError, match="patch_size"):
        apply_non_local_means(np.zeros((5, 5), dtype=float), patch_size=4)

    with pytest.raises(ValueError, match="patch_distance"):
        apply_non_local_means(np.zeros((5, 5), dtype=float), patch_distance=0)


def test_build_nlm_metadata_returns_serializable_fields():
    metadata = build_nlm_metadata(
        h=0.15,
        patch_size=7,
        patch_distance=8,
        fast_mode=False,
    )

    assert metadata == {
        "preprocess": "nlm",
        "nlm_h": 0.15,
        "nlm_patch_size": 7,
        "nlm_patch_distance": 8,
        "nlm_fast_mode": False,
    }


def test_apply_bm3d_returns_new_denoised_array():
    if not is_bm3d_available():
        pytest.skip("bm3d package not available in test environment")

    rng = np.random.default_rng(12345)
    image_data = np.zeros((16, 16), dtype=float)
    image_data[8, 8] = 5.0
    noisy = image_data + rng.normal(0.0, 0.05, size=image_data.shape)

    denoised = apply_bm3d(noisy, sigma_psd=0.05)

    assert denoised.shape == noisy.shape
    assert denoised.dtype == float
    assert not np.shares_memory(denoised, noisy)
    assert not np.allclose(denoised, noisy)


def test_apply_bm3d_rejects_invalid_inputs_and_missing_backend(monkeypatch):
    with pytest.raises(ValueError, match="2D"):
        apply_bm3d(np.zeros((3, 3, 3), dtype=float), sigma_psd=0.1)

    with pytest.raises(ValueError, match="sigma_psd"):
        apply_bm3d(np.zeros((5, 5), dtype=float), sigma_psd=0.0)

    monkeypatch.setattr("AtomMapper.app.preprocessing._BM3D_AVAILABLE", False)
    with pytest.raises(RuntimeError, match="BM3D package is not available"):
        apply_bm3d(np.zeros((5, 5), dtype=float), sigma_psd=0.1)


def test_build_bm3d_metadata_returns_serializable_fields():
    metadata = build_bm3d_metadata(
        sigma_psd=0.08,
        stage="all_stages",
    )

    assert metadata == {
        "preprocess": "bm3d",
        "bm3d_sigma_psd": 0.08,
        "bm3d_stage": "all_stages",
    }
