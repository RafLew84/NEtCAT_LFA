"""Tests for shared AtomMapper preprocessing state contracts."""

from __future__ import annotations

import numpy as np
import pytest

from AtomMapper.app.preprocessing_state import (
    BM3DParameters,
    BlurParameters,
    NonLocalMeansParameters,
    PreviewViewport,
    PreprocessingMethod,
    PreprocessingPreviewRequest,
    PreprocessingPreviewResult,
    PreprocessingState,
)


def test_preprocessing_method_and_parameter_normalization():
    state = PreprocessingState(
        method=" NLM ",
        blur=BlurParameters(sigma_px=-2.0, mode=" REFLECT "),
        nlm=NonLocalMeansParameters(h="0.25", patch_size=4, patch_distance=0, fast_mode=1),
        bm3d=BM3DParameters(sigma_psd="0.0", stage=" "),
    ).normalized()

    assert state.method is PreprocessingMethod.NLM
    assert state.blur.sigma_px == pytest.approx(1.0)
    assert state.blur.mode == "reflect"
    assert state.nlm.h == pytest.approx(0.25)
    assert state.nlm.patch_size == 5
    assert state.nlm.patch_distance == 6
    assert state.nlm.fast_mode is True
    assert state.bm3d.sigma_psd == pytest.approx(0.1)
    assert state.bm3d.stage == "all_stages"


def test_preprocessing_state_round_trip_dict_serialization():
    state = PreprocessingState(
        method=PreprocessingMethod.BM3D,
        blur=BlurParameters(sigma_px=1.6, mode="nearest"),
        nlm=NonLocalMeansParameters(h=0.22, patch_size=7, patch_distance=9, fast_mode=False),
        bm3d=BM3DParameters(sigma_psd=0.33, stage="hard_thresholding"),
    )

    payload = state.to_dict()
    restored = PreprocessingState.from_dict(payload)

    assert restored == state.normalized()
    assert restored.active_parameters == restored.bm3d


def test_preview_request_normalizes_viewport_and_state():
    request = PreprocessingPreviewRequest(
        image_id=" image-1 ",
        source_group_id=" group-1 ",
        state=PreprocessingState(
            method="blur",
            blur=BlurParameters(sigma_px="2.5", mode=" mirror "),
        ),
        viewport=PreviewViewport(x=-5, y="7", width="20", height=-10),
    ).normalized()

    assert request.image_id == "image-1"
    assert request.source_group_id == "group-1"
    assert request.state.method is PreprocessingMethod.BLUR
    assert request.state.blur.sigma_px == pytest.approx(2.5)
    assert request.state.blur.mode == "mirror"
    assert request.viewport is not None
    assert request.viewport.x == 0
    assert request.viewport.y == 7
    assert request.viewport.width == 20
    assert request.viewport.height == 0


def test_preview_result_helpers_cover_success_and_failure():
    request = PreprocessingPreviewRequest(
        image_id="image-2",
        source_group_id="group-2",
        state=PreprocessingState(),
    )

    success = PreprocessingPreviewResult.from_success(
        request,
        np.arange(16, dtype=float).reshape((4, 4)),
    )
    failure = PreprocessingPreviewResult.from_failure(request, "backend unavailable")

    assert success.success is True
    assert success.processed_image is not None
    assert success.processed_image.shape == (4, 4)
    assert success.request.state.method is PreprocessingMethod.BLUR
    assert failure.success is False
    assert failure.processed_image is None
    assert failure.error_message == "backend unavailable"


def test_preview_result_rejects_non_2d_images():
    request = PreprocessingPreviewRequest(
        image_id="image-3",
        source_group_id="group-3",
        state=PreprocessingState(),
    )

    with pytest.raises(ValueError, match="Expected 2D preview image data"):
        PreprocessingPreviewResult.from_success(request, np.zeros((2, 2, 2), dtype=float))
