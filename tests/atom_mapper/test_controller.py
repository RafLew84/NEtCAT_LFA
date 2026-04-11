"""Tests for AtomMapper controller state management."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from AtomMapper.app.controller import AtomMapperController
from AtomMapper.app.models import LoadedImage, ROIState
from AtomMapper.app.preprocessing import is_bm3d_available


def _make_loaded_image(name: str, width: int = 8, height: int = 6) -> LoadedImage:
    image_data = np.arange(width * height, dtype=float).reshape((height, width))
    return LoadedImage(
        source_path=str(Path("/tmp") / name),
        display_name=name,
        file_extension=Path(name).suffix.lower(),
        image_data=image_data,
        pixels_x=width,
        pixels_y=height,
        size_nm_x=float(width),
        size_nm_y=float(height),
        metadata={"image_type": "Topo"},
        raw_metadata={},
    )


def test_controller_tracks_loaded_images_and_selection():
    controller = AtomMapperController()
    first = _make_loaded_image("first.stp")
    second = _make_loaded_image("second.s94")

    controller.set_loaded_images([first, second])

    assert controller.active_image_index == 0
    assert controller.active_image == first
    assert controller.active_source_group_id == first.source_group_id
    assert controller.loaded_images == (first, second)
    assert controller.original_images == (first, second)
    assert controller.images_for_source_group(first.source_group_id) == (first,)

    selected = controller.select_image(1)
    assert selected == second
    assert controller.active_image_index == 1
    assert controller.active_image == second
    assert controller.active_source_group_id == second.source_group_id


def test_controller_rejects_invalid_selection():
    controller = AtomMapperController()
    controller.set_loaded_images([_make_loaded_image("only.stp")])

    with pytest.raises(IndexError, match="out of range"):
        controller.select_image(5)


def test_controller_creates_and_updates_active_roi_state():
    controller = AtomMapperController()
    image = _make_loaded_image("roi.stp", width=100, height=80)

    controller.set_loaded_images([image])

    default_roi = controller.active_roi_state
    assert default_roi is not None
    assert default_roi.width == 16
    assert default_roi.height == 16
    assert default_roi.x == 42
    assert default_roi.y == 32

    updated = controller.update_active_roi_state(ROIState(x=90, y=70, width=30, height=30))
    assert updated == ROIState(x=70, y=50, width=30, height=30)
    assert controller.active_roi_state == updated


def test_controller_clamps_active_roi_to_4px_minimum_size():
    controller = AtomMapperController()
    image = _make_loaded_image("roi-minimum.stp", width=100, height=80)

    controller.set_loaded_images([image])

    updated = controller.update_active_roi_state(ROIState(x=12, y=14, width=1, height=2))

    assert updated == ROIState(x=12, y=14, width=4, height=4)
    assert controller.active_roi_state == updated


def test_controller_adds_variant_and_tracks_family_without_overwriting_original_roi():
    controller = AtomMapperController()
    original = _make_loaded_image("sample.stp", width=100, height=80)
    variant = original.derive_variant(variant_name="blur", image_data=original.image_data + 1.0)

    controller.set_loaded_images([original])
    original_roi = controller.update_active_roi_state(ROIState(x=10, y=12, width=24, height=22))

    controller.add_loaded_variant(variant, make_active=True)

    assert controller.loaded_images == (original, variant)
    assert controller.active_image == variant
    assert controller.active_source_group_id == original.source_group_id
    assert controller.images_for_source_group(original.source_group_id) == (original, variant)
    assert controller.variant_images_for_source_group(original.source_group_id) == (variant,)

    variant_roi = controller.update_active_roi_state(ROIState(x=30, y=34, width=18, height=18))
    assert controller.active_roi_state == variant_roi

    controller.select_image(0)
    assert controller.active_image == original
    assert controller.active_roi_state == original_roi

    controller.select_image(1)
    assert controller.active_image == variant
    assert controller.active_roi_state == variant_roi


def test_controller_rejects_variant_with_missing_parent_or_wrong_source_group():
    controller = AtomMapperController()
    original = _make_loaded_image("sample.stp", width=32, height=24)
    controller.set_loaded_images([original])

    orphan_variant = LoadedImage(
        source_path=original.source_path,
        display_name="sample [blur].stp",
        file_extension=".stp",
        image_data=original.image_data + 1.0,
        pixels_x=original.pixels_x,
        pixels_y=original.pixels_y,
        size_nm_x=original.size_nm_x,
        size_nm_y=original.size_nm_y,
        parent_image_id="missing-parent",
        source_group_id=original.source_group_id,
        variant_name="blur",
    )
    wrong_group_variant = LoadedImage(
        source_path=original.source_path,
        display_name="sample [blur-2].stp",
        file_extension=".stp",
        image_data=original.image_data + 2.0,
        pixels_x=original.pixels_x,
        pixels_y=original.pixels_y,
        size_nm_x=original.size_nm_x,
        size_nm_y=original.size_nm_y,
        parent_image_id=original.image_id,
        source_group_id="other-group",
        variant_name="blur",
    )

    with pytest.raises(ValueError, match="parent"):
        controller.add_loaded_variant(orphan_variant)

    with pytest.raises(ValueError, match="source_group_id"):
        controller.add_loaded_variant(wrong_group_variant)


def test_controller_can_create_blur_variant_for_active_image():
    controller = AtomMapperController()
    original = _make_loaded_image("sample.stp", width=40, height=30)
    controller.set_loaded_images([original])

    variant = controller.create_blur_variant_for_active_image(sigma_px=1.25, make_active=True)

    assert variant.variant_name == "blur"
    assert variant.parent_image_id == original.image_id
    assert variant.source_group_id == original.source_group_id
    assert variant.metadata["preprocess"] == "blur"
    assert variant.metadata["blur_sigma_px"] == pytest.approx(1.25)
    assert variant.pixels_x == original.pixels_x
    assert variant.pixels_y == original.pixels_y
    assert variant.size_nm_x == original.size_nm_x
    assert variant.size_nm_y == original.size_nm_y
    assert np.array_equal(original.image_data, _make_loaded_image("sample.stp", width=40, height=30).image_data)
    assert not np.array_equal(variant.image_data, original.image_data)
    assert controller.loaded_images == (original, variant)
    assert controller.active_image == variant
    assert controller.images_for_source_group(original.source_group_id) == (original, variant)


def test_controller_rejects_blur_variant_without_active_image():
    controller = AtomMapperController()

    with pytest.raises(ValueError, match="active image"):
        controller.create_blur_variant_for_active_image()


def test_controller_can_create_nlm_variant_for_active_image():
    controller = AtomMapperController()
    original = _make_loaded_image("sample.stp", width=40, height=30)
    controller.set_loaded_images([original])

    variant = controller.create_nlm_variant_for_active_image(
        h=0.12,
        patch_size=5,
        patch_distance=7,
        fast_mode=False,
        make_active=True,
    )

    assert variant.variant_name == "nlm"
    assert variant.parent_image_id == original.image_id
    assert variant.source_group_id == original.source_group_id
    assert variant.metadata["preprocess"] == "nlm"
    assert variant.metadata["nlm_h"] == pytest.approx(0.12)
    assert variant.metadata["nlm_patch_size"] == 5
    assert variant.metadata["nlm_patch_distance"] == 7
    assert variant.metadata["nlm_fast_mode"] is False
    assert variant.pixels_x == original.pixels_x
    assert variant.pixels_y == original.pixels_y
    assert controller.loaded_images == (original, variant)
    assert controller.active_image == variant
    assert controller.images_for_source_group(original.source_group_id) == (original, variant)


def test_controller_rejects_nlm_variant_without_active_image():
    controller = AtomMapperController()

    with pytest.raises(ValueError, match="active image"):
        controller.create_nlm_variant_for_active_image()


def test_controller_can_create_bm3d_variant_for_active_image():
    if not is_bm3d_available():
        pytest.skip("bm3d package not available in test environment")

    controller = AtomMapperController()
    original = _make_loaded_image("sample.stp", width=24, height=24)
    controller.set_loaded_images([original])

    variant = controller.create_bm3d_variant_for_active_image(
        sigma_psd=0.09,
        stage="all_stages",
        make_active=True,
    )

    assert variant.variant_name == "bm3d"
    assert variant.parent_image_id == original.image_id
    assert variant.source_group_id == original.source_group_id
    assert variant.metadata["preprocess"] == "bm3d"
    assert variant.metadata["bm3d_sigma_psd"] == pytest.approx(0.09)
    assert variant.metadata["bm3d_stage"] == "all_stages"
    assert controller.loaded_images == (original, variant)
    assert controller.active_image == variant


def test_controller_rejects_bm3d_variant_without_active_image():
    controller = AtomMapperController()

    with pytest.raises(ValueError, match="active image"):
        controller.create_bm3d_variant_for_active_image()
