"""Tests for AtomMapper data models."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from AtomMapper.app.models import LoadedImage, ROIState


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
    )


def test_loaded_image_assigns_default_identity_fields():
    loaded = _make_loaded_image("sample.stp")

    assert loaded.image_id
    assert loaded.source_group_id == loaded.image_id
    assert loaded.parent_image_id is None
    assert loaded.variant_name == "original"
    assert loaded.is_original is True


def test_loaded_image_derive_variant_keeps_source_group_and_links_parent():
    loaded = _make_loaded_image("sample.stp", width=12, height=10)
    variant_data = loaded.image_data + 5.0

    variant = loaded.derive_variant(
        variant_name="blur",
        image_data=variant_data,
        metadata_updates={"preprocess": "blur"},
    )

    assert variant.image_id != loaded.image_id
    assert variant.source_group_id == loaded.source_group_id
    assert variant.parent_image_id == loaded.image_id
    assert variant.variant_name == "blur"
    assert variant.is_original is False
    assert variant.display_name == "sample [blur].stp"
    assert variant.pixels_x == loaded.pixels_x
    assert variant.pixels_y == loaded.pixels_y
    assert variant.size_nm_x == loaded.size_nm_x
    assert variant.size_nm_y == loaded.size_nm_y
    assert variant.metadata["preprocess"] == "blur"
    assert np.array_equal(variant.image_data, variant_data)


def test_loaded_image_derive_variant_rejects_empty_variant_name():
    loaded = _make_loaded_image("sample.stp")

    with pytest.raises(ValueError, match="variant_name"):
        loaded.derive_variant(variant_name="  ", image_data=loaded.image_data)


def test_roi_state_clamped_uses_4px_minimum_bbox():
    roi = ROIState(x=5, y=6, width=1, height=2)

    clamped = roi.clamped(image_width=20, image_height=18)

    assert clamped == ROIState(x=5, y=6, width=4, height=4)
