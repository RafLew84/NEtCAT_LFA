"""Tests for AtomMapper data models."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from AtomMapper.app.models import AtomPoint, AtomRow, LoadedImage, ROIState


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


def test_atom_point_serializes_and_restores_fit_payload():
    point = AtomPoint(
        row_id="row-1",
        image_id="image-1",
        source_group_id="group-1",
        point_index=2,
        x_px=12.5,
        y_px=9.75,
        amplitude=18.2,
        sigma_x_px=1.3,
        sigma_y_px=1.7,
        theta_deg=33.0,
        offset=0.4,
        fit_success=False,
        fit_error_message="unstable fit",
        metadata={"quality": "check"},
    )

    payload = point.to_dict()
    restored = AtomPoint.from_dict(payload)

    assert point.point_id
    assert restored == point
    assert payload["point_id"] == point.point_id
    assert payload["point_index"] == 2
    assert payload["x_nm"] is None
    assert payload["metadata"]["quality"] == "check"
    assert payload["manual_override"] is False
    assert payload["original_x_px"] is None
    assert payload["original_y_px"] is None


def test_atom_point_with_manual_position_preserves_original_fit_coordinates():
    point = AtomPoint(
        row_id="row-1",
        image_id="image-1",
        source_group_id="group-1",
        point_index=0,
        x_px=12.5,
        y_px=9.75,
        fit_success=True,
    )

    corrected = point.with_manual_position(x_px=14.0, y_px=8.5, source="drag")

    assert corrected.manual_override is True
    assert corrected.manual_override_source == "drag"
    assert corrected.x_px == 14.0
    assert corrected.y_px == 8.5
    assert corrected.original_x_px == 12.5
    assert corrected.original_y_px == 9.75
    assert corrected.fit_x_px == 12.5
    assert corrected.fit_y_px == 9.75


def test_atom_point_manual_override_roundtrip_restores_original_fit_coordinates():
    point = AtomPoint(
        row_id="row-1",
        image_id="image-1",
        source_group_id="group-1",
        point_index=1,
        x_px=15.0,
        y_px=20.0,
        manual_override=True,
        manual_override_source="drag",
        original_x_px=10.5,
        original_y_px=18.25,
    )

    restored = AtomPoint.from_dict(point.to_dict())

    assert restored == point
    assert restored.fit_x_px == 10.5
    assert restored.fit_y_px == 18.25


def test_atom_point_rejects_invalid_identity_or_index():
    with pytest.raises(ValueError, match="row_id"):
        AtomPoint(
            row_id="",
            image_id="image-1",
            source_group_id="group-1",
            point_index=0,
            x_px=1.0,
            y_px=2.0,
        )

    with pytest.raises(ValueError, match="point_index"):
        AtomPoint(
            row_id="row-1",
            image_id="image-1",
            source_group_id="group-1",
            point_index=-1,
            x_px=1.0,
            y_px=2.0,
        )


def test_atom_row_normalizes_points_and_tracks_next_index():
    point_late = AtomPoint(
        row_id="row-a",
        image_id="image-2",
        source_group_id="group-a",
        point_index=3,
        x_px=14.0,
        y_px=7.5,
    )
    point_early = AtomPoint(
        row_id="row-a",
        image_id="image-1",
        source_group_id="group-a",
        point_index=0,
        x_px=5.0,
        y_px=6.0,
    )

    row = AtomRow(
        row_id="row-a",
        source_group_id="group-a",
        display_name=" ",
        points=(point_late, point_early),
    )

    assert row.display_name == "Row row-a"
    assert row.point_count == 2
    assert row.next_point_index == 4
    assert tuple(point.point_index for point in row.points) == (0, 3)


def test_atom_row_with_point_validates_source_group_and_roundtrips_serialization():
    row = AtomRow(
        row_id="row-b",
        source_group_id="group-b",
        display_name="Domain wall row",
    )
    point = AtomPoint(
        row_id="row-b",
        image_id="variant-1",
        source_group_id="group-b",
        point_index=row.next_point_index,
        x_px=10.0,
        y_px=11.0,
        x_nm=0.44,
        y_nm=0.52,
    )

    updated_row = row.with_point(point)
    restored_row = AtomRow.from_dict(updated_row.to_dict())

    assert updated_row.point_count == 1
    assert updated_row.points[0] == point
    assert restored_row == updated_row

    foreign_point = AtomPoint(
        row_id="row-b",
        image_id="variant-2",
        source_group_id="group-c",
        point_index=1,
        x_px=12.0,
        y_px=15.0,
    )
    with pytest.raises(ValueError, match="source_group_id"):
        updated_row.with_point(foreign_point)
