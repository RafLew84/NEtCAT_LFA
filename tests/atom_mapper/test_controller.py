"""Tests for AtomMapper controller state management."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from AtomMapper.app.controller import AtomMapperController
from AtomMapper.app.io import load_loaded_image
from AtomMapper.app.models import AtomPoint, AtomRow, LoadedImage, ROIState
from AtomMapper.app.preprocessing import is_bm3d_available
from AtomMapper.app.session_model import AtomMapperSession, SessionViewState

PROJECT_ROOT = Path(__file__).resolve().parents[2]


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


def test_controller_can_create_rotate_variant_for_active_image():
    controller = AtomMapperController()
    original = _make_loaded_image("sample.stp", width=40, height=30)
    controller.set_loaded_images([original])

    variant = controller.create_rotate_variant_for_active_image(
        quarter_turns=1,
        make_active=True,
    )

    assert variant.variant_name == "rotate-90"
    assert variant.parent_image_id == original.image_id
    assert variant.source_group_id == original.source_group_id
    assert variant.metadata["preprocess"] == "rotate"
    assert variant.metadata["rotate_quarter_turns"] == 1
    assert variant.metadata["rotate_angle_deg"] == 90
    assert variant.pixels_x == original.pixels_y
    assert variant.pixels_y == original.pixels_x
    assert variant.size_nm_x == original.size_nm_y
    assert variant.size_nm_y == original.size_nm_x
    assert variant.image_data.shape == (original.pixels_x, original.pixels_y)
    assert np.array_equal(variant.image_data, np.rot90(original.image_data, k=1))
    assert controller.loaded_images == (original, variant)
    assert controller.active_image == variant


def test_controller_rejects_rotate_variant_without_active_image():
    controller = AtomMapperController()

    with pytest.raises(ValueError, match="active image"):
        controller.create_rotate_variant_for_active_image()


def test_controller_can_create_flip_variant_for_active_image():
    controller = AtomMapperController()
    original = _make_loaded_image("sample.stp", width=40, height=30)
    controller.set_loaded_images([original])

    variant = controller.create_flip_variant_for_active_image(
        flip_x=True,
        flip_y=True,
        make_active=True,
    )

    assert variant.variant_name == "flip-xy"
    assert variant.parent_image_id == original.image_id
    assert variant.source_group_id == original.source_group_id
    assert variant.metadata["preprocess"] == "flip"
    assert variant.metadata["flip_x"] is True
    assert variant.metadata["flip_y"] is True
    assert variant.pixels_x == original.pixels_x
    assert variant.pixels_y == original.pixels_y
    assert np.array_equal(variant.image_data, np.flipud(np.fliplr(original.image_data)))
    assert controller.loaded_images == (original, variant)
    assert controller.active_image == variant


def test_controller_rejects_flip_variant_without_active_image():
    controller = AtomMapperController()

    with pytest.raises(ValueError, match="active image"):
        controller.create_flip_variant_for_active_image()


def test_controller_tracks_rows_per_source_group_and_preserves_active_row_across_variants():
    controller = AtomMapperController()
    original = _make_loaded_image("sample.stp", width=40, height=30)
    other = _make_loaded_image("other.stp", width=32, height=24)
    variant = original.derive_variant(variant_name="blur", image_data=original.image_data + 1.0)

    controller.set_loaded_images([original, variant, other])

    first_row = controller.create_row_for_active_source_group(display_name="Row A")

    assert controller.atom_rows == (first_row,)
    assert controller.active_row == first_row
    assert controller.rows_for_source_group(original.source_group_id) == (first_row,)

    controller.select_image(1)
    assert controller.active_image == variant
    assert controller.active_source_group_id == original.source_group_id
    assert controller.active_row == first_row

    controller.select_image(2)
    assert controller.active_image == other
    assert controller.active_row is None

    second_row = controller.create_row_for_active_source_group(display_name="Row B")
    assert controller.active_row == second_row
    assert controller.rows_for_source_group(other.source_group_id) == (second_row,)

    controller.select_image(0)
    assert controller.active_image == original
    assert controller.active_row == first_row


def test_controller_add_point_to_row_and_emit_row_point_signals():
    controller = AtomMapperController()
    original = _make_loaded_image("sample.stp", width=40, height=30)
    controller.set_loaded_images([original])
    row = controller.create_row_for_active_source_group(display_name="Row A")

    active_row_events: list[AtomRow | None] = []
    point_events: list[AtomRow] = []
    controller.active_row_changed.connect(active_row_events.append)
    controller.row_points_changed.connect(point_events.append)

    point = AtomPoint(
        row_id=row.row_id,
        image_id=original.image_id,
        source_group_id=original.source_group_id,
        point_index=row.next_point_index,
        x_px=12.5,
        y_px=9.25,
        sigma_x_px=1.2,
        sigma_y_px=1.4,
    )

    updated_row = controller.add_point_to_row(point)

    assert updated_row.point_count == 1
    stored_point = updated_row.points[0]
    assert stored_point.x_px == point.x_px
    assert stored_point.y_px == point.y_px
    assert stored_point.sigma_x_px == point.sigma_x_px
    assert stored_point.sigma_y_px == point.sigma_y_px
    assert stored_point.x_nm == pytest.approx(point.x_px)
    assert stored_point.y_nm == pytest.approx(point.y_px)
    assert controller.active_row == updated_row
    assert point_events[-1] == updated_row
    assert active_row_events[-1] == updated_row


def test_controller_restore_from_session_reconstructs_runtime_state():
    controller = AtomMapperController()
    original = _make_loaded_image("session.stp", width=20, height=16)
    variant = original.derive_variant(variant_name="blur", image_data=original.image_data + 1.0)
    row = AtomRow(source_group_id=original.source_group_id, display_name="Row 1")
    point = AtomPoint(
        row_id=row.row_id,
        image_id=variant.image_id,
        source_group_id=original.source_group_id,
        point_index=0,
        x_px=7.5,
        y_px=8.5,
        point_id="point-1",
    )
    row = row.with_point(point)

    session = AtomMapperSession(
        loaded_images=(original, variant),
        active_image_id=variant.image_id,
        roi_states_by_image_id={
            original.image_id: ROIState(x=2, y=3, width=8, height=8),
            variant.image_id: ROIState(x=4, y=5, width=6, height=7),
        },
        rows=(row,),
        active_row_id_by_source_group={original.source_group_id: row.row_id},
        active_point_id_by_source_group={original.source_group_id: point.point_id},
        view_state=SessionViewState(),
    )

    controller.restore_from_session(session)

    assert controller.loaded_images == (original, variant)
    assert controller.active_image == variant
    assert controller.active_image_index == 1
    assert controller.active_roi_state == ROIState(x=4, y=5, width=6, height=7)
    assert controller.atom_rows == (row,)
    assert controller.active_row == row
    assert controller.active_row_id_by_source_group == {original.source_group_id: row.row_id}


@pytest.mark.parametrize("sample_name", ["8343.stp", "85291r.s94"])
def test_controller_populates_nm_coordinates_for_real_loaded_samples(sample_name: str):
    controller = AtomMapperController()
    loaded = load_loaded_image(PROJECT_ROOT / "data" / sample_name)
    controller.set_loaded_images([loaded])
    row = controller.create_row_for_active_source_group(display_name="Row 1")

    point = AtomPoint(
        row_id=row.row_id,
        image_id=loaded.image_id,
        source_group_id=loaded.source_group_id,
        point_index=0,
        x_px=10.5,
        y_px=12.25,
        point_id=f"{sample_name}-point",
    )

    updated_row = controller.add_point_to_row(point)
    stored_point = updated_row.points[0]
    calibration = loaded.physical_calibration

    assert calibration is not None
    assert stored_point.x_nm == pytest.approx(10.5 * calibration.pixel_size_nm_x)
    assert stored_point.y_nm == pytest.approx(12.25 * calibration.pixel_size_nm_y)


def test_controller_add_point_to_row_leaves_nm_empty_without_physical_calibration():
    controller = AtomMapperController()
    original = LoadedImage(
        source_path="/tmp/no-calibration.stp",
        display_name="no-calibration.stp",
        file_extension=".stp",
        image_data=np.zeros((6, 8), dtype=float),
        pixels_x=8,
        pixels_y=6,
        size_nm_x=0.0,
        size_nm_y=6.0,
    )
    controller.set_loaded_images([original])
    row = controller.create_row_for_active_source_group(display_name="Row A")

    point = AtomPoint(
        row_id=row.row_id,
        image_id=original.image_id,
        source_group_id=original.source_group_id,
        point_index=0,
        x_px=3.0,
        y_px=4.0,
    )

    updated_row = controller.add_point_to_row(point)

    assert updated_row.points[0].x_nm is None
    assert updated_row.points[0].y_nm is None


def test_controller_remove_single_point_from_row_without_deleting_row():
    controller = AtomMapperController()
    original = _make_loaded_image("sample.stp", width=40, height=30)
    controller.set_loaded_images([original])
    row = controller.create_row_for_active_source_group(display_name="Row A")

    first_point = AtomPoint(
        row_id=row.row_id,
        image_id=original.image_id,
        source_group_id=original.source_group_id,
        point_index=0,
        x_px=10.0,
        y_px=11.0,
    )
    second_point = AtomPoint(
        row_id=row.row_id,
        image_id=original.image_id,
        source_group_id=original.source_group_id,
        point_index=1,
        x_px=20.0,
        y_px=21.0,
    )
    controller.add_point_to_row(first_point)
    controller.add_point_to_row(second_point)

    updated_row = controller.remove_point_from_row(row.row_id, first_point.point_id)

    assert updated_row.row_id == row.row_id
    assert updated_row.point_count == 1
    remaining_point = updated_row.points[0]
    assert remaining_point.point_id == second_point.point_id
    assert remaining_point.x_px == second_point.x_px
    assert remaining_point.y_px == second_point.y_px
    assert remaining_point.x_nm == pytest.approx(second_point.x_px)
    assert remaining_point.y_nm == pytest.approx(second_point.y_px)
    assert controller.active_row == updated_row
    assert controller.rows_for_source_group(original.source_group_id) == (updated_row,)


def test_controller_move_point_in_row_marks_manual_override_and_preserves_fit_origin():
    controller = AtomMapperController()
    original = _make_loaded_image("sample.stp", width=40, height=30)
    controller.set_loaded_images([original])
    row = controller.create_row_for_active_source_group(display_name="Row A")

    point = AtomPoint(
        row_id=row.row_id,
        image_id=original.image_id,
        source_group_id=original.source_group_id,
        point_index=0,
        x_px=12.5,
        y_px=9.25,
        sigma_x_px=1.2,
        sigma_y_px=1.4,
    )
    controller.add_point_to_row(point)

    updated_row = controller.move_point_in_row(
        row_id=row.row_id,
        point_id=point.point_id,
        x_px=15.0,
        y_px=8.0,
        source="drag",
    )

    moved_point = updated_row.points[0]
    assert moved_point.manual_override is True
    assert moved_point.manual_override_source == "drag"
    assert moved_point.x_px == pytest.approx(15.0)
    assert moved_point.y_px == pytest.approx(8.0)
    assert moved_point.x_nm == pytest.approx(15.0)
    assert moved_point.y_nm == pytest.approx(8.0)
    assert moved_point.fit_x_px == pytest.approx(12.5)
    assert moved_point.fit_y_px == pytest.approx(9.25)
    assert controller.active_row == updated_row


def test_controller_replace_and_remove_point_validate_row_and_image_consistency():
    controller = AtomMapperController()
    original = _make_loaded_image("sample.stp", width=40, height=30)
    other = _make_loaded_image("other.stp", width=20, height=20)
    controller.set_loaded_images([original, other])
    row = controller.create_row_for_active_source_group(display_name="Row A")

    point = AtomPoint(
        row_id=row.row_id,
        image_id=original.image_id,
        source_group_id=original.source_group_id,
        point_index=0,
        x_px=10.0,
        y_px=11.0,
    )
    controller.add_point_to_row(point)

    with pytest.raises(ValueError, match="Unknown row_id"):
        controller.remove_point_from_row("missing-row", point.point_id)

    with pytest.raises(ValueError, match="not present in row"):
        controller.remove_point_from_row(row.row_id, "missing-point")

    foreign_image_replacement = AtomPoint(
        point_id=point.point_id,
        row_id=row.row_id,
        image_id=other.image_id,
        source_group_id=other.source_group_id,
        point_index=point.point_index,
        x_px=33.0,
        y_px=44.0,
    )
    with pytest.raises(ValueError, match="different source_group_id"):
        controller.replace_point_in_row(foreign_image_replacement)


def test_controller_rejects_rows_and_points_with_unknown_or_mismatched_source_group():
    controller = AtomMapperController()
    original = _make_loaded_image("sample.stp", width=40, height=30)
    other = _make_loaded_image("other.stp", width=20, height=20)
    controller.set_loaded_images([original, other])

    foreign_row = AtomRow(
        source_group_id="missing-group",
        display_name="Foreign",
    )
    with pytest.raises(ValueError, match="source_group_id"):
        controller.add_row(foreign_row)

    row = controller.create_row_for_active_source_group(display_name="Row A")
    wrong_group_point = AtomPoint(
        row_id=row.row_id,
        image_id=other.image_id,
        source_group_id=other.source_group_id,
        point_index=0,
        x_px=1.0,
        y_px=2.0,
    )
    with pytest.raises(ValueError, match="different source_group_id"):
        controller.add_point_to_row(wrong_group_point)

    missing_image_point = AtomPoint(
        row_id=row.row_id,
        image_id="missing-image",
        source_group_id=original.source_group_id,
        point_index=0,
        x_px=1.0,
        y_px=2.0,
    )
    with pytest.raises(ValueError, match="Unknown image_id"):
        controller.add_point_to_row(missing_image_point)
