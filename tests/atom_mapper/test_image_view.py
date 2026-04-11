"""Tests for the AtomMapper STM image viewport."""

from __future__ import annotations

import numpy as np
import pytest
from PyQt6.QtCore import QPoint, Qt

pytest.importorskip("PyQt6", reason="PyQt6 is required for AtomMapper GUI tests")
pytest.importorskip("pytestqt", reason="pytest-qt is required for AtomMapper GUI tests")

from AtomMapper.app.image_view import STMImageViewport
from AtomMapper.app.models import LoadedImage, ROIState


def _make_loaded_image(name: str, image_data: np.ndarray) -> LoadedImage:
    return LoadedImage(
        source_path=f"/tmp/{name}",
        display_name=name,
        file_extension=".stp",
        image_data=image_data,
        pixels_x=image_data.shape[1],
        pixels_y=image_data.shape[0],
        size_nm_x=float(image_data.shape[1]),
        size_nm_y=float(image_data.shape[0]),
        metadata={"image_type": "Topo"},
        raw_metadata={},
    )


def test_image_viewport_renders_selected_image(qtbot):
    viewport = STMImageViewport()
    qtbot.addWidget(viewport)

    image = _make_loaded_image(
        "gradient.stp",
        np.arange(64, dtype=float).reshape((8, 8)),
    )
    viewport.resize(500, 500)
    viewport.set_loaded_image(image)

    assert viewport.current_loaded_image == image
    assert viewport.image_label.pixmap() is not None
    assert not viewport.image_label.pixmap().isNull()
    assert "gradient.stp" in viewport.info_label.text()


def test_image_viewport_zoom_changes_scale(qtbot):
    viewport = STMImageViewport()
    qtbot.addWidget(viewport)
    viewport.resize(640, 520)

    image = _make_loaded_image(
        "zoom.stp",
        np.arange(300, dtype=float).reshape((15, 20)),
    )
    viewport.set_loaded_image(image)

    initial_zoom = viewport.zoom_factor
    initial_pixmap_size = viewport.image_label.pixmap().size()

    viewport.handle_wheel_delta(120)

    assert viewport.zoom_factor > initial_zoom
    assert viewport.image_label.pixmap().size().width() > initial_pixmap_size.width()
    assert "zoom 1.15x" in viewport.info_label.text()

    viewport.handle_wheel_delta(-120)

    assert viewport.zoom_factor == pytest.approx(1.0)
    assert viewport.scroll_area.alignment() == (
        Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignTop
    )


def test_image_viewport_fills_one_viewport_dimension_at_fit_scale(qtbot):
    viewport = STMImageViewport()
    qtbot.addWidget(viewport)
    viewport.resize(640, 520)
    viewport.show()

    image = _make_loaded_image(
        "fit-scale.stp",
        np.arange(1200, dtype=float).reshape((30, 40)),
    )
    viewport.set_loaded_image(image)
    qtbot.waitUntil(lambda: viewport.image_label.pixmap() is not None)

    viewport_rect = viewport.scroll_area.viewport().rect()
    pixmap_size = viewport.image_label.pixmap().size()

    assert (
        pixmap_size.width() == viewport_rect.width()
        or pixmap_size.height() == viewport_rect.height()
    )


def test_image_viewport_keeps_roi_aligned_while_zooming(qtbot):
    viewport = STMImageViewport()
    qtbot.addWidget(viewport)
    viewport.resize(640, 520)

    image = _make_loaded_image(
        "zoom-roi.stp",
        np.arange(1200, dtype=float).reshape((30, 40)),
    )
    viewport.set_loaded_image(image)
    viewport.set_roi_state(ROIState(x=8, y=6, width=10, height=12))

    rect_before = viewport.image_label.roi_display_rect()
    assert rect_before is not None
    before_ratios = (
        rect_before.x() / viewport.image_label.width(),
        rect_before.y() / viewport.image_label.height(),
        rect_before.width() / viewport.image_label.width(),
        rect_before.height() / viewport.image_label.height(),
    )

    viewport.handle_wheel_delta(120)

    rect_after = viewport.image_label.roi_display_rect()
    assert rect_after is not None
    after_ratios = (
        rect_after.x() / viewport.image_label.width(),
        rect_after.y() / viewport.image_label.height(),
        rect_after.width() / viewport.image_label.width(),
        rect_after.height() / viewport.image_label.height(),
    )

    assert viewport.image_label.margin() == 0
    assert "padding" not in viewport.image_label.styleSheet()
    assert after_ratios == pytest.approx(before_ratios)


def test_image_viewport_zoom_can_anchor_to_cursor_position(qtbot):
    viewport = STMImageViewport()
    qtbot.addWidget(viewport)
    viewport.resize(420, 320)
    viewport.show()

    image = _make_loaded_image(
        "zoom-anchor.stp",
        np.arange(12000, dtype=float).reshape((100, 120)),
    )
    viewport.set_loaded_image(image)
    viewport.set_roi_state(ROIState(x=45, y=40, width=16, height=16))

    for _ in range(4):
        viewport.handle_wheel_delta(120)

    horizontal_bar = viewport.scroll_area.horizontalScrollBar()
    vertical_bar = viewport.scroll_area.verticalScrollBar()
    horizontal_bar.setValue(45)
    vertical_bar.setValue(30)
    before_values = (horizontal_bar.value(), vertical_bar.value())

    viewport.handle_wheel_delta(120, QPoint(160, 110))

    after_values = (horizontal_bar.value(), vertical_bar.value())
    assert after_values[0] > before_values[0]
    assert after_values[1] > before_values[1]


def test_image_viewport_emits_roi_changes_for_move_and_resize(qtbot):
    viewport = STMImageViewport()
    qtbot.addWidget(viewport)
    viewport.resize(640, 520)
    viewport.show()

    image = _make_loaded_image(
        "roi.stp",
        np.arange(2000, dtype=float).reshape((40, 50)),
    )
    viewport.set_loaded_image(image)
    viewport.set_roi_state(ROIState(x=10, y=8, width=12, height=10))

    emitted: list[ROIState] = []
    viewport.roi_state_edited.connect(emitted.append)

    move_rect = viewport.image_label.roi_display_rect()
    assert move_rect is not None
    move_start = move_rect.center().toPoint()
    move_end = move_start + QPoint(30, 18)
    qtbot.mousePress(viewport.image_label, Qt.MouseButton.LeftButton, pos=move_start)
    qtbot.mouseMove(viewport.image_label, pos=move_end)
    qtbot.mouseRelease(viewport.image_label, Qt.MouseButton.LeftButton, pos=move_end)

    assert emitted
    moved_roi = emitted[-1]
    assert moved_roi.x > 10
    assert moved_roi.y > 8

    resize_rect = viewport.image_label.roi_display_rect()
    assert resize_rect is not None
    handle_pos = resize_rect.bottomRight().toPoint() - QPoint(2, 2)
    resize_end = handle_pos + QPoint(28, 24)
    qtbot.mousePress(viewport.image_label, Qt.MouseButton.LeftButton, pos=handle_pos)
    qtbot.mouseMove(viewport.image_label, pos=resize_end)
    qtbot.mouseRelease(viewport.image_label, Qt.MouseButton.LeftButton, pos=resize_end)

    resized_roi = emitted[-1]
    assert resized_roi.width >= moved_roi.width
    assert resized_roi.height >= moved_roi.height


def test_image_viewport_clamps_roi_drag_to_image_bounds(qtbot):
    viewport = STMImageViewport()
    qtbot.addWidget(viewport)
    viewport.resize(420, 320)
    viewport.show()

    image = _make_loaded_image(
        "roi-clamp.stp",
        np.arange(12000, dtype=float).reshape((100, 120)),
    )
    viewport.set_loaded_image(image)
    viewport.set_roi_state(ROIState(x=90, y=70, width=20, height=20))

    emitted: list[ROIState] = []
    viewport.roi_state_edited.connect(emitted.append)

    roi_rect = viewport.image_label.roi_display_rect()
    assert roi_rect is not None
    move_start = roi_rect.center().toPoint()
    move_end = QPoint(viewport.image_label.width() + 150, viewport.image_label.height() + 120)

    qtbot.mousePress(viewport.image_label, Qt.MouseButton.LeftButton, pos=move_start)
    qtbot.mouseMove(viewport.image_label, pos=move_end)
    qtbot.mouseRelease(viewport.image_label, Qt.MouseButton.LeftButton, pos=move_end)

    assert emitted
    moved_roi = emitted[-1]
    assert moved_roi.x + moved_roi.width <= image.pixels_x
    assert moved_roi.y + moved_roi.height <= image.pixels_y

    refreshed_rect = viewport.image_label.roi_display_rect()
    assert refreshed_rect is not None
    assert refreshed_rect.right() <= viewport.image_label.width()
    assert refreshed_rect.bottom() <= viewport.image_label.height()


def test_image_viewport_resize_can_shrink_roi_to_4px_minimum(qtbot):
    viewport = STMImageViewport()
    qtbot.addWidget(viewport)
    viewport.resize(420, 320)
    viewport.show()

    image = _make_loaded_image(
        "roi-minimum.stp",
        np.arange(12000, dtype=float).reshape((100, 120)),
    )
    viewport.set_loaded_image(image)
    viewport.set_roi_state(ROIState(x=40, y=30, width=18, height=18))

    emitted: list[ROIState] = []
    viewport.roi_state_edited.connect(emitted.append)

    qtbot.waitUntil(lambda: viewport.image_label._resize_handle.isVisible())
    handle_pos = viewport.image_label._resize_handle.geometry().center()
    resize_end = QPoint(1, 1)

    qtbot.mousePress(viewport.image_label, Qt.MouseButton.LeftButton, pos=handle_pos)
    qtbot.mouseMove(viewport.image_label, pos=resize_end)
    qtbot.mouseRelease(viewport.image_label, Qt.MouseButton.LeftButton, pos=resize_end)

    assert emitted
    resized_roi = emitted[-1]
    assert resized_roi.width == 4
    assert resized_roi.height == 4


def test_image_viewport_supports_panning_by_dragging_background(qtbot):
    viewport = STMImageViewport()
    qtbot.addWidget(viewport)
    viewport.resize(420, 320)
    viewport.show()

    image = _make_loaded_image(
        "pan.stp",
        np.arange(12000, dtype=float).reshape((100, 120)),
    )
    viewport.set_loaded_image(image)
    viewport.set_roi_state(ROIState(x=45, y=40, width=16, height=16))

    for _ in range(4):
        viewport.handle_wheel_delta(120)

    horizontal_bar = viewport.scroll_area.horizontalScrollBar()
    vertical_bar = viewport.scroll_area.verticalScrollBar()
    horizontal_bar.setValue(60)
    vertical_bar.setValue(55)

    start = QPoint(horizontal_bar.value() + 20, vertical_bar.value() + 20)
    end = start + QPoint(30, 24)
    before_values = (horizontal_bar.value(), vertical_bar.value())

    qtbot.mousePress(viewport.image_label, Qt.MouseButton.LeftButton, pos=start)
    qtbot.mouseMove(viewport.image_label, pos=end)
    qtbot.mouseRelease(viewport.image_label, Qt.MouseButton.LeftButton, pos=end)

    after_values = (horizontal_bar.value(), vertical_bar.value())
    assert after_values[0] < before_values[0]
    assert after_values[1] < before_values[1]
