"""Tests for the experimental pyqtgraph STM viewport skeleton."""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("PyQt6", reason="PyQt6 is required for AtomMapper GUI tests")
pytest.importorskip("pytestqt", reason="pytest-qt is required for AtomMapper GUI tests")
pytest.importorskip("pyqtgraph", reason="pyqtgraph is required for the 2B viewport refactor")

from AtomMapper.app.models import LoadedImage, ROIState
from AtomMapper.app.pyqtgraph_image_view import PyQtGraphSTMViewport


def _make_loaded_image(name: str, width: int = 8, height: int = 6) -> LoadedImage:
    data = np.arange(width * height, dtype=float).reshape((height, width))
    return LoadedImage(
        source_path=f"/tmp/{name}",
        display_name=name,
        file_extension=".stp",
        image_data=data,
        pixels_x=width,
        pixels_y=height,
        size_nm_x=float(width),
        size_nm_y=float(height),
        metadata={"image_type": "Topo"},
        raw_metadata={},
    )


def test_pyqtgraph_viewport_initializes_with_placeholder(qtbot):
    viewport = PyQtGraphSTMViewport()
    qtbot.addWidget(viewport)

    assert viewport.backend_available is True
    assert viewport.image_view is not None
    assert viewport.plot_widget is not None
    assert viewport.image_item is not None
    assert viewport.histogram_widget is not None
    assert viewport.current_loaded_image is None
    assert viewport.stack.currentWidget() is viewport.placeholder_label
    assert "No image selected" in viewport.placeholder_label.text()
    assert bool(viewport.view_box.state["aspectLocked"]) is True


def test_pyqtgraph_viewport_switches_from_placeholder_to_canvas(qtbot):
    viewport = PyQtGraphSTMViewport()
    qtbot.addWidget(viewport)

    image = _make_loaded_image("pg-skeleton.stp")
    viewport.set_loaded_image(image)

    assert viewport.current_loaded_image == image
    assert viewport.stack.currentWidget() is viewport.plot_widget
    assert "pg-skeleton.stp" in viewport.info_label.text()
    assert viewport.image_item.image is not None
    assert viewport.image_item.image.shape == image.image_data.T.shape
    rect = viewport.image_item.boundingRect()
    assert rect.width() == pytest.approx(image.pixels_x)
    assert rect.height() == pytest.approx(image.pixels_y)
    assert viewport.view_box.state["yInverted"] is True
    assert viewport.view_box.state["mouseEnabled"] == [True, True]
    assert bool(viewport.view_box.state["aspectLocked"]) is True
    assert viewport.histogram_widget is not None


def test_pyqtgraph_viewport_clears_canvas_when_image_is_unset(qtbot):
    viewport = PyQtGraphSTMViewport()
    qtbot.addWidget(viewport)

    image = _make_loaded_image("pg-clear.stp")
    viewport.set_loaded_image(image)
    assert viewport.image_item.image is not None

    viewport.set_loaded_image(None)

    assert viewport.current_loaded_image is None
    assert viewport.stack.currentWidget() is viewport.placeholder_label
    assert "No image selected" in viewport.placeholder_label.text()
    assert viewport.image_item.image is None


def test_pyqtgraph_viewport_sets_roi_overlay_in_image_coordinates(qtbot):
    viewport = PyQtGraphSTMViewport()
    qtbot.addWidget(viewport)

    image = _make_loaded_image("pg-roi.stp")
    viewport.set_loaded_image(image)
    viewport.set_roi_state(ROIState(x=2, y=1, width=4, height=3))

    assert viewport.current_roi_state == ROIState(x=2, y=1, width=4, height=4)
    assert viewport.roi_item is not None
    assert viewport.roi_item.isVisible() is True
    assert tuple(map(float, viewport.roi_item.pos())) == pytest.approx((2.0, 1.0))
    assert tuple(map(float, viewport.roi_item.size())) == pytest.approx((4.0, 4.0))


def test_pyqtgraph_viewport_emits_roi_edits_and_clamps_to_4px_minimum(qtbot):
    viewport = PyQtGraphSTMViewport()
    qtbot.addWidget(viewport)

    image = _make_loaded_image("pg-roi-edit.stp", width=20, height=20)
    viewport.set_loaded_image(image)
    viewport.set_roi_state(ROIState(x=1, y=1, width=6, height=6))

    emitted: list[ROIState] = []
    viewport.roi_state_edited.connect(emitted.append)

    viewport.roi_item.setPos((3.2, 2.7))
    qtbot.waitUntil(lambda: len(emitted) >= 1)
    moved_roi = emitted[-1]
    assert moved_roi == ROIState(x=3, y=3, width=6, height=6)

    viewport.roi_item.setSize((1.0, 2.0))
    qtbot.waitUntil(lambda: len(emitted) >= 2)
    resized_roi = emitted[-1]
    assert resized_roi == ROIState(x=3, y=3, width=4, height=4)
    assert viewport.current_roi_state == resized_roi
    assert tuple(map(float, viewport.roi_item.size())) == pytest.approx((4.0, 4.0))


def test_pyqtgraph_viewport_supports_zoom_and_pan_without_touching_roi_state(qtbot):
    viewport = PyQtGraphSTMViewport()
    qtbot.addWidget(viewport)

    image = _make_loaded_image("pg-nav.stp", width=20, height=10)
    viewport.set_loaded_image(image)
    viewport.set_roi_state(ROIState(x=2, y=2, width=5, height=5))

    initial_roi = viewport.current_roi_state
    initial_x_range, initial_y_range = viewport.view_box.viewRange()
    assert (initial_x_range[1] - initial_x_range[0]) > 0.0
    assert (initial_y_range[1] - initial_y_range[0]) > 0.0

    viewport.view_box.scaleBy((0.5, 0.5))
    zoomed_x_range, zoomed_y_range = viewport.view_box.viewRange()
    assert (zoomed_x_range[1] - zoomed_x_range[0]) < (initial_x_range[1] - initial_x_range[0])
    assert (zoomed_y_range[1] - zoomed_y_range[0]) < (initial_y_range[1] - initial_y_range[0])

    viewport.view_box.translateBy(x=1.5, y=-0.5)
    panned_x_range, panned_y_range = viewport.view_box.viewRange()
    assert panned_x_range != pytest.approx(zoomed_x_range)
    assert panned_y_range != pytest.approx(zoomed_y_range)
    assert viewport.current_roi_state == initial_roi


def test_pyqtgraph_viewport_exposes_histogram_lut_controls(qtbot):
    viewport = PyQtGraphSTMViewport()
    qtbot.addWidget(viewport)

    image = _make_loaded_image("pg-gamma.stp", width=16, height=10)
    raw_before = np.array(image.image_data, copy=True)
    viewport.set_loaded_image(image)

    assert np.array_equal(image.image_data, raw_before)
    assert viewport.histogram_widget is not None
    assert viewport.image_view is not None
    levels_before = viewport.histogram_widget.item.getLevels()
    viewport.image_view.setLevels(min=5.0, max=30.0)
    levels_after = viewport.histogram_widget.item.getLevels()
    assert levels_after != levels_before
    assert np.array_equal(viewport.current_loaded_image.image_data, raw_before)
    assert np.array_equal(viewport.image_item.image, image.image_data.T)


def test_pyqtgraph_viewport_histogram_levels_preserve_view_range_and_roi_state(qtbot):
    viewport = PyQtGraphSTMViewport()
    qtbot.addWidget(viewport)

    image = _make_loaded_image("pg-levels-nav.stp", width=20, height=10)
    viewport.set_loaded_image(image)
    viewport.set_roi_state(ROIState(x=2, y=2, width=5, height=5))

    viewport.view_box.scaleBy((0.5, 0.5))
    viewport.view_box.translateBy(x=1.5, y=-0.5)
    view_before = viewport.view_box.viewRange()
    roi_before = viewport.current_roi_state

    viewport.image_view.setLevels(min=2.0, max=50.0)

    view_after = viewport.view_box.viewRange()
    assert view_after[0] == pytest.approx(view_before[0])
    assert view_after[1] == pytest.approx(view_before[1])
    assert viewport.current_roi_state == roi_before
