"""Tests for the AtomMapper STM image viewport."""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("PyQt6", reason="PyQt6 is required for AtomMapper GUI tests")
pytest.importorskip("pytestqt", reason="pytest-qt is required for AtomMapper GUI tests")

from AtomMapper.app.image_view import STMImageViewport
from AtomMapper.app.models import LoadedImage


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
