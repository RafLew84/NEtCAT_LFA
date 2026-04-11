"""Tests for the AtomMapper live ROI preview widget."""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("PyQt6", reason="PyQt6 is required for AtomMapper GUI tests")
pytest.importorskip("pytestqt", reason="pytest-qt is required for AtomMapper GUI tests")

from AtomMapper.app.models import LoadedImage, ROIState
from AtomMapper.app.roi_preview import ROIPreviewWidget


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


def test_roi_preview_renders_and_updates_current_patch(qtbot):
    preview = ROIPreviewWidget()
    qtbot.addWidget(preview)

    image = _make_loaded_image(
        "preview.stp",
        np.arange(96, dtype=float).reshape((8, 12)),
    )
    preview.set_loaded_image(image)
    preview.set_roi_state(ROIState(x=2, y=1, width=5, height=4))

    expected_first = image.image_data[1:5, 2:7]
    assert preview.current_patch_data is not None
    assert np.array_equal(preview.current_patch_data, expected_first)
    assert preview.preview_label.pixmap() is not None
    assert not preview.preview_label.pixmap().isNull()
    assert "ROI x=2 y=1 w=5 h=4" in preview.info_label.text()

    preview.set_roi_state(ROIState(x=4, y=2, width=3, height=3))

    expected_second = image.image_data[2:5, 4:7]
    assert preview.current_patch_data is not None
    assert np.array_equal(preview.current_patch_data, expected_second)
    assert "patch 3x3 px" in preview.info_label.text()


def test_roi_preview_handles_out_of_range_roi(qtbot):
    preview = ROIPreviewWidget()
    qtbot.addWidget(preview)

    image = _make_loaded_image(
        "outside.stp",
        np.arange(48, dtype=float).reshape((6, 8)),
    )
    preview.set_loaded_image(image)
    preview.set_roi_state(ROIState(x=30, y=30, width=5, height=5))

    assert preview.current_patch_data is None
    assert preview.preview_label.text() == "ROI is outside the image bounds."
    assert "does not intersect the image" in preview.info_label.text()
