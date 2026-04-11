"""Bootstrap tests for the standalone AtomMapper application."""

import numpy as np
import pytest

pytest.importorskip("PyQt6", reason="PyQt6 is required for AtomMapper GUI tests")
pytest.importorskip("pytestqt", reason="pytest-qt is required for AtomMapper GUI tests")

from AtomMapper.app.controller import AtomMapperController
from AtomMapper.app.main import create_application, create_main_window
from AtomMapper.app.models import LoadedImage


def _make_loaded_image(name: str) -> LoadedImage:
    data = np.arange(48, dtype=float).reshape((6, 8))
    return LoadedImage(
        source_path=f"/tmp/{name}",
        display_name=name,
        file_extension=".stp",
        image_data=data,
        pixels_x=8,
        pixels_y=6,
        size_nm_x=8.0,
        size_nm_y=6.0,
        metadata={"image_type": "Topo"},
        raw_metadata={},
    )


def test_create_main_window(qtbot):
    app = create_application(["atommapper-tests"])
    assert app is not None

    window = create_main_window()
    qtbot.addWidget(window)

    assert window.windowTitle() == "AtomMapper"
    assert window.centralWidget() is not None
    assert window.isVisible() is False
    assert window.file_list_widget.count() == 0
    assert window.active_image_label.text() == "Active image: none"
    assert window.image_viewport.current_loaded_image is None


def test_main_window_syncs_file_list_with_controller(qtbot):
    controller = AtomMapperController()
    window = create_main_window()
    qtbot.addWidget(window)
    window.close()

    controlled_window = window.__class__(controller=controller)
    qtbot.addWidget(controlled_window)

    first = _make_loaded_image("first.stp")
    second = _make_loaded_image("second.stp")
    controller.set_loaded_images([first, second])

    assert controlled_window.file_list_widget.count() == 2
    assert controlled_window.file_list_widget.item(0).text() == "first.stp"
    assert controlled_window.file_list_widget.item(1).text() == "second.stp"
    assert controller.active_image_index == 0
    assert controlled_window.image_viewport.current_loaded_image == first
    assert controlled_window.image_viewport.image_label.pixmap() is not None
    assert not controlled_window.image_viewport.image_label.pixmap().isNull()

    controlled_window.file_list_widget.setCurrentRow(1)

    assert controller.active_image_index == 1
    assert controller.active_image == second
    assert "second.stp" in controlled_window.active_image_label.text()
    assert controlled_window.image_viewport.current_loaded_image == second
