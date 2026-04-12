"""Bootstrap tests for the standalone AtomMapper application."""

import numpy as np
import pytest
from PyQt6.QtCore import Qt

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
    assert window.export_csv_button.isEnabled() is False
    assert window.save_session_button.isEnabled() is True
    assert window.load_session_button.isEnabled() is True
    assert window.analysis_dock.widget() == window.analysis_dock_content


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
    assert controlled_window.image_viewport.image_item is not None
    assert controlled_window.image_viewport.image_item.image is not None

    controlled_window.file_list_widget.setCurrentRow(1)

    assert controller.active_image_index == 1
    assert controller.active_image == second
    assert "second.stp" in controlled_window.active_image_label.text()
    assert controlled_window.image_viewport.current_loaded_image == second


def test_main_window_places_saved_points_in_bottom_analysis_dock(qtbot):
    window = create_main_window()
    qtbot.addWidget(window)

    grid_layout = window.analysis_grid_panel.layout()
    assert grid_layout is not None

    roi_index = grid_layout.indexOf(window.roi_preview)
    fit_index = grid_layout.indexOf(window.gaussian_fit_preview)
    image_index = grid_layout.indexOf(window.image_viewport)

    assert roi_index >= 0
    assert fit_index >= 0
    assert image_index >= 0

    assert grid_layout.getItemPosition(roi_index) == (0, 0, 1, 1)
    assert grid_layout.getItemPosition(fit_index) == (1, 0, 1, 1)
    assert grid_layout.indexOf(window.saved_points_panel) == -1
    assert grid_layout.getItemPosition(image_index) == (0, 1, 2, 1)
    assert window.analysis_dock.widget() == window.analysis_dock_content
    analysis_layout = window.analysis_dock_content.layout()
    assert analysis_layout is not None
    assert analysis_layout.indexOf(window.saved_points_panel) >= 0
    assert analysis_layout.indexOf(window.row_plot_widget) >= 0
    assert analysis_layout.indexOf(window.global_scatter_plot_widget) >= 0
    assert analysis_layout.indexOf(window.row_metrics_widget) >= 0
    assert window.dockWidgetArea(window.analysis_dock) == Qt.DockWidgetArea.BottomDockWidgetArea
