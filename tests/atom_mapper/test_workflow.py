"""End-to-end GUI smoke tests for the AtomMapper foundation workflow."""

from __future__ import annotations

from pathlib import Path

import pytest
from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import QMessageBox

from AtomMapper.app.main import create_main_window

pytest.importorskip("PyQt6", reason="PyQt6 is required for AtomMapper GUI tests")
pytest.importorskip("pytestqt", reason="pytest-qt is required for AtomMapper GUI tests")

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def test_main_window_loads_sample_file_via_button(qtbot, monkeypatch):
    window = create_main_window()
    qtbot.addWidget(window)

    sample_path = str(PROJECT_ROOT / "data" / "8343.stp")

    monkeypatch.setattr(
        "AtomMapper.app.main_window.QFileDialog.getOpenFileNames",
        lambda *args, **kwargs: ([sample_path], "STM files (*.stp *.s94)"),
    )

    qtbot.mouseClick(window.load_button, Qt.MouseButton.LeftButton)

    assert len(window.controller.loaded_images) == 1
    assert window.file_list_widget.count() == 1
    assert window.file_list_widget.item(0).text() == "8343.stp"
    assert window.controller.active_image is not None
    assert window.controller.active_image.display_name == "8343.stp"
    assert window.image_viewport.current_loaded_image is not None
    assert window.image_viewport.current_loaded_image.display_name == "8343.stp"
    assert window.image_viewport.image_label.pixmap() is not None
    assert not window.image_viewport.image_label.pixmap().isNull()
    assert "Loaded 1 STM file." in window.statusBar().currentMessage()
    assert window.file_list_hint_label.text() == "1 STM file loaded."


def test_main_window_reports_load_errors(qtbot, monkeypatch, tmp_path: Path):
    window = create_main_window()
    qtbot.addWidget(window)

    bad_path = str(tmp_path / "bad.txt")
    Path(bad_path).write_text("dummy", encoding="utf-8")

    monkeypatch.setattr(
        "AtomMapper.app.main_window.QFileDialog.getOpenFileNames",
        lambda *args, **kwargs: ([bad_path], "STM files (*.stp *.s94)"),
    )

    captured: dict[str, str] = {}

    def fake_warning(parent, title, text):
        captured["title"] = title
        captured["text"] = text
        return QMessageBox.StandardButton.Ok

    monkeypatch.setattr("AtomMapper.app.main_window.QMessageBox.warning", fake_warning)

    qtbot.mouseClick(window.load_button, Qt.MouseButton.LeftButton)

    assert len(window.controller.loaded_images) == 0
    assert window.file_list_widget.count() == 0
    assert captured["title"] == "AtomMapper - Load Error"
    assert "Could not load one or more STM files" in captured["text"]
    assert "bad.txt" in captured["text"]
    assert "Some files failed to load." in window.statusBar().currentMessage()
