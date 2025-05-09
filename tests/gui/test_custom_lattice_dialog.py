# tests/gui/test_custom_lattice_dialog.py
"""
Unit tests for the CustomLatticeDialog using pytest-qt.
"""
import pytest
from PyQt6.QtWidgets import QDialog, QMessageBox

# Import the dialog class
try:
    from lfa.gui.custom_lattice_dialog import CustomLatticeDialog
except ImportError:
    pytest.fail("Could not import CustomLatticeDialog from lfa.gui.custom_lattice_dialog", pytrace=False)

# Requires PyQt6 and pytest-qt
pytest.importorskip("PyQt6", reason="PyQt6 not found, skipping GUI dialog tests")
pytest.importorskip("pytestqt.qt_compat", reason="pytest-qt not found, skipping GUI dialog tests")
from pytestqt.qt_compat import qt_api # Used by qtbot

# --- Test Functions ---

def test_custom_lattice_dialog_init(qtbot):
    """Test basic initialization of the dialog."""
    dialog = CustomLatticeDialog()
    qtbot.addWidget(dialog) # Register for cleanup
    assert dialog.windowTitle() == "Define Custom Lattice"
    assert dialog.name_edit.text() == "Custom Lattice"
    assert dialog.type_combo.currentText() == "hexagonal" # Check default
    assert dialog.a_surf_spinbox.value() > 0

def test_custom_lattice_dialog_get_definition_ok(qtbot):
    """Test getting lattice definition after setting values and clicking OK."""
    dialog = CustomLatticeDialog()
    qtbot.addWidget(dialog)

    # Set values
    dialog.name_edit.setText("My Hexagonal")
    dialog.type_combo.setCurrentText("hexagonal")
    dialog.a_surf_spinbox.setValue(0.350)

    # Simulate clicking OK
    # In a real test, we'd connect to accepted signal or click the button
    # For simplicity, call accept_input directly for this unit test
    dialog.accept_input() # This will call super().accept() if valid

    definition = dialog.get_lattice_definition()
    assert definition is not None
    assert definition["name"] == "My Hexagonal"
    assert definition["type"] == "hexagonal"
    assert definition["a_surf"] == 0.350
    assert definition["source"] == "User Defined"

def test_custom_lattice_dialog_get_definition_cancel(qtbot):
    """Test getting definition if dialog is cancelled."""
    dialog = CustomLatticeDialog()
    qtbot.addWidget(dialog)
    dialog.reject() # Simulate Cancel
    assert dialog.get_lattice_definition() is None

def test_custom_lattice_dialog_validation_empty_name(qtbot, mocker):
    dialog = CustomLatticeDialog(); qtbot.addWidget(dialog)
    # Mockuj statyczną metodę warning w module, w którym jest używana
    mock_qmessagebox_warning = mocker.patch("lfa.gui.custom_lattice_dialog.QMessageBox.warning")

    dialog.name_edit.setText("  "); dialog.accept_input()

    # Sprawdź, czy zamockowana metoda została wywołana
    mock_qmessagebox_warning.assert_called_once()
    args, _ = mock_qmessagebox_warning.call_args
    assert "Lattice name cannot be empty" in args[2] # args[2] to tekst wiadomości

    assert dialog.result() != QDialog.DialogCode.Accepted
    assert dialog.get_lattice_definition() is None