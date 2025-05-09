# lfa/gui/custom_lattice_dialog.py
"""
Dialog for defining a custom substrate lattice.
"""
import logging
from typing import Optional, Dict, Any

try:
    from PyQt6.QtWidgets import (
        QDialog, QVBoxLayout, QHBoxLayout, QFormLayout, QLabel, QLineEdit,
        QComboBox, QDialogButtonBox, QDoubleSpinBox, QMessageBox
    )
    from PyQt6.QtCore import pyqtSlot
except ImportError:
    logging.critical("Failed to import necessary PyQt6 modules for CustomLatticeDialog.")
    raise

logger = logging.getLogger(__name__)

class CustomLatticeDialog(QDialog):
    """Dialog for user to define custom lattice parameters."""
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Define Custom Lattice")
        self.setMinimumWidth(350)

        self._lattice_definition: Optional[Dict[str, Any]] = None

        layout = QVBoxLayout(self)
        form_layout = QFormLayout()

        self.name_edit = QLineEdit("Custom Lattice")
        form_layout.addRow("Lattice Name:", self.name_edit)

        self.type_combo = QComboBox()
        self.type_combo.addItems(["hexagonal", "square"]) # Supported types
        form_layout.addRow("Lattice Type:", self.type_combo)

        self.a_surf_spinbox = QDoubleSpinBox()
        self.a_surf_spinbox.setDecimals(4) # Precision for nm
        self.a_surf_spinbox.setRange(0.0001, 10.0) # Reasonable range in nm
        self.a_surf_spinbox.setSingleStep(0.001)
        self.a_surf_spinbox.setValue(0.300) # Default example value
        form_layout.addRow("Surface Constant 'a_surf' (nm):", self.a_surf_spinbox)

        layout.addLayout(form_layout)

        self.button_box = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        self.button_box.accepted.connect(self.accept_input)
        self.button_box.rejected.connect(self.reject)
        layout.addWidget(self.button_box)

    def accept_input(self):
        """Validate input and store the definition before accepting."""
        name = self.name_edit.text().strip()
        lattice_type = self.type_combo.currentText()
        a_surf = self.a_surf_spinbox.value()

        if not name:
            QMessageBox.warning(self, "Input Error", "Lattice name cannot be empty.")
            return
        if a_surf <= 0:
            QMessageBox.warning(self, "Input Error", "'a_surf' must be positive.")
            return

        self._lattice_definition = {
            "name": name,
            "type": lattice_type,
            "a_surf": a_surf,
            "source": "User Defined"
        }
        logger.info(f"Custom lattice defined: {self._lattice_definition}")
        super().accept() # Call the original accept to close with QDialog.Accepted

    def get_lattice_definition(self) -> Optional[Dict[str, Any]]:
        """Returns the defined lattice dictionary, or None if not accepted."""
        return self._lattice_definition