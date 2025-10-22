# lfa/gui/custom_lattice_dialog.py
"""
Dialog for defining a custom substrate lattice.

This module provides a dialog interface for users to define custom lattice parameters
for substrate analysis. It supports hexagonal and square lattice types and allows
users to specify the surface constant (a_surf) in nanometers.
"""
import logging
import math
from typing import Optional, Dict, Any

try:
    from PyQt6.QtWidgets import (
        QDialog, QVBoxLayout, QFormLayout, QLabel, QLineEdit,
        QComboBox, QDialogButtonBox, QDoubleSpinBox, QMessageBox
    )
except ImportError:
    logging.critical("Failed to import necessary PyQt6 modules for CustomLatticeDialog.")
    raise

logger = logging.getLogger(__name__)

class CustomLatticeDialog(QDialog):
    """
    Dialog for user to define custom lattice parameters.
    
    This dialog allows users to:
    - Define a custom name for the lattice
    - Select the lattice type (hexagonal or square)
    - Specify the surface constant (a_surf) in nanometers
    
    The dialog validates user input and returns a dictionary containing
    the lattice definition when accepted.
    
    Attributes:
        _lattice_definition (Optional[Dict[str, Any]]): The defined lattice parameters
        name_edit (QLineEdit): Input field for lattice name
        type_combo (QComboBox): Dropdown for selecting lattice type
        a_surf_spinbox (QDoubleSpinBox): Input for surface constant
    """
    def __init__(self, parent=None):
        """
        Initialize the custom lattice dialog.
        
        Args:
            parent: Parent widget for the dialog
        """
        super().__init__(parent)
        self.setWindowTitle("Define Custom Lattice")
        self.setMinimumWidth(350)

        self._lattice_definition: Optional[Dict[str, Any]] = None

        layout = QVBoxLayout(self)
        form_layout = QFormLayout()

        self.name_edit = QLineEdit("Custom Lattice")
        self.name_label = QLabel("Lattice Name:")
        form_layout.addRow(self.name_label, self.name_edit)

        self.mode_label = QLabel("Definition Mode:")
        self.mode_combo = QComboBox()
        self.mode_combo.addItems([
            "Lattice type with a_surf",
            "Manual vectors"
        ])
        form_layout.addRow(self.mode_label, self.mode_combo)

        self.type_label = QLabel("Lattice Type:")
        self.type_combo = QComboBox()
        self.type_combo.addItems(["hexagonal", "square"])
        form_layout.addRow(self.type_label, self.type_combo)

        self.a_surf_label = QLabel("Surface Constant 'a_surf' (nm):")
        self.a_surf_spinbox = QDoubleSpinBox()
        self.a_surf_spinbox.setDecimals(4)
        self.a_surf_spinbox.setRange(0.0001, 10.0)
        self.a_surf_spinbox.setSingleStep(0.001)
        self.a_surf_spinbox.setValue(0.300)
        form_layout.addRow(self.a_surf_label, self.a_surf_spinbox)

        self.vector_a_label = QLabel("Vector |a| (nm):")
        self.vector_a_spinbox = QDoubleSpinBox()
        self.vector_a_spinbox.setDecimals(4)
        self.vector_a_spinbox.setRange(0.0001, 20.0)
        self.vector_a_spinbox.setSingleStep(0.001)
        self.vector_a_spinbox.setValue(0.300)
        form_layout.addRow(self.vector_a_label, self.vector_a_spinbox)

        self.vector_b_label = QLabel("Vector |b| (nm):")
        self.vector_b_spinbox = QDoubleSpinBox()
        self.vector_b_spinbox.setDecimals(4)
        self.vector_b_spinbox.setRange(0.0001, 20.0)
        self.vector_b_spinbox.setSingleStep(0.001)
        self.vector_b_spinbox.setValue(0.300)
        form_layout.addRow(self.vector_b_label, self.vector_b_spinbox)

        self.vector_gamma_label = QLabel("Angle γ (deg):")
        self.vector_gamma_spinbox = QDoubleSpinBox()
        self.vector_gamma_spinbox.setDecimals(2)
        self.vector_gamma_spinbox.setRange(1.0, 179.0)
        self.vector_gamma_spinbox.setSingleStep(0.1)
        self.vector_gamma_spinbox.setValue(60.0)
        form_layout.addRow(self.vector_gamma_label, self.vector_gamma_spinbox)

        self.mode_combo.currentIndexChanged.connect(self._update_mode_widgets)
        self._update_mode_widgets()

        layout.addLayout(form_layout)

        self.button_box = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        self.button_box.accepted.connect(self.accept_input)
        self.button_box.rejected.connect(self.reject)
        layout.addWidget(self.button_box)

    def _update_mode_widgets(self):
        manual_mode = self.mode_combo.currentIndex() == 1

        for widget in (self.type_label, self.type_combo, self.a_surf_label, self.a_surf_spinbox):
            widget.setVisible(not manual_mode)
        for widget in (self.vector_a_label, self.vector_a_spinbox,
                       self.vector_b_label, self.vector_b_spinbox,
                       self.vector_gamma_label, self.vector_gamma_spinbox):
            widget.setVisible(manual_mode)

    def accept_input(self):
        """
        Validate input and store the definition before accepting.
        
        This method:
        1. Validates the lattice name is not empty
        2. Validates the surface constant is positive
        3. Creates a dictionary with the lattice definition
        4. Accepts the dialog if validation passes
        
        The lattice definition includes:
        - name: User-defined name for the lattice
        - type: Selected lattice type (hexagonal or square)
        - a_surf: Surface constant in nanometers
        - source: Indicates this is a user-defined lattice
        """
        name = self.name_edit.text().strip()
        if not name:
            QMessageBox.warning(self, "Input Error", "Lattice name cannot be empty.")
            return

        manual_mode = self.mode_combo.currentIndex() == 1

        if manual_mode:
            a_length = self.vector_a_spinbox.value()
            b_length = self.vector_b_spinbox.value()
            gamma_deg = self.vector_gamma_spinbox.value()

            if a_length <= 0 or b_length <= 0:
                QMessageBox.warning(self, "Input Error", "Vector lengths must be positive.")
                return

            gamma_rad = math.radians(gamma_deg)
            if abs(math.sin(gamma_rad)) < 1e-6:
                QMessageBox.warning(self, "Input Error", "Angle must not produce collinear vectors.")
                return

            a_vec = (float(a_length), 0.0)
            b_vec = (
                float(b_length * math.cos(gamma_rad)),
                float(b_length * math.sin(gamma_rad))
            )

            self._lattice_definition = {
                "name": name,
                "type": "custom",
                "a_length_nm": float(a_length),
                "b_length_nm": float(b_length),
                "gamma_deg": float(gamma_deg),
                "a_vec_nm": a_vec,
                "b_vec_nm": b_vec,
                "preferred_point_count": 6,
                "source": "User Defined"
            }
        else:
            lattice_type = self.type_combo.currentText()
            a_surf = self.a_surf_spinbox.value()
            if a_surf <= 0:
                QMessageBox.warning(self, "Input Error", "'a_surf' must be positive.")
                return
            self._lattice_definition = {
                "name": name,
                "type": lattice_type,
                "a_surf": float(a_surf),
                "source": "User Defined"
            }

        logger.info(f"Custom lattice defined: {self._lattice_definition}")
        super().accept()

    def get_lattice_definition(self) -> Optional[Dict[str, Any]]:
        """
        Returns the defined lattice dictionary, or None if not accepted.
        
        Returns:
            Optional[Dict[str, Any]]: Dictionary containing:
                - name: Lattice name
                - type: Lattice type (hexagonal or square)
                - a_surf: Surface constant in nanometers
                - source: "User Defined"
            Returns None if the dialog was not accepted
        """
        return self._lattice_definition
