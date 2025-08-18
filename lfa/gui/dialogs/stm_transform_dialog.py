# lfa/gui/dialogs/stm_transform_dialog.py
import logging
import numpy as np
from scipy.ndimage import affine_transform
from scipy.linalg import polar
from typing import Optional, Dict, Any

from PyQt6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QLabel, QDialogButtonBox, 
    QGroupBox, QCheckBox, QWidget
)
from PyQt6.QtCore import pyqtSlot
from PyQt6.QtGui import QTransform
import pyqtgraph as pg

logger = logging.getLogger(__name__)

class StmTransformDialog(QDialog):
    def __init__(self, input_data: np.ndarray, substrate_transform_F: np.ndarray, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Apply Substrate Transform to STM")
        self.setMinimumSize(900, 500)

        self.input_data = input_data
        self.substrate_F_m2i = substrate_transform_F
        self.transformed_data: Optional[np.ndarray] = None

        # Macierze do APLIKACJI transformacji (Idealny -> Zniekształcony)
        self.R_matrix_apply = np.eye(2)
        self.U_matrix_apply = np.eye(2)
        
        # Parametry do WYŚWIETLANIA (Zniekształcony -> Idealny)
        self.rotation_angle_deg_display = 0.0
        self.stretch_factors_display = (1.0, 1.0)
        
        self._decompose_transform()

        main_layout = QVBoxLayout(self)
        top_layout = QHBoxLayout()
        
        controls_panel = QWidget()
        controls_panel.setMaximumWidth(300)
        controls_layout = QVBoxLayout(controls_panel)
        self._create_controls(controls_layout)
        
        preview_widget = pg.GraphicsLayoutWidget()
        self.plot_preview = preview_widget.addPlot(title="Preview")
        self.plot_preview.getViewBox().invertY(True)
        self.img_preview = pg.ImageItem()
        self.img_preview.setImage(self.input_data.T)
        self.plot_preview.addItem(self.img_preview)
        self.plot_preview.setAspectLocked(True)

        top_layout.addWidget(controls_panel)
        top_layout.addWidget(preview_widget)
        main_layout.addLayout(top_layout)

        self.button_box = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        self.button_box.button(QDialogButtonBox.StandardButton.Ok).setText("Apply Transform")
        main_layout.addWidget(self.button_box)

        self._connect_signals()
        self._update_info_labels()
        self._update_preview()

    def _decompose_transform(self):
        """
        Analizuje macierz F_m2i do wyświetlenia, a jej odwrotność (F_i2m)
        przygotowuje do zastosowania na obrazie.
        """
        if self.substrate_F_m2i is None: return
        try:
            # --- CZĘŚĆ 1: Analiza F_m2i (Zniekształcony -> Idealny) do WYŚWIETLENIA ---
            R_display, U_display = polar(self.substrate_F_m2i)
            self.rotation_angle_deg_display = np.degrees(np.arctan2(R_display[1, 0], R_display[0, 0]))
            eigenvalues_display, _ = np.linalg.eig(U_display)
            self.stretch_factors_display = (eigenvalues_display[0], eigenvalues_display[1])

            # --- CZĘŚĆ 2: Obliczenie odwrotności F_i2m (Idealny -> Zniekształcony) do ZASTOSOWANIA ---
            F_i2m = np.linalg.inv(self.substrate_F_m2i)
            self.R_matrix_apply, self.U_matrix_apply = polar(F_i2m)
            
        except Exception as e:
            logger.error(f"Failed to decompose transform matrix F: {e}")

    def _create_controls(self, layout: QVBoxLayout):
        params_group = QGroupBox("Apply Transform Components")
        group_layout = QVBoxLayout(params_group)
        self.cb_apply_rotation = QCheckBox("Apply Rotation")
        self.cb_apply_rotation.setChecked(True)
        group_layout.addWidget(self.cb_apply_rotation)
        self.cb_apply_stretch = QCheckBox("Apply Stretch")
        self.cb_apply_stretch.setChecked(True)
        group_layout.addWidget(self.cb_apply_stretch)
        self.info_rot_label = QLabel()
        self.info_stretch_label = QLabel()
        group_layout.addWidget(QLabel("--- Detected Parameters (Correction) ---"))
        group_layout.addWidget(self.info_rot_label)
        group_layout.addWidget(self.info_stretch_label)
        layout.addWidget(params_group)
        layout.addStretch()

    def _update_info_labels(self):
        """Wyświetla poprawne, nieodwrócone wartości."""
        self.info_rot_label.setText(f"Rotation: {self.rotation_angle_deg_display:.2f}°")
        self.info_stretch_label.setText(f"Stretches: ({self.stretch_factors_display[0]:.3f}, {self.stretch_factors_display[1]:.3f})")

    def _connect_signals(self):
        self.cb_apply_rotation.stateChanged.connect(self._update_preview)
        self.cb_apply_stretch.stateChanged.connect(self._update_preview)
        self.button_box.accepted.connect(self.accept)
        self.button_box.rejected.connect(self.reject)
    
    @pyqtSlot()
    def _update_preview(self):
        h, w = self.input_data.shape
        
        # Budujemy macierz na podstawie macierzy do APLIKACJI
        F_eff = np.eye(2)
        if self.cb_apply_stretch.isChecked():
            F_eff = self.U_matrix_apply @ F_eff
        if self.cb_apply_rotation.isChecked():
            F_eff = self.R_matrix_apply @ F_eff

        transform = QTransform()
        transform.translate(w/2, h/2)
        # Tworzymy macierz QTransform z naszej macierzy NumPy F_eff
        # F_eff[0,0], F_eff[1,0], F_eff[0,1], F_eff[1,1] - poprawna kolejność dla QTransform
        q_matrix_part = QTransform(F_eff[0,0], F_eff[1,0], 0, F_eff[0,1], F_eff[1,1], 0, 0, 0, 1)
        transform = q_matrix_part * transform
        transform.translate(-w/2, -h/2)
        
        self.img_preview.setTransform(transform)
        self.plot_preview.autoRange()

    def accept(self):
        F_eff = np.eye(2)
        if self.cb_apply_stretch.isChecked(): F_eff = self.U_matrix_apply @ F_eff
        if self.cb_apply_rotation.isChecked(): F_eff = self.R_matrix_apply @ F_eff
        
        h, w = self.input_data.shape
        corners = np.array([[0, 0], [w, 0], [w, h], [0, h]]) - np.array([w/2, h/2])
        transformed_corners = corners @ F_eff.T
        min_coords = transformed_corners.min(axis=0); max_coords = transformed_corners.max(axis=0)
        new_w, new_h = (max_coords - min_coords)
        output_shape = (int(np.ceil(new_h)), int(np.ceil(new_w)))
        offset_correction = np.array([w/2, h/2]) - (F_eff @ np.array([w/2, h/2]))
        transform_offset = offset_correction - np.dot(F_eff, min_coords)

        try:
            F_eff_inv = np.linalg.inv(F_eff)
            self.transformed_data = affine_transform(
                self.input_data, matrix=F_eff_inv, offset=np.dot(F_eff_inv, transform_offset),
                output_shape=output_shape, order=3
            )
        except Exception as e:
            logger.error(f"Final transform calculation failed: {e}")
            self.transformed_data = None
        
        super().accept()
        
    def get_parameters(self) -> Dict[str, Any]:
        return {
            'applied_rotation': self.rotation_angle_deg_display if self.cb_apply_rotation.isChecked() else 0.0,
            'applied_stretch': self.stretch_factors_display if self.cb_apply_stretch.isChecked() else (1.0, 1.0),
        }

    def get_transformed_data(self) -> Optional[np.ndarray]:
        return self.transformed_data