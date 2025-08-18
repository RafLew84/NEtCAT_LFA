# lfa/gui/dialogs/stm_transform_dialog.py
import logging
import numpy as np
import os
from scipy.linalg import polar
from typing import Optional, Dict, Any

from PyQt6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QLabel, QDialogButtonBox, QWidget,
    QGroupBox, QCheckBox, QSplitter, QPushButton, QMessageBox, QFileDialog
)
from PyQt6.QtCore import pyqtSlot, Qt
from PyQt6.QtGui import QTransform
import pyqtgraph as pg
import pyqtgraph.exporters
from PIL import Image

logger = logging.getLogger(__name__)

class StmTransformDialog(QDialog):
    def __init__(self, input_data: np.ndarray, substrate_transform_F: np.ndarray, parent=None):
        super().__init__(parent)
        self.setWindowTitle("STM Transform Comparison & Export")
        self.setMinimumSize(1200, 600)

        self.input_data = input_data
        self.substrate_F_m2i = substrate_transform_F

        self.R_matrix_apply = np.eye(2)
        self.U_matrix_apply = np.eye(2)
        self.rotation_angle_deg_display = 0.0
        self.stretch_factors_display = (1.0, 1.0)
        
        self._decompose_transform()
        self._init_ui()
        self._connect_signals()
        self._update_info_labels()
        self._update_preview()

    def _init_ui(self):
        main_layout = QVBoxLayout(self)
        
        # --- Panel porównawczy ---
        display_splitter = QSplitter(Qt.Orientation.Horizontal)
        
        # Panel Oryginału
        original_widget = pg.GraphicsLayoutWidget()
        self.plot_original = original_widget.addPlot(title="Original STM")
        self.plot_original.getViewBox().invertY(True)
        self.img_original = pg.ImageItem()
        self.img_original.setImage(self.input_data.T)
        self.plot_original.addItem(self.img_original)
        self.plot_original.setAspectLocked(True)

        # Panel Podglądu Transformacji
        preview_widget = pg.GraphicsLayoutWidget()
        self.plot_preview = preview_widget.addPlot(title="Transformed Preview")
        self.plot_preview.getViewBox().invertY(True)
        self.img_preview = pg.ImageItem()
        self.img_preview.setImage(self.input_data.T) # Na starcie pokazujemy to samo
        self.plot_preview.addItem(self.img_preview)
        self.plot_preview.setAspectLocked(True)

        # Połączenie widoków dla synchronicznego zoomu i przesuwania
        self.plot_preview.setXLink(self.plot_original)
        self.plot_preview.setYLink(self.plot_original)

        display_splitter.addWidget(original_widget)
        display_splitter.addWidget(preview_widget)
        main_layout.addWidget(display_splitter)

        # --- Panel Kontrolny ---
        bottom_widget = QWidget()
        bottom_layout = QHBoxLayout(bottom_widget)
        
        controls_panel = QWidget()
        controls_layout = QVBoxLayout(controls_panel)
        self._create_controls(controls_layout)

        bottom_layout.addWidget(controls_panel)
        bottom_layout.addStretch()
        main_layout.addWidget(bottom_widget)

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
        
        # Nowy przycisk zapisu i przycisk zamknięcia
        self.save_button = QPushButton("Save Comparison...")
        self.close_button = QPushButton("Close")
        
        button_layout = QHBoxLayout()
        button_layout.addWidget(self.save_button)
        button_layout.addStretch()
        button_layout.addWidget(self.close_button)

        layout.addWidget(params_group)
        layout.addLayout(button_layout)
        layout.addStretch()

    def _connect_signals(self):
        self.cb_apply_rotation.stateChanged.connect(self._update_preview)
        self.cb_apply_stretch.stateChanged.connect(self._update_preview)
        self.save_button.clicked.connect(self._save_comparison)
        self.close_button.clicked.connect(self.accept)

    def _decompose_transform(self):
        # Ta metoda pozostaje bez zmian
        if self.substrate_F_m2i is None: return
        try:
            R_display, U_display = polar(self.substrate_F_m2i)
            self.rotation_angle_deg_display = np.degrees(np.arctan2(R_display[1, 0], R_display[0, 0]))
            eigenvalues_display, _ = np.linalg.eig(U_display)
            self.stretch_factors_display = (eigenvalues_display[0], eigenvalues_display[1])
            F_i2m = np.linalg.inv(self.substrate_F_m2i)
            self.R_matrix_apply, self.U_matrix_apply = polar(F_i2m)
        except Exception as e:
            logger.error(f"Failed to decompose transform matrix F: {e}")

    def _update_info_labels(self):
        # Ta metoda pozostaje bez zmian
        self.info_rot_label.setText(f"Rotation: {self.rotation_angle_deg_display:.2f}°")
        self.info_stretch_label.setText(f"Stretches: ({self.stretch_factors_display[0]:.3f}, {self.stretch_factors_display[1]:.3f})")
    
    @pyqtSlot()
    def _update_preview(self):
        # Ta metoda pozostaje prawie bez zmian, ale na końcu dopasowuje oba widoki
        h, w = self.input_data.shape
        F_eff = np.eye(2)
        if self.cb_apply_stretch.isChecked(): F_eff = self.U_matrix_apply @ F_eff
        if self.cb_apply_rotation.isChecked(): F_eff = self.R_matrix_apply @ F_eff
        
        transform = QTransform()
        transform.translate(w/2, h/2)
        q_matrix_part = QTransform(F_eff[0,0], F_eff[1,0], 0, F_eff[0,1], F_eff[1,1], 0, 0, 0, 1)
        transform = q_matrix_part * transform
        transform.translate(-w/2, -h/2)
        
        self.img_preview.setTransform(transform)
        self.plot_original.autoRange() # Dopasuj oba widoki
        self.plot_preview.autoRange()

    @pyqtSlot()
    def _save_comparison(self):
        """Zapisuje obraz oryginalny i przetransformowany do plików, oba z osiami."""
        file_path, _ = QFileDialog.getSaveFileName(self, "Save Comparison Images", "", "PNG Image (*.png)")
        if not file_path:
            return

        base, ext = os.path.splitext(file_path)
        original_path = f"{base}_original.png"
        transformed_path = f"{base}_transformed.png"

        try:
            # --- POCZĄTEK ZMIANY ---
            # 1. Zapis obrazu oryginalnego (z osiami) za pomocą eksportera
            exporter_original = pg.exporters.ImageExporter(self.plot_original.scene())
            exporter_original.export(original_path)
            logger.info(f"Oryginalny obraz zapisany w: {original_path}")
            # --- KONIEC ZMIANY ---

            # 2. Zapis obrazu przetransformowanego (bez zmian, już działał poprawnie)
            exporter_transformed = pg.exporters.ImageExporter(self.plot_preview.scene())
            exporter_transformed.export(transformed_path)
            logger.info(f"Przetransformowany obraz zapisany w: {transformed_path}")
            
            QMessageBox.information(self, "Zapisano", f"Obrazy zostały pomyślnie zapisane:\n- {os.path.basename(original_path)}\n- {os.path.basename(transformed_path)}")

        except Exception as e:
            logger.exception(f"Błąd podczas zapisu porównania: {e}")
            QMessageBox.critical(self, "Błąd Zapisu", f"Nie można było zapisać obrazów:\n{e}")