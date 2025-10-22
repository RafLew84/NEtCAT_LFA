# lfa/gui/dialogs/stm_transform_dialog.py
import logging
import numpy as np
import os
from scipy.linalg import polar
from typing import Optional, Dict, Any
from scipy.ndimage import affine_transform

from PyQt6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QLabel, QDialogButtonBox, QWidget,
    QGroupBox, QCheckBox, QSplitter, QPushButton, QMessageBox, QFileDialog
)
from PyQt6.QtCore import pyqtSlot, Qt
from PyQt6.QtGui import QTransform, QImage
import pyqtgraph as pg
import pyqtgraph.exporters
from PIL import Image

from ...core.history import HistoryNode
from ...io.write_stp import write_STP_file

logger = logging.getLogger(__name__)

def qimage_to_numpy(qimage: QImage) -> np.ndarray:
    """Convert a grayscale QImage into a NumPy array."""
    qimage = qimage.convertToFormat(QImage.Format.Format_Grayscale8)
    ptr = qimage.constBits()
    ptr.setsize(qimage.sizeInBytes())
    return np.array(ptr).reshape(qimage.height(), qimage.width())

class StmTransformDialog(QDialog):
    def __init__(self, input_data: np.ndarray, original_node: HistoryNode, substrate_transform_F: np.ndarray, parent=None):
        super().__init__(parent)
        self.setWindowTitle("STM Transform Comparison & Export")
        self.setMinimumSize(1200, 600)

        self.input_data = input_data
        self.substrate_F_m2i = substrate_transform_F

        self.R_matrix_apply = np.eye(2)
        self.U_matrix_apply = np.eye(2)
        self.rotation_angle_deg_display = 0.0
        self.stretch_factors_display = (1.0, 1.0)
        self.original_node = original_node
        
        self._decompose_transform()
        self._init_ui()
        self._connect_signals()
        self._update_info_labels()
        self._update_preview()

    def _init_ui(self):
        main_layout = QVBoxLayout(self)
        
        display_splitter = QSplitter(Qt.Orientation.Horizontal)
        
        original_widget = pg.GraphicsLayoutWidget()
        self.plot_original = original_widget.addPlot(title="Original STM")
        self.plot_original.getViewBox().invertY(True)
        self.img_original = pg.ImageItem()
        self.img_original.setImage(self.input_data.T)
        self.plot_original.addItem(self.img_original)
        self.plot_original.setAspectLocked(True)

        preview_widget = pg.GraphicsLayoutWidget()
        self.plot_preview = preview_widget.addPlot(title="Transformed Preview")
        self.plot_preview.getViewBox().invertY(True)
        self.img_preview = pg.ImageItem()
        self.img_preview.setImage(self.input_data.T)
        self.plot_preview.addItem(self.img_preview)
        self.plot_preview.setAspectLocked(True)

        self.plot_preview.setXLink(self.plot_original)
        self.plot_preview.setYLink(self.plot_original)

        display_splitter.addWidget(original_widget)
        display_splitter.addWidget(preview_widget)
        main_layout.addWidget(display_splitter)

        bottom_widget = QWidget()
        bottom_layout = QHBoxLayout(bottom_widget)
        
        controls_panel = QWidget()
        controls_layout = QVBoxLayout(controls_panel)
        self._create_controls(controls_layout)

        bottom_layout.addWidget(controls_panel)
        bottom_layout.addStretch()
        main_layout.addWidget(bottom_widget)

    def _calculate_final_transformed_data(self) -> Optional[np.ndarray]:
        """
        Compute the final NumPy array after applying the transform, using the corrected,
        centered affine transformation and consistent coordinate system.
        """
        if not self.cb_apply_stretch.isChecked() and not self.cb_apply_rotation.isChecked():
            logger.info("No transformation selected. Returning a copy of the original data.")
            return self.input_data.copy()

        try:
            F_eff_xy = np.eye(2)
            if self.cb_apply_stretch.isChecked(): F_eff_xy = self.U_matrix_apply @ F_eff_xy
            if self.cb_apply_rotation.isChecked(): F_eff_xy = self.R_matrix_apply @ F_eff_xy
            
            F_eff_rc = np.array([[F_eff_xy[1,1], F_eff_xy[1,0]],
                                [F_eff_xy[0,1], F_eff_xy[0,0]]])

            h, w = self.input_data.shape
            corners_rc = np.array([[0, 0], [0, w], [h, w], [h, 0]]) - np.array([h/2, w/2])
            transformed_corners_rc = corners_rc @ F_eff_rc.T
            
            min_coords = transformed_corners_rc.min(axis=0)
            max_coords = transformed_corners_rc.max(axis=0)
            new_h, new_w = (max_coords - min_coords)
            output_shape = (int(np.ceil(new_h)), int(np.ceil(new_w)))
            
            c_in_rc = np.array([h/2, w/2])
            c_out_rc = np.array(output_shape) / 2
            F_eff_rc_inv = np.linalg.inv(F_eff_rc)
            offset = c_in_rc - np.dot(F_eff_rc_inv, c_out_rc)
            
            transformed_image = affine_transform(
                self.input_data,
                matrix=F_eff_rc_inv,
                offset=offset,
                output_shape=output_shape,
                order=3,
                cval=np.min(self.input_data)
            )
            return transformed_image

        except Exception as e:
            logger.error(f"Final transform data calculation failed: {e}")
            return None

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
        
        self.save_button = QPushButton("Save Comparison (PNG)...")
        self.save_stp_button = QPushButton("Save as STP...")
        self.close_button = QPushButton("Close")
        
        button_layout = QHBoxLayout()
        button_layout.addWidget(self.save_stp_button)
        button_layout.addWidget(self.save_button)
        button_layout.addStretch()
        button_layout.addWidget(self.close_button)

        layout.addWidget(params_group)
        layout.addLayout(button_layout)
        layout.addStretch()

    def _connect_signals(self):
        self.cb_apply_rotation.stateChanged.connect(self._update_preview)
        self.save_stp_button.clicked.connect(self._save_as_stp)
        self.cb_apply_stretch.stateChanged.connect(self._update_preview)
        self.save_button.clicked.connect(self._save_comparison)
        self.close_button.clicked.connect(self.accept)

    def _decompose_transform(self):
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
        self.info_rot_label.setText(f"Rotation: {self.rotation_angle_deg_display:.2f}°")
        self.info_stretch_label.setText(f"Stretches: ({self.stretch_factors_display[0]:.3f}, {self.stretch_factors_display[1]:.3f})")

    @pyqtSlot()
    def _save_as_stp(self):
        """Compute the final transform, update WSxM metadata, and write an .stp file."""
        transformed_data = self._calculate_final_transformed_data()
        if transformed_data is None:
            QMessageBox.critical(self, "Computation Error", "Could not compute the image for saving.")
            return

        file_path, _ = QFileDialog.getSaveFileName(self, "Save Transformed STM as STP", "", "WSxM STP Files (*.stp)")
        if not file_path:
            return

        try:
            import copy
            new_header = copy.deepcopy(self.original_node.parameters.get("raw_header", {}))

            _, orig_w = self.input_data.shape
            orig_nm_x = self.original_node.parameters.get("size_nm_x", 0.0)
            
            new_h, new_w = transformed_data.shape
            
            nm_per_px = orig_nm_x / orig_w if orig_w > 0 else 0
            new_nm_x = new_w * nm_per_px
            new_nm_y = new_h * nm_per_px # Assume square pixels
            
            if "General Info" in new_header and isinstance(new_header["General Info"], dict):
                new_header["General Info"]["Number of columns"] = str(new_w)
                new_header["General Info"]["Number of rows"] = str(new_h)
                
                applied_processes = []
                if self.cb_apply_stretch.isChecked(): applied_processes.append("stretch")
                if self.cb_apply_rotation.isChecked(): applied_processes.append("rotate")
                if applied_processes:
                    process_str = ", ".join(applied_processes)
                    existing = new_header["General Info"].get("Image processes", "")
                    new_header["General Info"]["Image processes"] = f"{existing}, {process_str}".strip(", ")
            
            if "Control" in new_header and isinstance(new_header["Control"], dict):
                new_header["Control"]["X Amplitude"] = f"{new_nm_x:.6f} nm"
                new_header["Control"]["Y Amplitude"] = f"{new_nm_y:.6f} nm"

            write_STP_file(
                file_path=file_path,
                data_array=transformed_data,
                header_info=new_header
            )
            QMessageBox.information(self, "Saved", f"Image saved as:\n{os.path.basename(file_path)}")

        except Exception as e:
            logger.exception(f"Error while writing STP file: {e}")
            QMessageBox.critical(self, "Save Error", f"Could not save the STP file:\n{e}")
    
    @pyqtSlot()
    def _update_preview(self):
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
        self.plot_original.autoRange() # Align both views
        self.plot_preview.autoRange()

    @pyqtSlot()
    def _save_comparison(self):
        """Save the original and transformed images to disk, each with axis overlays."""
        file_path, _ = QFileDialog.getSaveFileName(self, "Save Comparison Images", "", "PNG Image (*.png)")
        if not file_path:
            return

        base, ext = os.path.splitext(file_path)
        original_path = f"{base}_original.png"
        transformed_path = f"{base}_transformed.png"

        try:
            exporter_original = pg.exporters.ImageExporter(self.plot_original.scene())
            exporter_original.export(original_path)
            logger.info(f"Oryginalny obraz zapisany w: {original_path}")

            exporter_transformed = pg.exporters.ImageExporter(self.plot_preview.scene())
            exporter_transformed.export(transformed_path)
            logger.info(f"Przetransformowany obraz zapisany w: {transformed_path}")
            
            QMessageBox.information(
                self,
                "Saved",
                f"Images saved successfully:\n- {os.path.basename(original_path)}\n- {os.path.basename(transformed_path)}"
            )

        except Exception as e:
            logger.exception(f"Error while saving comparison images: {e}")
            QMessageBox.critical(self, "Save Error", f"Could not save the images:\n{e}")
