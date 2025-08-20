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
    """Konwertuje QImage (grayscale) na tablicę NumPy."""
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

    def _calculate_final_transformed_data(self) -> Optional[np.ndarray]:
        """
        Oblicza finalną tablicę NumPy z transformacją, używając poprawnej,
        scentrowanej transformacji afinicznej i spójnego systemu współrzędnych.
        """
        if not self.cb_apply_stretch.isChecked() and not self.cb_apply_rotation.isChecked():
            logger.info("No transformation selected. Returning a copy of the original data.")
            return self.input_data.copy()

        try:
            # Krok 1: Zbuduj macierz transformacji F_eff dla współrzędnych (x, y)
            F_eff_xy = np.eye(2)
            if self.cb_apply_stretch.isChecked(): F_eff_xy = self.U_matrix_apply @ F_eff_xy
            if self.cb_apply_rotation.isChecked(): F_eff_xy = self.R_matrix_apply @ F_eff_xy
            
            # --- NOWY, KLUCZOWY KROK ---
            # Konwertujemy macierz z systemu (x, y) na system (rząd, kolumna) używany przez Scipy
            F_eff_rc = np.array([[F_eff_xy[1,1], F_eff_xy[1,0]],
                                [F_eff_xy[0,1], F_eff_xy[0,0]]])
            # --- KONIEC NOWEGO KROKU ---

            # Krok 2: Oblicz wymiary wyjściowe używając już poprawnej macierzy
            h, w = self.input_data.shape
            # Nasze narożniki są w formacie (rząd, kolumna)
            corners_rc = np.array([[0, 0], [0, w], [h, w], [h, 0]]) - np.array([h/2, w/2])
            transformed_corners_rc = corners_rc @ F_eff_rc.T
            
            min_coords = transformed_corners_rc.min(axis=0)
            max_coords = transformed_corners_rc.max(axis=0)
            new_h, new_w = (max_coords - min_coords)
            output_shape = (int(np.ceil(new_h)), int(np.ceil(new_w)))
            
            # Krok 3: Oblicz macierz odwrotną i poprawne przesunięcie (offset)
            c_in_rc = np.array([h/2, w/2])
            c_out_rc = np.array(output_shape) / 2
            F_eff_rc_inv = np.linalg.inv(F_eff_rc)
            offset = c_in_rc - np.dot(F_eff_rc_inv, c_out_rc)
            
            # Krok 4: Wykonaj transformację
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

    # def _calculate_final_transformed_data(self) -> Optional[np.ndarray]:
    #     """
    #     Oblicza finalną tablicę NumPy z transformacją, używając poprawnej,
    #     scentrowanej transformacji afinicznej.
    #     """
    #     if not self.cb_apply_stretch.isChecked() and not self.cb_apply_rotation.isChecked():
    #         logger.info("No transformation selected. Returning a copy of the original data.")
    #         return self.input_data.copy()

    #     try:
    #         # Krok 1: Zbuduj efektywną macierz transformacji (Idealny -> Zniekształcony)
    #         F_eff = np.eye(2)
    #         if self.cb_apply_stretch.isChecked(): F_eff = self.U_matrix_apply @ F_eff
    #         if self.cb_apply_rotation.isChecked(): F_eff = self.R_matrix_apply @ F_eff
            
    #         # Krok 2: Oblicz wymiary nowego, opasującego prostokąta (output_shape)
    #         h, w = self.input_data.shape
    #         corners = np.array([[0, 0], [w, 0], [w, h], [0, h]]) - np.array([w/2, h/2])
    #         transformed_corners = corners @ F_eff.T
    #         min_coords = transformed_corners.min(axis=0)
    #         max_coords = transformed_corners.max(axis=0)
    #         new_w, new_h = (max_coords - min_coords)
    #         output_shape = (int(np.ceil(new_h)), int(np.ceil(new_w)))
            
    #         # Krok 3: Oblicz macierz odwrotną i poprawne przesunięcie (offset)
    #         # Ta nowa, poprawna formuła centruje transformację.
    #         c_in = np.array([h/2, w/2])
    #         c_out = np.array(output_shape) / 2
    #         F_eff_inv = np.linalg.inv(F_eff)

    #         # Przesunięcie jest obliczane tak, aby środek oryginalnego obrazu
    #         # po transformacji znalazł się w środku nowego obrazu.
    #         offset = c_in - np.dot(F_eff_inv, c_out)
            
    #         # Krok 4: Wykonaj transformację z poprawnymi parametrami
    #         transformed_image = affine_transform(
    #             self.input_data,
    #             matrix=F_eff_inv,
    #             offset=offset,
    #             output_shape=output_shape,
    #             order=3, # Interpolacja sześcienna dla wysokiej jakości
    #             cval=np.min(self.input_data) # Wypełnij tło najciemniejszą wartością z obrazu
    #         )
    #         return transformed_image

    #     except Exception as e:
    #         logger.error(f"Final transform data calculation failed: {e}")
    #         return None

    # def _calculate_final_transformed_data(self) -> Optional[np.ndarray]:
    #     """Oblicza finalną tablicę NumPy z transformacją."""
        
    #     # --- NOWY KOD: Obsługa przypadku braku transformacji ---
    #     if not self.cb_apply_stretch.isChecked() and not self.cb_apply_rotation.isChecked():
    #         logger.info("No transformation selected. Returning a copy of the original data.")
    #         return self.input_data.copy()
    #     # --- KONIEC NOWEGO KODU ---

    #     # Istniejąca logika, która jest teraz wywoływana tylko, gdy jest to potrzebne
    #     F_eff = np.eye(2)
    #     if self.cb_apply_stretch.isChecked(): F_eff = self.U_matrix_apply @ F_eff
    #     if self.cb_apply_rotation.isChecked(): F_eff = self.R_matrix_apply @ F_eff
        
    #     h, w = self.input_data.shape
    #     corners = np.array([[0, 0], [w, 0], [w, h], [0, h]]) - np.array([w/2, h/2])
    #     transformed_corners = corners @ F_eff.T
    #     min_coords = transformed_corners.min(axis=0); max_coords = transformed_corners.max(axis=0)
    #     new_w, new_h = (max_coords - min_coords)
    #     output_shape = (int(np.ceil(new_h)), int(np.ceil(new_w)))
    #     offset_correction = np.array([w/2, h/2]) - (F_eff @ np.array([w/2, h/2]))
    #     transform_offset = offset_correction - np.dot(F_eff, min_coords)

    #     try:
    #         F_eff_inv = np.linalg.inv(F_eff)
    #         return affine_transform(
    #             self.input_data, matrix=F_eff_inv, offset=np.dot(F_eff_inv, transform_offset),
    #             output_shape=output_shape, order=3
    #         )
    #     except Exception as e:
    #         logger.error(f"Final transform calculation for save failed: {e}")
    #         return None

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
    def _save_as_stp(self):
        """Oblicza finalną transformację, aktualizuje metadane WSxM i zapisuje do pliku .stp."""
        transformed_data = self._calculate_final_transformed_data()
        if transformed_data is None:
            QMessageBox.critical(self, "Błąd Obliczeń", "Nie udało się obliczyć obrazu do zapisu.")
            return

        file_path, _ = QFileDialog.getSaveFileName(self, "Save Transformed STM as STP", "", "WSxM STP Files (*.stp)")
        if not file_path:
            return

        try:
            # 1. Stwórz głęboką kopię oryginalnego, zagnieżdżonego nagłówka
            import copy
            new_header = copy.deepcopy(self.original_node.parameters.get("raw_header", {}))

            # 2. Pobierz oryginalne wymiary
            orig_h, orig_w = self.input_data.shape
            orig_nm_x = self.original_node.parameters.get("size_nm_x", 0.0)
            
            # 3. Pobierz nowe wymiary w pikselach
            new_h, new_w = transformed_data.shape
            
            # 4. Oblicz nowe wymiary fizyczne
            nm_per_px = orig_nm_x / orig_w if orig_w > 0 else 0
            new_nm_x = new_w * nm_per_px
            new_nm_y = new_h * nm_per_px # Zakładamy kwadratowe piksele
            
            # 5. Zaktualizuj pola w odpowiednich sekcjach
            if "General Info" in new_header and isinstance(new_header["General Info"], dict):
                new_header["General Info"]["Number of columns"] = str(new_w)
                new_header["General Info"]["Number of rows"] = str(new_h)
                
                # Dodaj informację o wykonanych procesach
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

            # 6. Wywołaj nową funkcję zapisu
            write_STP_file(
                file_path=file_path,
                data_array=transformed_data,
                header_info=new_header
            )
            QMessageBox.information(self, "Zapisano", f"Obraz został zapisany jako:\n{os.path.basename(file_path)}")

        except Exception as e:
            logger.exception(f"Błąd podczas zapisu do pliku STP: {e}")
            QMessageBox.critical(self, "Błąd Zapisu", f"Nie można było zapisać pliku STP:\n{e}")
    
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