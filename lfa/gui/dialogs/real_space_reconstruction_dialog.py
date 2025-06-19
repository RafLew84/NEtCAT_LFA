# lfa/gui/dialogs/real_space_reconstruction_dialog.py
import logging
from typing import Optional, Dict, Any, List, Tuple

import numpy as np
from PyQt6.QtCore import Qt, pyqtSlot
from PyQt6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout,
    QLabel, QWidget, QGroupBox, QMessageBox, QSpinBox,
    QFormLayout, QRadioButton, QSplitter, QDialogButtonBox, QPushButton, QCheckBox, QComboBox
)

try:
    import pyqtgraph as pg
    from pyqtgraph import GraphicsLayoutWidget, ImageItem, ViewBox, RectROI
    PYQTGRAPH_AVAILABLE = True
except ImportError:
    pg = None; GraphicsLayoutWidget = None; ImageItem = None; PYQTGRAPH_AVAILABLE = False
    logging.error("RealSpaceReconstructionDialog: PyQtGraph not found.")

from ...core.history import HistoryNode
from ...analysis.fft_engine import calculate_fft
from ...analysis.peak_fitting import _gaussian_2d, fit_2d_gaussian_in_roi_with_all_data

logger = logging.getLogger(__name__)


class RealSpaceReconstructionDialog(QDialog):
    def __init__(self,
                 magnitude_fft_data: np.ndarray,
                 parent=None):
        super().__init__(parent)
        self.setWindowTitle("Real Space Reconstruction from FFT")
        self.setMinimumSize(1200, 700)

        self.magnitude_fft_data = magnitude_fft_data
        
        self.current_mode = "autocorrelation"
        self.mask_array: Optional[np.ndarray] = None
        self.roi_items: list[RectROI] = []

        self.refinement_roi = RectROI(pos=(0,0), size=(7,7), pen=pg.mkPen('y', width=2), movable=True)
        self.selected_spots_px: List[Tuple[float, float]] = []
        self.spot_markers: Optional[ScatterPlotItem] = None

        self._init_ui()
        self._connect_signals()
        
        self._on_mode_changed()
        if self.original_fft_item:
            self.original_fft_item.setImage(self.magnitude_fft_data.T)

    def _init_ui(self):
        main_layout = QVBoxLayout(self)
        
        controls_group = QGroupBox("Reconstruction Controls")
        controls_layout = QHBoxLayout(controls_group)
        self._create_controls(controls_layout)
        main_layout.addWidget(controls_group)
        
        display_splitter = QSplitter(Qt.Orientation.Horizontal)
        fft_widget = GraphicsLayoutWidget(); self.fft_plot = fft_widget.addPlot(title="Original FFT (Select regions here)"); self.original_fft_item = ImageItem(); self.fft_plot.addItem(self.original_fft_item); self.fft_plot.setAspectLocked(True)
        mask_widget = GraphicsLayoutWidget(); self.mask_plot = mask_widget.addPlot(title="Mask Preview"); self.mask_item = ImageItem(); self.mask_plot.addItem(self.mask_item); self.mask_plot.setAspectLocked(True)
        reco_widget = GraphicsLayoutWidget(); self.reco_plot = reco_widget.addPlot(title="Reconstructed Real Space"); self.reco_item = ImageItem(); self.reco_plot.addItem(self.reco_item); self.reco_plot.setAspectLocked(True)
        
        self.refinement_roi = RectROI(pos=(0,0), size=(7,7), pen=pg.mkPen('y', width=2), movable=True, resizable=False)
        self.fft_plot.addItem(self.refinement_roi)
        self.refinement_roi.setVisible(False) # Domyślnie ukryte

        display_splitter.addWidget(fft_widget); display_splitter.addWidget(mask_widget); display_splitter.addWidget(reco_widget)
        main_layout.addWidget(display_splitter, 1)

        self.button_box = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        main_layout.addWidget(self.button_box)

    def _create_controls(self, layout: QHBoxLayout):
        mode_group = QGroupBox("Mode"); mode_layout = QVBoxLayout(mode_group)
        self.rb_autocorrelation = QRadioButton("Calculate Autocorrelation"); self.rb_autocorrelation.setChecked(True)
        self.rb_roi_mask = QRadioButton("Mask with ROIs")
        self.rb_spot_mask = QRadioButton("Mask with Spots")
        mode_layout.addWidget(self.rb_autocorrelation); mode_layout.addWidget(self.rb_roi_mask); mode_layout.addWidget(self.rb_spot_mask); mode_layout.addStretch()
        layout.addWidget(mode_group)

        # --- ZMIANA: Nowa grupa dla opcji autokorelacji ---
        autocorr_options_group = QGroupBox("Autocorrelation Options")
        autocorr_layout = QFormLayout(autocorr_options_group)
        self.remove_dc_checkbox = QCheckBox("Remove DC Component")
        self.remove_dc_checkbox.setChecked(True)
        self.remove_dc_checkbox.setToolTip("Removes the central (0,0) peak from FFT before calculating autocorrelation to avoid a bright, broad background.")
        self.scaling_combo = QComboBox()
        self.scaling_combo.addItems(["Log Power", "Power Spectrum", "Linear Magnitude", "Square Root"])
        self.scaling_combo.setToolTip("Selects the scaling for the FFT display.\n'Log Power' is best for viewing weak spots.\n'Power Spectrum' represents physical intensity.")
        autocorr_layout.addRow(self.remove_dc_checkbox)
        autocorr_layout.addRow("Display Scale:", self.scaling_combo)
        layout.addWidget(autocorr_options_group)
        self.autocorr_options_group = autocorr_options_group # Zapisz referencję
        # --- KONIEC ZMIANY ---

        action_group = QGroupBox("Actions"); action_layout = QVBoxLayout(action_group)
        self.add_roi_button = QPushButton("Add ROI")
        self.symmetric_roi_checkbox = QCheckBox("Add Symmetric ROI")
        self.add_spot_button = QPushButton("Select Spot")

        self.spot_mask_size_spinbox = QSpinBox()
        self.spot_mask_size_spinbox.setMinimum(1) # Minimalny rozmiar (sigma)
        self.spot_mask_size_spinbox.setMaximum(20)
        self.spot_mask_size_spinbox.setValue(2) # Domyślna wartość (sigma=2.0)
        self.spot_mask_size_spinbox.setToolTip("Sets the size (sigma) of the Gaussian mask for each spot.")

        self.clear_mask_button = QPushButton("Clear Mask")
        self.reconstruct_button = QPushButton("Reconstruct / Calculate")
        action_layout.addWidget(self.add_roi_button); action_layout.addWidget(self.symmetric_roi_checkbox)
        action_layout.addWidget(self.add_spot_button); action_layout.addWidget(self.clear_mask_button)
        action_layout.addWidget(self.spot_mask_size_spinbox)
        action_layout.addWidget(self.reconstruct_button); action_layout.addStretch()
        layout.addWidget(action_group)
        layout.addStretch()

    def _connect_signals(self):
        self.button_box.accepted.connect(self.accept); self.button_box.rejected.connect(self.reject)

        if self.fft_plot and self.fft_plot.scene():
            self.fft_plot.scene().sigMouseClicked.connect(self._handle_fft_image_click)

        self.rb_autocorrelation.toggled.connect(self._on_mode_changed)
        self.rb_roi_mask.toggled.connect(self._on_mode_changed)
        self.rb_spot_mask.toggled.connect(self._on_mode_changed)
        self.reconstruct_button.clicked.connect(self._on_reconstruct_clicked)
        self.add_roi_button.clicked.connect(self._on_add_roi_clicked)
        self.clear_mask_button.clicked.connect(self._on_clear_mask_clicked)
        self.symmetric_roi_checkbox.stateChanged.connect(self._update_mask_from_rois)
        self.add_spot_button.clicked.connect(self._on_add_spot_clicked)
        self.spot_mask_size_spinbox.valueChanged.connect(self._update_mask_from_spots)

    @pyqtSlot(object)
    def _handle_fft_image_click(self, event):
        """Umieszcza odpowiednie ROI na obrazie w zależności od trybu."""
        if event.button() != Qt.MouseButton.LeftButton: return

        # Logika jest aktywna tylko w trybie "Mask with Spots"
        if self.current_mode == 'spot_mask':
            pos_viewbox = self.fft_plot.vb.mapSceneToView(event.scenePos())
            # Sprawdź, czy kliknięcie jest w granicach obrazu
            if self.fft_plot.vb.sceneBoundingRect().contains(pos_viewbox):
                roi_size = self.refinement_roi.size()
                # Wycentruj ROI na pozycji kliknięcia
                roi_pos = (pos_viewbox.x() - roi_size.x() / 2, pos_viewbox.y() - roi_size.y() / 2)
                self.refinement_roi.setPos(roi_pos)
                self.refinement_roi.setVisible(True)
                event.accept()
        else:
            event.ignore() # Ignoruj kliknięcia w innych trybach

    @pyqtSlot()
    def _on_mode_changed(self):
        is_autocorr = self.rb_autocorrelation.isChecked()
        is_roi_mode = self.rb_roi_mask.isChecked()
        is_spot_mode = self.rb_spot_mask.isChecked()

        self.autocorr_options_group.setVisible(is_autocorr)

        self.add_roi_button.setEnabled(is_roi_mode)
        self.symmetric_roi_checkbox.setEnabled(is_roi_mode)
        self.add_spot_button.setEnabled(is_spot_mode)
        self.spot_mask_size_spinbox.setVisible(is_spot_mode)
        # self.spot_mask_size_spinbox.parent().setVisible(is_spot_mode)
        self.clear_mask_button.setEnabled(is_roi_mode or is_spot_mode)
        self.reconstruct_button.setText("Calculate Autocorrelation" if is_autocorr else "Reconstruct from Mask")
        
        self.current_mode = "autocorrelation" if is_autocorr else ("roi_mask" if is_roi_mode else "spot_mask")

        self._on_clear_mask_clicked() # Wyczyść maskę i wynik przy zmianie trybu

    @pyqtSlot()
    def _on_reconstruct_clicked(self):
        logger.info(f"'{self.reconstruct_button.text()}' button clicked.")
        if self.current_mode == "autocorrelation": self._perform_autocorrelation()
        elif self.current_mode in ["roi_mask", "spot_mask"]: self._perform_reconstruction_from_mask()
        else: QMessageBox.information(self, "Not Implemented", f"Mode '{self.current_mode}' is not yet implemented.")

    @pyqtSlot()
    def _on_add_roi_clicked(self):
        """Dodaje nowy, interaktywny ROI do obrazu FFT."""
        h, w = self.magnitude_fft_data.shape
        roi_size = (w // 8, h // 8)
        roi_pos = (w // 2 - roi_size[0] // 2, h // 2 - roi_size[1] // 2)
        
        new_roi = RectROI(pos=roi_pos, size=roi_size, pen=pg.mkPen('y', width=2), movable=True, resizable=True)
        self.roi_items.append(new_roi)
        self.fft_plot.addItem(new_roi)
        new_roi.sigRegionChanged.connect(self._update_mask_from_rois)
        self._update_mask_from_rois() # Od razu zaktualizuj maskę

    @pyqtSlot()
    def _on_add_spot_clicked(self):
        """Uściśla i dodaje nowy pik do listy na podstawie `selection_roi_spot`."""
        if not self.refinement_roi.isVisible():
            QMessageBox.warning(self, "No ROI", "Please click on the FFT image to place a refinement ROI first.")
            return

        roi_state=self.refinement_roi.getState()
        x0r,y0r=round(roi_state['pos'].x()),round(roi_state['pos'].y())
        w,h=round(roi_state['size'].x()),round(roi_state['size'].y());
        center_yx = (y0r + h//2, x0r + w//2)
        
        if fit_2d_gaussian_in_roi_with_all_data:
            fit_res = fit_2d_gaussian_in_roi_with_all_data(self.magnitude_fft_data, center_yx, w//2)
            if fit_res:
                _, (refined_ky, refined_kx), _ = fit_res
                refined_pos = (float(refined_kx), float(refined_ky))
                self.selected_spots_px.append(refined_pos)
                logger.info(f"Added refined spot at: {refined_pos}")
                self._update_mask_from_spots()
                self._redraw_spot_markers()
            else:
                QMessageBox.warning(self, "Fit Failed", "Could not refine spot position with Gaussian fit.")
        else:
            # Fallback, jeśli dopasowanie Gaussa nie jest dostępne
            self.selected_spots_px.append((center_yx[1], center_yx[0]))
            self._update_mask_from_spots(); self._redraw_spot_markers()

    def _redraw_spot_markers(self):
        """Rysuje markery dla wybranych pików na obrazie FFT."""
        if self.spot_markers:
            self.fft_plot.removeItem(self.spot_markers)
            self.spot_markers = None
            
        if self.selected_spots_px:
            self.spot_markers = pg.ScatterPlotItem(
                pos=np.array(self.selected_spots_px), symbol='+', size=15,
                pen=pg.mkPen('r', width=2)
            )
            self.fft_plot.addItem(self.spot_markers)

    def _update_mask_from_spots(self):
        """Tworzy i wyświetla maskę z profili Gaussa w miejscach wybranych pików."""
        if self.magnitude_fft_data is None: return
        h, w = self.magnitude_fft_data.shape
        mask = np.zeros((h, w), dtype=np.float32)
        
        # Przygotuj siatkę współrzędnych tylko raz
        Y, X = np.mgrid[0:h, 0:w]

        sigma = float(self.spot_mask_size_spinbox.value() / 10)

        xy_tuple = (Y.flatten(), X.flatten())
        
        for (kx, ky) in self.selected_spots_px:
            # "Maluje" znormalizowanego Gaussa dla każdego piku
            # Sigma określa "rozmiar" piku na masce
            # sigma = 2.0 
            amplitude = 1.0
            popt = [amplitude, ky, kx, sigma, sigma, 0, 0]
            gauss_flat = _gaussian_2d(xy_tuple, *popt)
            mask += gauss_flat.reshape(h, w)
            
        if mask.max() > 0:
            mask /= mask.max() # Normalizuj maskę do zakresu [0, 1]
            
        self.mask_array = mask
        self.mask_item.setImage(self.mask_array.T, autoLevels=True)
        self.mask_plot.autoRange()

    @pyqtSlot()
    def _on_clear_mask_clicked(self):
        """Usuwa wszystkie ROI i czyści maskę."""
        for roi in self.roi_items:
            self.fft_plot.removeItem(roi)
        self.roi_items.clear()
        self.mask_item.clear()
        self.reco_item.clear()
        self.mask_array = None
        self.selected_spots_px.clear()
        self._redraw_spot_markers()
        logger.debug("Cleared all ROIs and masks.")

    def _perform_autocorrelation(self):
        if self.magnitude_fft_data is None: return

        try:
            power_spectrum = self.magnitude_fft_data.copy()
            
            # --- ZMIANA: Warunkowe usuwanie składowej stałej ---
            if self.remove_dc_checkbox.isChecked():
                rows, cols = power_spectrum.shape
                center_y, center_x = rows // 2, cols // 2
                power_spectrum[center_y, center_x] = 0
            # --- KONIEC ZMIANY ---

            logger.debug("Calculating FFT of power spectrum...")
            autocorr_complex = np.fft.fft2(power_spectrum)
            autocorr_shifted = np.fft.fftshift(autocorr_complex)
            autocorr_map = np.abs(autocorr_shifted)
            
            # --- ZMIANA: Warunkowe skalowanie wizualizacji ---
            scaling_mode = self.scaling_combo.currentText()
            if scaling_mode == "Logarithmic":
                display_data = np.log1p(autocorr_map**2)
            elif "Power Spectrum" in scaling_mode:
                display_data = autocorr_map**2
            elif "Square Root" in scaling_mode:
                display_data = np.sqrt(autocorr_map)
            else: # Linear
                display_data = autocorr_map
            # --- KONIEC ZMIANY ---

            self.reco_item.setImage(display_data.T, autoLevels=True)
            self.reco_plot.setTitle("Autocorrelation (Patterson Map)")
            self.reco_plot.autoRange()
            
            logger.info("Autocorrelation calculated and displayed successfully.")
            
        except Exception as e:
            logger.exception(f"Error during autocorrelation calculation: {e}")
            QMessageBox.critical(self, "Calculation Error", f"Could not calculate autocorrelation: {e}")

    @pyqtSlot()
    def _update_mask_from_rois(self):
        """Tworzy i wyświetla maskę na podstawie wszystkich ROI."""
        if self.magnitude_fft_data is None: return
        h, w = self.magnitude_fft_data.shape
        mask = np.zeros((h, w), dtype=np.float32)
        center_x, center_y = w / 2, h / 2

        for roi in self.roi_items:
            pos = roi.pos()
            size = roi.size()
            x0, y0 = int(round(pos.x())), int(round(pos.y()))
            x1, y1 = x0 + int(round(size.x())), y0 + int(round(size.y()))
            
            # Ograniczenie do wymiarów obrazu
            x0, x1 = np.clip([x0, x1], 0, w)
            y0, y1 = np.clip([y0, y1], 0, h)
            
            if x1 > x0 and y1 > y0:
                # mask[y0:y1, x0:x1] = 1.0
                mask[y0:y1, x0:x1] = self.magnitude_fft_data[y0:y1, x0:x1]

            # Jeśli tryb symetryczny jest włączony, dodaj symetryczne ROI
            if self.symmetric_roi_checkbox.isChecked():
                sym_x = 2 * center_x - (pos.x() + size.x())
                sym_y = 2 * center_y - (pos.y() + size.y())
                
                x0_s, y0_s = int(round(sym_x)), int(round(sym_y))
                x1_s, y1_s = x0_s + int(round(size.x())), y0_s + int(round(size.y()))

                x0_s, x1_s = np.clip([x0_s, x1_s], 0, w)
                y0_s, y1_s = np.clip([y0_s, y1_s], 0, h)

                if x1_s > x0_s and y1_s > y0_s:
                    # mask[y0_s:y1_s, x0_s:x1_s] = 1.0
                    mask[y0_s:y1_s, x0_s:x1_s] = self.magnitude_fft_data[y0_s:y1_s, x0_s:x1_s]
                
        self.mask_array = mask
        self.mask_item.setImage(self.mask_array.T, autoLevels=True)
        self.mask_plot.autoRange()

    def _perform_reconstruction_from_mask(self):
        """Wykonuje iFFT na zamaskowanych danych zespolonych."""
        
        try:
            logger.debug("Performing reconstruction from ROI mask.")
            
            # Krok 1: Zastosuj maskę do ORYGINALNYCH, ZESPOLONYCH danych FFT.
            # Wynik jest nadal w formacie "przesuniętym" (shifted).
            masked_fft_shifted = self.magnitude_fft_data * self.mask_array

            # Krok 2: Przesuń dane z powrotem do formatu obliczeniowego (odwrotność fftshift).
            # masked_fft_unshifted = np.fft.ifftshift(masked_fft_shifted)

            # Krok 3: Wykonaj ODWROTNĄ transformatę Fouriera.
            reconstructed_complex = np.fft.fft2(masked_fft_shifted)
            reconstructed_complex = np.fft.fftshift(reconstructed_complex)

            # Krok 4: Wyświetl moduł (amplitudę) wyniku.
            reconstructed_image = np.abs(reconstructed_complex)
            
            self.reco_item.setImage(reconstructed_image.T, autoLevels=True)
            self.reco_plot.setTitle("Reconstructed from Mask")
            self.reco_plot.autoRange()
            
            logger.info("Reconstruction from mask successful.")

        except Exception as e:
            logger.exception(f"Error during reconstruction from mask: {e}")
            QMessageBox.critical(self, "Reconstruction Error", f"Could not reconstruct image: {e}")

            