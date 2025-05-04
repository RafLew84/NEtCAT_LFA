# lfa/gui/preprocessing_dialogs.py
import logging
# Usunięto import abc
import numpy as np
from typing import Optional, Tuple, Dict, Any

try:
    from PyQt6.QtWidgets import (
        QDialog, QVBoxLayout, QHBoxLayout, QSlider, QLabel, QCheckBox,
        QDialogButtonBox, QWidget, QSizePolicy, QSpacerItem, QFrame, QMessageBox
    )
    from PyQt6.QtCore import Qt, pyqtSlot
    import pyqtgraph as pg
    from pyqtgraph import PlotItem, RectROI, ROI, ImageItem
except ImportError as e:
    logging.critical(f"Failed to import necessary Qt or pyqtgraph modules: {e}")
    raise

# Importuj funkcję przetwarzania
try:
    from ..preprocessing.filtering import gaussian_blur
except ImportError:
    logging.error("Could not import gaussian_blur function.")
    def gaussian_blur(image, sigma): return image # Dummy function

logger = logging.getLogger(__name__)


# --- Standalone Gaussian Blur Dialog ---

class GaussianBlurDialog(QDialog): # Dziedziczy bezpośrednio z QDialog
    """
    Dialog window for applying Gaussian Blur.

    Includes side-by-side views, ROI selection, ROI/Whole image mode toggle,
    and live preview functionality, implemented as a standalone dialog.
    """

    def __init__(self, original_data: np.ndarray, parent=None):
        """Initializes the dialog."""
        super().__init__(parent)
        if original_data is None: raise ValueError("Original data cannot be None")

        self.operation_name = "Gaussian Blur" # Nazwa operacji
        self.original_data = original_data.astype(np.float32)
        self.preview_data = self.original_data.copy()
        self._final_processed_data: Optional[np.ndarray] = None
        self._final_params: Dict[str, Any] = {}
        self._final_is_roi_applied_only: bool = False

        self.setWindowTitle(f"{self.operation_name} Settings")
        self.setMinimumSize(900, 500)
        current_flags=self.windowFlags(); self.setWindowFlags(current_flags | Qt.WindowType.WindowMinimizeButtonHint | Qt.WindowType.WindowMaximizeButtonHint)

        # --- Layouts ---
        main_layout=QVBoxLayout(self); top_layout=QHBoxLayout(); controls_area_layout=QVBoxLayout(); bottom_layout=QHBoxLayout()

        # --- Graphics Views ---
        pg.setConfigOption('background', 'w'); pg.setConfigOption('foreground', 'k'); self.win = pg.GraphicsLayoutWidget()
        self.plot_original = self.win.addPlot(row=0, col=0, title="Original", name="plot_orig"); self.img_original = ImageItem(); self.plot_original.addItem(self.img_original); self.plot_original.hideAxis('left'); self.plot_original.hideAxis('bottom'); self.plot_original.setAspectLocked(True)
        self.plot_processed = self.win.addPlot(row=0, col=1, title="Preview", name="plot_proc"); self.img_processed = ImageItem(); self.plot_processed.addItem(self.img_processed); self.plot_processed.hideAxis('left'); self.plot_processed.hideAxis('bottom'); self.plot_processed.setAspectLocked(True)
        self.plot_processed.vb.setXLink(self.plot_original.vb); self.plot_processed.vb.setYLink(self.plot_original.vb)
        self.plot_original.vb.invertY(True); self.plot_processed.vb.invertY(True)
        top_layout.addWidget(self.win, stretch=3)

        # --- Controls Panel ---
        controls_panel = QWidget(); controls_panel.setMaximumWidth(250); controls_panel.setLayout(controls_area_layout)

        # --- Parameter Controls (Specific to Gaussian Blur) ---
        parameter_widget_container = QWidget()
        specific_param_layout = QVBoxLayout(parameter_widget_container)
        specific_param_layout.setContentsMargins(0,0,0,0)
        self._create_parameter_controls(specific_param_layout) # Wywołanie metody tworzącej kontrolki
        controls_area_layout.addWidget(parameter_widget_container)
        # -----------------------------------------------------

        controls_area_layout.addWidget(QFrame(frameShape=QFrame.Shape.HLine, frameShadow=QFrame.Shadow.Sunken))

        # --- ROI and Mode Controls ---
        self.apply_to_roi_only_checkbox = QCheckBox("Apply only to ROI area")
        self.apply_to_roi_only_checkbox.setChecked(False)
        self.live_preview_checkbox = QCheckBox("Live Preview")
        self.live_preview_checkbox.setChecked(True)
        controls_area_layout.addWidget(self.apply_to_roi_only_checkbox)
        controls_area_layout.addWidget(self.live_preview_checkbox)
        # ---------------------------

        controls_area_layout.addWidget(QFrame(frameShape=QFrame.Shape.HLine, frameShadow=QFrame.Shadow.Sunken))

        # --- ROI Info and Item ---
        self.roi_info_label = QLabel("ROI: Not selected")
        controls_area_layout.addWidget(self.roi_info_label)
        h, w = self.original_data.shape; roi_w, roi_h = w//4, h//4; roi_x, roi_y = w//2 - roi_w//2, h//2 - roi_h//2
        self.roi = RectROI(pos=(roi_x, roi_y), size=(roi_w, roi_h), pen=pg.mkPen('y', width=2), translateSnap=True, scaleSnap=True); self.plot_original.addItem(self.roi)
        # Widoczność kontrolek ROI zależy od checkboxa
        is_roi_mode = self.apply_to_roi_only_checkbox.isChecked()
        self.roi.setVisible(is_roi_mode)
        self.roi_info_label.setVisible(is_roi_mode)
        self._on_roi_changed() # Aktualizacja etykiety
        # -------------------------

        controls_area_layout.addStretch()
        top_layout.addWidget(controls_panel, stretch=1)

        # --- Dialog Buttons ---
        self.button_box = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel); self.button_box.button(QDialogButtonBox.StandardButton.Ok).setText("Apply Changes"); bottom_layout.addWidget(self.button_box)

        # --- Assemble Layout ---
        main_layout.addLayout(top_layout); main_layout.addLayout(bottom_layout)

        # --- Initial Display & Connections ---
        self.update_original_view(); self._update_preview()
        # Connect signals - uwaga: _on_parameter_or_preview_changed jest teraz metodą tej klasy
        self.apply_to_roi_only_checkbox.stateChanged.connect(self._on_parameter_or_preview_changed)
        self.live_preview_checkbox.stateChanged.connect(self._on_parameter_or_preview_changed)
        self.roi.sigRegionChanged.connect(self._on_roi_changed)
        self.button_box.accepted.connect(self.accept); self.button_box.rejected.connect(self.reject)

        logger.debug(f"Standalone {self.operation_name} dialog initialized.")

    # --- Implementacja "abstrakcyjnych" metod z poprzedniej bazy ---
    def _create_parameter_controls(self, layout: QVBoxLayout):
        """Adds controls specific to Gaussian Blur (sigma slider)."""
        sigma_controls_layout = QHBoxLayout()
        self.sigma_label = QLabel(f"Sigma: {0.0:.1f}")
        self.sigma_slider = QSlider(Qt.Orientation.Horizontal); self.sigma_slider.setMinimum(0); self.sigma_slider.setMaximum(100); self.sigma_slider.setValue(0); self.sigma_slider.setTickInterval(10); self.sigma_slider.setTickPosition(QSlider.TickPosition.TicksBelow)
        sigma_controls_layout.addWidget(QLabel("Sigma:")); sigma_controls_layout.addWidget(self.sigma_slider); sigma_controls_layout.addWidget(self.sigma_label)
        # Podłącz slider do slotu obsługi zmian parametrów
        self.sigma_slider.valueChanged.connect(self._on_parameter_or_preview_changed) # Podłącz do wspólnego slotu
        layout.addLayout(sigma_controls_layout)

    def _get_current_parameters(self) -> Dict[str, Any]:
        """Returns the current sigma value AND the state of the ROI checkbox."""
        sigma = self.sigma_slider.value() / 10.0
        return {
            'sigma': round(sigma, 2),
            'apply_roi_only': self.apply_to_roi_only_checkbox.isChecked() # Stan checkboxa
        }

    def _apply_operation(self, image: np.ndarray, params: Dict[str, Any]) -> Optional[np.ndarray]:
        """Applies gaussian_blur, potentially only to ROI based on params."""
        sigma = params.get('sigma', 0.0)
        apply_roi_only = params.get('apply_roi_only', False)
        logger.debug(f"Applying Gaussian Blur. Sigma={sigma}, ROI Only={apply_roi_only}")
        try:
            # Zawsze obliczamy pełne rozmycie jako bazę
            processed_full = gaussian_blur(image, sigma)
            if processed_full is None: return None # Błąd w funkcji rozmycia

            if apply_roi_only:
                roi_slice = self._get_roi_slice()
                if roi_slice:
                    # Zastosuj tylko do ROI
                    result_image = image.copy()
                    result_image[roi_slice] = processed_full[roi_slice]
                    return result_image
                else: # Błąd ROI
                    logger.warning("Cannot apply Gaussian Blur to ROI only: Invalid ROI.")
                    return image # Zwróć oryginał w razie błędu ROI
            else:
                # Zastosuj do całości
                return processed_full
        except Exception as e:
            logger.exception(f"Error applying gaussian_blur: {e}")
            return None # Zwróć None w razie błędu

    # --- Sloty i Metody skopiowane/zaadaptowane z BasePreprocessingDialog ---
    @pyqtSlot()
    def _on_parameter_or_preview_changed(self):
        """Slot for parameter, roi checkbox or live preview checkbox changes."""
        # Aktualizacja etykiety sigma, jeśli slider istnieje (specyficzne dla Gaussa)
        if hasattr(self, 'sigma_slider'):
             sigma = self.sigma_slider.value() / 10.0
             self.sigma_label.setText(f"Sigma: {sigma:.1f}")

        # Aktualizacja widoczności ROI/etykiety
        is_roi_mode = self.apply_to_roi_only_checkbox.isChecked()
        self.roi.setVisible(is_roi_mode)
        self.roi_info_label.setVisible(is_roi_mode)

        # Aktualizacja podglądu
        if self.live_preview_checkbox.isChecked():
            self._update_preview()

    @pyqtSlot()
    def _on_roi_changed(self):
        """Updates ROI info label and preview if needed."""
        pos=self.roi.pos(); size=self.roi.size(); info_text = f"ROI: ({pos.x():.1f}, {pos.y():.1f}) Size: ({size.x():.1f}, {size.y():.1f})"; self.roi_info_label.setText(info_text)
        # Aktualizuj podgląd tylko jeśli oba checkboxy są zaznaczone
        if self.apply_to_roi_only_checkbox.isChecked() and self.live_preview_checkbox.isChecked():
             self._update_preview()

    def _get_roi_slice(self) -> Optional[Tuple[slice, slice]]:
        if not self.roi.isVisible() or not self.roi.size().x() > 0 or not self.roi.size().y() > 0: return None
        pos=self.roi.pos(); size=self.roi.size(); h,w=self.original_data.shape; x0,y0=int(round(pos.x())),int(round(pos.y())); width,height=int(round(size.x())),int(round(size.y())); x1=min(x0+width,w); y1=min(y0+height,h); x0=max(0,x0); y0=max(0,y0)
        if x1>x0 and y1>y0: return slice(y0,y1), slice(x0,x1)
        else: logger.warning("Invalid ROI dimensions."); return None

    def _update_preview(self):
        """Calculates and updates the preview image."""
        # Ta metoda jest teraz prostsza, bo cała logika ROI/Whole jest w _apply_operation
        if not self.live_preview_checkbox.isChecked():
            self.preview_data = self.original_data.copy(); self.update_preview_view()
            return
        params = self._get_current_parameters()
        logger.debug(f"Updating preview. Params: {params}")
        try:
            self.preview_data = self._apply_operation(self.original_data, params)
            if self.preview_data is None: # Handle operation failure
                 self.preview_data = self.original_data.copy()
            self.update_preview_view()
        except Exception as e: logger.exception(f"Error during preview update: {e}")

    def update_original_view(self):
        if self.original_data is not None and self.img_original: self.img_original.setImage(self.original_data.T); self.plot_original.autoRange()
    def update_preview_view(self):
        if not self.img_processed: return
        if self.preview_data is not None: self.img_processed.setImage(self.preview_data.T); logger.debug("Preview view updated.")
        else: self.img_processed.clear(); logger.debug("Preview view cleared.")

    def accept(self):
        """Calculate final result and close dialog."""
        params = self._get_current_parameters()
        self._final_is_roi_applied_only = params.get('apply_roi_only', False)
        logger.info(f"Dialog accepted. Finalizing '{self.operation_name}'. Apply ROI Only: {self._final_is_roi_applied_only}, Params: {params}")
        try:
            base_image = self.original_data
            # Oblicz finalny wynik
            self._final_processed_data = self._apply_operation(base_image, params)
            if self._final_processed_data is None: raise ValueError("Processing operation failed.")
            if np.allclose(self._final_processed_data, self.original_data): logger.info("Data not modified."); self._final_processed_data = None; super().reject(); return
            logger.info("Final processing calculated."); super().accept()
        except Exception as e: logger.exception(f"Error final processing: {e}"); QMessageBox.critical(self, "Error", f"... {e}"); self._final_processed_data = None; self._final_is_roi_applied_only = False; super().reject()

    def reject(self): logger.info(f"{self.operation_name} dialog rejected."); self._final_processed_data = None; super().reject()
    def get_processed_data(self) -> Optional[np.ndarray]: return self._final_processed_data.copy() if self._final_processed_data is not None else None
    def get_parameters(self) -> dict: return self._get_current_parameters()
    def was_roi_applied_only(self) -> bool: return self._final_is_roi_applied_only

