# lfa/gui/preprocessing_dialogs.py
import logging
import numpy as np
from typing import Optional, Tuple

try:
    from PyQt6.QtWidgets import (
        QDialog, QVBoxLayout, QHBoxLayout, QSlider, QLabel, QCheckBox,
        QDialogButtonBox, QWidget, QSizePolicy, QSpacerItem, QMessageBox
    )
    from PyQt6.QtCore import Qt, pyqtSlot, QRectF, QPointF
    import pyqtgraph as pg
    from pyqtgraph import PlotItem, RectROI, ROI
except ImportError as e:
    logging.critical(f"Failed to import necessary Qt or pyqtgraph modules: {e}")
    raise

try:
    from ..preprocessing.filtering import gaussian_blur
except ImportError:
    logging.error("Could not import preprocessing functions.")
    def gaussian_blur(image, sigma): logging.error("gaussian_blur function not available!"); return image

logger = logging.getLogger(__name__)

class GaussianBlurDialog(QDialog):
    """
    Dialog window for applying Gaussian Blur with integrated ROI/Whole Image mode.

    Features:
    - Side-by-side view of Original and Preview/Processed images.
    - ROI selection on the Original image.
    - "Process ROI Only" checkbox to switch between modes.
    - "Live Preview" checkbox to enable/disable real-time updates.
    - Final application via OK/Apply Changes button reflects the selected mode.
    """

    def __init__(self, original_data: np.ndarray, parent=None):
        super().__init__(parent)
        if original_data is None: raise ValueError("Original data cannot be None")

        self.original_data = original_data.astype(np.float32)
        # Bufor na wynik podglądu (live preview)
        self.preview_data = self.original_data.copy()
        # Flaga i zmienna na finalny wynik po kliknięciu Apply/OK
        self._final_processed_data: Optional[np.ndarray] = None
        self._final_is_roi: bool = False

        self.setWindowTitle("Gaussian Blur")
        self.setMinimumSize(900, 500)

        # --- Layouts ---
        main_layout = QVBoxLayout(self)
        top_layout = QHBoxLayout()
        controls_layout_v = QVBoxLayout()
        bottom_layout = QHBoxLayout()

        # --- Graphics Layout for Images ---
        pg.setConfigOption('background', 'w'); pg.setConfigOption('foreground', 'k')
        self.win = pg.GraphicsLayoutWidget()
        self.plot_original = self.win.addPlot(row=0, col=0, title="Original (Select ROI here)", name="plot_orig")
        self.img_original = pg.ImageItem(); self.plot_original.addItem(self.img_original)
        self.plot_original.hideAxis('left'); self.plot_original.hideAxis('bottom'); self.plot_original.setAspectLocked(True)
        self.plot_processed = self.win.addPlot(row=0, col=1, title="Preview", name="plot_proc") # Zmieniono tytuł
        self.img_processed = pg.ImageItem(); self.plot_processed.addItem(self.img_processed)
        self.plot_processed.hideAxis('left'); self.plot_processed.hideAxis('bottom'); self.plot_processed.setAspectLocked(True)
        self.plot_processed.vb.setXLink(self.plot_original.vb); self.plot_processed.vb.setYLink(self.plot_original.vb)
        top_layout.addWidget(self.win, stretch=3)

        # --- Controls Panel ---
        controls_panel = QWidget(); controls_panel.setMaximumWidth(250)
        controls_panel.setLayout(controls_layout_v)

        # Sigma Controls
        sigma_group_layout = QHBoxLayout()
        self.sigma_label = QLabel(f"Sigma: {0.0:.1f}")
        self.sigma_slider = QSlider(Qt.Orientation.Horizontal)
        self.sigma_slider.setMinimum(0); self.sigma_slider.setMaximum(100); self.sigma_slider.setValue(0)
        self.sigma_slider.setTickInterval(10); self.sigma_slider.setTickPosition(QSlider.TickPosition.TicksBelow)
        sigma_group_layout.addWidget(QLabel("Sigma:")); sigma_group_layout.addWidget(self.sigma_slider); sigma_group_layout.addWidget(self.sigma_label)
        controls_layout_v.addLayout(sigma_group_layout)

        # --- Checkboxes for Mode Control ---
        self.roi_mode_checkbox = QCheckBox("Process ROI Only")
        self.roi_mode_checkbox.setChecked(False) # Domyślnie przetwarzaj cały obraz
        controls_layout_v.addWidget(self.roi_mode_checkbox)

        self.live_preview_checkbox = QCheckBox("Live Preview")
        self.live_preview_checkbox.setChecked(True)
        controls_layout_v.addWidget(self.live_preview_checkbox)
        # ------------------------------------

        controls_layout_v.addSpacing(15)

        # ROI Info Label
        self.roi_info_label = QLabel("ROI: Not selected")
        controls_layout_v.addWidget(self.roi_info_label)

        # Ukryj/pokaż etykietę ROI w zależności od trybu
        self.roi_info_label.setVisible(self.roi_mode_checkbox.isChecked())

        controls_layout_v.addStretch()
        top_layout.addWidget(controls_panel, stretch=1)

        # --- ROI ---
        h, w = self.original_data.shape
        roi_w, roi_h = w // 4, h // 4; roi_x, roi_y = w // 2 - roi_w // 2, h // 2 - roi_h // 2
        self.roi = RectROI(pos=(roi_x, roi_y), size=(roi_w, roi_h), pen=pg.mkPen('y', width=2), translateSnap=True, scaleSnap=True)
        self.plot_original.addItem(self.roi)
        # Ukryj/pokaż ROI w zależności od trybu
        self.roi.setVisible(self.roi_mode_checkbox.isChecked())
        self._on_roi_changed() # Aktualizuj etykietę na starcie

        # --- Dialog Buttons ---
        self.button_box = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        self.button_box.button(QDialogButtonBox.StandardButton.Ok).setText("Apply Changes")
        bottom_layout.addWidget(self.button_box)

        # --- Assemble Main Layout ---
        main_layout.addLayout(top_layout)
        main_layout.addLayout(bottom_layout)

        # --- Initial Display ---
        self.update_original_view()
        self._update_preview() # Wywołaj wstępną aktualizację podglądu

        # --- Connect Signals ---
        self.sigma_slider.valueChanged.connect(self._on_parameter_changed)
        self.live_preview_checkbox.stateChanged.connect(self._on_parameter_changed)
        self.roi_mode_checkbox.stateChanged.connect(self._on_mode_changed) # Nowy slot
        self.roi.sigRegionChanged.connect(self._on_roi_changed)
        self.button_box.accepted.connect(self.accept) # OK/Apply Changes
        self.button_box.rejected.connect(self.reject)

        logger.debug("GaussianBlurDialog with integrated modes initialized.")

    def _get_sigma_value(self) -> float:
        return self.sigma_slider.value() / 10.0

    @pyqtSlot()
    def _on_parameter_changed(self):
        """Slot called when sigma slider or live preview checkbox changes."""
        sigma = self._get_sigma_value()
        self.sigma_label.setText(f"Sigma: {sigma:.1f}")
        # Always update preview if live preview is on
        if self.live_preview_checkbox.isChecked():
            self._update_preview()
        # else: # Optionally clear preview or leave the last state? Leaving is fine.
        #    pass

    @pyqtSlot(int) # Slot dla stateChanged checkboxa trybu
    def _on_mode_changed(self, state):
        """Slot called when 'Process ROI Only' checkbox state changes."""
        is_roi_mode = state == Qt.CheckState.Checked.value # Czy tryb ROI jest aktywny
        self.roi.setVisible(is_roi_mode) # Pokaż/ukryj ROI
        self.roi_info_label.setVisible(is_roi_mode) # Pokaż/ukryj etykietę ROI
        # Zaktualizuj podgląd, aby odzwierciedlić nowy tryb (jeśli live preview włączone)
        if self.live_preview_checkbox.isChecked():
            self._update_preview()
        logger.debug(f"Processing mode changed. ROI mode active: {is_roi_mode}")


    @pyqtSlot()
    def _on_roi_changed(self):
        """Slot called when the ROI is moved or resized."""
        pos = self.roi.pos(); size = self.roi.size()
        info_text = f"ROI: ({pos.x():.1f}, {pos.y():.1f}) Size: ({size.x():.1f}, {size.y():.1f})"
        self.roi_info_label.setText(info_text)
        # Jeśli jesteśmy w trybie ROI i live preview jest włączone, odśwież podgląd
        if self.roi_mode_checkbox.isChecked() and self.live_preview_checkbox.isChecked():
             self._update_preview()

    def _get_roi_slice(self) -> Optional[Tuple[slice, slice]]:
        # ... (bez zmian - kalkulacja wycinka) ...
        pos = self.roi.pos(); size = self.roi.size(); h, w = self.original_data.shape
        x0, y0 = int(round(pos.x())), int(round(pos.y())); width, height = int(round(size.x())), int(round(size.y()))
        x1 = min(x0 + width, w); y1 = min(y0 + height, h); x0 = max(0, x0); y0 = max(0, y0)
        if x1 > x0 and y1 > y0: row_slice = slice(y0, y1); col_slice = slice(x0, x1); return row_slice, col_slice
        else: logger.warning("Invalid ROI dimensions after clamping/rounding."); return None


    def _update_preview(self):
        """Calculates and updates the preview image based on current settings."""
        if not self.live_preview_checkbox.isChecked():
             # If live preview turned off, maybe revert preview to last applied state?
             # For now, let's just show the last calculated preview or processed data.
             # Or simply return - the view won't update. Let's return.
             # logger.debug("Live preview off, preview update skipped.")
             # Revert preview to the actual processed data state
             self.preview_data = self.processed_data.copy()
             self.update_processed_view()
             return

        sigma = self._get_sigma_value()
        is_roi_mode = self.roi_mode_checkbox.isChecked()
        logger.debug(f"Updating preview. ROI mode: {is_roi_mode}, Sigma: {sigma:.2f}")

        try:
            # Base for calculation is always the original data for preview consistency
            base_image = self.original_data
            # Calculate the effect of the filter on the *whole* base image first
            processed_full = gaussian_blur(base_image, sigma)

            if is_roi_mode:
                roi_slice = self._get_roi_slice()
                if roi_slice:
                    # Start preview with original data
                    self.preview_data = base_image.copy()
                    # Apply processed data *only* within the ROI slice
                    self.preview_data[roi_slice] = processed_full[roi_slice]
                else:
                    # If ROI is invalid, show the full processed image? Or original? Show original.
                    self.preview_data = base_image.copy()
            else:
                # If not ROI mode, the preview is the fully processed image
                self.preview_data = processed_full

            # Update the view
            self.update_processed_view()

        except Exception as e:
            logger.exception(f"Error during preview update: {e}")
            # Optionally show error to user or just log

    # Update/Display methods
    def update_original_view(self):
        if self.original_data is not None and self.img_original:
            self.img_original.setImage(self.original_data.T); self.plot_original.autoRange()

    def update_processed_view(self):
        """Updates the 'Processed' image view with self.preview_data."""
        # This view now always shows the content of self.preview_data
        if not self.img_processed: return

        if self.preview_data is not None:
            self.img_processed.setImage(self.preview_data.T) # Transpose
            # self.plot_processed.autoRange() # Keep autoRange? Might be annoying on live update
            logger.debug("Preview view updated.")
        else:
             self.img_processed.clear()
             logger.debug("Preview view cleared.")


    # Accept/Reject/Getters
    def accept(self):
        """Calculate final result based on mode and close dialog."""
        sigma = self._get_sigma_value()
        is_roi_mode = self.roi_mode_checkbox.isChecked()
        self._final_is_roi = is_roi_mode # Store flag for main window

        logger.info(f"Dialog accepted. Finalizing processing. ROI mode: {is_roi_mode}, Sigma: {sigma:.2f}")

        try:
            # Always calculate the fully processed version first
            base_image = self.original_data
            processed_full = gaussian_blur(base_image, sigma)

            if is_roi_mode:
                roi_slice = self._get_roi_slice()
                if roi_slice:
                    # Create final result by applying effect only within ROI
                    self._final_processed_data = base_image.copy()
                    self._final_processed_data[roi_slice] = processed_full[roi_slice]
                else:
                    # Invalid ROI selected at time of apply? Return original.
                    logger.warning("Cannot apply ROI mode with invalid ROI. Returning original data.")
                    self._final_processed_data = base_image.copy()
                    self._final_is_roi = False # Mark as not ROI applied
            else:
                # Whole image mode - final result is the fully processed image
                self._final_processed_data = processed_full

            logger.info("Final processing calculated.")
            super().accept() # Close the dialog

        except Exception as e:
            logger.exception(f"Error during final processing calculation: {e}")
            QMessageBox.critical(self, "Processing Error", f"Failed to calculate final result:\n{e}")
            # Don't close the dialog on error? Or close with reject? Let's reject.
            super().reject()


    def reject(self):
        logger.info("Gaussian Blur dialog rejected (Cancel clicked).")
        self._final_processed_data = None # Ensure no data is returned
        super().reject()

    def get_processed_data(self) -> Optional[np.ndarray]:
        # Return the final data calculated during accept()
        return self._final_processed_data.copy() if self._final_processed_data is not None else None

    def get_parameters(self) -> dict:
        # Return parameters used for the final calculation
        return {'sigma': round(self._get_sigma_value(), 2)}

    def was_roi_applied(self) -> bool:
        """Returns True if the final accepted result was ROI-based."""
        return self._final_is_roi