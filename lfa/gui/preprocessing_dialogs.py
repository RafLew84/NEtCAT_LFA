# lfa/gui/preprocessing_dialogs.py
import logging
# Removed import abc
import numpy as np
from typing import Optional, Tuple, Dict, Any, List

try:
    from PyQt6.QtWidgets import (
        QDialog, QVBoxLayout, QHBoxLayout, QSlider, QLabel, QCheckBox,
        QDialogButtonBox, QWidget, QSizePolicy, QSpacerItem, QFrame, QMessageBox, 
        QGroupBox, QRadioButton, QPushButton, QComboBox, QSpinBox, QDoubleSpinBox,
        QApplication
    )
    from PyQt6.QtGui import QIntValidator
    from PyQt6.QtCore import Qt, pyqtSlot
    import pyqtgraph as pg
    from pyqtgraph import PlotItem, RectROI, ROI, ImageItem
except ImportError as e:
    logging.critical(f"Failed to import necessary Qt or pyqtgraph modules: {e}")
    raise

# Import processing function
try:
    from ..preprocessing.filtering import gaussian_blur, median_filter_lfa
except ImportError:
    logging.error("Could not import gaussian_blur function.")
    def gaussian_blur(image, sigma): return image 
    def median_filter_lfa(image, size, mode='reflect', cval=0.0): return image 

try:
    from ..preprocessing.leveling import fit_plane, fit_plane_3pts, level_by_plane
except ImportError:
    logging.error("Could not import fit_plane, fit_plane_3pts, or level_by_plane functions.")
    def fit_plane(*args, **kwargs): return None
    def fit_plane_3pts(*args, **kwargs): return None
    def level_by_plane(*args, **kwargs): return None

try: from ..preprocessing.denoising import denoise_nlmeans_skimage, denoise_bm3d_lfa
except ImportError: 
    logging.error("Could not import denoising functions."); 
    def denoise_nlmeans_skimage(**kwargs): return kwargs.get('image')
    def denoise_bm3d_lfa(**kwargs): return kwargs.get('image')

logger = logging.getLogger(__name__)

# --- Standalone Gaussian Blur Dialog ---

class GaussianBlurDialog(QDialog): # Inherits directly from QDialog
    """
    Dialog window for applying Gaussian Blur.

    Includes side-by-side views, ROI selection, ROI/Whole image mode toggle,
    and live preview functionality, implemented as a standalone dialog.
    """

    def __init__(self, original_data: np.ndarray, parent=None):
        """Initializes the dialog."""
        super().__init__(parent)
        if original_data is None: raise ValueError("Original data cannot be None")

        self.operation_name = "Gaussian Blur" # Operation name
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
        self._create_parameter_controls(specific_param_layout) # Call method creating controls
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
        # ROI controls visibility depends on checkbox
        is_roi_mode = self.apply_to_roi_only_checkbox.isChecked()
        self.roi.setVisible(is_roi_mode)
        self.roi_info_label.setVisible(is_roi_mode)
        self._on_roi_changed() # Update label
        # -------------------------

        controls_area_layout.addStretch()
        top_layout.addWidget(controls_panel, stretch=1)

        # --- Dialog Buttons ---
        self.button_box = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel); self.button_box.button(QDialogButtonBox.StandardButton.Ok).setText("Apply Changes"); bottom_layout.addWidget(self.button_box)

        # --- Assemble Layout ---
        main_layout.addLayout(top_layout); main_layout.addLayout(bottom_layout)

        # --- Initial Display & Connections ---
        self.update_original_view(); self._update_preview()
        # Connect signals - note: _on_parameter_or_preview_changed is now a method of this class
        self.apply_to_roi_only_checkbox.stateChanged.connect(self._on_parameter_or_preview_changed)
        self.live_preview_checkbox.stateChanged.connect(self._on_parameter_or_preview_changed)
        self.roi.sigRegionChanged.connect(self._on_roi_changed)
        self.button_box.accepted.connect(self.accept); self.button_box.rejected.connect(self.reject)

        logger.debug(f"Standalone {self.operation_name} dialog initialized.")

    # --- Implementation of "abstract" methods from previous base ---
    def _create_parameter_controls(self, layout: QVBoxLayout):
        """Adds controls specific to Gaussian Blur (sigma slider)."""
        sigma_controls_layout = QHBoxLayout()
        self.sigma_label = QLabel(f"Sigma: {0.0:.1f}")
        self.sigma_slider = QSlider(Qt.Orientation.Horizontal); self.sigma_slider.setMinimum(0); self.sigma_slider.setMaximum(100); self.sigma_slider.setValue(0); self.sigma_slider.setTickInterval(10); self.sigma_slider.setTickPosition(QSlider.TickPosition.TicksBelow)
        sigma_controls_layout.addWidget(QLabel("Sigma:")); sigma_controls_layout.addWidget(self.sigma_slider); sigma_controls_layout.addWidget(self.sigma_label)
        # Connect slider to parameter change handler
        self.sigma_slider.valueChanged.connect(self._on_parameter_or_preview_changed) # Connect to common slot
        layout.addLayout(sigma_controls_layout)

    def _get_current_parameters(self) -> Dict[str, Any]:
        """Returns the current sigma value AND the state of the ROI checkbox."""
        sigma = self.sigma_slider.value() / 10.0
        return {
            'sigma': round(sigma, 2),
            'apply_roi_only': self.apply_to_roi_only_checkbox.isChecked() # Checkbox state
        }

    def _apply_operation(self, image: np.ndarray, params: Dict[str, Any]) -> Optional[np.ndarray]:
        """Applies gaussian_blur, potentially only to ROI based on params."""
        sigma = params.get('sigma', 0.0)
        apply_roi_only = params.get('apply_roi_only', False)
        logger.debug(f"Applying Gaussian Blur. Sigma={sigma}, ROI Only={apply_roi_only}")
        try:
            # Always calculate full blur as base
            processed_full = gaussian_blur(image, sigma)
            if processed_full is None: return None # Error in blur function

            if apply_roi_only:
                roi_slice = self._get_roi_slice()
                if roi_slice:
                    # Apply only to ROI
                    result_image = image.copy()
                    result_image[roi_slice] = processed_full[roi_slice]
                    return result_image
                else: # ROI error
                    logger.warning("Cannot apply Gaussian Blur to ROI only: Invalid ROI.")
                    return image # Return original on ROI error
            else:
                # Apply to whole image
                return processed_full
        except Exception as e:
            logger.exception(f"Error applying gaussian_blur: {e}")
            return None # Return None on error

    # --- Slots and Methods copied/adapted from BasePreprocessingDialog ---
    @pyqtSlot()
    def _on_parameter_or_preview_changed(self):
        """Slot for parameter, roi checkbox or live preview checkbox changes."""
        # Update sigma label if slider exists (specific to Gaussian)
        if hasattr(self, 'sigma_slider'):
             sigma = self.sigma_slider.value() / 10.0
             self.sigma_label.setText(f"Sigma: {sigma:.1f}")

        # Update ROI/label visibility
        is_roi_mode = self.apply_to_roi_only_checkbox.isChecked()
        self.roi.setVisible(is_roi_mode)
        self.roi_info_label.setVisible(is_roi_mode)

        # Update preview
        if self.live_preview_checkbox.isChecked():
            self._update_preview()

    @pyqtSlot()
    def _on_roi_changed(self):
        """Updates ROI info label and preview if needed."""
        pos=self.roi.pos(); size=self.roi.size(); info_text = f"ROI: ({pos.x():.1f}, {pos.y():.1f}) Size: ({size.x():.1f}, {size.y():.1f})"; self.roi_info_label.setText(info_text)
        # Update preview only if both checkboxes are checked
        if self.apply_to_roi_only_checkbox.isChecked() and self.live_preview_checkbox.isChecked():
             self._update_preview()

    def _get_roi_slice(self) -> Optional[Tuple[slice, slice]]:
        if not self.roi.isVisible() or not self.roi.size().x() > 0 or not self.roi.size().y() > 0: return None
        pos=self.roi.pos(); size=self.roi.size(); h,w=self.original_data.shape; x0,y0=int(round(pos.x())),int(round(pos.y())); width,height=int(round(size.x())),int(round(size.y())); x1=min(x0+width,w); y1=min(y0+height,h); x0=max(0,x0); y0=max(0,y0)
        if x1>x0 and y1>y0: return slice(y0,y1), slice(x0,x1)
        else: logger.warning("Invalid ROI dimensions."); return None

    def _update_preview(self):
        """Calculates and updates the preview image."""
        # This method is now simpler as all ROI/Whole logic is in _apply_operation
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
            # Calculate final result
            self._final_processed_data = self._apply_operation(base_image, params)
            if self._final_processed_data is None: raise ValueError("Processing operation failed.")
            if np.allclose(self._final_processed_data, self.original_data): logger.info("Data not modified."); self._final_processed_data = None; super().reject(); return
            logger.info("Final processing calculated."); super().accept()
        except Exception as e: logger.exception(f"Error final processing: {e}"); QMessageBox.critical(self, "Error", f"... {e}"); self._final_processed_data = None; self._final_is_roi_applied_only = False; super().reject()

    def reject(self): logger.info(f"{self.operation_name} dialog rejected."); self._final_processed_data = None; super().reject()
    def get_processed_data(self) -> Optional[np.ndarray]: return self._final_processed_data.copy() if self._final_processed_data is not None else None
    def get_parameters(self) -> dict: return self._get_current_parameters()
    def was_roi_applied_only(self) -> bool: return self._final_is_roi_applied_only

class NLMeansDialog(QDialog):
    """
    Standalone dialog window for applying Non-Local Means denoising (skimage).
    Includes ROI/Whole image mode toggle and live preview.
    """
    def __init__(self, original_data: np.ndarray, parent=None):
        super().__init__(parent)
        if original_data is None: raise ValueError("Original data cannot be None")

        self.operation_name = "NL-Means Denoising (skimage)"
        self.original_data = original_data.astype(np.float32)
        self.preview_data = self.original_data.copy()
        self._final_processed_data: Optional[np.ndarray] = None
        self._final_params: Dict[str, Any] = {}
        self._final_is_roi_applied_only: bool = False

        self.setWindowTitle(f"{self.operation_name} Settings")
        self.setMinimumSize(900, 600) 
        current_flags=self.windowFlags(); self.setWindowFlags(current_flags | Qt.WindowType.WindowMinimizeButtonHint | Qt.WindowType.WindowMaximizeButtonHint)

        # --- Layouts ---
        main_layout=QVBoxLayout(self); top_layout=QHBoxLayout(); controls_area_layout=QVBoxLayout(); bottom_layout=QHBoxLayout()

        # --- Graphics Views ---
        pg.setConfigOption('background', 'w'); pg.setConfigOption('foreground', 'k'); self.win = pg.GraphicsLayoutWidget()
        self.plot_original = self.win.addPlot(row=0, col=0, title="Original", name="plot_orig"); self.img_original = ImageItem()
        self.plot_original.addItem(self.img_original); self.plot_original.hideAxis('left'); self.plot_original.hideAxis('bottom')
        self.plot_original.setAspectLocked(True)
        self.plot_processed = self.win.addPlot(row=0, col=1, title="Preview", name="plot_proc"); self.img_processed = ImageItem()
        self.plot_processed.addItem(self.img_processed); self.plot_processed.hideAxis('left')
        self.plot_processed.hideAxis('bottom'); self.plot_processed.setAspectLocked(True)
        self.plot_processed.vb.setXLink(self.plot_original.vb)
        self.plot_processed.vb.setYLink(self.plot_original.vb)
        self.plot_original.vb.invertY(True); self.plot_processed.vb.invertY(True)
        top_layout.addWidget(self.win, stretch=3)

        # --- Controls Panel ---
        controls_panel = QWidget(); controls_panel.setMaximumWidth(250); controls_panel.setLayout(controls_area_layout)
        parameter_widget_container = QWidget()
        specific_param_layout = QVBoxLayout(parameter_widget_container); specific_param_layout.setContentsMargins(0,0,0,0)
        self._create_parameter_controls(specific_param_layout)
        controls_area_layout.addWidget(parameter_widget_container)

        controls_area_layout.addWidget(QFrame(frameShape=QFrame.Shape.HLine, frameShadow=QFrame.Shadow.Sunken))

        # ROI and Mode Controls
        self.apply_to_roi_only_checkbox = QCheckBox("Apply only to ROI area"); self.apply_to_roi_only_checkbox.setChecked(False)
        self.live_preview_checkbox = QCheckBox("Live Preview"); self.live_preview_checkbox.setChecked(True)
        controls_area_layout.addWidget(self.apply_to_roi_only_checkbox)
        controls_area_layout.addWidget(self.live_preview_checkbox)

        controls_area_layout.addWidget(QFrame(frameShape=QFrame.Shape.HLine, frameShadow=QFrame.Shadow.Sunken))

        # ROI Info and Item
        self.roi_info_label = QLabel("ROI: Not selected"); controls_area_layout.addWidget(self.roi_info_label)
        h, w = self.original_data.shape; roi_w, roi_h = w//4, h//4; roi_x, roi_y = w//2-roi_w//2, h//2-roi_h//2
        self.roi = RectROI(pos=(roi_x, roi_y), size=(roi_w, roi_h), pen=pg.mkPen('b', width=2), translateSnap=True, scaleSnap=True); # Blue ROI
        self.plot_original.addItem(self.roi)
        is_roi_mode = self.apply_to_roi_only_checkbox.isChecked()
        self.roi.setVisible(is_roi_mode); self.roi_info_label.setVisible(is_roi_mode)
        self._on_roi_changed()

        controls_area_layout.addStretch()
        top_layout.addWidget(controls_panel, stretch=1)

        # Dialog Buttons
        self.button_box = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel); self.button_box.button(QDialogButtonBox.StandardButton.Ok).setText("Apply Changes"); bottom_layout.addWidget(self.button_box)

        # Assemble Layout
        main_layout.addLayout(top_layout); main_layout.addLayout(bottom_layout)

        # Initial Display & Connections
        self.update_original_view(); self._update_preview()
        self.apply_to_roi_only_checkbox.stateChanged.connect(self._on_parameter_or_preview_changed)
        self.live_preview_checkbox.stateChanged.connect(self._on_parameter_or_preview_changed)
        self.roi.sigRegionChanged.connect(self._on_roi_changed)
        self.button_box.accepted.connect(self.accept); self.button_box.rejected.connect(self.reject)

        logger.debug(f"Standalone {self.operation_name} dialog initialized.")


    # --- UI and logic implementation ---
    def _create_parameter_controls(self, layout: QVBoxLayout):
        """Adds controls specific to NL-Means Filter."""
        # Sigma (Noise StDev estimate)
        sigma_layout = QHBoxLayout()
        sigma_layout.addWidget(QLabel("Noise Sigma (σ):"))
        self.sigma_spinbox = QDoubleSpinBox()
        self.sigma_spinbox.setDecimals(4)
        self.sigma_spinbox.setRange(0.0001, 1e6) 
        # TODO: Estimate default sigma from the image? For now, set a small value
        estimated_sigma = np.std(self.original_data) / 10.0 
        self.sigma_spinbox.setValue(max(0.001, estimated_sigma)) 
        self.sigma_spinbox.setSingleStep(0.01)
        sigma_layout.addWidget(self.sigma_spinbox)
        layout.addLayout(sigma_layout)

        h_mult_layout = QHBoxLayout()
        h_mult_layout.addWidget(QLabel("H Multiplier (h = mult * σ):"))
        self.h_mult_spinbox = QDoubleSpinBox()
        self.h_mult_spinbox.setDecimals(2)
        self.h_mult_spinbox.setRange(0.1, 5.0) 
        self.h_mult_spinbox.setValue(1.0) 
        self.h_mult_spinbox.setSingleStep(0.05)
        h_mult_layout.addWidget(self.h_mult_spinbox)
        layout.addLayout(h_mult_layout)

        # Patch Size
        psize_layout = QHBoxLayout()
        psize_layout.addWidget(QLabel("Patch Size:"))
        self.patch_size_spinbox = QSpinBox()
        self.patch_size_spinbox.setMinimum(3)
        self.patch_size_spinbox.setMaximum(21) 
        self.patch_size_spinbox.setSingleStep(2) 
        self.patch_size_spinbox.setValue(7) 
        psize_layout.addWidget(self.patch_size_spinbox)
        layout.addLayout(psize_layout)

        # Patch Distance
        pdist_layout = QHBoxLayout()
        pdist_layout.addWidget(QLabel("Patch Distance:"))
        self.patch_dist_spinbox = QSpinBox()
        self.patch_dist_spinbox.setMinimum(1)
        self.patch_dist_spinbox.setMaximum(100)
        self.patch_dist_spinbox.setValue(11) 
        pdist_layout.addWidget(self.patch_dist_spinbox)
        layout.addLayout(pdist_layout)

        # Fast Mode
        self.fast_mode_checkbox = QCheckBox("Use Fast Mode")
        self.fast_mode_checkbox.setChecked(True) 
        layout.addWidget(self.fast_mode_checkbox)

        # Connect signals
        self.sigma_spinbox.valueChanged.connect(self._on_parameter_or_preview_changed)
        self.h_mult_spinbox.valueChanged.connect(self._on_parameter_or_preview_changed)
        self.patch_size_spinbox.valueChanged.connect(self._on_parameter_or_preview_changed)
        self.patch_dist_spinbox.valueChanged.connect(self._on_parameter_or_preview_changed)
        self.fast_mode_checkbox.stateChanged.connect(self._on_parameter_or_preview_changed)


    def _get_current_parameters(self) -> Dict[str, Any]:
        """Gathers parameters for NL-Means Filter."""
        return {
            'sigma': self.sigma_spinbox.value(),
            'h_param_mult': self.h_mult_spinbox.value(),
            'patch_size': self.patch_size_spinbox.value(),
            'patch_distance': self.patch_dist_spinbox.value(),
            'fast_mode': self.fast_mode_checkbox.isChecked(),
            'apply_roi_only': self.apply_to_roi_only_checkbox.isChecked()
        }

    def _apply_operation(self, image: np.ndarray, params: Dict[str, Any]) -> Optional[np.ndarray]:
        """Applies denoise_nlmeans_skimage based on parameters."""
        sigma = params.get('sigma', 0.01)
        h_mult = params.get('h_param_mult', 1.0)
        p_size = params.get('patch_size', 7)
        p_dist = params.get('patch_distance', 11)
        f_mode = params.get('fast_mode', True)
        apply_roi_only = params.get('apply_roi_only', False)

        logger.debug(f"NL-Means _apply_operation called. Params: {params}")

        try:
            processed_full = denoise_nlmeans_skimage(
                image, sigma=sigma, h_param_mult=h_mult, patch_size=p_size,
                patch_distance=p_dist, fast_mode=f_mode
            )
            if processed_full is None: return None

            if apply_roi_only:
                roi_slice = self._get_roi_slice()
                if roi_slice:
                    result_image = image.copy()
                    result_image[roi_slice] = processed_full[roi_slice]
                    return result_image
                else:
                    logger.warning("Cannot apply NL-Means to ROI only: Invalid ROI.")
                    return image 
            else:
                return processed_full
        except Exception as e:
            logger.exception(f"Error applying denoise_nlmeans_skimage: {e}")
            return None


    # --- Slots  ---
    @pyqtSlot()
    def _on_parameter_or_preview_changed(self):
        is_roi_mode = self.apply_to_roi_only_checkbox.isChecked()
        self.roi.setVisible(is_roi_mode)
        self.roi_info_label.setVisible(is_roi_mode)
        if self.live_preview_checkbox.isChecked():
            self._update_preview()

    @pyqtSlot()
    def _on_roi_changed(self):
        pos=self.roi.pos(); size=self.roi.size(); info_text = f"ROI: ({pos.x():.1f}, {pos.y():.1f}) Size: ({size.x():.1f}, {size.y():.1f})"; self.roi_info_label.setText(info_text)
        if self.apply_to_roi_only_checkbox.isChecked() and self.live_preview_checkbox.isChecked():
             self._update_preview()

    def _get_roi_slice(self) -> Optional[Tuple[slice, slice]]:
        if not self.roi.isVisible() or not self.roi.size().x() > 0 or not self.roi.size().y() > 0: return None
        pos=self.roi.pos(); size=self.roi.size(); h,w=self.original_data.shape; x0,y0=int(round(pos.x())),int(round(pos.y())); width,height=int(round(size.x())),int(round(size.y())); x1=min(x0+width,w); y1=min(y0+height,h); x0=max(0,x0); y0=max(0,y0)
        if x1>x0 and y1>y0: return slice(y0,y1), slice(x0,x1)
        else: logger.warning("Invalid ROI dimensions."); return None

    def _update_preview(self):
        if not self.live_preview_checkbox.isChecked():
            self.preview_data = self.original_data.copy(); self.update_preview_view(); return
        params = self._get_current_parameters()
        logger.debug(f"Updating preview. Params: {params}")
        try:
            self.preview_data = self._apply_operation(self.original_data, params)
            if self.preview_data is None: self.preview_data = self.original_data.copy()
            self.update_preview_view()
        except Exception as e: logger.exception(f"Error during preview update: {e}")

    def update_original_view(self):
        if self.original_data is not None and self.img_original: self.img_original.setImage(self.original_data.T); self.plot_original.autoRange()
    def update_preview_view(self):
        if not self.img_processed: return
        if self.preview_data is not None: self.img_processed.setImage(self.preview_data.T); logger.debug("Preview view updated.")
        else: self.img_processed.clear(); logger.debug("Preview view cleared.")

    def accept(self):
        params = self._get_current_parameters()
        self._final_is_roi_applied_only = params.get('apply_roi_only', False)
        logger.info(f"Dialog accepted. Finalizing '{self.operation_name}'. Apply ROI Only: {self._final_is_roi_applied_only}, Params: {params}")
        try:
            self._final_processed_data = self._apply_operation(self.original_data, params)
            if self._final_processed_data is None: raise ValueError("Processing failed.")
            if np.allclose(self._final_processed_data, self.original_data): logger.info("Data not modified."); self._final_processed_data = None; super().reject(); return
            logger.info("Final processing calculated."); super().accept()
        except Exception as e: logger.exception(f"Error final processing: {e}"); QMessageBox.critical(self, "Error", f"... {e}"); self._final_processed_data = None; self._final_is_roi_applied_only = False; super().reject()

    def reject(self): logger.info(f"{self.operation_name} dialog rejected."); self._final_processed_data = None; super().reject()
    def get_processed_data(self) -> Optional[np.ndarray]: return self._final_processed_data.copy() if self._final_processed_data is not None else None
    def get_parameters(self) -> dict: return self._get_current_parameters()
    def was_roi_applied_only(self) -> bool: return self._final_is_roi_applied_only


class PlaneLevelingDialog(QDialog):
    """
    Standalone dialog window for Plane Leveling with multiple modes.
    """
    def __init__(self, original_data: np.ndarray, parent=None):
        super().__init__(parent)
        if original_data is None: raise ValueError("Original data cannot be None")

        self.operation_name = "Plane Leveling"
        self.original_data = original_data.astype(np.float32)
        self.preview_data = self.original_data.copy()
        self._final_processed_data: Optional[np.ndarray] = None
        self._final_params: Dict[str, Any] = {}
        self._final_is_roi_applied_only: bool = False

        # State for point selection
        self._is_selecting_points = False
        self._selected_points: List[Tuple[int, int]] = []
        self._mouse_click_connection = None

        self.setWindowTitle(f"{self.operation_name} Settings")
        self.setMinimumSize(900, 550) # Slightly taller for additional controls
        current_flags=self.windowFlags(); self.setWindowFlags(current_flags | Qt.WindowType.WindowMinimizeButtonHint | Qt.WindowType.WindowMaximizeButtonHint)

        # --- Layouts ---
        main_layout=QVBoxLayout(self); top_layout=QHBoxLayout(); controls_area_layout=QVBoxLayout(); bottom_layout=QHBoxLayout()

        # --- Graphics Views (like in GaussianBlurDialog) ---
        pg.setConfigOption('background', 'w'); pg.setConfigOption('foreground', 'k'); self.win = pg.GraphicsLayoutWidget()
        self.plot_original = self.win.addPlot(row=0, col=0, title="Original", name="plot_orig")
        self.img_original = ImageItem(); self.plot_original.addItem(self.img_original)
        self.plot_original.hideAxis('left'); self.plot_original.hideAxis('bottom')
        self.plot_original.setAspectLocked(True)
        self.plot_processed = self.win.addPlot(row=0, col=1, title="Preview", name="plot_proc")
        self.img_processed = ImageItem(); self.plot_processed.addItem(self.img_processed)
        self.plot_processed.hideAxis('left'); self.plot_processed.hideAxis('bottom')
        self.plot_processed.setAspectLocked(True)
        self.plot_processed.vb.setXLink(self.plot_original.vb)
        self.plot_processed.vb.setYLink(self.plot_original.vb)
        self.plot_original.vb.invertY(True); self.plot_processed.vb.invertY(True)
        top_layout.addWidget(self.win, stretch=3)

        # --- Controls Panel ---
        controls_panel = QWidget(); controls_panel.setMaximumWidth(250); controls_panel.setLayout(controls_area_layout)

        # --- Parameter Controls (Specific to Plane Leveling) ---
        self._create_parameter_controls(controls_area_layout) # Fill panel
        # -----------------------------------------------------

        controls_area_layout.addWidget(QFrame(frameShape=QFrame.Shape.HLine, frameShadow=QFrame.Shadow.Sunken))

        # --- ROI Application Mode & Live Preview ---
        self.apply_to_roi_only_checkbox = QCheckBox("Apply plane only to ROI area")
        self.apply_to_roi_only_checkbox.setChecked(False)
        self.apply_to_roi_only_checkbox.setVisible(False) # Hidden initially

        self.live_preview_checkbox = QCheckBox("Live Preview")
        self.live_preview_checkbox.setChecked(True)

        controls_area_layout.addWidget(self.apply_to_roi_only_checkbox)
        controls_area_layout.addWidget(self.live_preview_checkbox)
        # -------------------------------------------

        controls_area_layout.addWidget(QFrame(frameShape=QFrame.Shape.HLine, frameShadow=QFrame.Shadow.Sunken))

        # --- ROI Info ---
        self.roi_info_label = QLabel("ROI: Not selected")
        controls_area_layout.addWidget(self.roi_info_label)
        self.roi_info_label.setVisible(False) # Hidden initially

        controls_area_layout.addStretch()
        top_layout.addWidget(controls_panel, stretch=1)

        # --- ROI Item ---
        h, w = self.original_data.shape; roi_w, roi_h = w//4, h//4; roi_x, roi_y = w//2 - roi_w//2, h//2 - roi_h//2
        self.roi = RectROI(pos=(roi_x, roi_y), size=(roi_w, roi_h), pen=pg.mkPen('g', width=2), translateSnap=True, scaleSnap=True); # Different color for distinction
        self.plot_original.addItem(self.roi)
        self.roi.setVisible(False) # Hidden initially
        self._on_roi_changed() # Update label

        # --- Dialog Buttons ---
        self.button_box = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel); self.button_box.button(QDialogButtonBox.StandardButton.Ok).setText("Apply Changes"); bottom_layout.addWidget(self.button_box)

        # --- Assemble Layout ---
        main_layout.addLayout(top_layout); main_layout.addLayout(bottom_layout)

        # --- Initial Display & Connections ---
        self.update_original_view(); self._update_preview()
        # Connect signals
        self.live_preview_checkbox.stateChanged.connect(self._update_preview_slot) # Connect to preview update slot
        self.apply_to_roi_only_checkbox.stateChanged.connect(self._update_preview_slot) # Checkbox change also updates preview
        self.roi.sigRegionChanged.connect(self._on_roi_changed)
        self.button_box.accepted.connect(self.accept); self.button_box.rejected.connect(self.reject)

        logger.debug(f"Standalone {self.operation_name} dialog initialized.")


    # --- Implementation of UI and Logic Methods ---

    def _create_parameter_controls(self, layout: QVBoxLayout):
        """Adds controls specific to Plane Leveling."""
        mode_groupbox = QGroupBox("Leveling Mode")
        mode_layout = QVBoxLayout()
        self.rb_whole = QRadioButton("Fit to Whole Image"); self.rb_roi = QRadioButton("Fit to ROI"); self.rb_3pt = QRadioButton("Use 3 Points")
        mode_layout.addWidget(self.rb_whole); mode_layout.addWidget(self.rb_roi); mode_layout.addWidget(self.rb_3pt)
        mode_groupbox.setLayout(mode_layout); layout.addWidget(mode_groupbox)

        self.points_groupbox = QGroupBox("3 Point Selection"); points_layout = QVBoxLayout()
        self.points_label = QLabel("Click 'Select Points', then click 3 points on 'Original'.")
        self.point_coords_labels = [QLabel(f"P{i+1}: -") for i in range(3)]; self.select_points_button = QPushButton("Select Points"); self.clear_points_button = QPushButton("Clear Points"); self.clear_points_button.setEnabled(False)
        points_layout.addWidget(self.points_label); [points_layout.addWidget(lbl) for lbl in self.point_coords_labels]; btn_layout = QHBoxLayout(); btn_layout.addWidget(self.select_points_button); btn_layout.addWidget(self.clear_points_button); points_layout.addLayout(btn_layout)
        self.points_groupbox.setLayout(points_layout); layout.addWidget(self.points_groupbox); self.points_groupbox.setVisible(False)

        # Connect RadioButton signals to slot
        self.rb_whole.toggled.connect(self._on_mode_changed)
        self.rb_roi.toggled.connect(self._on_mode_changed)
        self.rb_3pt.toggled.connect(self._on_mode_changed)
        self.select_points_button.clicked.connect(self._toggle_point_selection); self.clear_points_button.clicked.connect(self._clear_points)
        self.rb_whole.setChecked(True) # Default mode


    @pyqtSlot() # Slot for live/roi_only checkboxes and parameters (in future)
    def _update_preview_slot(self):
         if self.live_preview_checkbox.isChecked():
              self._update_preview()

    @pyqtSlot(bool) # Slot for RadioButtons
    def _on_mode_changed(self):
        is_roi_mode = self.rb_roi.isChecked()
        is_3pt_mode = self.rb_3pt.isChecked()
        # Manage control visibility
        self.roi.setVisible(is_roi_mode)
        self.roi_info_label.setVisible(is_roi_mode)
        self.apply_to_roi_only_checkbox.setVisible(is_roi_mode)
        self.points_groupbox.setVisible(is_3pt_mode)
        if not is_3pt_mode and self._is_selecting_points: self._toggle_point_selection(force_off=True)
        elif not is_3pt_mode: self._clear_points()
        # Update preview after mode change
        if self.live_preview_checkbox.isChecked(): self._update_preview()

    @pyqtSlot()
    def _on_roi_changed(self):
        """Updates ROI info label and preview if live preview is on and ROI mode active."""
        pos=self.roi.pos(); size=self.roi.size(); info_text = f"ROI: ({pos.x():.1f}, {pos.y():.1f}) Size: ({size.x():.1f}, {size.y():.1f})"; self.roi_info_label.setText(info_text)
        # Update preview only if live and ROI mode
        if self.rb_roi.isChecked() and self.live_preview_checkbox.isChecked():
             self._update_preview()

    # --- Point Selection Logic ---
    def _toggle_point_selection(self, checked=False, force_off=False):
        if force_off:
            self._is_selecting_points = False
        else:
            self._is_selecting_points = not self._is_selecting_points

        if self._is_selecting_points:
            logger.info("Start 3pt select")
            self.select_points_button.setText("Stop")
            self._clear_points()
            if self._mouse_click_connection is None:
                vb = self.plot_original.getViewBox()
                scene = vb.scene()
                if scene and hasattr(scene, 'sigMouseClicked') and callable(getattr(scene, 'sigMouseClicked')):
                    self._mouse_click_connection = scene.sigMouseClicked.connect(self._handle_mouse_click)
                    logger.debug("Click connected.")
                else:
                    logger.error("Cannot connect click.")
                    self._is_selecting_points = False
                    self.select_points_button.setText("Select")
        else:
            logger.info("Stop 3pt select")
            self.select_points_button.setText("Select")
            if self._mouse_click_connection is not None:
                try:
                    vb = self.plot_original.getViewBox()
                    scene = vb.scene()
                    if scene and hasattr(scene, 'sigMouseClicked') and callable(getattr(scene, 'sigMouseClicked')):
                        scene.sigMouseClicked.disconnect(self._mouse_click_connection)
                        logger.debug("Click disconnected.")
                except Exception as e:
                    logger.warning(f"Cannot disconnect click: {e}")
                self._mouse_click_connection = None

    def _clear_points(self): self._selected_points = []; [lbl.setText(f"P{i+1}: -") for i, lbl in enumerate(self.point_coords_labels)]; self.clear_points_button.setEnabled(False); logger.info("Points cleared."); self._update_preview_slot()
    def _handle_mouse_click(self, event):
        if not self._is_selecting_points or len(self._selected_points) >= 3: return
        pos_scene = event.scenePos(); pos_data = self.plot_original.vb.mapSceneToView(pos_scene); x=int(round(pos_data.x())); y=int(round(pos_data.y())); h, w = self.original_data.shape
        if 0 <= x < w and 0 <= y < h: point = (x, y); self._selected_points.append(point); idx = len(self._selected_points); self.point_coords_labels[idx-1].setText(f"P{idx}: ({x}, {y})"); self.clear_points_button.setEnabled(True); logger.info(f"Point {idx}: ({x}, {y})")
        if len(self._selected_points) == 3: self._toggle_point_selection(force_off=True); self._update_preview_slot()
        else: logger.warning(f"Clicked outside bounds: ({x}, {y})")

    def _get_roi_slice(self) -> Optional[Tuple[slice, slice]]:
        if not self.roi.isVisible() or not self.roi.size().x() > 0 or not self.roi.size().y() > 0: return None
        pos=self.roi.pos(); size=self.roi.size(); h,w=self.original_data.shape; x0,y0=int(round(pos.x())),int(round(pos.y())); width,height=int(round(size.x())),int(round(size.y())); x1=min(x0+width,w); y1=min(y0+height,h); x0=max(0,x0); y0=max(0,y0)
        if x1>x0 and y1>y0: return slice(y0,y1), slice(x0,x1)
        else: logger.warning("Invalid ROI dimensions."); return None

    def _get_current_parameters(self) -> Dict[str, Any]:
        """Collects parameters for Plane Leveling."""
        params = {'apply_roi_only': self.apply_to_roi_only_checkbox.isChecked()}
        if self.rb_whole.isChecked(): params['mode'] = 'whole'
        elif self.rb_roi.isChecked(): params['mode'] = 'roi'
        elif self.rb_3pt.isChecked(): params['mode'] = '3pt'; params['points'] = self._selected_points.copy()
        else: params['mode'] = 'unknown'
        return params

    def _calculate_leveled_image(self, image_in: np.ndarray, params: Dict[str, Any]) -> Optional[np.ndarray]:
         """Helper method to calculate the leveled image based on params."""
         mode = params.get('mode', 'whole')
         apply_roi_only = params.get('apply_roi_only', False) and mode == 'roi'
         fitted_plane = None; roi_slice = None

         try:
             # Fit Plane
             if mode == 'whole': fitted_plane = fit_plane(image_in, roi_slice=None)
             elif mode == 'roi':
                 roi_slice = self._get_roi_slice()
                 if roi_slice: fitted_plane = fit_plane(image_in, roi_slice=roi_slice)
                 else: logger.warning("Invalid ROI for fit."); return image_in # Return original
             elif mode == '3pt':
                 points = params.get('points', [])
                 if len(points) == 3: fitted_plane = fit_plane_3pts(image_in, points)
                 else: logger.warning("Not enough points for 3pt fit."); return image_in # Return original

             if fitted_plane is None: logger.error("Plane fitting failed."); return image_in

             # Apply Leveling
             if apply_roi_only and roi_slice:
                  logger.debug("Leveling: Applying subtraction only within ROI.")
                  leveled_image = image_in.copy()
                  leveled_image[roi_slice] = image_in[roi_slice] - fitted_plane[roi_slice]
                  return leveled_image
             else:
                  logger.debug("Leveling: Applying subtraction to whole image.")
                  leveled_image = level_by_plane(image_in, fitted_plane)
                  return leveled_image

         except Exception as e:
             logger.exception(f"Error in plane leveling calculation: {e}")
             return None # Return None on calculation error


    def _update_preview(self):
        """Updates preview."""
        if not self.live_preview_checkbox.isChecked():
            self.preview_data = self.original_data.copy(); self.update_preview_view()
            return

        params = self._get_current_parameters()
        logger.debug(f"Updating preview. Params: {params}")
        # Use helper method for calculations
        calculated_preview = self._calculate_leveled_image(self.original_data, params)
        self.preview_data = calculated_preview if calculated_preview is not None else self.original_data.copy()
        self.update_preview_view()


    def update_original_view(self):
        if self.original_data is not None and self.img_original: self.img_original.setImage(self.original_data.T); self.plot_original.autoRange()
    def update_preview_view(self):
        if not self.img_processed: return
        if self.preview_data is not None: self.img_processed.setImage(self.preview_data.T); logger.debug("Preview view updated.")
        else: self.img_processed.clear(); logger.debug("Preview view cleared.")


    def accept(self):
        """Calculates final result and closes dialog."""
        params = self._get_current_parameters()
        # Check 3-point condition before calculations
        if params.get('mode') == '3pt' and len(params.get('points', [])) != 3:
            QMessageBox.warning(self, "Missing Points", "Please select exactly 3 points."); return

        self._final_is_roi_applied_only = params.get('apply_roi_only', False) and params.get('mode') == 'roi'
        logger.info(f"Dialog accepted. Finalizing '{self.operation_name}'. Apply ROI Only: {self._final_is_roi_applied_only}, Params: {params}")
        try:
            # Calculate final result
            self._final_processed_data = self._calculate_leveled_image(self.original_data, params)
            if self._final_processed_data is None: raise ValueError("Processing operation failed.")
            if np.allclose(self._final_processed_data, self.original_data): logger.info("Data not modified."); self._final_processed_data = None; super().reject(); return
            logger.info("Final processing calculated."); super().accept()
        except Exception as e: logger.exception(f"Error final processing: {e}"); QMessageBox.critical(self, "Error", f"... {e}"); self._final_processed_data = None; self._final_is_roi_applied_only = False; super().reject()

    def reject(self):
        if self._is_selecting_points: self._toggle_point_selection(force_off=True)
        logger.info(f"{self.operation_name} dialog rejected."); self._final_processed_data = None; super().reject()

    def get_processed_data(self) -> Optional[np.ndarray]: return self._final_processed_data.copy() if self._final_processed_data is not None else None
    def get_parameters(self) -> dict: return self._get_current_parameters()
    def was_roi_applied_only(self) -> bool: return self._final_is_roi_applied_only

class BM3DDialog(QDialog):
    """
    Standalone dialog window for applying BM3D denoising.
    Includes ROI/Whole image mode toggle. Live preview is disabled by default.
    """
    def __init__(self, original_data: np.ndarray, parent=None):
        super().__init__(parent)
        if original_data is None: raise ValueError("Original data cannot be None")

        self.operation_name = "BM3D Denoising"
        self.original_data = original_data.astype(np.float32)
        self.preview_data = self.original_data.copy()
        self._final_processed_data: Optional[np.ndarray] = None
        self._final_params: Dict[str, Any] = {}
        self._final_is_roi_applied_only: bool = False

        self.setWindowTitle(f"{self.operation_name} Settings")
        self.setMinimumSize(900, 550)
        current_flags=self.windowFlags(); self.setWindowFlags(current_flags | Qt.WindowType.WindowMinimizeButtonHint | Qt.WindowType.WindowMaximizeButtonHint)

        # --- Layouts ---
        main_layout=QVBoxLayout(self); top_layout=QHBoxLayout(); controls_area_layout=QVBoxLayout() 
        bottom_layout=QHBoxLayout()

        # --- Graphics Views ---
        pg.setConfigOption('background', 'w'); pg.setConfigOption('foreground', 'k')
        self.win = pg.GraphicsLayoutWidget()
        self.plot_original = self.win.addPlot(row=0, col=0, title="Original", name="plot_orig")
        self.img_original = ImageItem(); self.plot_original.addItem(self.img_original)
        self.plot_original.hideAxis('left'); self.plot_original.hideAxis('bottom')
        self.plot_original.setAspectLocked(True)
        self.plot_processed = self.win.addPlot(row=0, col=1, title="Preview (Click Update)", name="plot_proc")
        self.img_processed = ImageItem(); self.plot_processed.addItem(self.img_processed)
        self.plot_processed.hideAxis('left'); self.plot_processed.hideAxis('bottom')
        self.plot_processed.setAspectLocked(True)
        self.plot_processed.vb.setXLink(self.plot_original.vb)
        self.plot_processed.vb.setYLink(self.plot_original.vb)
        self.plot_original.vb.invertY(True); self.plot_processed.vb.invertY(True)
        top_layout.addWidget(self.win, stretch=3)

        # --- Controls Panel ---
        controls_panel = QWidget(); controls_panel.setMaximumWidth(250)
        controls_panel.setLayout(controls_area_layout)
        parameter_widget_container = QWidget()
        specific_param_layout = QVBoxLayout(parameter_widget_container)
        specific_param_layout.setContentsMargins(0,0,0,0)
        self._create_parameter_controls(specific_param_layout)
        controls_area_layout.addWidget(parameter_widget_container)

        controls_area_layout.addWidget(QFrame(frameShape=QFrame.Shape.HLine, frameShadow=QFrame.Shadow.Sunken))

        # --- ROI / Mode Controls ---
        self.apply_to_roi_only_checkbox = QCheckBox("Apply only to ROI area")
        self.apply_to_roi_only_checkbox.setChecked(False)
        # Live Preview - domyślnie wyłączone i nieaktywne
        self.live_preview_checkbox = QCheckBox("Live Preview")
        self.live_preview_checkbox.setChecked(False)
        self.live_preview_checkbox.setEnabled(False)
        self.live_preview_checkbox.setToolTip("Live preview disabled for BM3D due to performance.")
        # Przycisk do ręcznej aktualizacji podglądu
        self.update_preview_button = QPushButton("Update Preview")

        controls_area_layout.addWidget(self.apply_to_roi_only_checkbox)
        controls_area_layout.addWidget(self.live_preview_checkbox)
        controls_area_layout.addWidget(self.update_preview_button) # Dodaj przycisk

        controls_area_layout.addWidget(QFrame(frameShape=QFrame.Shape.HLine, frameShadow=QFrame.Shadow.Sunken))

        # ROI Info and Item
        self.roi_info_label = QLabel("ROI: Not selected"); controls_area_layout.addWidget(self.roi_info_label)
        h, w = self.original_data.shape; roi_w, roi_h = w//4, h//4; roi_x, roi_y = w//2-roi_w//2, h//2-roi_h//2
        self.roi = RectROI(pos=(roi_x, roi_y), size=(roi_w, roi_h), pen=pg.mkPen('r', width=2), translateSnap=True, scaleSnap=True); # Czerwone ROI
        self.plot_original.addItem(self.roi)
        is_roi_mode = self.apply_to_roi_only_checkbox.isChecked()
        self.roi.setVisible(is_roi_mode); self.roi_info_label.setVisible(is_roi_mode)
        self._on_roi_changed()

        controls_area_layout.addStretch()
        top_layout.addWidget(controls_panel, stretch=1)

        # Dialog Buttons
        self.button_box = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        self.button_box.button(QDialogButtonBox.StandardButton.Ok).setText("Apply Changes")
        bottom_layout.addWidget(self.button_box)

        # Assemble Layout
        main_layout.addLayout(top_layout); main_layout.addLayout(bottom_layout)

        # Initial Display & Connections
        self.update_original_view(); self._update_preview()
        self.apply_to_roi_only_checkbox.stateChanged.connect(self._on_settings_changed)
        self.roi.sigRegionChanged.connect(self._on_roi_changed) # ROI nadal aktualizuje etykietę
        self.update_preview_button.clicked.connect(self._update_preview) # Przycisk aktualizuje podgląd
        self.button_box.accepted.connect(self.accept); self.button_box.rejected.connect(self.reject)

        logger.debug(f"Standalone {self.operation_name} dialog initialized.")


    # --- Implementacja Metod UI i Logiki ---
    def _create_parameter_controls(self, layout: QVBoxLayout):
        """Adds controls specific to BM3D Filter."""
        sigma_layout = QHBoxLayout()
        sigma_layout.addWidget(QLabel("Noise Sigma (σ) [0-1]:")) # Etykieta wskazująca zakres
        self.sigma_spinbox = QDoubleSpinBox()
        self.sigma_spinbox.setDecimals(4)
        self.sigma_spinbox.setRange(0.0001, 1.0) # Sigma względem zakresu [0, 1]
        self.sigma_spinbox.setValue(0.05) # Typowa wartość startowa
        self.sigma_spinbox.setSingleStep(0.005)
        self.sigma_spinbox.setToolTip("Estimated noise standard deviation relative to the [0, 1] data range.")
        sigma_layout.addWidget(self.sigma_spinbox)
        layout.addLayout(sigma_layout)

        # Podłącz zmianę parametru do ogólnego slotu (chociaż live preview jest wyłączone)
        self.sigma_spinbox.valueChanged.connect(self._on_settings_changed)

    def _get_current_parameters(self) -> Dict[str, Any]:
        """Gathers parameters for BM3D Filter."""
        return {
            'sigma_psd': self.sigma_spinbox.value(),
            'apply_roi_only': self.apply_to_roi_only_checkbox.isChecked()
        }

    def _apply_operation(self, image: np.ndarray, params: Dict[str, Any]) -> Optional[np.ndarray]:
        """Applies denoise_bm3d_lfa based on parameters."""
        sigma = params.get('sigma_psd', 0.05)
        apply_roi_only = params.get('apply_roi_only', False)
        logger.debug(f"BM3D _apply_operation called. Sigma={sigma}, ROI Only={apply_roi_only}")
        try:
            # Zawsze obliczaj pełny wynik najpierw
            processed_full = denoise_bm3d_lfa(image, sigma_psd=sigma)
            if processed_full is None: return None # Błąd w funkcji backendu

            if apply_roi_only:
                roi_slice = self._get_roi_slice()
                if roi_slice:
                    result_image = image.copy()
                    result_image[roi_slice] = processed_full[roi_slice]
                    return result_image
                else: logger.warning("Cannot apply BM3D to ROI only: Invalid ROI."); return image
            else: return processed_full # Zastosuj do całości
        except Exception as e: logger.exception(f"Error applying denoise_bm3d_lfa: {e}"); return None

    # --- Sloty i Metody ---
    @pyqtSlot()
    def _on_settings_changed(self):
        """Slot for parameter or ROI checkbox changes."""
        is_roi_mode = self.apply_to_roi_only_checkbox.isChecked()
        self.roi.setVisible(is_roi_mode)
        self.roi_info_label.setVisible(is_roi_mode)

    @pyqtSlot()
    def _on_roi_changed(self):
        """Updates ROI info label."""
        pos=self.roi.pos(); size=self.roi.size(); info_text = f"ROI: ({pos.x():.1f}, {pos.y():.1f}) Size: ({size.x():.1f}, {size.y():.1f})"; self.roi_info_label.setText(info_text)

    def _get_roi_slice(self) -> Optional[Tuple[slice, slice]]:
        if not self.roi.isVisible() or not self.roi.size().x()>0 or not self.roi.size().y()>0: 
            return None
        pos=self.roi.pos()
        size=self.roi.size()
        h,w=self.original_data.shape
        x0,y0=int(round(pos.x())),int(round(pos.y()))
        width,height=int(round(size.x())),int(round(size.y()))
        x1=min(x0+width,w); y1=min(y0+height,h); x0=max(0,x0); y0=max(0,y0);
        if x1>x0 and y1>y0: 
            return slice(y0,y1), slice(x0,x1)
        else: 
            logger.warning("Invalid ROI dims.")
            return None

    @pyqtSlot()
    def _update_preview(self):
        """Calculates and updates the preview image MANUALLY."""
        params = self._get_current_parameters()
        logger.debug(f"Manually updating preview. Params: {params}")
        try:
            self.preview_data = self._apply_operation(self.original_data, params)
            if self.preview_data is None: self.preview_data = self.original_data.copy()
            self.update_preview_view()
        except Exception as e:
            logger.exception(f"Error during manual preview update: {e}")
            QMessageBox.warning(self, "Preview Error", f"Could not update preview:\n{e}")


    def update_original_view(self):
        if self.original_data is not None and self.img_original: self.img_original.setImage(self.original_data.T); self.plot_original.autoRange()
    def update_preview_view(self):
        if not self.img_processed: return
        if self.preview_data is not None: self.img_processed.setImage(self.preview_data.T); logger.debug("Preview view updated.")
        else: self.img_processed.clear(); logger.debug("Preview view cleared.")

    def accept(self):
        params = self._get_current_parameters()
        self._final_is_roi_applied_only = params.get('apply_roi_only', False)
        logger.info(f"Dialog accepted. Finalizing '{self.operation_name}'. Apply ROI Only: {self._final_is_roi_applied_only}, Params: {params}")
        try:
            self._final_processed_data = self._apply_operation(self.original_data, params)
            if self._final_processed_data is None: raise ValueError("Processing failed.")
            if np.allclose(self._final_processed_data, self.original_data): logger.info("Data not modified."); self._final_processed_data = None; super().reject(); return
            logger.info("Final processing calculated."); super().accept()
        except Exception as e: logger.exception(f"Error final processing: {e}"); QMessageBox.critical(self, "Error", f"... {e}"); self._final_processed_data = None; self._final_is_roi_applied_only = False; super().reject()


    def reject(self): logger.info(f"{self.operation_name} dialog rejected."); self._final_processed_data = None; super().reject()
    def get_processed_data(self) -> Optional[np.ndarray]: return self._final_processed_data.copy() if self._final_processed_data is not None else None
    def get_parameters(self) -> dict: return self._get_current_parameters()
    def was_roi_applied_only(self) -> bool: return self._final_is_roi_applied_only


class MedianFilterDialog(QDialog):
    """
    Standalone dialog window for applying Median Filter.
    Includes ROI/Whole image mode toggle and live preview.
    """
    def __init__(self, original_data: np.ndarray, parent=None):
        super().__init__(parent)
        if original_data is None: raise ValueError("Original data cannot be None")

        self.operation_name = "Median Filter"
        self.original_data = original_data.astype(np.float32)
        self.preview_data = self.original_data.copy()
        self._final_processed_data: Optional[np.ndarray] = None
        self._final_params: Dict[str, Any] = {}
        self._final_is_roi_applied_only: bool = False

        self.setWindowTitle(f"{self.operation_name} Settings")
        self.setMinimumSize(900, 550) # Nieco wyższe dla dodatkowych kontrolek
        current_flags=self.windowFlags(); self.setWindowFlags(current_flags | Qt.WindowType.WindowMinimizeButtonHint | Qt.WindowType.WindowMaximizeButtonHint)

        # --- Layouts ---
        main_layout=QVBoxLayout(self); top_layout=QHBoxLayout(); controls_area_layout=QVBoxLayout(); bottom_layout=QHBoxLayout()

        # --- Graphics Views ---
        pg.setConfigOption('background', 'w'); pg.setConfigOption('foreground', 'k')
        self.win = pg.GraphicsLayoutWidget()
        self.plot_original = self.win.addPlot(row=0, col=0, title="Original", name="plot_orig")
        self.img_original = ImageItem(); self.plot_original.addItem(self.img_original)
        self.plot_original.hideAxis('left'); self.plot_original.hideAxis('bottom')
        self.plot_original.setAspectLocked(True)
        self.plot_processed = self.win.addPlot(row=0, col=1, title="Preview", name="plot_proc")
        self.img_processed = ImageItem(); self.plot_processed.addItem(self.img_processed)
        self.plot_processed.hideAxis('left'); self.plot_processed.hideAxis('bottom')
        self.plot_processed.setAspectLocked(True)
        self.plot_processed.vb.setXLink(self.plot_original.vb)
        self.plot_processed.vb.setYLink(self.plot_original.vb)
        self.plot_original.vb.invertY(True); self.plot_processed.vb.invertY(True)
        top_layout.addWidget(self.win, stretch=3)

        # --- Controls Panel ---
        controls_panel = QWidget(); controls_panel.setMaximumWidth(250); controls_panel.setLayout(controls_area_layout)
        parameter_widget_container = QWidget()
        specific_param_layout = QVBoxLayout(parameter_widget_container); specific_param_layout.setContentsMargins(0,0,0,0)
        self._create_parameter_controls(specific_param_layout) # Wywołanie metody tworzącej kontrolki
        controls_area_layout.addWidget(parameter_widget_container)

        controls_area_layout.addWidget(QFrame(frameShape=QFrame.Shape.HLine, frameShadow=QFrame.Shadow.Sunken))

        # ROI and Mode Controls
        self.apply_to_roi_only_checkbox = QCheckBox("Apply only to ROI area"); self.apply_to_roi_only_checkbox.setChecked(False)
        self.live_preview_checkbox = QCheckBox("Live Preview"); self.live_preview_checkbox.setChecked(True)
        controls_area_layout.addWidget(self.apply_to_roi_only_checkbox)
        controls_area_layout.addWidget(self.live_preview_checkbox)

        controls_area_layout.addWidget(QFrame(frameShape=QFrame.Shape.HLine, frameShadow=QFrame.Shadow.Sunken))

        # ROI Info and Item
        self.roi_info_label = QLabel("ROI: Not selected"); controls_area_layout.addWidget(self.roi_info_label)
        h, w = self.original_data.shape; roi_w, roi_h = w//4, h//4; roi_x, roi_y = w//2 - roi_w//2, h//2 - roi_h//2
        self.roi = RectROI(pos=(roi_x, roi_y), size=(roi_w, roi_h), pen=pg.mkPen('y', width=2), translateSnap=True, scaleSnap=True); self.plot_original.addItem(self.roi)
        # Initial visibility based on checkbox
        is_roi_mode = self.apply_to_roi_only_checkbox.isChecked()
        self.roi.setVisible(is_roi_mode); self.roi_info_label.setVisible(is_roi_mode)
        self._on_roi_changed()

        controls_area_layout.addStretch()
        top_layout.addWidget(controls_panel, stretch=1)

        # Dialog Buttons
        self.button_box = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel); self.button_box.button(QDialogButtonBox.StandardButton.Ok).setText("Apply Changes"); bottom_layout.addWidget(self.button_box)

        # Assemble Layout
        main_layout.addLayout(top_layout); main_layout.addLayout(bottom_layout)

        # Initial Display & Connections
        self.update_original_view(); self._update_preview()
        self.apply_to_roi_only_checkbox.stateChanged.connect(self._on_parameter_or_preview_changed)
        self.live_preview_checkbox.stateChanged.connect(self._on_parameter_or_preview_changed)
        self.roi.sigRegionChanged.connect(self._on_roi_changed)
        self.button_box.accepted.connect(self.accept); self.button_box.rejected.connect(self.reject)

        logger.debug(f"Standalone {self.operation_name} dialog initialized.")


    # --- Implementacja Metod UI i Logiki ---
    def _create_parameter_controls(self, layout: QVBoxLayout):
        """Adds controls specific to Median Filter."""
        # Size control (SpinBox for odd numbers)
        size_layout = QHBoxLayout()
        self.size_label = QLabel("Filter Size:")
        self.size_spinbox = QSpinBox()
        self.size_spinbox.setMinimum(3)  # Min size 3x3
        self.size_spinbox.setMaximum(31) # Max size (adjust as needed)
        self.size_spinbox.setSingleStep(2) # Step by 2 to keep it odd
        self.size_spinbox.setValue(3)     # Default size
        size_layout.addWidget(self.size_label)
        size_layout.addWidget(self.size_spinbox)
        layout.addLayout(size_layout)

        # Mode control (ComboBox)
        mode_layout = QHBoxLayout()
        self.mode_label = QLabel("Boundary Mode:")
        self.mode_combobox = QComboBox()
        self.valid_modes = ['reflect', 'constant', 'nearest', 'mirror', 'wrap']
        self.mode_combobox.addItems(self.valid_modes)
        self.mode_combobox.setCurrentText('reflect') # Default mode
        mode_layout.addWidget(self.mode_label)
        mode_layout.addWidget(self.mode_combobox)
        layout.addLayout(mode_layout)

        # Cval control (DoubleSpinBox, enabled only for 'constant' mode)
        cval_layout = QHBoxLayout()
        self.cval_label = QLabel("Constant Value (cval):")
        self.cval_spinbox = QDoubleSpinBox() # Use QDoubleSpinBox for float
        self.cval_spinbox.setRange(-1e6, 1e6) # Set appropriate range
        self.cval_spinbox.setDecimals(3)
        self.cval_spinbox.setValue(0.0)
        self.cval_label.setVisible(False) # Initially hidden
        self.cval_spinbox.setVisible(False) # Initially hidden
        cval_layout.addWidget(self.cval_label)
        cval_layout.addWidget(self.cval_spinbox)
        layout.addLayout(cval_layout)

        # Connect signals for parameter changes
        self.size_spinbox.valueChanged.connect(self._on_parameter_or_preview_changed)
        self.mode_combobox.currentIndexChanged.connect(self._on_mode_combobox_changed) # Separate slot for mode
        self.cval_spinbox.valueChanged.connect(self._on_parameter_or_preview_changed)


    @pyqtSlot(int)
    def _on_mode_combobox_changed(self, index):
        """Handles mode change, enables/disables cval control."""
        selected_mode = self.mode_combobox.itemText(index)
        is_constant_mode = (selected_mode == 'constant')
        self.cval_label.setVisible(is_constant_mode)
        self.cval_spinbox.setVisible(is_constant_mode)
        # Trigger preview update
        self._on_parameter_or_preview_changed()

    def _get_current_parameters(self) -> Dict[str, Any]:
        """Gathers parameters for Median Filter."""
        return {
            'size': self.size_spinbox.value(),
            'mode': self.mode_combobox.currentText(),
            'cval': self.cval_spinbox.value(),
            'apply_roi_only': self.apply_to_roi_only_checkbox.isChecked()
        }

    def _apply_operation(self, image: np.ndarray, params: Dict[str, Any]) -> Optional[np.ndarray]:
        """Applies median_filter_lfa based on parameters."""
        size = params.get('size', 3)
        mode = params.get('mode', 'reflect')
        cval = params.get('cval', 0.0)
        apply_roi_only = params.get('apply_roi_only', False)
        logger.debug(f"Median Filter _apply_operation: size={size}, mode='{mode}', cval={cval}, ROI Only={apply_roi_only}")

        try:
            # Always calculate the filter effect on the whole image first
            processed_full = median_filter_lfa(image, size=size, mode=mode, cval=cval)
            if processed_full is None: return None # Handle backend error

            if apply_roi_only:
                roi_slice = self._get_roi_slice()
                if roi_slice:
                    result_image = image.copy()
                    result_image[roi_slice] = processed_full[roi_slice]
                    return result_image
                else:
                    logger.warning("Cannot apply Median Filter to ROI only: Invalid ROI.")
                    return image # Return original if ROI invalid
            else:
                # Apply to whole image
                return processed_full
        except Exception as e:
            logger.exception(f"Error applying median_filter_lfa: {e}")
            return None

    # --- Sloty i Metody skopiowane/zaadaptowane (jak w GaussianBlurDialog) ---
    # Te metody są teraz częścią tej klasy, nie dziedziczone
    @pyqtSlot()
    def _on_parameter_or_preview_changed(self):
        is_roi_mode = self.apply_to_roi_only_checkbox.isChecked()
        self.roi.setVisible(is_roi_mode)
        self.roi_info_label.setVisible(is_roi_mode)
        if self.live_preview_checkbox.isChecked():
            self._update_preview()

    @pyqtSlot()
    def _on_roi_changed(self):
        pos=self.roi.pos(); size=self.roi.size(); info_text = f"ROI: ({pos.x():.1f}, {pos.y():.1f}) Size: ({size.x():.1f}, {size.y():.1f})"; self.roi_info_label.setText(info_text)
        if self.apply_to_roi_only_checkbox.isChecked() and self.live_preview_checkbox.isChecked():
             self._update_preview()

    def _get_roi_slice(self) -> Optional[Tuple[slice, slice]]:
        if not self.roi.isVisible() or not self.roi.size().x() > 0 or not self.roi.size().y() > 0: return None
        pos=self.roi.pos(); size=self.roi.size(); h,w=self.original_data.shape; x0,y0=int(round(pos.x())),int(round(pos.y())); width,height=int(round(size.x())),int(round(size.y())); x1=min(x0+width,w); y1=min(y0+height,h); x0=max(0,x0); y0=max(0,y0)
        if x1>x0 and y1>y0: return slice(y0,y1), slice(x0,x1)
        else: logger.warning("Invalid ROI dimensions."); return None

    def _update_preview(self):
        if not self.live_preview_checkbox.isChecked():
            self.preview_data = self.original_data.copy(); self.update_preview_view(); return
        params = self._get_current_parameters()
        logger.debug(f"Updating preview. Params: {params}")
        try:
            self.preview_data = self._apply_operation(self.original_data, params)
            if self.preview_data is None: self.preview_data = self.original_data.copy()
            self.update_preview_view()
        except Exception as e: logger.exception(f"Error during preview update: {e}")

    def update_original_view(self):
        if self.original_data is not None and self.img_original: self.img_original.setImage(self.original_data.T); self.plot_original.autoRange()
    def update_preview_view(self):
        if not self.img_processed: return
        if self.preview_data is not None: self.img_processed.setImage(self.preview_data.T); logger.debug("Preview view updated.")
        else: self.img_processed.clear(); logger.debug("Preview view cleared.")

    def accept(self):
        params = self._get_current_parameters()
        self._final_is_roi_applied_only = params.get('apply_roi_only', False)
        logger.info(f"Dialog accepted. Finalizing '{self.operation_name}'. Apply ROI Only: {self._final_is_roi_applied_only}, Params: {params}")
        try:
            self._final_processed_data = self._apply_operation(self.original_data, params)
            if self._final_processed_data is None: raise ValueError("Processing failed.")
            if np.allclose(self._final_processed_data, self.original_data): logger.info("Data not modified."); self._final_processed_data = None; super().reject(); return
            logger.info("Final processing calculated."); super().accept()
        except Exception as e: logger.exception(f"Error final processing: {e}"); QMessageBox.critical(self, "Error", f"... {e}"); self._final_processed_data = None; self._final_is_roi_applied_only = False; super().reject()

    def reject(self): logger.info(f"{self.operation_name} dialog rejected."); self._final_processed_data = None; super().reject()
    def get_processed_data(self) -> Optional[np.ndarray]: return self._final_processed_data.copy() if self._final_processed_data is not None else None
    def get_parameters(self) -> dict: return self._get_current_parameters()
    def was_roi_applied_only(self) -> bool: return self._final_is_roi_applied_only

