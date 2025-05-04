# lfa/gui/preprocessing_dialogs.py
import logging
import abc # Import Abstract Base Classes module
import numpy as np
from typing import Optional, Tuple, Dict, Any

try:
    from PyQt6.QtWidgets import (
        QDialog, QVBoxLayout, QHBoxLayout, QSlider, QLabel, QCheckBox,
        QDialogButtonBox, QWidget, QSizePolicy, QSpacerItem, QFrame, QMessageBox # Add QFrame
    )
    from PyQt6.QtCore import Qt, pyqtSlot, QRectF, QPointF
    import pyqtgraph as pg
    from pyqtgraph import PlotItem, RectROI, ROI, ImageItem # Make sure ImageItem is imported
except ImportError as e:
    logging.critical(f"Failed to import necessary Qt or pyqtgraph modules: {e}")
    raise

# Import processing functions that will be used
try:
    from ..preprocessing.filtering import gaussian_blur
    # Add other function imports as they are implemented
    # from ..preprocessing.leveling import plane_level
except ImportError:
    logging.error("Could not import preprocessing functions.")
    # Define dummy functions if needed for development without backend
    def gaussian_blur(image, sigma): return image
    # def plane_level(image, points): return image


logger = logging.getLogger(__name__)

# --- Base Class for Preprocessing Dialogs ---

class BasePreprocessingDialog(QDialog):
    """
    Abstract base class for preprocessing operation dialogs.

    Provides common UI elements (original/preview views, ROI, mode checkboxes)
    and handles the core logic for live preview and applying changes based on mode.
    Subclasses must implement methods to create parameter controls and apply
    the specific operation.
    """
    __metaclass__ = abc.ABCMeta # Define as abstract base class

    def __init__(self, original_data: np.ndarray, operation_name: str, parent=None):
        """
        Initializes the base dialog.

        Args:
            original_data (np.ndarray): The input image data.
            operation_name (str): Name of the operation for the window title.
            parent: Parent widget.
        """
        super().__init__(parent)
        if original_data is None: raise ValueError("Original data cannot be None")

        self.operation_name = operation_name
        self.original_data = original_data.astype(np.float32)
        self.preview_data = self.original_data.copy()
        self._final_processed_data: Optional[np.ndarray] = None
        self._final_is_roi: bool = False

        self._initial_levels_set = False

        self.setWindowTitle(f"{self.operation_name} Settings")
        self.setMinimumSize(900, 500)

        current_flags = self.windowFlags()

        self.setWindowFlags(current_flags |
                            Qt.WindowType.WindowMinimizeButtonHint |
                            Qt.WindowType.WindowMaximizeButtonHint)

        # --- Main Layouts ---
        main_layout = QVBoxLayout(self)
        top_layout = QHBoxLayout()       # For images and controls panel
        controls_area_layout = QVBoxLayout() # Vertical layout for the entire controls panel
        self.param_controls_layout = QVBoxLayout() # Layout specific for parameters (filled by subclass)
        mode_controls_layout = QVBoxLayout()   # Layout for mode checkboxes
        roi_info_layout = QVBoxLayout()      # Layout for ROI info
        bottom_layout = QHBoxLayout()    # For OK/Cancel buttons

        # --- Graphics Layout for Images ---
        pg.setConfigOption('background', 'w'); pg.setConfigOption('foreground', 'k')
        self.win = pg.GraphicsLayoutWidget()
        self.plot_original = self.win.addPlot(row=0, col=0, title="Original (Select ROI here)", name="plot_orig")
        self.img_original = ImageItem(); self.plot_original.addItem(self.img_original)
        self.plot_original.hideAxis('left'); self.plot_original.hideAxis('bottom'); self.plot_original.setAspectLocked(True)
        self.plot_processed = self.win.addPlot(row=0, col=1, title="Preview", name="plot_proc")
        self.img_processed = ImageItem(); self.plot_processed.addItem(self.img_processed)
        self.plot_processed.hideAxis('left'); self.plot_processed.hideAxis('bottom'); self.plot_processed.setAspectLocked(True)
        self.plot_processed.vb.setXLink(self.plot_original.vb); self.plot_processed.vb.setYLink(self.plot_original.vb)
        top_layout.addWidget(self.win, stretch=3)

        # --- Controls Panel Structure ---
        controls_panel = QWidget(); controls_panel.setMaximumWidth(250)
        controls_panel.setLayout(controls_area_layout)

        parameter_widget_container = QWidget()

        self.specific_param_layout = QVBoxLayout(parameter_widget_container)
        self.specific_param_layout.setContentsMargins(0,0,0,0) 


        self._create_parameter_controls(self.specific_param_layout)

        controls_area_layout.addWidget(parameter_widget_container)

        # Separator line
        line1 = QFrame(); line1.setFrameShape(QFrame.Shape.HLine); line1.setFrameShadow(QFrame.Shadow.Sunken)
        controls_area_layout.addWidget(line1)

        # Mode Controls
        self.roi_mode_checkbox = QCheckBox("Process ROI Only")
        self.roi_mode_checkbox.setChecked(False)
        self.live_preview_checkbox = QCheckBox("Live Preview")
        self.live_preview_checkbox.setChecked(True)
        mode_controls_layout.addWidget(self.roi_mode_checkbox)
        mode_controls_layout.addWidget(self.live_preview_checkbox)
        controls_area_layout.addLayout(mode_controls_layout)

        # Separator line
        line2 = QFrame(); line2.setFrameShape(QFrame.Shape.HLine); line2.setFrameShadow(QFrame.Shadow.Sunken)
        controls_area_layout.addWidget(line2)

        # ROI Info
        self.roi_info_label = QLabel("ROI: Not selected")
        roi_info_layout.addWidget(self.roi_info_label)
        controls_area_layout.addLayout(roi_info_layout)
        self.roi_info_label.setVisible(self.roi_mode_checkbox.isChecked()) # Initial visibility

        controls_area_layout.addStretch() # Spacer at the bottom
        top_layout.addWidget(controls_panel, stretch=1)

        # --- ROI Item ---
        h, w = self.original_data.shape
        roi_w, roi_h = w // 4, h // 4; roi_x, roi_y = w // 2 - roi_w // 2, h // 2 - roi_h // 2
        self.roi = RectROI(pos=(roi_x, roi_y), size=(roi_w, roi_h), pen=pg.mkPen('y', width=2), translateSnap=True, scaleSnap=True)
        self.plot_original.addItem(self.roi)
        self.roi.setVisible(self.roi_mode_checkbox.isChecked()) # Initial visibility
        self._on_roi_changed() # Update label

        # --- Dialog Buttons ---
        self.button_box = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        self.button_box.button(QDialogButtonBox.StandardButton.Ok).setText("Apply Changes")
        bottom_layout.addWidget(self.button_box)

        # --- Assemble Main Layout ---
        main_layout.addLayout(top_layout)
        main_layout.addLayout(bottom_layout)

        # --- Initial Display ---
        self.update_original_view()
        self._update_preview() # Initial preview update
        

        # --- Connect Base Signals ---
        self.live_preview_checkbox.stateChanged.connect(self._on_parameter_or_preview_changed)
        self.roi_mode_checkbox.stateChanged.connect(self._on_mode_changed)
        self.roi.sigRegionChanged.connect(self._on_roi_changed)
        self.button_box.accepted.connect(self.accept)
        self.button_box.rejected.connect(self.reject)

        self.plot_original.vb.invertY(True)
        self.plot_processed.vb.invertY(True)

        logger.debug(f"BasePreprocessingDialog for '{self.operation_name}' initialized.")

    # --- Abstract Methods (to be implemented by subclasses) ---
    @abc.abstractmethod
    def _create_parameter_controls(self, layout: QVBoxLayout):
        """Subclasses must implement this to add their specific parameter controls."""
        raise NotImplementedError

    @abc.abstractmethod
    def _get_current_parameters(self) -> Dict[str, Any]:
        """Subclasses must implement this to return current parameter values."""
        raise NotImplementedError

    @abc.abstractmethod
    def _apply_operation(self, image: np.ndarray, params: Dict[str, Any]) -> np.ndarray:
        """Subclasses must implement this to call their specific processing function."""
        raise NotImplementedError

    # --- Common Slots and Methods ---
    @pyqtSlot()
    def _on_parameter_or_preview_changed(self):
        """Slot called when any parameter control OR live preview checkbox changes."""
        # Subclasses should connect their parameter widget signals (like valueChanged) here.
        # This base implementation only reacts to the live preview checkbox state.
        if self.live_preview_checkbox.isChecked():
            self._update_preview()

    @pyqtSlot(int)
    def _on_mode_changed(self, state):
        """Handles changes to the 'Process ROI Only' checkbox."""
        is_roi_mode = state == Qt.CheckState.Checked.value
        self.roi.setVisible(is_roi_mode)
        self.roi_info_label.setVisible(is_roi_mode)
        if self.live_preview_checkbox.isChecked():
            self._update_preview() # Update preview based on new mode
        logger.debug(f"Processing mode changed. ROI mode active: {is_roi_mode}")

    @pyqtSlot()
    def _on_roi_changed(self):
        """Updates ROI info label and potentially live preview."""
        pos = self.roi.pos(); size = self.roi.size()
        info_text = f"ROI: ({pos.x():.1f}, {pos.y():.1f}) Size: ({size.x():.1f}, {size.y():.1f})"
        self.roi_info_label.setText(info_text)
        if self.roi_mode_checkbox.isChecked() and self.live_preview_checkbox.isChecked():
             self._update_preview()

    def _get_roi_slice(self) -> Optional[Tuple[slice, slice]]:
        pos = self.roi.pos(); size = self.roi.size(); h, w = self.original_data.shape
        x0, y0 = int(round(pos.x())), int(round(pos.y())); width, height = int(round(size.x())), int(round(size.y()))
        x1 = min(x0 + width, w); y1 = min(y0 + height, h); x0 = max(0, x0); y0 = max(0, y0)
        if x1 > x0 and y1 > y0: row_slice = slice(y0, y1); col_slice = slice(x0, x1); return row_slice, col_slice
        else: logger.warning("Invalid ROI dimensions after clamping/rounding."); return None

    def _update_preview(self):
        """Calculates and updates the preview image based on current settings."""
        if not self.live_preview_checkbox.isChecked():
            # If live preview is off, ensure the preview shows the original data
            # (as no changes have been "applied" yet in this model)
            self.preview_data = self.original_data.copy()
            self.update_preview_view()
            logger.debug("Live preview off. Preview shows original.")
            return

        params = self._get_current_parameters()
        is_roi_mode = self.roi_mode_checkbox.isChecked()
        logger.debug(f"Updating preview. ROI mode: {is_roi_mode}, Params: {params}")

        try:
            base_image = self.original_data # Preview always based on original
            # Calculate the effect of the operation on the *whole* base image
            processed_full = self._apply_operation(base_image, params)

            if is_roi_mode:
                roi_slice = self._get_roi_slice()
                if roi_slice:
                    # Start preview with original data
                    self.preview_data = base_image.copy()
                    # Apply processed data *only* within the ROI slice for preview
                    self.preview_data[roi_slice] = processed_full[roi_slice]
                else: # Invalid ROI, show original in preview
                    self.preview_data = base_image.copy()
            else: # Whole image mode
                self.preview_data = processed_full

            self.update_preview_view() # Display self.preview_data

        except Exception as e:
            logger.exception(f"Error during preview update: {e}")

    def update_original_view(self):
        if self.original_data is not None and self.img_original:
            self.img_original.setImage(self.original_data.T); self.plot_original.autoRange()

    def update_preview_view(self):
        """Updates the 'Preview' image view."""
        if not self.img_processed: return
        if self.preview_data is not None:
            self.img_processed.setImage(self.preview_data.T)
            logger.debug("Preview view updated.")
        else:
             self.img_processed.clear()
             logger.debug("Preview view cleared (no data).")

    def accept(self):
        """Calculate final result based on mode/parameters and close dialog."""
        params = self._get_current_parameters()
        is_roi_mode = self.roi_mode_checkbox.isChecked()
        self._final_is_roi = is_roi_mode

        logger.info(f"Dialog accepted. Finalizing '{self.operation_name}'. ROI mode: {is_roi_mode}, Params: {params}")
        try:
            base_image = self.original_data
            processed_full = self._apply_operation(base_image, params)

            if is_roi_mode:
                roi_slice = self._get_roi_slice()
                if roi_slice and (processed_full.shape == base_image.shape): # Check processing didn't fail
                    self._final_processed_data = base_image.copy()
                    self._final_processed_data[roi_slice] = processed_full[roi_slice]
                else:
                    logger.warning("Cannot apply ROI mode with invalid ROI or processing failed. Returning original.")
                    self._final_processed_data = base_image.copy()
                    self._final_is_roi = False
            else: # Whole image mode
                self._final_processed_data = processed_full

            # Check if data actually changed compared to original
            if np.allclose(self._final_processed_data, self.original_data):
                logger.info("Data was not significantly modified. Treating as cancel.")
                self._final_processed_data = None # Signal no change occurred
                super().reject() # Treat as cancel if no change
                return

            logger.info("Final processing calculated.")
            super().accept()

        except Exception as e:
            logger.exception(f"Error during final processing calculation: {e}")
            QMessageBox.critical(self, "Processing Error", f"Failed to calculate final result:\n{e}")
            super().reject()

    def reject(self):
        logger.info(f"{self.operation_name} dialog rejected (Cancel clicked).")
        self._final_processed_data = None
        super().reject()

    def get_processed_data(self) -> Optional[np.ndarray]:
        return self._final_processed_data.copy() if self._final_processed_data is not None else None

    def get_parameters(self) -> dict:
        # Return parameters at the moment OK was clicked
        return self._get_current_parameters()

    def was_roi_applied(self) -> bool:
        """Returns True if the final accepted result was ROI-based."""
        return self._final_is_roi


# --- Concrete Implementation: Gaussian Blur Dialog ---

class GaussianBlurDialog(BasePreprocessingDialog):
    """Concrete dialog for Gaussian Blur, inheriting from BasePreprocessingDialog."""

    def __init__(self, original_data: np.ndarray, parent=None):
        super().__init__(original_data, "Gaussian Blur", parent)

    # 1. Implement parameter controls creation
    def _create_parameter_controls(self, layout: QVBoxLayout):
        """
        Adds controls specific to Gaussian Blur (sigma slider) to the provided layout.

        Args:
            layout (QVBoxLayout): The layout provided by the base class to add controls to.
        """
        # Create a *local* horizontal layout for these specific controls
        sigma_controls_layout = QHBoxLayout()

        self.sigma_label = QLabel(f"Sigma: {0.0:.1f}")
        self.sigma_slider = QSlider(Qt.Orientation.Horizontal)
        self.sigma_slider.setMinimum(0); self.sigma_slider.setMaximum(100); self.sigma_slider.setValue(0)
        self.sigma_slider.setTickInterval(10); self.sigma_slider.setTickPosition(QSlider.TickPosition.TicksBelow)

        # Add widgets to the local horizontal layout
        sigma_controls_layout.addWidget(QLabel("Sigma:"))
        sigma_controls_layout.addWidget(self.sigma_slider)
        sigma_controls_layout.addWidget(self.sigma_label)

        # Connect the slider's signal to the slot that handles parameter changes
        self.sigma_slider.valueChanged.connect(self._on_specific_parameter_changed)

        # *** Add the local layout to the main parameter layout provided by the base class ***
        layout.addLayout(sigma_controls_layout)

    # Slot to update label and trigger preview update via base method
    @pyqtSlot(int)
    def _on_specific_parameter_changed(self, value):
        """Updates the sigma label AND calls the base class handler."""
        sigma = value / 10.0
        self.sigma_label.setText(f"Sigma: {sigma:.1f}")
        # Call the base class method that handles live preview logic
        super()._on_parameter_or_preview_changed()

    # 2. Implement parameter retrieval
    def _get_current_parameters(self) -> Dict[str, Any]:
        """Returns the current sigma value."""
        sigma = self.sigma_slider.value() / 10.0
        return {'sigma': round(sigma, 2)}

    # 3. Implement the operation call
    def _apply_operation(self, image: np.ndarray, params: Dict[str, Any]) -> np.ndarray:
        """Applies the gaussian_blur function."""
        sigma = params.get('sigma', 0.0)
        # Ensure the imported gaussian_blur is used correctly
        try:
             from ..preprocessing.filtering import gaussian_blur as gb_func
             return gb_func(image, sigma)
        except ImportError:
             logger.error("gaussian_blur function could not be re-imported in _apply_operation.")
             return image # Return original if import fails here