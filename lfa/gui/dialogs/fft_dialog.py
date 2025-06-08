# lfa/gui/fft_dialog.py
"""
Dialog window for calculating and previewing Fast Fourier Transform (FFT).

This module provides a dialog interface for performing FFT calculations on STM image data.
Features include:
- FFT calculation on full image or selected ROI
- Window function application options
- Live preview with various scaling modes
- Interactive ROI selection
- Support for different magnitude scaling methods (log, linear, power, sqrt)
"""
import logging
import numpy as np
from typing import Optional, Tuple, Dict, Any

try:
    # Import necessary PyQt6 components
    from PyQt6.QtWidgets import (
        QDialog, QVBoxLayout, QHBoxLayout, QCheckBox, QComboBox,
        QDialogButtonBox, QWidget, QSizePolicy, QSpacerItem, QFrame, QMessageBox,
        QLabel, QPushButton, QGroupBox, QRadioButton, QSplitter
    )
    from PyQt6.QtCore import Qt, pyqtSlot
    # Import necessary pyqtgraph components
    import pyqtgraph as pg
    from pyqtgraph import PlotItem, RectROI, ROI, ImageItem, ImageView
except ImportError as e:
    logging.critical(f"Failed to import necessary Qt or pyqtgraph modules: {e}")
    # Depending on how critical this is, you might exit or disable features
    raise # Re-raise for now

# Import the backend FFT calculation function and available window types
try:
    from lfa.analysis.fft_engine import calculate_fft, AVAILABLE_WINDOWS
except ImportError:
    logging.error("Could not import calculate_fft function from lfa.analysis.fft_engine.")
    # Define a dummy function to allow the UI to load, but calculations will fail
    def calculate_fft(image_data, apply_window=True, window_type='hann') -> Optional[np.ndarray]:
        logging.error("calculate_fft backend function is missing!")
        return None # Return None to indicate failure
    AVAILABLE_WINDOWS = {'none': None} # Placeholder

logger = logging.getLogger(__name__)

class FFTDialog(QDialog):
    """
    Standalone dialog for FFT Calculation with scaling options.
    
    This dialog provides a comprehensive interface for performing FFT calculations
    on STM image data with the following features:
    - Interactive ROI selection for partial FFT calculation
    - Window function application options
    - Multiple magnitude scaling modes (log, linear, power, sqrt)
    - Live preview of FFT results
    - Support for both full image and ROI-based calculations
    
    Attributes:
        input_data (np.ndarray): The input STM image data
        preview_display_data (Optional[np.ndarray]): Scaled magnitude for preview
        _final_processed_data (Optional[np.ndarray]): Final scaled magnitude result
        _final_params (Dict[str, Any]): Parameters used for final calculation
        _final_source_roi_slice (Optional[Tuple[slice, slice]]): ROI slice if used
    """

    def __init__(self, input_stm_data: np.ndarray, parent=None):
        """
        Initialize the FFT dialog.
        
        Args:
            input_stm_data: 2D numpy array containing the STM image data
            parent: Parent widget for the dialog
        """
        super().__init__(parent)
        if input_stm_data is None or input_stm_data.ndim != 2:
            raise ValueError("FFTDialog requires valid 2D input_stm_data")

        self.operation_name = "FFT Calculation"
        self.input_data = input_stm_data.astype(np.float32)
        # Stores the scaled magnitude for preview display
        self.preview_display_data: Optional[np.ndarray] = None
        # Stores the final scaled magnitude result after accept
        self._final_processed_data: Optional[np.ndarray] = None
        self._final_params: Dict[str, Any] = {}
        self._final_source_roi_slice: Optional[Tuple[slice, slice]] = None

        self.setWindowTitle(self.operation_name)
        self.setMinimumSize(950, 550)
        current_flags=self.windowFlags()
        self.setWindowFlags(current_flags | Qt.WindowType.WindowMinimizeButtonHint | Qt.WindowType.WindowMaximizeButtonHint)

        # --- Layouts ---
        main_layout=QVBoxLayout(self)
        top_layout=QHBoxLayout()
        controls_area_layout=QVBoxLayout()
        bottom_layout=QHBoxLayout()

        # --- Splitter and Views ---
        splitter = QSplitter(Qt.Orientation.Horizontal)
        # Left Panel: Input Image + ROI (using GraphicsLayoutWidget for PlotItem)
        left_widget = pg.GraphicsLayoutWidget()
        self.plot_input = left_widget.addPlot(title="Input Data (Select ROI here)")
        self.img_input = ImageItem(); self.plot_input.addItem(self.img_input)
        self.plot_input.hideAxis('left'); self.plot_input.hideAxis('bottom')
        self.plot_input.setAspectLocked(True); self.plot_input.vb.invertY(True)
        splitter.addWidget(left_widget)
        # Right Panel: FFT Preview (using ImageView)
        self.fft_image_view = ImageView(self)
        self.fft_image_view.ui.menuBtn.hide(); self.fft_image_view.ui.roiBtn.hide()
        self.fft_image_view.getView().invertY(False) # No Y inversion for FFT
        splitter.addWidget(self.fft_image_view)
        top_layout.addWidget(splitter, stretch=3)

        # --- ROI Item ---
        h, w = self.input_data.shape; roi_w, roi_h = w//4, h//4; roi_x, roi_y = w//2-roi_w//2, h//2-roi_h//2
        self.roi = RectROI(pos=(roi_x, roi_y), size=(roi_w, roi_h), pen=pg.mkPen('m', width=2), translateSnap=True, scaleSnap=True)
        self.plot_input.addItem(self.roi)

        # --- Controls Panel ---
        controls_panel = QWidget(); controls_panel.setMaximumWidth(250); controls_panel.setLayout(controls_area_layout)
        self._create_parameter_controls(controls_area_layout) # Populate controls
        controls_area_layout.addStretch()
        top_layout.addWidget(controls_panel, stretch=1)

        # --- Dialog Buttons ---
        self.button_box = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel); self.button_box.button(QDialogButtonBox.StandardButton.Ok).setText("Apply FFT"); bottom_layout.addWidget(self.button_box)

        # --- Assemble Layout ---
        main_layout.addLayout(top_layout); main_layout.addLayout(bottom_layout)
        splitter.setSizes([450, 500])

        # --- Initial State & Connections ---
        self.update_input_view()
        self._on_mode_changed() # Set initial visibility of ROI controls
        self._update_preview() # Calculate initial preview

        # Connect signals
        self.roi_mode_checkbox.stateChanged.connect(self._on_mode_changed)
        self.live_preview_checkbox.stateChanged.connect(self._update_preview_slot)
        self.window_checkbox.stateChanged.connect(self._on_parameter_changed)
        self.window_combo.currentIndexChanged.connect(self._on_parameter_changed)
        self.rb_log.toggled.connect(self._on_parameter_changed)
        self.rb_linear.toggled.connect(self._on_parameter_changed)
        self.rb_power.toggled.connect(self._on_parameter_changed)
        self.rb_sqrt.toggled.connect(self._on_parameter_changed)
        self.roi.sigRegionChanged.connect(self._on_roi_changed)
        self.button_box.accepted.connect(self.accept)
        self.button_box.rejected.connect(self.reject)

        logger.debug(f"Standalone {self.operation_name} dialog initialized.")


    def _create_parameter_controls(self, layout: QVBoxLayout):
        """
        Creates and adds controls specific to FFT calculation.
        
        This method sets up:
        - ROI mode selection
        - Window function controls
        - Display scaling options
        - Live preview toggle
        """
        self.roi_mode_checkbox = QCheckBox("Calculate FFT only for ROI")
        self.roi_mode_checkbox.setChecked(False)
        layout.addWidget(self.roi_mode_checkbox)
        self.roi_info_label = QLabel("ROI: Not selected")
        layout.addWidget(self.roi_info_label)

        layout.addWidget(QFrame(frameShape=QFrame.Shape.HLine, frameShadow=QFrame.Shadow.Sunken))

        self.window_checkbox = QCheckBox("Apply Window Function")
        self.window_checkbox.setChecked(True)
        self.window_combo = QComboBox()
        self.window_options = ['None'] + sorted([k for k, v in AVAILABLE_WINDOWS.items() if v is not None])
        self.window_combo.addItems(self.window_options)
        default_window = 'hann'
        if default_window in self.window_options: self.window_combo.setCurrentText(default_window)
        elif len(self.window_options) > 1: self.window_combo.setCurrentIndex(1)
        self.window_combo.setEnabled(self.window_checkbox.isChecked())
        layout.addWidget(self.window_checkbox)
        layout.addWidget(self.window_combo)

        layout.addWidget(QFrame(frameShape=QFrame.Shape.HLine, frameShadow=QFrame.Shadow.Sunken))

        scale_groupbox = QGroupBox("Display Scaling")
        scale_layout = QVBoxLayout()
        self.rb_log = QRadioButton("Log Magnitude (log1p(abs))"); self.rb_log.setChecked(True)
        self.rb_linear = QRadioButton("Linear Magnitude (abs)")
        self.rb_power = QRadioButton("Power Spectrum (abs^2)")
        self.rb_sqrt = QRadioButton("Sqrt Magnitude (sqrt(abs))")
        scale_layout.addWidget(self.rb_log); scale_layout.addWidget(self.rb_linear)
        scale_layout.addWidget(self.rb_power); scale_layout.addWidget(self.rb_sqrt)
        scale_groupbox.setLayout(scale_layout); layout.addWidget(scale_groupbox)

        layout.addWidget(QFrame(frameShape=QFrame.Shape.HLine, frameShadow=QFrame.Shadow.Sunken))
        self.live_preview_checkbox = QCheckBox("Live Preview"); self.live_preview_checkbox.setChecked(True); layout.addWidget(self.live_preview_checkbox)


    # --- Slots and Helper Methods ---

    @pyqtSlot()
    def _on_parameter_changed(self):
        """
        Handles changes in windowing or scaling settings.
        Updates the preview if live preview is enabled.
        """
        if hasattr(self, 'window_combo') and hasattr(self, 'window_checkbox'):
            self.window_combo.setEnabled(self.window_checkbox.isChecked())
        self._update_preview_slot() # Trigger preview update

    @pyqtSlot(int)
    def _on_mode_changed(self, state=None):
        """
        Handles changes to the 'Calculate FFT only for ROI' checkbox.
        Updates ROI visibility and preview accordingly.
        """
        is_roi_mode = self.roi_mode_checkbox.isChecked()
        self.roi.setVisible(is_roi_mode)
        self.roi_info_label.setVisible(is_roi_mode)
        self._update_preview_slot() # Trigger preview update
        logger.debug(f"FFT mode changed. ROI mode active: {is_roi_mode}")

    @pyqtSlot()
    def _on_roi_changed(self):
        """
        Updates ROI info label and preview if ROI mode and live preview are active.
        """
        pos=self.roi.pos(); size=self.roi.size()
        info_text = f"ROI: ({pos.x():.1f}, {pos.y():.1f}) Size: ({size.x():.1f}, {size.y():.1f})"
        self.roi_info_label.setText(info_text)
        # Update preview only if ROI mode AND live preview are active
        if self.roi_mode_checkbox.isChecked() and self.live_preview_checkbox.isChecked():
             self._update_preview()

    @pyqtSlot()
    def _update_preview_slot(self):
         """
         Triggers preview update if live preview checkbox is checked.
         """
         if self.live_preview_checkbox.isChecked():
              self._update_preview()

    def _get_roi_slice(self) -> Optional[Tuple[slice, slice]]:
        """
        Calculates integer numpy slices from the current ROI state.
        
        Returns:
            Optional[Tuple[slice, slice]]: Tuple of (y_slice, x_slice) if ROI is valid,
                                         None otherwise
        """
        if not self.roi.isVisible() or not self.roi.size().x()>0 or not self.roi.size().y()>0:
            return None
        pos=self.roi.pos()
        size=self.roi.size()
        # Use input_data shape as reference
        h,w=self.input_data.shape
        x0=int(round(pos.x()))
        y0=int(round(pos.y()))
        width=int(round(size.x()))
        height=int(round(size.y()))
        x1=min(x0+width,w)
        y1=min(y0+height,h)
        x0=max(0,x0)
        y0=max(0,y0)
        # Check for valid slice AFTER clamping
        if x1 > x0 and y1 > y0:
            return slice(y0,y1), slice(x0,x1)
        else:
            logger.warning("Invalid ROI dimensions after clamping/rounding.")
            return None

    def _get_current_parameters(self) -> Dict[str, Any]:
        """
        Gathers current parameters for FFT calculation.
        
        Returns:
            Dict[str, Any]: Dictionary containing:
                - apply_window: Whether to apply window function
                - window_type: Selected window type
                - apply_roi_only: Whether to use ROI only
                - scaling_mode: Selected scaling mode (log/linear/power/sqrt)
        """
        apply_win = self.window_checkbox.isChecked()
        win_type = self.window_combo.currentText().lower() if apply_win else 'none'
        if win_type == 'none':
             apply_win = False

        if self.rb_log.isChecked(): scaling_mode = 'log'
        elif self.rb_linear.isChecked(): scaling_mode = 'linear'
        elif self.rb_power.isChecked(): scaling_mode = 'power'
        elif self.rb_sqrt.isChecked(): scaling_mode = 'sqrt'
        else: scaling_mode = 'log' # Default

        return {
            'apply_window': apply_win,
            'window_type': win_type if apply_win else None,
            'apply_roi_only': self.roi_mode_checkbox.isChecked(),
            'scaling_mode': scaling_mode
        }

    def _calculate_scaled_fft_magnitude(self, input_fft_data: np.ndarray, params: Dict[str, Any]) -> Optional[np.ndarray]:
         """
         Calculates complex FFT and returns the scaled magnitude.
         
         Args:
             input_fft_data: Input data for FFT calculation
             params: Dictionary of FFT parameters
             
         Returns:
             Optional[np.ndarray]: Scaled magnitude of FFT result, or None if calculation fails
         """
         try:
             is_roi = params.get('apply_roi_only', False)
             target_shape = self.input_data.shape if is_roi else None

             # 1. Calculate complex FFT (with padding for ROI)
             complex_fft = calculate_fft(
                 input_fft_data,
                 apply_window=params.get('apply_window', False),
                 window_type=params.get('window_type', 'hann'),
                 pad_to_shape=target_shape
             )
             if complex_fft is None:
                 logger.error("Backend FFT calculation returned None.")
                 return None

             # 2. Calculate magnitude
             magnitude = np.abs(complex_fft)

             # 3. Apply selected scaling
             scaling_mode = params.get('scaling_mode', 'log')
             logger.debug(f"Applying scaling mode: {scaling_mode}")
             if scaling_mode == 'log':
                 display_data = np.log1p(magnitude)
             elif scaling_mode == 'linear':
                 display_data = magnitude
             elif scaling_mode == 'power':
                 display_data = magnitude**2
             elif scaling_mode == 'sqrt':
                 display_data = np.sqrt(magnitude)
             else: # Default to log
                 display_data = np.log1p(magnitude)

             return display_data.astype(np.float32)

         except Exception as e:
              logger.exception(f"Error in _calculate_scaled_fft_magnitude: {e}")
              return None


    def _update_preview(self):
        """
        Calculates and updates the scaled FFT magnitude preview image.
        Handles both full image and ROI-based calculations.
        """
        if not self.live_preview_checkbox.isChecked():
            self.preview_display_data = None # Clear preview buffer
            self.update_fft_view() # Update display to show empty
            return

        params = self._get_current_parameters()
        is_roi = params.get('apply_roi_only', False)
        logger.debug(f"Updating FFT preview. ROI: {is_roi}, Params: {params}")

        input_for_calc = self.input_data
        if is_roi:
            roi_slice = self._get_roi_slice()
            if roi_slice:
                input_for_calc = self.input_data[roi_slice]
            else: # Invalid ROI
                logger.warning("Preview skipped: Invalid ROI selected for FFT.")
                self.preview_display_data = None; self.update_fft_view(); return
        if input_for_calc.size == 0:
             logger.warning("Preview skipped: Input data for FFT is empty."); self.preview_display_data = None; self.update_fft_view(); return

        # Calculate scaled magnitude using the helper
        self.preview_display_data = self._calculate_scaled_fft_magnitude(input_for_calc, params)

        if self.preview_display_data is None: # Handle calculation error
             logger.error("Preview calculation failed.")
             # Optionally show placeholder or leave view empty
             # self.preview_display_data = np.zeros((50,50), dtype=np.float32)

        self.update_fft_view() # Display the result (or clear if None)


    def update_input_view(self):
        """
        Updates the Input image view with the current input data.
        """
        if self.input_data is not None and hasattr(self, 'img_input') and self.img_input:
             self.img_input.setImage(self.input_data.T) # Transpose for STM view
             self.plot_input.autoRange()

    def update_fft_view(self):
        """
        Updates the FFT preview view using self.fft_image_view.
        Handles both valid data display and clearing the view.
        """
        if not hasattr(self, 'fft_image_view') or self.fft_image_view is None: return
        if self.preview_display_data is not None:
            logger.debug(f"Displaying FFT preview data shape: {self.preview_display_data.shape}")
            # Use setImage on ImageView; Use .T if user preferred that orientation
            self.fft_image_view.setImage(
                np.fliplr(self.preview_display_data.astype(np.float32).T), # Using .T
                autoLevels=True, # Let histogram handle levels/gamma
                autoRange=True
            )
        else:
            self.fft_image_view.clear() # Clear view if data is None
            logger.debug("FFT Preview view cleared.")


    # --- Methods to retrieve results after dialog acceptance ---
    def get_processed_data(self) -> Optional[np.ndarray]:
        """
        Returns the final calculated scaled FFT magnitude data.
        
        Returns:
            Optional[np.ndarray]: Copy of the final processed data, or None if not available
        """
        return self._final_processed_data.copy() if self._final_processed_data is not None else None

    def get_fft_parameters(self) -> dict:
        """
        Returns the parameters used for the final FFT calculation.
        
        Returns:
            dict: Copy of the parameters used for FFT calculation
        """
        return self._final_params.copy()

    def get_source_roi_slice(self) -> Optional[Tuple[slice, slice]]:
        """
        Returns the source ROI slice if FFT was calculated on ROI.
        
        Returns:
            Optional[Tuple[slice, slice]]: ROI slice used for calculation, or None if full image was used
        """
        return self._final_source_roi_slice

    # --- Dialog Actions ---
    def accept(self):
        """
        Calculate final SCALED FFT MAGNITUDE based on settings and close.
        Handles both full image and ROI-based calculations.
        """
        params = self._get_current_parameters()
        is_roi = params.get('apply_roi_only', False)
        self._final_params = params # Store final parameters used
        self._final_source_roi_slice = None # Reset

        logger.info(f"FFT Dialog accepted. ROI mode: {is_roi}, Params: {params}")

        input_for_calc = self.input_data
        if is_roi:
            roi_slice = self._get_roi_slice()
            if roi_slice:
                input_for_calc = self.input_data[roi_slice]
                self._final_source_roi_slice = roi_slice # Store the slice used
            else:
                QMessageBox.critical(self, "Error", "Invalid ROI selected."); super().reject(); return
        if input_for_calc.size == 0:
             QMessageBox.critical(self, "Error", "Input data for FFT is empty."); super().reject(); return

        try:
            # Calculate the final SCALED MAGNITUDE using the helper
            self._final_processed_data = self._calculate_scaled_fft_magnitude(input_for_calc, params)
            if self._final_processed_data is None:
                raise ValueError("FFT calculation or scaling failed.")

            # No need to check allclose for FFT
            logger.info("Final scaled FFT magnitude calculated successfully.")
            super().accept() # Close the dialog with Accepted state

        except Exception as e:
            logger.exception(f"Error during final FFT calculation/scaling: {e}")
            QMessageBox.critical(self, "Calculation Error", f"Failed to calculate final FFT result:\n{e}")
            self._final_processed_data = None # Ensure no data returned on error
            self._final_source_roi_slice = None
            super().reject() # Close with Rejected state

    def reject(self):
        """
        Called when Cancel button is clicked or dialog is closed.
        Cleans up any stored data.
        """
        logger.info(f"{self.operation_name} dialog rejected (Cancel clicked).")
        self._final_processed_data = None
        self._final_source_roi_slice = None
        super().reject()

    def was_roi_applied_only(self) -> bool:
        """
        Returns True if the final result was calculated from an ROI.
        
        Returns:
            bool: True if ROI was used for calculation, False otherwise
        """
        # For FFT, this means _final_source_roi_slice is set
        return self._final_source_roi_slice is not None