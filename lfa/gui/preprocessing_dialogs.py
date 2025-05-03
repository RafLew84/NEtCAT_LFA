# lfa/gui/preprocessing_dialogs.py
"""
Dialog windows for configuring and previewing preprocessing operations.
"""
import logging
import numpy as np
from typing import Optional

try:
    from PyQt6.QtWidgets import (
        QDialog, QVBoxLayout, QHBoxLayout, QSlider, QLabel, QCheckBox, QDialogButtonBox,
        QWidget, QSizePolicy, QMessageBox
    )
    from PyQt6.QtCore import Qt, pyqtSignal, pyqtSlot
    import pyqtgraph as pg
    from pyqtgraph import PlotItem 
except ImportError as e:
    logging.critical(f"Failed to import necessary Qt or pyqtgraph modules: {e}")
    PlotItem = None
    # Handle the error appropriately, maybe exit or disable functionality
    raise

# Import the processing function
try:
    from ..preprocessing.filtering import gaussian_blur
except ImportError:
    logging.error("Could not import preprocessing functions.")
    # Define dummy function to avoid crash if import fails
    def gaussian_blur(image, sigma):
        logging.error("gaussian_blur function not available!")
        return image


logger = logging.getLogger(__name__)

# --- Gaussian Blur Dialog ---

class GaussianBlurDialog(QDialog):
    """
    Dialog window for applying Gaussian Blur.

    Shows original and processed images side-by-side.
    Allows adjusting the sigma parameter with a slider and live preview.
    """
    # Signal emitted when processing parameters change significantly (e.g., slider released)
    # Could be used for more complex updates if needed.
    # processing_updated = pyqtSignal()

    def __init__(self, original_data: np.ndarray, parent=None):
        """
        Initializes the dialog.

        Args:
            original_data (np.ndarray): The initial 2D image data to process.
            parent: The parent widget (usually the MainWindow).
        """
        super().__init__(parent)
        if original_data is None:
            raise ValueError("Original data cannot be None for GaussianBlurDialog")

        self.original_data = original_data.astype(np.float32) # Store a float copy
        self.processed_data = self.original_data.copy() # Start with a copy for the processed view

        self.setWindowTitle("Gaussian Blur")
        self.setMinimumSize(800, 450) # Set a reasonable minimum size

        # --- Main Layouts ---
        # Main vertical layout
        main_layout = QVBoxLayout(self)
        # Horizontal layout for image views
        image_layout = QHBoxLayout()
        # Horizontal layout for controls
        controls_layout = QHBoxLayout()

        # --- Image Views (using GraphicsLayoutWidget for more control) ---
        pg.setConfigOption('background', 'w')
        pg.setConfigOption('foreground', 'k')

        # Graphics Layout Widget to hold the two image views
        self.win = pg.GraphicsLayoutWidget()

        # PlotItem for Original Image
        self.plot_original = self.win.addPlot(row=0, col=0, title="Original", name="plot_orig") # Tworzy PlotItem i ustawia tytuł
        self.img_original = pg.ImageItem()
        self.plot_original.addItem(self.img_original)
        self.plot_original.hideAxis('left')
        self.plot_original.hideAxis('bottom')
        self.plot_original.setAspectLocked(True)

        # PlotItem for Processed Image
        self.plot_processed = self.win.addPlot(row=0, col=1, title="Processed", name="plot_proc") # Tworzy PlotItem i ustawia tytuł
        self.img_processed = pg.ImageItem()
        self.plot_processed.addItem(self.img_processed)
        self.plot_processed.hideAxis('left')
        self.plot_processed.hideAxis('bottom')
        self.plot_processed.setAspectLocked(True)

        # Link the views for synchronized zoom/pan
        self.plot_processed.vb.setXLink(self.plot_original.vb)
        self.plot_processed.vb.setYLink(self.plot_original.vb)

        image_layout.addWidget(self.win) # Add graphics layout to the horizontal image layout

        # --- Controls ---
        # Sigma Slider
        controls_widget = QWidget() # Widget to hold controls layout
        controls_widget.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Maximum) # Limit vertical size
        self.sigma_label = QLabel(f"Sigma: {0.0:.1f}")
        self.sigma_slider = QSlider(Qt.Orientation.Horizontal)
        self.sigma_slider.setMinimum(0)  # Min sigma = 0 (no blur)
        self.sigma_slider.setMaximum(100) # Max sigma (scaled, e.g., 10.0) - adjust range as needed
        self.sigma_slider.setValue(0)     # Initial value
        self.sigma_slider.setTickInterval(10) # Optional ticks
        self.sigma_slider.setTickPosition(QSlider.TickPosition.TicksBelow)

        # Live Preview Checkbox
        self.live_preview_checkbox = QCheckBox("Live Preview")
        self.live_preview_checkbox.setChecked(True) # Default to on

        # Add controls to their layout
        controls_layout.addWidget(QLabel("Sigma:"))
        controls_layout.addWidget(self.sigma_slider)
        controls_layout.addWidget(self.sigma_label)
        controls_layout.addStretch() # Add space
        controls_layout.addWidget(self.live_preview_checkbox)
        controls_widget.setLayout(controls_layout)


        # --- Dialog Buttons ---
        self.button_box = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        # Rename OK to Apply for clarity
        self.button_box.button(QDialogButtonBox.StandardButton.Ok).setText("Apply")

        # --- Assemble Main Layout ---
        main_layout.addLayout(image_layout) # Add image views layout (takes most space)
        main_layout.addWidget(controls_widget) # Add controls layout
        main_layout.addWidget(self.button_box) # Add standard buttons

        # --- Initial Display ---
        self.update_original_view()
        self.update_processed_view() # Show original initially in processed view
        # self._update_views_range() # Set initial zoom/pan

        # --- Connect Signals ---
        self.sigma_slider.valueChanged.connect(self._on_parameter_changed)
        # Use sliderReleased if valueChanged is too slow/resource intensive
        # self.sigma_slider.sliderReleased.connect(self._on_slider_released)
        self.live_preview_checkbox.stateChanged.connect(self._on_parameter_changed)
        self.button_box.accepted.connect(self.accept) # Connect OK/Apply button
        self.button_box.rejected.connect(self.reject) # Connect Cancel button

        logger.debug("GaussianBlurDialog initialized.")


    def _get_sigma_value(self) -> float:
        """Gets the sigma value from the slider, scaled appropriately."""
        # Scale slider value (0-100) to a float sigma range (e.g., 0.0-10.0)
        return self.sigma_slider.value() / 10.0

    @pyqtSlot()
    def _on_parameter_changed(self):
        """Slot called when sigma slider value or live preview checkbox changes."""
        sigma = self._get_sigma_value()
        self.sigma_label.setText(f"Sigma: {sigma:.1f}") # Update label

        if self.live_preview_checkbox.isChecked():
            self._apply_processing()

    # Optional slot if using sliderReleased signal:
    # @pyqtSlot()
    # def _on_slider_released(self):
    #    """Slot called when slider is released (less frequent update)."""
    #    if self.live_preview_checkbox.isChecked():
    #        self._apply_processing()

    def _apply_processing(self):
        """Applies the Gaussian blur based on current parameters."""
        if self.original_data is None:
            return

        sigma = self._get_sigma_value()
        logger.debug(f"Applying Gaussian Blur with sigma={sigma:.2f}...")

        # Run the processing function
        # For potentially slow operations, consider running this in a background thread
        try:
             self.processed_data = gaussian_blur(self.original_data, sigma)
             # Update the "Processed" image view
             self.update_processed_view()
        except Exception as e:
             logger.exception(f"Error applying gaussian blur in dialog: {e}")
             QMessageBox.warning(self, "Processing Error", f"Failed to apply Gaussian blur:\n{e}")


    def update_original_view(self):
        """Updates the 'Original' image view."""
        if self.original_data is not None and self.img_original:
            self.img_original.setImage(np.fliplr(self.original_data.T)) # Transpose and flip vertically
            self.plot_original.autoRange()
            logger.debug("Original view updated.")

    def update_processed_view(self):
        """Updates the 'Processed' image view."""
        if self.processed_data is not None and self.img_processed:
            self.img_processed.setImage(np.fliplr(self.processed_data.T)) # Transpose and flip vertically
            self.plot_processed.autoRange()
            logger.debug("Processed view updated.")

    # def _update_views_range(self):
    #      """Sets the zoom/pan range for both views based on original data."""
    #      if self.vb_original and self.vb_processed:
    #           self.vb_original.autoRange()
    #           self.vb_processed.autoRange()
    #           logger.debug("Views autoranged.")

    def accept(self):
        """Called when the 'Apply' (OK) button is clicked."""
        logger.info("Gaussian Blur dialog accepted (Apply clicked).")
        # Ensure the final processing is applied if live preview was off
        if not self.live_preview_checkbox.isChecked():
            self._apply_processing()
        # Let the main window know the dialog was accepted
        super().accept()

    def reject(self):
        """Called when the 'Cancel' button is clicked."""
        logger.info("Gaussian Blur dialog rejected (Cancel clicked).")
        # Discard changes, processed_data is not returned
        super().reject()

    def get_processed_data(self) -> Optional[np.ndarray]:
        """
        Returns the final processed data.
        Call this *after* the dialog has been accepted (dialog.exec() == QDialog.Accepted).
        """
        # Return a copy to avoid external modification if dialog is reused (though unlikely here)
        return self.processed_data.copy() if self.processed_data is not None else None