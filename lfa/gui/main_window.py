# lfa/gui/main_window.py
"""
Defines the main window for the Lattice Fourier Analyzer (LFA) application.
"""

import logging
import os
import numpy as np
from typing import Optional

from PyQt6.QtWidgets import (
    QMainWindow, QVBoxLayout, QWidget, QFileDialog, QMessageBox, QApplication
)
from PyQt6.QtGui import QAction, QIcon # Import QIcon if you plan to use icons
from PyQt6.QtCore import Qt, pyqtSlot

# Import pyqtgraph after checking for installation
try:
    import pyqtgraph as pg
except ImportError:
    logging.error("PyQtGraph not found. Please install it: pip install pyqtgraph")
    # Optionally raise an error or exit if pyqtgraph is essential
    pg = None # Set to None to handle its absence gracefully later

# Import LFA core components
from ..core.data_models import STMImage
from ..io.factory import load_stm_file

logger = logging.getLogger(__name__)

# --- Main Window Class ---
class MainWindow(QMainWindow):
    """
    The main application window inheriting from QMainWindow.
    Provides menu bar, status bar, and central widget area.
    """
    def __init__(self, parent=None):
        """Initializes the main window."""
        super().__init__(parent)

        self.current_file_path: Optional[str] = None
        self.stm_image: Optional[STMImage] = None

        self.setWindowTitle("Lattice Fourier Analyzer (LFA)")
        # Set initial size (optional)
        self.resize(800, 600)

        # --- Central Widget and Layout ---
        central_widget = QWidget(self)
        self.setCentralWidget(central_widget)
        layout = QVBoxLayout(central_widget) # Use QVBoxLayout for vertical arrangement

        # --- Image View Widget (using pyqtgraph) ---
        if pg:
            # Set background to white for better contrast with grayscale
            pg.setConfigOption('background', 'w')
            pg.setConfigOption('foreground', 'k')

            self.image_view = pg.ImageView(self)
            # Add the ImageView to the layout
            layout.addWidget(self.image_view)
            logger.debug("PyQtGraph ImageView created.")
        else:
            self.image_view = None
            logger.error("Cannot create ImageView because PyQtGraph is not available.")
            # Optionally add a placeholder label
            # placeholder_label = QLabel("PyQtGraph not installed. Image display unavailable.")
            # layout.addWidget(placeholder_label)

        # --- Menu Bar ---
        self.create_menus()

        # --- Status Bar (optional) ---
        self.statusBar().showMessage("Ready") # Initial status message

        logger.info("Main window initialized.")


    def create_menus(self):
        """Creates the main menu bar and its actions."""
        menu_bar = self.menuBar()

        # --- File Menu ---
        file_menu = menu_bar.addMenu("&File") # '&' defines shortcut (Alt+F)

        # Open Action
        open_action = QAction("&Open...", self) # Pass 'self' as parent
        open_action.setStatusTip("Open an STM data file") # Tooltip for status bar
        open_action.triggered.connect(self.open_file_dialog) # Connect signal to slot
        # Add shortcut (optional)
        open_action.setShortcut("Ctrl+O")
        file_menu.addAction(open_action)

        # Add Separator
        file_menu.addSeparator()

        # Exit Action
        exit_action = QAction("&Exit", self)
        exit_action.setStatusTip("Exit the application")
        exit_action.triggered.connect(self.close) # Connect to QMainWindow's close method
        exit_action.setShortcut("Ctrl+Q")
        file_menu.addAction(exit_action)

        # --- Help Menu ---
        help_menu = menu_bar.addMenu("&Help")

        # About Action
        about_action = QAction("&About LFA...", self)
        about_action.setStatusTip("Show information about LFA")
        about_action.triggered.connect(self.show_about_dialog)
        help_menu.addAction(about_action)

        logger.debug("Menu bar created.")

    # --- Slots (methods connected to signals) ---
    @pyqtSlot() # Decorator marking this as a slot
    def open_file_dialog(self):
        """Opens a file dialog to select an STM file and loads it."""
        logger.debug("Open file dialog triggered.")
        # Define supported file types for the dialog filter
        # Format: "Description (*.ext1 *.ext2);;Another Description (*.ext3)"
        file_filter = "STM Files (*.stp *.s94);;All Files (*)"
        # Keep track of the last directory opened (optional, improves usability)
        start_dir = os.path.dirname(self.current_file_path) if self.current_file_path else os.path.expanduser("~")

        file_path, _ = QFileDialog.getOpenFileName(
            self,                   # Parent widget
            "Open STM File",        # Dialog title
            start_dir,              # Starting directory
            file_filter             # File type filter
        )

        # Check if a file was selected (dialog not cancelled)
        if file_path:
            logger.info(f"File selected: {file_path}")
            self.statusBar().showMessage(f"Loading file: {os.path.basename(file_path)}...")
            QApplication.processEvents() # Update UI to show status message

            # Load the file using the IO factory
            self.stm_image = load_stm_file(file_path)

            if self.stm_image:
                self.current_file_path = file_path
                logger.info("File loaded successfully via factory.")
                self.statusBar().showMessage(f"Loaded: {os.path.basename(file_path)}", 5000) # Message disappears after 5s
                # Update window title
                self.setWindowTitle(f"LFA - {os.path.basename(self.current_file_path)}")
                # Display the image
                self.display_image_data()
            else:
                # Loading failed (error logged by factory)
                self.statusBar().showMessage("Failed to load file.", 5000)
                QMessageBox.warning(
                    self,
                    "Loading Error",
                    f"Could not load the selected file:\n{file_path}\n\n"
                    "Check logs for details. The file might be corrupted or in an unsupported format."
                )
                self.setWindowTitle("Lattice Fourier Analyzer (LFA)") # Reset title
        else:
            logger.debug("File dialog cancelled.")
            self.statusBar().showMessage("File open cancelled.", 3000)

    @pyqtSlot()
    def show_about_dialog(self):
        """Displays the 'About' information message box."""
        logger.debug("About dialog triggered.")
        about_text = """
        <H2>Lattice Fourier Analyzer (LFA)</H2>
        <p>Version: 0.1.0 (Development)</p>
        <p>A tool for analyzing Scanning Tunneling Microscopy (STM) images,
        focusing on lattice determination using Fourier methods.</p>
        <p>(Further development ongoing)</p>
        <p><b>Note:</b> This is developmental software.</p>
        """
        QMessageBox.about(self, "About LFA", about_text) # Use QMessageBox.about static method

    # --- Other Methods ---
    def display_image_data(self):
        """Displays the current self.stm_image.data in the ImageView."""
        if not self.image_view:
             logger.error("ImageView widget not available. Cannot display image.")
             return
        if self.stm_image and self.stm_image.data is not None:
            logger.info(f"Displaying image data of shape: {self.stm_image.data.shape}")
            try:
                # pyqtgraph.ImageView handles data scaling and orientation.
                # Pass the raw data (preferably float32).
                # Sometimes data needs transposing (.T) depending on source vs display convention.
                # Start without transpose:
                self.image_view.setImage(self.stm_image.data.astype(np.float32).T)
                # If orientation is wrong (rotated 90 deg), try:
                # self.image_view.setImage(self.stm_image.data.astype(np.float32).T)

                # Auto-adjust contrast/levels initially (optional)
                self.image_view.autoLevels()
                # Reset zoom/pan (optional)
                self.image_view.autoRange()

            except Exception as e:
                logger.exception(f"Error setting image in ImageView: {e}")
                QMessageBox.critical(self,"Display Error", f"Could not display image data.\nError: {e}")
        else:
            logger.warning("No valid image data available to display.")
            # Clear the view if no data?
            self.image_view.clear()

    def closeEvent(self, event):
        """Handle the event when the user tries to close the window."""
        # Example: Add a confirmation dialog
        # reply = QMessageBox.question(self, 'Exit Confirmation',
        #                              "Are you sure you want to exit LFA?",
        #                              QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
        #                              QMessageBox.StandardButton.No)
        # if reply == QMessageBox.StandardButton.Yes:
        #     logger.info("Exiting application.")
        #     event.accept() # Allow closing
        # else:
        #     logger.debug("Exit cancelled by user.")
        #     event.ignore() # Prevent closing

        # For now, just log and accept closing
        logger.info("Close event triggered. Exiting application.")
        event.accept()