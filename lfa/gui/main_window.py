# lfa/gui/main_window.py
"""
Defines the main window for the Lattice Fourier Analyzer (LFA) application.
"""

import logging
import os
import numpy as np
from typing import Optional, Dict

from PyQt6.QtWidgets import (
    QMainWindow, QVBoxLayout, QWidget, QFileDialog, QMessageBox, QApplication, 
    QDialog, QHBoxLayout, QSplitter, QListWidget, QListWidgetItem
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
from ..core.history import HistoryNode

try:
    from .preprocessing_dialogs import GaussianBlurDialog
except ImportError:
    # This might happen if the file doesn't exist yet
    GaussianBlurDialog = None
    logging.warning("Could not import GaussianBlurDialog. Preprocessing options may be unavailable.")


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

        # --- History Management ---
        self.history: Dict[str, HistoryNode] = {} # Store all nodes by ID
        self.current_node_id: Optional[str] = None

        self.setWindowTitle("Lattice Fourier Analyzer (LFA)")
        # Set initial size (optional)
        self.resize(1000, 700)

        # --- Central Widget and Layout ---
        central_widget = QWidget(self)
        self.setCentralWidget(central_widget)
        main_layout = QHBoxLayout(central_widget) # Use QVBoxLayout for vertical arrangement

        splitter = QSplitter(Qt.Orientation.Horizontal)

        # --- History List Widget (Left Panel) ---
        self.history_list_widget = QListWidget()
        self.history_list_widget.currentItemChanged.connect(self.on_history_selection_changed)
        self.history_list_widget.setMaximumWidth(250) # Ogranicz szerokość listy
        splitter.addWidget(self.history_list_widget)

        # --- Image View Widget (Right Panel) ---
        image_view_container = QWidget() # Kontener dla ImageView
        image_view_layout = QVBoxLayout(image_view_container)
        image_view_layout.setContentsMargins(0,0,0,0) # Usuń marginesy

        # --- Image View Widget (using pyqtgraph) ---
        if pg:
            pg.setConfigOption('background', 'w')
            pg.setConfigOption('foreground', 'k')
            self.image_view = pg.ImageView(self)
            image_view_layout.addWidget(self.image_view)
        else:
            self.image_view = None
            logger.error("Cannot create ImageView because PyQtGraph is not available.")
            # Optionally add a placeholder label
            # placeholder_label = QLabel("PyQtGraph not installed. Image display unavailable.")
            # layout.addWidget(placeholder_label)
        
        splitter.addWidget(image_view_container)
        splitter.setSizes([250, 750])
        main_layout.addWidget(splitter)

        # --- Menu Bar ---
        self.create_menus()

        # --- Status Bar (optional) ---
        self.statusBar().showMessage("Ready - Load an image using File -> Open")
        self._update_action_states() # Wyłącz akcje na starcie

        logger.info("Main window initialized with history panel.")
    
    def _clear_history(self):
        """Clears the history tree and list widget."""
        self.history.clear()
        self.current_node_id = None
        self.history_list_widget.clear()
        if self.image_view:
            self.image_view.clear()
        logger.info("History cleared.")
        self._update_action_states()
    
    def _add_history_node(self, node: HistoryNode):
        """Adds a node to the history dict and the list widget."""
        if not node or not node.node_id:
            return
        self.history[node.node_id] = node
        item = QListWidgetItem(node.get_display_text())
        item.setData(Qt.ItemDataRole.UserRole, node.node_id) # Store node ID in item data
        self.history_list_widget.addItem(item)
        logger.debug(f"Added history node: {node.get_display_text()} (ID: {node.node_id})")
        return item # Return the list item
    
    def _set_current_node(self, node_id: Optional[str]):
        """Sets the current node ID and updates selection in the list."""
        if node_id not in self.history and node_id is not None:
            logger.error(f"Cannot set current node: ID {node_id} not found in history.")
            return

        self.current_node_id = node_id
        logger.info(f"Current history node set to: {node_id}")

        # Update selection in QListWidget without triggering signal again
        self.history_list_widget.blockSignals(True)
        found_item = None
        for i in range(self.history_list_widget.count()):
            item = self.history_list_widget.item(i)
            if item.data(Qt.ItemDataRole.UserRole) == node_id:
                item.setSelected(True)
                self.history_list_widget.setCurrentItem(item) # Ensure it's the current item
                found_item = item
                break
        if not found_item and self.history_list_widget.count() > 0:
             # Fallback or handle error if node exists but item doesn't
             pass
        self.history_list_widget.blockSignals(False)

        # Update displayed image and action states
        self.display_image_data()
        self._update_action_states()

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

        # --- Preprocessing Menu ---
        preprocessing_menu = menu_bar.addMenu("&Preprocessing")

        # Gaussian Blur Action
        self.gaussian_blur_action = QAction("&Gaussian Blur...", self)
        self.gaussian_blur_action.setStatusTip("Apply Gaussian blur filter")
        self.gaussian_blur_action.triggered.connect(self.open_gaussian_blur_dialog)
        # Disable action initially, enable only when an image is loaded
        self.gaussian_blur_action.setEnabled(False)
        preprocessing_menu.addAction(self.gaussian_blur_action)

        # --- Help Menu ---
        help_menu = menu_bar.addMenu("&Help")

        # About Action
        about_action = QAction("&About LFA...", self)
        about_action.setStatusTip("Show information about LFA")
        about_action.triggered.connect(self.show_about_dialog)
        help_menu.addAction(about_action)

        logger.debug("Menu bar created.")
    
    def _update_action_states(self):
        """Enables/disables actions based on the current state."""
        has_image = self.current_node_id is not None and self.current_node_id in self.history
        self.gaussian_blur_action.setEnabled(has_image)

    # --- Slots (methods connected to signals) ---
    @pyqtSlot()
    def open_file_dialog(self):
        logger.debug("Open file dialog triggered.")
        file_filter = "STM Files (*.stp *.s94);;All Files (*)"
        # Get the directory of the currently displayed file (if any)
        start_dir = ""
        if self.current_node_id and self.current_node_id in self.history:
            # Find the root node to get the original file path
            curr = self.history[self.current_node_id]
            while curr.parent_id and curr.parent_id in self.history:
                curr = self.history[curr.parent_id]
            # Assumes root node stores original file path in parameters or similar
            # For now, let's just use the last successful path if available
            try:
                 # We need to store the original file path somewhere accessible
                 # Let's assume the root node (or STMImage object we remove) stores it.
                 # We need to rethink where to store original_file_path persistently.
                 # Temporarily store it as an instance variable:
                 if hasattr(self, 'original_file_path') and self.original_file_path:
                      start_dir = os.path.dirname(self.original_file_path)

            except Exception: # Catch potential errors if path is weird
                pass
        if not start_dir:
             start_dir = os.path.expanduser("~") # Default to home dir

        file_path, _ = QFileDialog.getOpenFileName(self, "Open STM File", start_dir, file_filter)

        if file_path:
            logger.info(f"File selected: {file_path}")
            self.statusBar().showMessage(f"Loading file: {os.path.basename(file_path)}...")
            QApplication.processEvents()

            # Use load_stm_file which now returns STMImage object
            stm_image_obj = load_stm_file(file_path) # Now returns STMImage

            if stm_image_obj and stm_image_obj.data is not None:
                self.original_file_path = file_path # Store for next dialog path
                # --- Initialize History ---
                self._clear_history() # Clear previous history first
                root_node = HistoryNode(
                    operation_name="Original",
                    image_data=stm_image_obj.data.copy(), # Store a copy!
                    parameters={"filename": os.path.basename(file_path)} # Store filename
                )
                root_item = self._add_history_node(root_node)
                self._set_current_node(root_node.node_id) # Sets current node and displays image
                self.history_list_widget.setCurrentItem(root_item) # Ensure selection visually
                # --------------------------

                logger.info("File loaded successfully and history initialized.")
                self.statusBar().showMessage(f"Loaded: {os.path.basename(file_path)}", 5000)
                self.setWindowTitle(f"LFA - {os.path.basename(file_path)}")
                # self.display_image_data() # Called by _set_current_node
                # self._update_action_states() # Called by _set_current_node
            else:
                self._clear_history() # Clear history on failure
                self.statusBar().showMessage("Failed to load file.", 5000)
                QMessageBox.warning(self, "Loading Error", f"Could not load file: {file_path}")
                self.setWindowTitle("Lattice Fourier Analyzer (LFA)")
        else:
            logger.debug("File dialog cancelled.")
            self.statusBar().showMessage("File open cancelled.", 3000)

    @pyqtSlot(QListWidgetItem, QListWidgetItem) # previous, current
    def on_history_selection_changed(self, current_item: QListWidgetItem, previous_item: QListWidgetItem):
        """Slot called when the selection in the history list changes."""
        if current_item:
            node_id = current_item.data(Qt.ItemDataRole.UserRole)
            if node_id != self.current_node_id: # Avoid recursion if selection is set programmatically
                logger.info(f"History item selected: {current_item.text()}")
                self._set_current_node(node_id)
        # else: # No item selected (e.g., list cleared)
        #     self._set_current_node(None) # Handled by _clear_history

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

    @pyqtSlot()
    def open_gaussian_blur_dialog(self):
        """Opens the dialog for applying Gaussian Blur based on the current node."""
        if self.current_node_id is None or self.current_node_id not in self.history:
            QMessageBox.warning(self, "No Image", "No history state selected or available.")
            return
        if not GaussianBlurDialog:
             QMessageBox.critical(self,"Error", "GaussianBlurDialog could not be imported."); return

        current_node = self.history[self.current_node_id]
        # --- Get data for the dialog (handle lazy loading later) ---
        if current_node.image_data is None:
             # This part is for future lazy loading - for now, data should exist
             logger.error(f"Image data is missing for node {self.current_node_id}. Cannot open dialog.")
             QMessageBox.critical(self, "Internal Error", "Image data is missing for the selected history state.")
             return
        # Pass a copy of the *current* node's data to the dialog
        dialog_input_data = current_node.image_data.copy()
        # --------------------------------------------------------

        logger.info(f"Opening Gaussian Blur dialog based on node: {current_node.get_display_text()}")
        dialog = GaussianBlurDialog(dialog_input_data, parent=self)
        result = dialog.exec()

        if result == QDialog.DialogCode.Accepted:
            processed_data = dialog.get_processed_data() # Assumes dialog returns data
            # Retrieve parameters used from the dialog (needs modification in dialog)
            # For now, let's hardcode retrieval or assume dialog stores them
            # We need to modify the dialog to return this info.
            # Let's assume the dialog has a method get_params() for now:
            # params = dialog.get_params() # e.g., {'sigma': 1.5}
            # op_name = "Gaussian Blur" # Or get from dialog

            # TEMPORARY: Get sigma from dialog directly for now
            sigma_val = dialog._get_sigma_value() # Accessing protected member - needs refactor later
            params = {'sigma': round(sigma_val, 2)}
            op_name = "Gaussian Blur"


            if processed_data is not None:
                logger.info("Gaussian Blur dialog accepted. Creating new history node.")
                # --- Create new history node ---
                new_node = HistoryNode(
                    parent_id=self.current_node_id, # Parent is the node we started from
                    operation_name=op_name,
                    parameters=params,
                    image_data=processed_data # Store the result
                )
                new_item = self._add_history_node(new_node)
                self._set_current_node(new_node.node_id) # Make the new node current
                self.history_list_widget.setCurrentItem(new_item) # Select in list
                # -------------------------------
                self.statusBar().showMessage(f"{op_name} applied.", 3000)
            else:
                 logger.warning("Dialog accepted, but no processed data returned.")
        else:
            logger.info("Gaussian Blur dialog cancelled.")
            self.statusBar().showMessage("Gaussian Blur cancelled.", 3000)

    # --- Other Methods ---
    def display_image_data(self):
        """Displays the image data from the currently selected history node."""
        if not self.image_view: logger.error("ImageView widget not available."); return

        if self.current_node_id and self.current_node_id in self.history:
            node_to_display = self.history[self.current_node_id]
            # --- Get data (handle lazy loading later) ---
            display_data = node_to_display.image_data
            # -------------------------------------------

            if display_data is not None:
                logger.info(f"Displaying image for node {self.current_node_id}: {node_to_display.get_display_text()} (Shape: {display_data.shape})")
                try:
                    self.image_view.setImage(display_data.astype(np.float32).T) # Transpose
                    # Auto levels only on first display? Maybe not on history switch? User decision.
                    # self.image_view.autoLevels()
                    # self.image_view.autoRange() # Maybe keep autoRange
                except Exception as e: logger.exception(f"Error setting image in ImageView: {e}"); QMessageBox.critical(self,"Display Error", f"Could not display image data.\nError: {e}")
            else:
                # This case means data is missing (e.g., lazy loading not implemented or failed)
                logger.error(f"No image data stored or computed for node {self.current_node_id}!")
                self.image_view.clear()
                self.statusBar().showMessage(f"Error: Image data not available for selected state.", 5000)
        else:
            logger.debug("No current history node selected. Clearing image view.")
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