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
from PyQt6.QtGui import QAction, QIcon
from PyQt6.QtCore import Qt, pyqtSlot

# Import pyqtgraph after checking for installation
try:
    import pyqtgraph as pg
except ImportError:
    logging.error("PyQtGraph not found. Please install it: pip install pyqtgraph")
    pg = None

# Import LFA core components
from ..core.data_models import STMImage
from ..io.factory import load_stm_file
from ..core.history import HistoryNode

try:
    from .preprocessing_dialogs import (GaussianBlurDialog, PlaneLevelingDialog, 
    MedianFilterDialog, NLMeansDialog, BM3DDialog)
except ImportError:
    GaussianBlurDialog = None
    PlaneLevelingDialog = None
    MedianFilterDialog = None
    NLMeansDialog = None
    BM3DDialog = None
    logging.warning("Could not import preprocessing dialogs. Preprocessing options may be unavailable.")

logger = logging.getLogger(__name__)

class MainWindow(QMainWindow):
    """
    The main application window inheriting from QMainWindow.
    Provides menu bar, status bar, and central widget area.
    """
    def __init__(self, parent=None):
        super().__init__(parent)

        # --- History Management ---
        self.history: Dict[str, HistoryNode] = {}
        self.current_node_id: Optional[str] = None

        self.setWindowTitle("Lattice Fourier Analyzer (LFA)")
        self.resize(1000, 700)

        # --- Central Widget and Layout ---
        central_widget = QWidget(self)
        self.setCentralWidget(central_widget)
        main_layout = QHBoxLayout(central_widget)

        splitter = QSplitter(Qt.Orientation.Horizontal)

        # --- History List Widget (Left Panel) ---
        self.history_list_widget = QListWidget()
        self.history_list_widget.currentItemChanged.connect(self.on_history_selection_changed)
        self.history_list_widget.setMaximumWidth(250)
        splitter.addWidget(self.history_list_widget)

        # --- Image View Widget (Right Panel) ---
        image_view_container = QWidget()
        image_view_layout = QVBoxLayout(image_view_container)
        image_view_layout.setContentsMargins(0,0,0,0)

        if pg:
            pg.setConfigOption('background', 'w')
            pg.setConfigOption('foreground', 'k')
            self.image_view = pg.ImageView(self)
            image_view_layout.addWidget(self.image_view)
        else:
            self.image_view = None
            logger.error("Cannot create ImageView because PyQtGraph is not available.")
        
        splitter.addWidget(image_view_container)
        splitter.setSizes([250, 750])
        main_layout.addWidget(splitter)

        # --- Menu Bar ---
        self.create_menus()

        # --- Status Bar ---
        self.statusBar().showMessage("Ready - Load an image using File -> Open")
        self._update_action_states()

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
        item.setData(Qt.ItemDataRole.UserRole, node.node_id)
        self.history_list_widget.addItem(item)
        logger.debug(f"Added history node: {node.get_display_text()} (ID: {node.node_id})")
        return item
    
    def _set_current_node(self, node_id: Optional[str]):
        """Sets the current node ID and updates selection in the list."""
        if node_id not in self.history and node_id is not None:
            logger.error(f"Cannot set current node: ID {node_id} not found in history.")
            return

        self.current_node_id = node_id
        logger.info(f"Current history node set to: {node_id}")

        self.history_list_widget.blockSignals(True)
        found_item = None
        for i in range(self.history_list_widget.count()):
            item = self.history_list_widget.item(i)
            if item.data(Qt.ItemDataRole.UserRole) == node_id:
                item.setSelected(True)
                self.history_list_widget.setCurrentItem(item)
                found_item = item
                break
        if not found_item and self.history_list_widget.count() > 0:
            pass
        self.history_list_widget.blockSignals(False)

        self.display_image_data()
        self._update_action_states()

    def create_menus(self):
        """Creates the main menu bar and its actions."""
        menu_bar = self.menuBar()

        # --- File Menu ---
        file_menu = menu_bar.addMenu("&File")

        open_action = QAction("&Open...", self)
        open_action.setStatusTip("Open an STM data file")
        open_action.triggered.connect(self.open_file_dialog)
        open_action.setShortcut("Ctrl+O")
        file_menu.addAction(open_action)

        file_menu.addSeparator()

        exit_action = QAction("&Exit", self)
        exit_action.setStatusTip("Exit the application")
        exit_action.triggered.connect(self.close)
        exit_action.setShortcut("Ctrl+Q")
        file_menu.addAction(exit_action)

        # --- Preprocessing Menu ---
        preprocessing_menu = menu_bar.addMenu("&Preprocessing")

        self.gaussian_blur_action = QAction("&Gaussian Blur...", self)
        self.gaussian_blur_action.setStatusTip("Apply Gaussian blur filter")
        self.gaussian_blur_action.triggered.connect(self.open_gaussian_blur_dialog)
        self.gaussian_blur_action.setEnabled(False)
        preprocessing_menu.addAction(self.gaussian_blur_action)

        self.plane_level_action = QAction("&Plane Leveling...", self)
        self.plane_level_action.setStatusTip("Level image by subtracting a fitted plane")
        self.plane_level_action.triggered.connect(self.open_plane_leveling_dialog) 
        preprocessing_menu.addAction(self.plane_level_action)

        self.median_filter_action = QAction("&Median Filter...", self)
        self.median_filter_action.setStatusTip("Apply median filter for noise reduction")
        self.median_filter_action.triggered.connect(self.open_median_filter_dialog) 
        preprocessing_menu.addAction(self.median_filter_action)

        self.nlmeans_action = QAction("&NL-Means Denoising...", self)
        self.nlmeans_action.setStatusTip("Apply Non-Local Means denoising (skimage)")
        self.nlmeans_action.triggered.connect(self.open_nlmeans_dialog) 
        preprocessing_menu.addAction(self.nlmeans_action)

        self.bm3d_action = QAction("&BM3D Denoising...", self)
        self.bm3d_action.setStatusTip("Apply BM3D denoising (Computationally intensive)")
        self.bm3d_action.triggered.connect(self.open_bm3d_dialog) 
        preprocessing_menu.addAction(self.bm3d_action)

        # --- Help Menu ---
        help_menu = menu_bar.addMenu("&Help")

        about_action = QAction("&About LFA...", self)
        about_action.setStatusTip("Show information about LFA")
        about_action.triggered.connect(self.show_about_dialog)
        help_menu.addAction(about_action)

        logger.debug("Menu bar created.")
    
    def _update_action_states(self):
        """Enables/disables actions based on the current state."""
        has_image = self.current_node_id is not None and self.current_node_id in self.history
        self.gaussian_blur_action.setEnabled(has_image)
        self.plane_level_action.setEnabled(has_image)
        self.median_filter_action.setEnabled(has_image)
        self.nlmeans_action.setEnabled(has_image)
        self.bm3d_action.setEnabled(has_image)

    @pyqtSlot()
    def open_file_dialog(self):
        logger.debug("Open file dialog triggered.")
        file_filter = "STM Files (*.stp *.s94);;All Files (*)"
        start_dir = ""
        if self.current_node_id and self.current_node_id in self.history:
            curr = self.history[self.current_node_id]
            while curr.parent_id and curr.parent_id in self.history:
                curr = self.history[curr.parent_id]
            try:
                if hasattr(self, 'original_file_path') and self.original_file_path:
                    start_dir = os.path.dirname(self.original_file_path)
            except Exception:
                pass
        if not start_dir:
            start_dir = os.path.expanduser("~")

        file_path, _ = QFileDialog.getOpenFileName(self, "Open STM File", start_dir, file_filter)

        if file_path:
            logger.info(f"File selected: {file_path}")
            self.statusBar().showMessage(f"Loading file: {os.path.basename(file_path)}...")
            QApplication.processEvents()

            stm_image_obj = load_stm_file(file_path)

            if stm_image_obj and stm_image_obj.data is not None:
                self.original_file_path = file_path
                self._clear_history()
                root_node = HistoryNode(
                    operation_name="Original",
                    image_data=stm_image_obj.data.copy(),
                    parameters={"filename": os.path.basename(file_path)}
                )
                root_item = self._add_history_node(root_node)
                self._set_current_node(root_node.node_id)
                self.history_list_widget.setCurrentItem(root_item)

                logger.info("File loaded successfully and history initialized.")
                self.statusBar().showMessage(f"Loaded: {os.path.basename(file_path)}", 5000)
                self.setWindowTitle(f"LFA - {os.path.basename(file_path)}")
            else:
                self._clear_history()
                self.statusBar().showMessage("Failed to load file.", 5000)
                QMessageBox.warning(self, "Loading Error", f"Could not load file: {file_path}")
                self.setWindowTitle("Lattice Fourier Analyzer (LFA)")
        else:
            logger.debug("File dialog cancelled.")
            self.statusBar().showMessage("File open cancelled.", 3000)

    @pyqtSlot()
    def open_bm3d_dialog(self):
        """Opens the dialog for applying BM3D Denoising."""
        if self.current_node_id is None or self.current_node_id not in self.history: QMessageBox.warning(self, "No Image", "..."); return
        if not BM3DDialog: QMessageBox.critical(self, "Error", "BM3DDialog not available."); return
        try: import bm3d
        except ImportError: QMessageBox.critical(self, "Missing Dependency", "The 'bm3d' package is required for this feature.\nPlease install it (pip install bm3d)."); return


        current_node = self.history[self.current_node_id]
        if current_node.image_data is None: QMessageBox.critical(self, "Internal Error", "..."); return
        dialog_input_data = current_node.image_data.copy()

        logger.info(f"Opening BM3D dialog based on node: {current_node.get_display_text()}")
        dialog = BM3DDialog(dialog_input_data, parent=self)
        result = dialog.exec()

        if result == QDialog.DialogCode.Accepted:
            processed_data = dialog.get_processed_data()
            params = dialog.get_parameters()
            was_roi_only = dialog.was_roi_applied_only()
            op_name = "BM3D"

            if processed_data is not None:
                if np.allclose(processed_data, current_node.image_data): 
                    logger.info("Data not modified.")
                    self.statusBar().showMessage("No changes applied.", 3000)
                    return

                logger.info(f"BM3D accepted. ROI Only: {was_roi_only}. Creating history node.")
                new_node = HistoryNode(
                    parent_id=self.current_node_id,
                    operation_name=op_name,
                    parameters=params,
                    image_data=processed_data,
                    is_roi_applied=was_roi_only
                )
                new_item = self._add_history_node(new_node)
                self._set_current_node(new_node.node_id)
                self.history_list_widget.setCurrentItem(new_item)
                display_name = new_node.get_display_text()
                self.statusBar().showMessage(f"{display_name} applied.", 3000)
            else: logger.warning("Dialog accepted, but no processed data returned.")
        else: logger.info("BM3D dialog cancelled."); self.statusBar().showMessage("BM3D cancelled.", 3000)


    @pyqtSlot()
    def open_nlmeans_dialog(self):
        """Opens the dialog for applying NL-Means Denoising."""
        if self.current_node_id is None or self.current_node_id not in self.history: QMessageBox.warning(self, "No Image", "..."); return
        if not NLMeansDialog: QMessageBox.critical(self, "Error", "NLMeansDialog not available."); return

        current_node = self.history[self.current_node_id]
        if current_node.image_data is None: QMessageBox.critical(self, "Internal Error", "..."); return
        dialog_input_data = current_node.image_data.copy()

        logger.info(f"Opening NL-Means dialog based on node: {current_node.get_display_text()}")
        dialog = NLMeansDialog(dialog_input_data, parent=self)
        result = dialog.exec()

        if result == QDialog.DialogCode.Accepted:
            processed_data = dialog.get_processed_data()
            params = dialog.get_parameters()
            was_roi_only = dialog.was_roi_applied_only()
            op_name = "NL-Means" # Można dodać parametry do nazwy, jeśli trzeba

            if processed_data is not None:
                 # Sprawdzenie allclose jest w dialog.accept()
                logger.info(f"NL-Means accepted. ROI Only: {was_roi_only}. Creating history node.")
                new_node = HistoryNode(
                    parent_id=self.current_node_id,
                    operation_name=op_name,
                    parameters=params,
                    image_data=processed_data,
                    is_roi_applied=was_roi_only
                )
                new_item = self._add_history_node(new_node)
                self._set_current_node(new_node.node_id)
                self.history_list_widget.setCurrentItem(new_item)
                display_name = new_node.get_display_text()
                self.statusBar().showMessage(f"{display_name} applied.", 3000)
            else: logger.warning("Dialog accepted, but no processed data returned.")
        else: logger.info("NL-Means dialog cancelled."); self.statusBar().showMessage("NL-Means cancelled.", 3000)


    @pyqtSlot()
    def open_median_filter_dialog(self):
        """Opens the dialog for applying Median Filter."""
        if self.current_node_id is None or self.current_node_id not in self.history: QMessageBox.warning(self, "No Image", "..."); return
        if not MedianFilterDialog: QMessageBox.critical(self, "Error", "MedianFilterDialog not available."); return

        current_node = self.history[self.current_node_id]
        if current_node.image_data is None: QMessageBox.critical(self, "Internal Error", "..."); return
        dialog_input_data = current_node.image_data.copy()

        logger.info(f"Opening Median Filter dialog based on node: {current_node.get_display_text()}")
        dialog = MedianFilterDialog(dialog_input_data, parent=self)
        result = dialog.exec()

        if result == QDialog.DialogCode.Accepted:
            processed_data = dialog.get_processed_data()
            params = dialog.get_parameters()
            was_roi_only = dialog.was_roi_applied_only()
            op_name = "Median Filter"

            if processed_data is not None:
                if np.allclose(processed_data, current_node.image_data): logger.info("Data not modified."); self.statusBar().showMessage("No changes applied.", 3000); return

                logger.info(f"Median Filter accepted. ROI Only: {was_roi_only}. Creating history node.")
                new_node = HistoryNode(
                    parent_id=self.current_node_id,
                    operation_name=op_name,
                    parameters=params,
                    image_data=processed_data,
                    is_roi_applied=was_roi_only
                )
                new_item = self._add_history_node(new_node)
                self._set_current_node(new_node.node_id)
                self.history_list_widget.setCurrentItem(new_item)
                display_name = new_node.get_display_text() # Powinno zawierać "(ROI Only)"
                self.statusBar().showMessage(f"{display_name} applied.", 3000)
            else: logger.warning("Dialog accepted, but no processed data returned.")
        else: logger.info("Median Filter dialog cancelled."); self.statusBar().showMessage("Median Filter cancelled.", 3000)


    @pyqtSlot()
    def open_plane_leveling_dialog(self):
        """Opens the dialog for applying Plane Leveling."""
        if self.current_node_id is None or self.current_node_id not in self.history: QMessageBox.warning(self, "No Image", "..."); return
        if not PlaneLevelingDialog: QMessageBox.critical(self, "Error", "PlaneLevelingDialog not available."); return

        current_node = self.history[self.current_node_id]
        if current_node.image_data is None: QMessageBox.critical(self, "Internal Error", "..."); return
        dialog_input_data = current_node.image_data.copy()

        logger.info(f"Opening Plane Leveling dialog based on node: {current_node.get_display_text()}")
        dialog = PlaneLevelingDialog(dialog_input_data, parent=self)
        result = dialog.exec()

        if result == QDialog.DialogCode.Accepted:
            processed_data = dialog.get_processed_data()
            params = dialog.get_parameters()
            was_roi_only = dialog.was_roi_applied_only() 
            op_name = "Plane Leveling"

            if processed_data is not None:
                logger.info(f"Plane Leveling accepted. ROI Only: {was_roi_only}. Creating history node.")
                new_node = HistoryNode(
                    parent_id=self.current_node_id,
                    operation_name=op_name,
                    parameters=params,
                    image_data=processed_data,
                    is_roi_applied=was_roi_only 
                )
                new_item = self._add_history_node(new_node)
                self._set_current_node(new_node.node_id)
                self.history_list_widget.setCurrentItem(new_item)
                display_name = new_node.get_display_text() 
                self.statusBar().showMessage(f"{display_name} applied.", 3000)
            else: logger.warning("Dialog accepted, but no processed data returned.")
        else: logger.info("Plane Leveling dialog cancelled."); self.statusBar().showMessage("Plane Leveling cancelled.", 3000)


    @pyqtSlot(QListWidgetItem, QListWidgetItem)
    def on_history_selection_changed(self, current_item: QListWidgetItem, previous_item: QListWidgetItem):
        """Slot called when the selection in the history list changes."""
        if current_item:
            node_id = current_item.data(Qt.ItemDataRole.UserRole)
            if node_id != self.current_node_id:
                logger.info(f"History item selected: {current_item.text()}")
                self._set_current_node(node_id)

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
        QMessageBox.about(self, "About LFA", about_text)

    @pyqtSlot()
    def open_gaussian_blur_dialog(self):
        if self.current_node_id is None or self.current_node_id not in self.history:
            QMessageBox.warning(self, "No Image", "Please load an image first.")
            return
        if not GaussianBlurDialog:
            QMessageBox.critical(self, "Error", "Gaussian Blur functionality is not available.")
            return
        current_node = self.history[self.current_node_id]
        if current_node.image_data is None:
            QMessageBox.critical(self, "Internal Error", "No image data available.")
            return
        dialog_input_data = current_node.image_data.copy()

        logger.info(f"Opening Gaussian Blur dialog based on node: {current_node.get_display_text()}")
        dialog = GaussianBlurDialog(dialog_input_data, parent=self)
        result = dialog.exec()

        if result == QDialog.DialogCode.Accepted:
            processed_data = dialog.get_processed_data()
            params = dialog.get_parameters()
            was_roi = dialog.was_roi_applied_only()
            op_name = "Gaussian Blur"

            if processed_data is not None:
                if np.allclose(processed_data, current_node.image_data):
                    logger.info("Data was not modified. No history node created.")
                    self.statusBar().showMessage("No changes applied.", 3000)
                    return

                logger.info(f"Dialog accepted. Final apply was ROI: {was_roi}. Creating history node.")
                new_node = HistoryNode(
                    parent_id=self.current_node_id,
                    operation_name=op_name,
                    parameters=params,
                    image_data=processed_data,
                    is_roi_applied=was_roi
                )
                new_item = self._add_history_node(new_node)
                self._set_current_node(new_node.node_id)
                self.history_list_widget.setCurrentItem(new_item)
                display_name = new_node.get_display_text()
                self.statusBar().showMessage(f"{display_name} applied.", 3000)
            else:
                logger.warning("Dialog accepted, but processed data is None.")
        else:
            logger.info("Gaussian Blur dialog cancelled.")
            self.statusBar().showMessage("Gaussian Blur cancelled.", 3000)

    def display_image_data(self):
        """Displays the image data from the currently selected history node."""
        if not self.image_view:
            logger.error("ImageView widget not available.")
            return

        if self.current_node_id and self.current_node_id in self.history:
            node_to_display = self.history[self.current_node_id]
            display_data = node_to_display.image_data

            if display_data is not None:
                logger.info(f"Displaying image for node {self.current_node_id}: {node_to_display.get_display_text()} (Shape: {display_data.shape})")
                try:
                    self.image_view.setImage(display_data.astype(np.float32).T)
                except Exception as e:
                    logger.exception(f"Error setting image in ImageView: {e}")
                    QMessageBox.critical(self, "Display Error", f"Could not display image data.\nError: {e}")
            else:
                logger.error(f"No image data stored or computed for node {self.current_node_id}!")
                self.image_view.clear()
                self.statusBar().showMessage(f"Error: Image data not available for selected state.", 5000)
        else:
            logger.debug("No current history node selected. Clearing image view.")
            self.image_view.clear()

    def closeEvent(self, event):
        """Handle the event when the user tries to close the window."""
        logger.info("Close event triggered. Exiting application.")
        event.accept()