# lfa/gui/dialogs/adsorbate_spot_dialog.py
import logging
from typing import List, Tuple, Optional
import numpy as np

from PyQt6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QPushButton, QDialogButtonBox,
    QLabel, QListWidget, QAbstractItemView, QWidget, QSplitter
)
from PyQt6.QtCore import Qt, pyqtSlot

try:
    import pyqtgraph as pg
    ImageView = pg.ImageView
except ImportError:
    pg = None
    ImageView = None
    logging.error("AdsorbateSpotSelectionDialog: PyQtGraph not found.")

logger = logging.getLogger(__name__)

class AdsorbateSpotSelectionDialog(QDialog):
    def __init__(self, fft_image_data: Optional[np.ndarray],
                 current_spots: Optional[List[Tuple[float, float]]] = None,
                 adsorbate_set_index: int = 0, # Informacja o edytowanym zestawie
                 parent=None):
        super().__init__(parent)
        self.setWindowTitle(f"Select Adsorbate Spots (Set {adsorbate_set_index + 1})")
        self.setMinimumSize(800, 600)

        self.fft_data = fft_image_data
        self.selected_spots: List[Tuple[float, float]] = list(current_spots) if current_spots else []
        self.adsorbate_set_index = adsorbate_set_index
        self.temp_spots_on_image: List[pg.ScatterPlotItem] = []


        main_layout = QVBoxLayout(self)
        top_splitter = QSplitter(Qt.Orientation.Horizontal)

        self.image_view: Optional[ImageView] = None
        if ImageView:
            self.image_view = ImageView()
            if self.fft_data is not None:
                self.image_view.setImage(self.fft_data.T)
                self.image_view.getView().setAspectLocked(True)
                self.image_view.getView().invertY(True)
            top_splitter.addWidget(self.image_view)
        else:
            top_splitter.addWidget(QLabel("PyQtGraph ImageView not available."))

        list_control_widget = QWidget()
        list_control_layout = QVBoxLayout(list_control_widget)
        list_control_layout.addWidget(QLabel(f"Selected Adsorbate Spots (Set {adsorbate_set_index + 1}):"))
        self.spots_list_widget = QListWidget()
        self.spots_list_widget.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        list_control_layout.addWidget(self.spots_list_widget)
        buttons_layout = QHBoxLayout()
        self.remove_spot_button = QPushButton("Remove Selected Spot")
        self.clear_all_spots_button = QPushButton("Clear All Spots")
        buttons_layout.addWidget(self.remove_spot_button)
        buttons_layout.addWidget(self.clear_all_spots_button)
        list_control_layout.addLayout(buttons_layout)
        top_splitter.addWidget(list_control_widget)
        top_splitter.setSizes([600, 200])
        main_layout.addWidget(top_splitter)

        # TODO: Kontrolki uściślania, podglądy

        self.button_box = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        main_layout.addWidget(self.button_box)

        self.button_box.accepted.connect(self.accept)
        self.button_box.rejected.connect(self.reject)
        self.remove_spot_button.clicked.connect(self._remove_selected_spot)
        self.clear_all_spots_button.clicked.connect(self._clear_all_spots)
        
        if self.image_view:
            # self.image_view.getImageItem().scene().sigMouseClicked.connect(self._handle_fft_click)
            pass

        self._update_spots_list_widget()
        logger.debug(f"AdsorbateSpotSelectionDialog for set {adsorbate_set_index} initialized.")

    def _handle_fft_click(self, event):
        # TODO: Implementacja w Kroku 6
        logger.debug(f"Adsorbate dialog: FFT view clicked (to be implemented): {event.pos()}")
        # self.selected_spots.append(...)
        # self._update_spots_list_widget()

    def _update_spots_list_widget(self):
        self.spots_list_widget.clear()
        for i, (kx, ky) in enumerate(self.selected_spots):
            self.spots_list_widget.addItem(f"A{i+1}: ({kx:.2f}, {ky:.2f})") # "A" dla adsorbatu

    @pyqtSlot()
    def _remove_selected_spot(self):
        current_item = self.spots_list_widget.currentItem()
        if current_item:
            row = self.spots_list_widget.row(current_item)
            if 0 <= row < len(self.selected_spots):
                del self.selected_spots[row]
                self._update_spots_list_widget()
                logger.debug(f"Removed adsorbate spot at index {row}")

    @pyqtSlot()
    def _clear_all_spots(self):
        self.selected_spots.clear()
        self._update_spots_list_widget()
        logger.debug("Cleared all adsorbate spots in dialog.")

    def get_selected_spots(self) -> List[Tuple[float, float]]:
        return list(self.selected_spots)

    def accept(self):
        logger.info(f"AdsorbateSpotSelectionDialog for set {self.adsorbate_set_index} accepted with {len(self.selected_spots)} spots.")
        super().accept()

    def reject(self):
        logger.info(f"AdsorbateSpotSelectionDialog for set {self.adsorbate_set_index} rejected.")
        super().reject()