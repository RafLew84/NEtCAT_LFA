# lfa/gui/dialogs/real_space_reconstruction_dialog.py
import logging
from typing import Optional, Dict, Any

import numpy as np
from PyQt6.QtCore import Qt, pyqtSlot
from PyQt6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout,
    QLabel, QWidget, QGroupBox, QCheckBox,
    QFormLayout, QComboBox, QSplitter, QDialogButtonBox, QPushButton, QRadioButton
)

try:
    import pyqtgraph as pg
    from pyqtgraph import GraphicsLayoutWidget, ImageItem
    PYQTGRAPH_AVAILABLE = True
except ImportError:
    pg = None; GraphicsLayoutWidget = None; ImageItem = None; PYQTGRAPH_AVAILABLE = False
    logging.error("RealSpaceReconstructionDialog: PyQtGraph not found.")

logger = logging.getLogger(__name__)

# class RealSpaceReconstructionDialog(QDialog):
#     """
#     Dialog for reconstructing a real-space image from a modified FFT.
#     (Szkielet - implementacja w kolejnych krokach)
#     """
#     def __init__(self, parent=None):
#         super().__init__(parent)

#         self.setWindowTitle("Real Space Reconstruction")
#         self.setMinimumSize(800, 600)

#         self._init_ui()
#         self._connect_signals()

#         logger.debug("RealSpaceReconstructionDialog initialized.")

#     def _init_ui(self):
#         """Tworzy szkielet interfejsu użytkownika."""
#         layout = QVBoxLayout(self)
#         label = QLabel("Real Space Reconstruction Dialog\n\n(UI and logic to be implemented)")
#         label.setAlignment(Qt.AlignmentFlag.AlignCenter)
#         layout.addWidget(label)
        
#         self.button_box = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
#         layout.addWidget(self.button_box)

#     def _connect_signals(self):
#         """Podłącza sygnały z widgetów do slotów."""
#         self.button_box.accepted.connect(self.accept)
#         self.button_box.rejected.connect(self.reject)

# # lfa/gui/dialogs/real_space_reconstruction_dialog.py
# import logging
# from typing import Optional, Dict, Any

# import numpy as np
# from PyQt6.QtCore import Qt, pyqtSlot
# from PyQt6.QtWidgets import (
#     QDialog, QVBoxLayout, QHBoxLayout,
#     QLabel, QWidget, QGroupBox,
#     QFormLayout, QComboBox, QSplitter, QDialogButtonBox, QPushButton, QRadioButton
# )

# try:
#     import pyqtgraph as pg
#     from pyqtgraph import GraphicsLayoutWidget, ImageItem
#     PYQTGRAPH_AVAILABLE = True
# except ImportError:
#     pg = None; GraphicsLayoutWidget = None; ImageItem = None; PYQTGRAPH_AVAILABLE = False
#     logging.error("RealSpaceReconstructionDialog: PyQtGraph not found.")

# logger = logging.getLogger(__name__)

class RealSpaceReconstructionDialog(QDialog):
    def __init__(self,
                 magnitude_fft_data: np.ndarray,
                 complex_fft_data: np.ndarray,
                 parent=None):
        super().__init__(parent)
        self.setWindowTitle("Real Space Reconstruction from FFT")
        self.setMinimumSize(1200, 700)
        current_flags=self.windowFlags()
        self.setWindowFlags(current_flags | Qt.WindowType.WindowMinimizeButtonHint | Qt.WindowType.WindowMaximizeButtonHint)


        self.magnitude_fft_data = magnitude_fft_data # Do wyświetlania i interakcji
        self.complex_fft_data = complex_fft_data   # Do obliczeń iFFT

        self._init_ui()
        self._connect_signals()
        
        # Wyświetl początkowy obraz FFT
        if self.original_fft_item:
            self.original_fft_item.setImage(self.magnitude_fft_data.T)

    def _init_ui(self):
        main_layout = QVBoxLayout(self)
        
        # Górny panel z kontrolkami
        controls_group = QGroupBox("Reconstruction Controls")
        controls_layout = QHBoxLayout(controls_group)
        self._create_controls(controls_layout)
        main_layout.addWidget(controls_group)
        
        # Dolny panel z wizualizacjami
        display_splitter = QSplitter(Qt.Orientation.Horizontal)
        
        # Panel 1: Oryginalne FFT (do interakcji)
        fft_widget = GraphicsLayoutWidget(); self.fft_plot = fft_widget.addPlot(title="Original FFT (Select regions here)"); self.original_fft_item = ImageItem(); self.fft_plot.addItem(self.original_fft_item); self.fft_plot.setAspectLocked(True)
        display_splitter.addWidget(fft_widget)
        
        # Panel 2: Maska
        mask_widget = GraphicsLayoutWidget(); self.mask_plot = mask_widget.addPlot(title="Mask Preview"); self.mask_item = ImageItem(); self.mask_plot.addItem(self.mask_item); self.mask_plot.setAspectLocked(True)
        display_splitter.addWidget(mask_widget)
        
        # Panel 3: Zrekonstruowany Obraz
        reco_widget = GraphicsLayoutWidget(); self.reco_plot = reco_widget.addPlot(title="Reconstructed Real Space"); self.reco_item = ImageItem(); self.reco_plot.addItem(self.reco_item); self.reco_plot.setAspectLocked(True)
        display_splitter.addWidget(reco_widget)
        
        main_layout.addWidget(display_splitter, 1)

        self.button_box = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        main_layout.addWidget(self.button_box)
        
    def _create_controls(self, layout: QHBoxLayout):
        """Tworzy panel kontrolek."""
        mode_group = QGroupBox("Mode"); mode_layout = QVBoxLayout(mode_group)
        self.rb_autocorrelation = QRadioButton("Calculate Autocorrelation"); self.rb_autocorrelation.setChecked(True)
        self.rb_roi_mask = QRadioButton("Mask with ROIs")
        self.rb_spot_mask = QRadioButton("Mask with Spots")
        mode_layout.addWidget(self.rb_autocorrelation); mode_layout.addWidget(self.rb_roi_mask); mode_layout.addWidget(self.rb_spot_mask)
        mode_layout.addStretch()
        layout.addWidget(mode_group)

        action_group = QGroupBox("Actions"); action_layout = QVBoxLayout(action_group)
        self.add_roi_button = QPushButton("Add ROI"); self.add_roi_button.setEnabled(False)
        self.symmetric_roi_checkbox = QCheckBox("Add Symmetric ROI"); self.symmetric_roi_checkbox.setEnabled(False)
        self.add_spot_button = QPushButton("Select Spot"); self.add_spot_button.setEnabled(False)
        self.clear_mask_button = QPushButton("Clear Mask")
        self.reconstruct_button = QPushButton("Reconstruct Image")
        action_layout.addWidget(self.add_roi_button); action_layout.addWidget(self.symmetric_roi_checkbox)
        action_layout.addWidget(self.add_spot_button); action_layout.addWidget(self.clear_mask_button)
        action_layout.addWidget(self.reconstruct_button)
        action_layout.addStretch()
        layout.addWidget(action_group)
        layout.addStretch()

    def _connect_signals(self):
        self.button_box.accepted.connect(self.accept)
        self.button_box.rejected.connect(self.reject)
        # TODO: Połączenie reszty kontrolek w kolejnym kroku