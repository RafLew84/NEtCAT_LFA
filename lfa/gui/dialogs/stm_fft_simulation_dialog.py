# lfa/gui/dialogs/stm_fft_simulation_dialog.py
import logging
from PyQt6.QtWidgets import QDialog, QVBoxLayout, QLabel, QDialogButtonBox
from PyQt6.QtCore import Qt

logger = logging.getLogger(__name__)

class StmFftSimulationDialog(QDialog):
    """
    Dialog for creating simulated STM/FFT data based on user-defined parameters.
    (Szkielet - implementacja w kolejnych krokach)
    """
    def __init__(self, parent=None):
        super().__init__(parent)

        self.setWindowTitle("STM/FFT Simulation")
        self.setMinimumSize(800, 600)

        self._init_ui()
        self._connect_signals()

        logger.debug("StmFftSimulationDialog initialized.")

    def _init_ui(self):
        """
        Tworzy szkielet interfejsu użytkownika.
        Na razie jest to tylko puste okno z informacją.
        """
        layout = QVBoxLayout(self)
        label = QLabel("STM/FFT Simulation Dialog\n\n(UI and logic will be implemented here)")
        label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(label)
        
        button_box = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        layout.addWidget(button_box)
        self.button_box = button_box

    def _connect_signals(self):
        """Podłącza sygnały z widgetów do slotów."""
        self.button_box.accepted.connect(self.accept)
        self.button_box.rejected.connect(self.reject)