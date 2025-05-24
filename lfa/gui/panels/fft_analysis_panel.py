# lfa/gui/panels/fft_analysis_panel.py
import logging
import numpy as np
from typing import Optional, Dict, Any
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QGroupBox, QFormLayout, QComboBox,
    QCheckBox, QRadioButton, QSpinBox, QPushButton, QHBoxLayout, QLabel
)
from PyQt6.QtCore import pyqtSignal, Qt

# Dostosuj ścieżki importu, jeśli KNOWN_LATTICES lub CustomLatticeDialog są w innych miejscach
# Zakładam, że KNOWN_LATTICES jest w lfa.analysis.lattice
# a CustomLatticeDialog w lfa.gui.dialogs.custom_lattice_dialog
try:
    # Próba importu z czterema kropkami, jeśli 'panels' jest na tym samym poziomie co 'analysis' i 'dialogs'
    # wewnątrz 'gui', a 'gui' jest w 'lfa'
    from lfa.analysis.lattice import KNOWN_LATTICES
    # from ..dialogs.custom_lattice_dialog import CustomLatticeDialog # Odkomentuj, jeśli potrzebne bezpośrednio tutaj
except ImportError:
    # Fallback, jeśli struktura jest inna lub testujesz ten plik bezpośrednio
    # W rzeczywistej aplikacji te importy muszą być poprawne.
    logging.warning("FFTAnalysisPanel: Could not perform standard relative imports for KNOWN_LATTICES. Using placeholders.")
    KNOWN_LATTICES = {"Placeholder (Error)": {}}
    # CustomLatticeDialog = None


logger = logging.getLogger(__name__)

class FFTAnalysisPanel(QWidget):
    """
    Widget panel containing all UI controls for FFT analysis settings.
    This includes ideal lattice overlay, spot selection, and spot refinement.
    """

    # --- Sygnały emitowane przez panel ---
    # Sygnały dla sekcji "Ideal Lattice Overlay"
    substrate_changed = pyqtSignal(str) # Emituje nazwę wybranego substratu lub specjalny tekst dla custom
    custom_lattice_define_requested = pyqtSignal() # Sygnał, gdy użytkownik wybierze "<Custom Define...>"
    show_ideal_lattice_changed = pyqtSignal(bool) # Emituje stan checkboxa (True/False)

    # Sygnały dla sekcji "Spot Selection"
    spot_selection_mode_changed = pyqtSignal(str) # Emituje "Substrate" lub "Adsorbate"
    # Sygnały dla zarządzania zestawami adsorbatu
    current_adsorbate_set_changed = pyqtSignal(str) # Emituje tekst aktualnie wybranego zestawu
    add_new_adsorbate_set_requested = pyqtSignal() # Gdy użytkownik wybierze "<Add New Set...>"

    reselect_current_adsorbate_set_triggered = pyqtSignal() # Przycisk "Reselect Set"
    clear_all_adsorbate_sets_triggered = pyqtSignal() # Przycisk "Clear All Sets"
    clear_last_adsorbate_point_triggered = pyqtSignal() # Przycisk "Clear Last Adsorbate Point"
    # Sygnał dla przycisku czyszczenia pików substratu
    select_edit_substrate_spots_requested = pyqtSignal()
    select_edit_adsorbate_spots_requested = pyqtSignal()

    fitted_substrate_spots_visibility_changed = pyqtSignal(bool)
    # Sygnały dla widoczności markerów pików
    substrate_spots_visibility_changed = pyqtSignal(bool)
    adsorbate_spots_visibility_changed = pyqtSignal(bool)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._init_ui()
        self._connect_internal_signals() # Dedykowana metoda do podłączania wewnętrznych sygnałów

    def _init_ui(self):
        """Initializes the user interface of the panel."""
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(5, 5, 5, 5) # Mniejsze marginesy dla panelu w docku
        main_layout.setSpacing(8)

        # --- Ideal Lattice Overlay Controls ---
        self._create_lattice_overlay_group(main_layout)

        # --- Spot Selection Controls ---
        self._create_spot_selection_group(main_layout)

        main_layout.addStretch(1) # Dodaj rozciągliwą przestrzeń na dole
        self.setLayout(main_layout)

    def _create_lattice_overlay_group(self, parent_layout: QVBoxLayout):
        """Creates the 'Ideal Lattice Overlay' group box and its controls."""
        self.lattice_group = QGroupBox("Ideal Lattice Overlay") # self.lattice_group
        lattice_layout = QFormLayout()

        self.substrate_combo = QComboBox()
        self.predefined_substrates = sorted(KNOWN_LATTICES.keys())
        self.substrate_combo.addItem("None")
        self.substrate_combo.addItems(self.predefined_substrates)
        self.custom_option_text = "<Custom Define...>"
        self.substrate_combo.addItem(self.custom_option_text)
        lattice_layout.addRow("Substrate:", self.substrate_combo)

        self.show_ideal_lattice_checkbox = QCheckBox("Show Ideal Lattice")
        self.show_ideal_lattice_checkbox.setChecked(True)
        lattice_layout.addRow(self.show_ideal_lattice_checkbox)

        self.lattice_group.setLayout(lattice_layout)
        parent_layout.addWidget(self.lattice_group)

    def _create_spot_selection_group(self, parent_layout: QVBoxLayout):
        """Creates the 'Spot Selection' group box and its controls."""
        self.spot_selection_group = QGroupBox("Spot Selection") # self.spot_selection_group
        spot_selection_layout = QVBoxLayout()

        # Spot Type Radio Buttons
        spot_type_layout = QHBoxLayout()
        self.rb_select_substrate = QRadioButton("Substrate")
        self.rb_select_substrate.setChecked(True) # Domyślnie zaznaczone
        self.rb_select_adsorbate = QRadioButton("Adsorbate")
        spot_type_layout.addWidget(self.rb_select_substrate)
        spot_type_layout.addWidget(self.rb_select_adsorbate)
        spot_selection_layout.addLayout(spot_type_layout)

        # --- Substrate Set Panel ---
        self.substrate_set_panel = QWidget()
        substrate_set_form_layout = QFormLayout(self.substrate_set_panel) # Użyj QFormLayout
        substrate_set_form_layout.setContentsMargins(0, 5, 0, 5)
        substrate_buttons_layout = QHBoxLayout()
        self.edit_substrate_spots_button = QPushButton("Edit/Select Substrate Spots")
        substrate_buttons_layout.addWidget(self.edit_substrate_spots_button)
        substrate_set_form_layout.addRow(substrate_buttons_layout) # Dodaj QHBoxLayout do QFormLayout

        self.rotation_angle_label = QLabel("Transform Rotation: -")
        self.rmse_label = QLabel("Transform RMSE: -")
        self.scale_factor_label = QLabel("Transform Stretches: -")
        substrate_set_form_layout.addRow(self.rotation_angle_label) # type: ignore
        substrate_set_form_layout.addRow(self.rmse_label) # type: ignore
        substrate_set_form_layout.addRow(self.scale_factor_label) # type: ignore
        spot_selection_layout.addWidget(self.substrate_set_panel)
        self.substrate_set_panel.setVisible(True) # Widoczne domyślnie

        # --- Adsorbate Set Panel ---
        self.adsorbate_set_panel = QWidget()
        adsorbate_set_form_layout = QFormLayout(self.adsorbate_set_panel) # Użyj QFormLayout
        adsorbate_set_form_layout.setContentsMargins(0, 5, 0, 5)

        self.adsorbate_set_combo = QComboBox()
        self.adsorbate_set_combo.addItem("Set 1")
        self.adsorbate_set_combo.addItem("<Add New Set...>")
        adsorbate_set_form_layout.addRow("Current Set:", self.adsorbate_set_combo)

        adsorbate_buttons_layout_top = QHBoxLayout()
        self.edit_adsorbate_spots_button = QPushButton("Edit/Select Current Set Spots")
        adsorbate_buttons_layout_top.addWidget(self.edit_adsorbate_spots_button)

        adsorbate_buttons_layout_bottom = QHBoxLayout()
        self.reselect_adsorbate_set_button = QPushButton("Clear Current Set") # Zmiana nazwy dla jasności
        self.clear_last_adsorbate_point_button = QPushButton("Clear Last Point in Set") # Zmiana nazwy
        self.clear_all_adsorbate_sets_button = QPushButton("Clear All Sets")
        adsorbate_buttons_layout_bottom.addWidget(self.reselect_adsorbate_set_button)
        adsorbate_buttons_layout_bottom.addWidget(self.clear_last_adsorbate_point_button)
        adsorbate_buttons_layout_bottom.addWidget(self.clear_all_adsorbate_sets_button)
        adsorbate_set_form_layout.addRow(adsorbate_buttons_layout_bottom)
        spot_selection_layout.addWidget(self.adsorbate_set_panel)
        self.adsorbate_set_panel.setVisible(False)

        # Spot Visibility Checkboxes
        self.show_fitted_substrate_spots_checkbox  = QCheckBox("Show Fitted Substrate Spots")
        self.show_fitted_substrate_spots_checkbox .setChecked(True)
        self.show_fitted_substrate_spots_checkbox .setToolTip("Show substrate spots after affine transformation to match ideal lattice (if calculated in dialog).")
        self.show_adsorbate_spots_checkbox = QCheckBox("Show Adsorbate Spots")
        self.show_adsorbate_spots_checkbox.setChecked(True)
        spot_selection_layout.addWidget(self.show_fitted_substrate_spots_checkbox)
        spot_selection_layout.addWidget(self.show_adsorbate_spots_checkbox)

        self.spot_selection_group.setLayout(spot_selection_layout)
        parent_layout.addWidget(self.spot_selection_group)

    def update_transform_results_display(self, analysis_results: Optional[Dict[str, Any]]):
        if analysis_results:
            self.rotation_angle_label.setText(f"Rotation (M->I): {analysis_results.get('rotation_angle_deg', 'N/A'):.2f}°")
            s_x = analysis_results.get('principal_stretches', [np.nan, np.nan])[0]
            s_y = analysis_results.get('principal_stretches', [np.nan, np.nan])[1]
            self.scale_factor_label.setText(f"Stretches (M->I): ({s_x:.3f}, {s_y:.3f})")
            self.rmse_label.setText(f"Fit RMSE (M->I, px): {analysis_results.get('rmse', 'N/A'):.3f}")
        else:
            self.rotation_angle_label.setText("Rotation: -")
            self.scale_factor_label.setText("Stretches: -")
            self.rmse_label.setText("RMSE: -")

    def _connect_internal_signals(self):
        """Connects internal widget signals to slots or directly to emitting class signals."""
        # Ideal Lattice Overlay
        self.substrate_combo.currentTextChanged.connect(self._handle_substrate_combo_change)
        self.show_ideal_lattice_checkbox.stateChanged.connect(
            lambda state: self.show_ideal_lattice_changed.emit(state == Qt.CheckState.Checked.value)
        )

        if hasattr(self, 'show_fitted_substrate_spots_checkbox'):
            self.show_fitted_substrate_spots_checkbox.stateChanged.connect(
                lambda state: self.fitted_substrate_spots_visibility_changed.emit(state == Qt.CheckState.Checked.value)
            )

        # Spot Selection Mode
        self.rb_select_substrate.toggled.connect(self._handle_spot_selection_mode_toggle)
        # self.rb_select_adsorbate jest połączony niejawnie, bo to grupa radio buttonów

        # Adsorbate Set Management
        self.adsorbate_set_combo.currentTextChanged.connect(self._handle_adsorbate_set_combo_change)

        self.edit_substrate_spots_button.clicked.connect(self.select_edit_substrate_spots_requested)
        self.edit_adsorbate_spots_button.clicked.connect(self.select_edit_adsorbate_spots_requested)

        self.reselect_adsorbate_set_button.clicked.connect(self.reselect_current_adsorbate_set_triggered)
        self.clear_last_adsorbate_point_button.clicked.connect(self.clear_last_adsorbate_point_triggered)
        self.clear_all_adsorbate_sets_button.clicked.connect(self.clear_all_adsorbate_sets_triggered)

        # Spot Visibility
        self.show_fitted_substrate_spots_checkbox.stateChanged.connect(
            lambda state: self.substrate_spots_visibility_changed.emit(state == Qt.CheckState.Checked.value)
        )
        self.show_adsorbate_spots_checkbox.stateChanged.connect(
            lambda state: self.adsorbate_spots_visibility_changed.emit(state == Qt.CheckState.Checked.value)
        )

    # --- Wewnętrzne Sloty ---
    def _handle_substrate_combo_change(self, text: str):
        logger.debug(f"FFTAnalysisPanel: Substrate combo changed to '{text}'")
        if text == self.custom_option_text:
            self.custom_lattice_define_requested.emit()
        else:
            self.substrate_changed.emit(text)

    def _handle_spot_selection_mode_toggle(self, checked: bool):
        # Ten slot jest wywoływany, gdy rb_select_substrate zmienia stan.
        # Jeśli rb_select_substrate jest zaznaczony, to `checked` będzie True.
        if checked: # Substrate selected
            self.substrate_set_panel.setVisible(True)
            self.adsorbate_set_panel.setVisible(False)
            self.spot_selection_mode_changed.emit("Substrate")
            logger.debug("FFTAnalysisPanel: Mode changed to Substrate")
        else: # Adsorbate selected (bo rb_select_substrate nie jest checked)
            self.substrate_set_panel.setVisible(False)
            self.adsorbate_set_panel.setVisible(True)
            self.spot_selection_mode_changed.emit("Adsorbate")
            logger.debug("FFTAnalysisPanel: Mode changed to Adsorbate")

    def _handle_adsorbate_set_combo_change(self, text: str):
        if text == "<Add New Set...>":
            self.add_new_adsorbate_set_requested.emit()
            # MainWindow doda nowy set i zaktualizuje to combo.
        else:
            self.current_adsorbate_set_changed.emit(text)

    # --- Metody publiczne do zarządzania stanem UI z zewnątrz (jeśli potrzebne) ---
    def set_substrate_combo_text(self, text: str):
        """Ustawia tekst w substrate_combo, blokując sygnały, aby uniknąć pętli."""
        self.substrate_combo.blockSignals(True)
        self.substrate_combo.setCurrentText(text)
        self.substrate_combo.blockSignals(False)

    def update_adsorbate_set_combo(self, set_names: list[str], current_set_text: str):
        """Aktualizuje elementy w adsorbate_set_combo i ustawia bieżący."""
        self.adsorbate_set_combo.blockSignals(True)
        self.adsorbate_set_combo.clear()
        self.adsorbate_set_combo.addItems(set_names)
        self.adsorbate_set_combo.addItem("<Add New Set...>")
        idx = self.adsorbate_set_combo.findText(current_set_text)
        if idx != -1:
            self.adsorbate_set_combo.setCurrentIndex(idx)
        elif set_names: # Jeśli current_set_text nie znaleziono, ale są inne, ustaw pierwszy
             self.adsorbate_set_combo.setCurrentIndex(0)
        self.adsorbate_set_combo.blockSignals(False)

    def get_current_substrate(self) -> str:
        return self.substrate_combo.currentText()

    def is_show_ideal_lattice_checked(self) -> bool:
        return self.show_ideal_lattice_checkbox.isChecked()
    
    def get_spot_selection_mode(self) -> str:
        """Zwraca aktualny tryb selekcji pików ('Substrate' lub 'Adsorbate')."""
        if self.rb_select_substrate.isChecked():
            return "Substrate"
        else:
            return "Adsorbate"
    
    def set_spot_selection_mode(self, mode: str):
        """Ustawia tryb selekcji pików (Substrate/Adsorbate)."""
        self.rb_select_substrate.blockSignals(True)
        self.rb_select_adsorbate.blockSignals(True)
        if mode == "Substrate":
            self.rb_select_substrate.setChecked(True)
            self.substrate_set_panel.setVisible(True)
            self.adsorbate_set_panel.setVisible(False)
        elif mode == "Adsorbate":
            self.rb_select_adsorbate.setChecked(True)
            self.substrate_set_panel.setVisible(False)
            self.adsorbate_set_panel.setVisible(True)
        else:
            logger.warning(f"FFTAnalysisPanel: Unknown spot selection mode '{mode}'")
        self.rb_select_substrate.blockSignals(False)
        self.rb_select_adsorbate.blockSignals(False)

    def is_substrate_spots_visible(self) -> bool:
        return self.show_fitted_substrate_spots_checkbox.isChecked()

    def is_adsorbate_spots_visible(self) -> bool:
        return self.show_adsorbate_spots_checkbox.isChecked()
    
    def set_edit_substrate_spots_button_enabled(self, enabled: bool): # Nowa nazwa
        if hasattr(self, 'edit_substrate_spots_button'):
            self.edit_substrate_spots_button.setEnabled(enabled)

    def set_edit_adsorbate_spots_button_enabled(self, enabled: bool): # Nowa nazwa
        if hasattr(self, 'edit_adsorbate_spots_button'):
            self.edit_adsorbate_spots_button.setEnabled(enabled)

    def set_clear_last_adsorbate_point_button_enabled(self, enabled: bool):
        if hasattr(self, 'clear_last_adsorbate_point_button'):
            self.clear_last_adsorbate_point_button.setEnabled(enabled)

    def set_reselect_adsorbate_set_button_enabled(self, enabled: bool):
        if hasattr(self, 'reselect_adsorbate_set_button'):
            self.reselect_adsorbate_set_button.setEnabled(enabled)

    def set_clear_all_adsorbate_sets_button_enabled(self, enabled: bool):
        if hasattr(self, 'clear_all_adsorbate_sets_button'):
            self.clear_all_adsorbate_sets_button.setEnabled(enabled)