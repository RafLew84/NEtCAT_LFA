# lfa/gui/panels/fft_analysis_panel.py
import logging
import numpy as np
from typing import Optional, Dict, Any
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QGroupBox, QFormLayout, QComboBox, QTextEdit,
    QCheckBox, QRadioButton, QSpinBox, QPushButton, QHBoxLayout, QLabel
)
from PyQt6.QtCore import pyqtSignal, Qt, pyqtSlot


try:
    from lfa.analysis.lattice import KNOWN_LATTICES
except ImportError:
    logging.warning("FFTAnalysisPanel: Could not perform standard relative imports for KNOWN_LATTICES. Using placeholders.")
    KNOWN_LATTICES = {"Placeholder (Error)": {}}
    # CustomLatticeDialog = None

ADSORBATE_LATTICE_TYPE_UNKNOWN = "Unknown"
ADSORBATE_LATTICE_TYPE_HEXAGONAL = "Hexagonal"
ADSORBATE_LATTICE_TYPE_SQUARE = "Square"


logger = logging.getLogger(__name__)

class FFTAnalysisPanel(QWidget):
    """
    Widget panel containing all UI controls for FFT analysis settings.
    This includes ideal lattice overlay, spot selection, and spot refinement.
    """

    # --- Signals emitted by the panel ---
    # Signals for the "Ideal Lattice Overlay" section
    substrate_changed = pyqtSignal(str) # Emits the name of the selected substrate or special text for custom
    custom_lattice_define_requested = pyqtSignal() # Signal when the user selects "<Custom Define...>"
    show_ideal_lattice_changed = pyqtSignal(bool) # Emits the state of the checkbox (True/False)

    # Signals for the "Spot Selection" section
    spot_selection_mode_changed = pyqtSignal(str) # Emits "Substrate" or "Adsorbate"
    # Signals for adsorbate set management
    current_adsorbate_set_changed = pyqtSignal(str) # Emits the text of the currently selected set
    add_new_adsorbate_set_requested = pyqtSignal() # When the user selects "<Add New Set...>"

    reselect_current_adsorbate_set_triggered = pyqtSignal() # "Reselect Set" button
    clear_all_adsorbate_sets_triggered = pyqtSignal() # "Clear All Sets" button
    # select_edit_substrate_spots_requested = pyqtSignal() # Button for clearing substrate peaks
    select_edit_substrate_spots_requested = pyqtSignal()
    select_edit_adsorbate_spots_requested = pyqtSignal()

    fitted_substrate_spots_visibility_changed = pyqtSignal(bool)
    # Signals for marker visibility
    substrate_spots_visibility_changed = pyqtSignal(bool)
    adsorbate_spots_visibility_changed = pyqtSignal(bool)

    calculate_substrate_real_space_params_requested = pyqtSignal()
    calculate_adsorbate_real_space_params_requested = pyqtSignal(int)

    expected_adsorbate_lattice_type_changed = pyqtSignal(int, str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.current_selected_expected_adsorbate_type = ADSORBATE_LATTICE_TYPE_UNKNOWN
        self._init_ui()
        self._connect_internal_signals() # Dedicated method for wiring internal signals

    def _init_ui(self):
        """Initializes the user interface of the panel."""
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(5, 5, 5, 5) # Smaller margins for the panel in the dock
        main_layout.setSpacing(8)

        # --- Ideal Lattice Overlay Controls ---
        self._create_lattice_overlay_group(main_layout)

        # --- Spot Selection Controls ---
        self._create_spot_selection_group(main_layout)

        self._create_real_space_params_group(main_layout)
        self._create_domain_wall_results_group(main_layout)

        main_layout.addStretch(1) # Add stretchable space at the bottom
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

    def _create_domain_wall_results_group(self, parent_layout: QVBoxLayout):
        """Creates a group for displaying domain wall analysis results."""
        self.domain_wall_results_group = QGroupBox("Domain Wall Analysis Results")
        layout = QFormLayout(self.domain_wall_results_group)
        
        self.dw_dist_kspace_label = QLabel("-")
        self.dw_periodicity_label = QLabel("-")
        self.dw_intensity_ratio_label = QLabel("-")
        self.dw_amplitude_ratio_label = QLabel("-")
        self.dw_max_value_ratio_label = QLabel("-")

        layout.addRow("k-space Distance (Δg*):", self.dw_dist_kspace_label)
        layout.addRow("Real Space Periodicity (P):", self.dw_periodicity_label)
        layout.addRow("Intensity Ratio (Sat/Main):", self.dw_intensity_ratio_label)
        layout.addRow("Amplitude Ratio (Sat/Main):", self.dw_amplitude_ratio_label)
        layout.addRow("Max Value Ratio (Sat/Main):", self.dw_max_value_ratio_label)
        
        parent_layout.addWidget(self.domain_wall_results_group)
        self.domain_wall_results_group.setVisible(False) # Visible only when domain-wall results exist

    def update_domain_wall_results_display(self, results: Optional[Dict[str, Any]]):
        """Updates the labels with the domain wall analysis results."""
        if results:
            self.dw_dist_kspace_label.setText(f"{results.get('dist_px', '-'):.2f} px | {results.get('dist_nm_inv', '-'):.4f} nm⁻¹")
            self.dw_periodicity_label.setText(f"{results.get('periodicity_nm', '-'):.3f} nm")
            self.dw_intensity_ratio_label.setText(f"{results.get('intensity_ratio', '-'):.3f}")
            self.dw_amplitude_ratio_label.setText(f"{results.get('amplitude_ratio', '-'):.3f}")
            self.dw_max_value_ratio_label.setText(f"{results.get('max_value_ratio', '-'):.3f}")
            self.domain_wall_results_group.setVisible(True)
        else:
            self.domain_wall_results_group.setVisible(False)

    def _create_real_space_params_group(self, parent_layout: QVBoxLayout):
        self.real_space_group = QGroupBox("Real Space Lattice Parameters")
        real_space_layout = QVBoxLayout(self.real_space_group)

        # Substrate section
        substrate_params_group = QGroupBox("Substrate")
        substrate_params_form = QFormLayout(substrate_params_group)
        self.sub_rs_a1_label = QLabel("a1: - nm")
        self.sub_rs_a2_label = QLabel("a2: - nm")
        self.sub_rs_alpha_label = QLabel("Angle: - °")
        substrate_params_form.addRow("Vector 1 |a1|:", self.sub_rs_a1_label)
        substrate_params_form.addRow("Vector 2 |a2|:", self.sub_rs_a2_label)
        substrate_params_form.addRow("Angle α (a1,a2):", self.sub_rs_alpha_label)
        self.calculate_substrate_rs_button = QPushButton("Calculate Substrate Parameters")
        self.calculate_substrate_rs_button.setEnabled(True) # Initially disabled
        substrate_params_form.addRow(self.calculate_substrate_rs_button)
        real_space_layout.addWidget(substrate_params_group)

        # Adsorbate section (for the current set)
        adsorbate_params_group = QGroupBox("Adsorbate (Current Set)")
        adsorbate_params_form = QFormLayout(adsorbate_params_group)
        self.ads_rs_a1_label = QLabel("a1: - nm")
        self.ads_rs_a2_label = QLabel("a2: - nm")
        self.ads_rs_alpha_label = QLabel("Angle: - °")
        adsorbate_params_form.addRow("Vector 1 |a1|:", self.ads_rs_a1_label)
        adsorbate_params_form.addRow("Vector 2 |a2|:", self.ads_rs_a2_label)
        adsorbate_params_form.addRow("Angle α (a1,a2):", self.ads_rs_alpha_label)
        self.calculate_adsorbate_rs_button = QPushButton("Calculate Adsorbate Parameters (Current Set)")
        self.calculate_adsorbate_rs_button.setEnabled(True) # Initially disabled
        adsorbate_params_form.addRow(self.calculate_adsorbate_rs_button)
        real_space_layout.addWidget(adsorbate_params_group)

        parent_layout.addWidget(self.real_space_group)

    def _create_spot_selection_group(self, parent_layout: QVBoxLayout):
        """Creates the 'Spot Selection' group box and its controls."""
        self.spot_selection_group = QGroupBox("Spot Selection") # self.spot_selection_group
        spot_selection_layout = QVBoxLayout()

        # Spot Type Radio Buttons
        spot_type_layout = QHBoxLayout()
        self.rb_select_substrate = QRadioButton("Substrate")
        self.rb_select_substrate.setChecked(True) # Checked by default
        self.rb_select_adsorbate = QRadioButton("Adsorbate")
        spot_type_layout.addWidget(self.rb_select_substrate)
        spot_type_layout.addWidget(self.rb_select_adsorbate)
        spot_selection_layout.addLayout(spot_type_layout)

        # --- Substrate Set Panel ---
        self.substrate_set_panel = QWidget()
        substrate_set_form_layout = QFormLayout(self.substrate_set_panel)
        substrate_set_form_layout.setContentsMargins(0, 5, 0, 5)
        substrate_buttons_layout = QHBoxLayout()
        self.edit_substrate_spots_button = QPushButton("Edit/Select Substrate Spots")
        substrate_buttons_layout.addWidget(self.edit_substrate_spots_button)
        substrate_set_form_layout.addRow(substrate_buttons_layout)

        self.rotation_angle_label = QLabel("Transform Rotation: -")
        self.rmse_label = QLabel("Transform RMSE: -")
        self.scale_factor_label = QLabel("Transform Stretches: -")
        substrate_set_form_layout.addRow(self.rotation_angle_label) # type: ignore
        substrate_set_form_layout.addRow(self.rmse_label) # type: ignore
        substrate_set_form_layout.addRow(self.scale_factor_label) # type: ignore
        spot_selection_layout.addWidget(self.substrate_set_panel)
        self.substrate_set_panel.setVisible(True) # Visible by default

        # --- Adsorbate Set Panel ---
        self.adsorbate_set_panel = QWidget()
        adsorbate_set_form_layout = QFormLayout(self.adsorbate_set_panel)
        adsorbate_set_form_layout.setContentsMargins(0, 5, 0, 5)

        self.adsorbate_set_combo = QComboBox()
        self.adsorbate_set_combo.addItem("Set 1")
        self.adsorbate_set_combo.addItem("<Add New Set...>")
        adsorbate_set_form_layout.addRow("Current Set:", self.adsorbate_set_combo)

        self.expected_adsorbate_type_combo = QComboBox()
        self.expected_adsorbate_type_combo.addItems([
            ADSORBATE_LATTICE_TYPE_UNKNOWN,
            ADSORBATE_LATTICE_TYPE_HEXAGONAL,
            ADSORBATE_LATTICE_TYPE_SQUARE
        ])
        self.expected_adsorbate_type_combo.setCurrentText(ADSORBATE_LATTICE_TYPE_UNKNOWN)
        adsorbate_set_form_layout.addRow("Expected Adsorbate Type:", self.expected_adsorbate_type_combo)

        adsorbate_buttons_layout_top = QHBoxLayout()
        self.edit_adsorbate_spots_button = QPushButton("Edit/Select Current Set Spots")
        adsorbate_buttons_layout_top.addWidget(self.edit_adsorbate_spots_button)

        adsorbate_buttons_layout_bottom = QHBoxLayout()
        self.reselect_adsorbate_set_button = QPushButton("Clear Current Set")
        self.clear_all_adsorbate_sets_button = QPushButton("Clear All Sets")
        adsorbate_buttons_layout_bottom.addWidget(self.reselect_adsorbate_set_button)
        adsorbate_buttons_layout_bottom.addWidget(self.clear_all_adsorbate_sets_button)
        adsorbate_set_form_layout.addRow(adsorbate_buttons_layout_bottom)
        spot_selection_layout.addWidget(self.adsorbate_set_panel)
        self.adsorbate_set_panel.setVisible(False)

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

        # Spot Selection Mode
        self.rb_select_substrate.toggled.connect(self._handle_spot_selection_mode_toggle)

        # Adsorbate Set Management
        self.adsorbate_set_combo.currentTextChanged.connect(self._handle_adsorbate_set_combo_change)

        self.edit_substrate_spots_button.clicked.connect(self.select_edit_substrate_spots_requested)
        self.edit_adsorbate_spots_button.clicked.connect(self.select_edit_adsorbate_spots_requested)

        self.reselect_adsorbate_set_button.clicked.connect(self.reselect_current_adsorbate_set_triggered)
        self.clear_all_adsorbate_sets_button.clicked.connect(self.clear_all_adsorbate_sets_triggered)

        if hasattr(self, 'expected_adsorbate_type_combo'):
            self.expected_adsorbate_type_combo.currentTextChanged.connect(self._handle_expected_adsorbate_type_changed)

        if hasattr(self, 'calculate_substrate_rs_button'):
            self.calculate_substrate_rs_button.clicked.connect(
                self.calculate_substrate_real_space_params_requested
            )
        if hasattr(self, 'calculate_adsorbate_rs_button'):
            self.calculate_adsorbate_rs_button.clicked.connect(
                lambda: self.calculate_adsorbate_real_space_params_requested.emit(
                    self.adsorbate_set_combo.currentIndex()
                )
            )

    def _handle_substrate_combo_change(self, text: str):
        logger.debug(f"FFTAnalysisPanel: Substrate combo changed to '{text}'")
        if text == self.custom_option_text:
            self.custom_lattice_define_requested.emit()
        else:
            self.substrate_changed.emit(text)

    def _handle_spot_selection_mode_toggle(self, checked: bool):
        # This slot is called when rb_select_substrate changes state.
        # If rb_select_substrate is checked, then `checked` will be True.
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
            # After adding a new set, `update_adsorbate_set_combo` will set the type to `current_selected_expected_adsorbate_type`
            # or based on data from AppController, if available.
        else:
            self.current_adsorbate_set_changed.emit(text)
            # After changing the set, update `expected_adsorbate_type_combo`
            # based on the value for this set from AppController
            # This will be done by MainWindow in response to a signal from AppController


    def update_substrate_real_space_display(self, params: Optional[Dict[str, Any]]):
        if hasattr(self, 'sub_rs_a1_label'): # Ensure UI has been initialized
            if params and "a1_nm" in params:
                self.sub_rs_a1_label.setText(f"{params['a1_nm']:.3f} nm")
                self.sub_rs_a2_label.setText(f"{params.get('a2_nm', 'N/A'):.3f} nm")
                self.sub_rs_alpha_label.setText(f"{params.get('alpha_deg', 'N/A'):.2f} °")
            else:
                self.sub_rs_a1_label.setText("- nm")
                self.sub_rs_a2_label.setText("- nm")
                self.sub_rs_alpha_label.setText("- °")

    def set_calculate_substrate_rs_button_enabled(self, enabled: bool):
        if hasattr(self, 'calculate_substrate_rs_button'):
            self.calculate_substrate_rs_button.setEnabled(True)

    def set_calculate_adsorbate_rs_button_enabled(self, enabled: bool):
        if hasattr(self, 'calculate_adsorbate_rs_button'):
            self.calculate_adsorbate_rs_button.setEnabled(True)

    def update_adsorbate_real_space_display(self, params: Optional[Dict[str, Any]]):
        if hasattr(self, 'ads_rs_a1_label'):
            if params and "a1_nm" in params:
                self.ads_rs_a1_label.setText(f"{params['a1_nm']:.3f} nm")
                self.ads_rs_a2_label.setText(f"{params.get('a2_nm', 'N/A'):.3f} nm")
                self.ads_rs_alpha_label.setText(f"{params.get('alpha_deg', 'N/A'):.2f} °")
            else:
                self.ads_rs_a1_label.setText("- nm")
                self.ads_rs_a2_label.setText("- nm")
                self.ads_rs_alpha_label.setText("- °")

    def set_substrate_combo_text(self, text: str):
        """Sets the text in substrate_combo, blocking signals to avoid loops."""
        self.substrate_combo.blockSignals(True)
        self.substrate_combo.setCurrentText(text)
        self.substrate_combo.blockSignals(False)

    def update_adsorbate_set_combo(self, set_names: list[str], current_set_text: str):
        """Updates the elements in adsorbate_set_combo and sets the current."""
        self.adsorbate_set_combo.blockSignals(True)
        self.adsorbate_set_combo.clear()
        self.adsorbate_set_combo.addItems(set_names)
        self.adsorbate_set_combo.addItem("<Add New Set...>")
        idx = self.adsorbate_set_combo.findText(current_set_text)
        if idx != -1:
            self.adsorbate_set_combo.setCurrentIndex(idx)
        elif set_names: # If current_set_text is not found, but there are others, set the first
             self.adsorbate_set_combo.setCurrentIndex(0)
        self.adsorbate_set_combo.blockSignals(False)

    def get_current_substrate(self) -> str:
        return self.substrate_combo.currentText()

    def is_show_ideal_lattice_checked(self) -> bool:
        return self.show_ideal_lattice_checkbox.isChecked()
    
    def get_spot_selection_mode(self) -> str:
        """Returns the current spot selection mode ('Substrate' or 'Adsorbate')."""
        if self.rb_select_substrate.isChecked():
            return "Substrate"
        else:
            return "Adsorbate"
    
    def set_spot_selection_mode(self, mode: str):
        """Sets the spot selection mode (Substrate/Adsorbate)."""
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
    
    def set_edit_substrate_spots_button_enabled(self, enabled: bool): # Nowa nazwa
        if hasattr(self, 'edit_substrate_spots_button'):
            self.edit_substrate_spots_button.setEnabled(enabled)

    def set_edit_adsorbate_spots_button_enabled(self, enabled: bool): # Nowa nazwa
        if hasattr(self, 'edit_adsorbate_spots_button'):
            self.edit_adsorbate_spots_button.setEnabled(enabled)

    def set_reselect_adsorbate_set_button_enabled(self, enabled: bool):
        if hasattr(self, 'reselect_adsorbate_set_button'):
            self.reselect_adsorbate_set_button.setEnabled(enabled)

    def set_clear_all_adsorbate_sets_button_enabled(self, enabled: bool):
        if hasattr(self, 'clear_all_adsorbate_sets_button'):
            self.clear_all_adsorbate_sets_button.setEnabled(enabled)

    def set_expected_adsorbate_type(self, type_name: str):
        """Sets the text in expected_adsorbate_type_combo without emitting a signal."""
        if hasattr(self, 'expected_adsorbate_type_combo'):
            self.expected_adsorbate_type_combo.blockSignals(True)
            idx = self.expected_adsorbate_type_combo.findText(type_name)
            if idx != -1:
                self.expected_adsorbate_type_combo.setCurrentIndex(idx)
            else: # pragma: no cover
                logger.warning(f"FFTAnalysisPanel: Could not find text '{type_name}' in expected_adsorbate_type_combo.")
                self.expected_adsorbate_type_combo.setCurrentText(ADSORBATE_LATTICE_TYPE_UNKNOWN) # Fallback
            self.expected_adsorbate_type_combo.blockSignals(False)
            self.current_selected_expected_adsorbate_type = self.expected_adsorbate_type_combo.currentText()

    @pyqtSlot(str)
    def _handle_expected_adsorbate_type_changed(self, selected_type: str):
        current_set_idx = self.adsorbate_set_combo.currentIndex()
        # Check if this is not the "<Add New Set...>" position
        # Assume the last position is "<Add New Set...>"
        # If only "Set 1" (without Add option), handle accordingly
        if current_set_idx >= 0 and current_set_idx < (self.adsorbate_set_combo.count() -1 ):
             self.current_selected_expected_adsorbate_type = selected_type
             logger.debug(f"FFTAnalysisPanel: Expected adsorbate type for set index {current_set_idx} changed to '{selected_type}'. Emitting signal.")
             self.expected_adsorbate_lattice_type_changed.emit(current_set_idx, selected_type)
        elif self.adsorbate_set_combo.count() == 1 and current_set_idx == 0 :
             self.current_selected_expected_adsorbate_type = selected_type
             logger.debug(f"FFTAnalysisPanel: Expected adsorbate type for set index {current_set_idx} changed to '{selected_type}'. Emitting signal.")
             self.expected_adsorbate_lattice_type_changed.emit(current_set_idx, selected_type)
