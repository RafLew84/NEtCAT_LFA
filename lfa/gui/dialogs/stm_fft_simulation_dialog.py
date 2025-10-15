# lfa/gui/dialogs/stm_fft_simulation_dialog.py
import logging
import numpy as np
from typing import Optional, Dict, Any, Tuple
from scipy.ndimage import gaussian_filter, median_filter

from PyQt6.QtCore import Qt, pyqtSlot, pyqtSignal, QRectF
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget,
    QVBoxLayout, QHBoxLayout, QGroupBox,
    QCheckBox, QLabel, QFormLayout, QPushButton,
    QSlider, QGroupBox,
    QComboBox, QDialog, QSplitter, QDialogButtonBox,
    QMessageBox
)

try:
    import pyqtgraph as pg
    from pyqtgraph import GraphicsLayoutWidget, ImageItem
    PYQTGRAPH_AVAILABLE = True
except ImportError: # pragma: no cover
    pg = None; GraphicsLayoutWidget = None; ImageItem = None; PYQTGRAPH_AVAILABLE = False
    logging.error("StmFftSimulationDialog: PyQtGraph not found.")

from ...core.history import HistoryNode
from ...logic.history_manager import HistoryManager
from ...analysis.peak_fitting import fit_2d_gaussian_in_roi_with_all_data, find_max_pixel_in_roi

try:
    from ...analysis.lattice import KNOWN_LATTICES, get_reciprocal_points, convert_g_vector_px_to_nm_inv, get_reciprocal_vectors
    from ...analysis.fft_engine import calculate_fft
except ImportError:
    KNOWN_LATTICES = {"Error": {"type": "hexagonal", "a_surf": 0.3}}
    def calculate_fft(*args, **kwargs): return None
    logging.error("StmFftSimulationDialog: Could not import KNOWN_LATTICES.")

logger = logging.getLogger(__name__)

class StmFftSimulationDialog(QDialog):
    """
    Dialog for creating simulated STM/FFT data based on user-defined parameters.
    """
    simulation_accepted = pyqtSignal(HistoryNode)

    def __init__(self, 
                 experimental_fft_image: np.ndarray,
                 experimental_data: Dict[str, Any],
                 simulation_params: Dict[str, Any],
                 history_manager: HistoryManager,
                 current_node_id: str,
                 parent=None):
        super().__init__(parent)
        self.setWindowTitle("STM/FFT Simulation & Comparison")
        self.setMinimumSize(1400, 800)
        current_flags=self.windowFlags()
        self.setWindowFlags(current_flags | Qt.WindowType.WindowMinimizeButtonHint | Qt.WindowType.WindowMaximizeButtonHint)


        self.experimental_fft_image = experimental_fft_image
        self.experimental_data = experimental_data
        self.sim_params = simulation_params
        self.history_manager = history_manager
        self.current_node_id = current_node_id
        # self.sim_params = simulation_params

        self.substrate_name = "Au(111)"
        self.adsorbate_name = "Iodine (predefined)"
        self.compression = 1.0
        self.stripe_width = 5.0
        self.relax_width = 2.0
        self.domain_type = 'Super Heavy'
        self.symmetry = 'Striped'
        self.atom_size_sub = 50
        self.atom_size_ads = 50
        self.fft_window_type = 'None'

        self._init_ui()
        self._connect_signals()

        self._display_experimental_data()

        self._initial_update_done = False
        
        if self.exp_fft_image_item:
            self.exp_fft_image_item.setImage(self.experimental_fft_image.T)

        self._update_simulation()

    def _init_ui(self):
        main_layout = QVBoxLayout(self)
        
        # --- TOP SECTION: CONTROLS AND EXPERIMENTAL RESULTS ---
        top_widget = QWidget()
        top_layout = QHBoxLayout(top_widget)
        top_layout.setContentsMargins(0, 0, 0, 0)
        
        sim_controls_group = QGroupBox("Simulation Controls")
        self._create_simulation_controls(sim_controls_group)
        top_layout.addWidget(sim_controls_group, 2) # Allocate more space to the controls panel

        exp_group = QGroupBox("Experimental Parameters (Reference)")
        exp_main_layout = QHBoxLayout(exp_group)
        top_layout.addWidget(exp_group, 3) # Allocate more space to results

        # Prepare empty layouts for subgroups
        self.exp_sub_layout = QFormLayout()
        self.exp_transform_layout = QFormLayout()
        self.exp_ads_layout = QVBoxLayout()
        self.exp_dw_layout = QFormLayout()

        sub_group = QGroupBox("Substrate"); sub_group.setLayout(self.exp_sub_layout)
        transform_group = QGroupBox("Transform"); transform_group.setLayout(self.exp_transform_layout)
        ads_group = QGroupBox("Adsorbate Sets"); ads_group.setLayout(self.exp_ads_layout)
        dw_group = QGroupBox("Domain Walls"); dw_group.setLayout(self.exp_dw_layout)

        exp_main_layout.addWidget(sub_group)
        exp_main_layout.addWidget(transform_group)
        exp_main_layout.addWidget(ads_group)
        exp_main_layout.addWidget(dw_group)
        
        main_layout.addWidget(top_widget)

        # --- DOLNA SEKCJA: WIZUALIZACJE ---
        bottom_splitter = QSplitter(Qt.Orientation.Horizontal)
        sim_stm_widget = GraphicsLayoutWidget()
        self.sim_stm_plot = sim_stm_widget.addPlot(title="Simulated STM")
        self.sim_stm_plot.setAspectLocked(True)
        self.sim_stm_image_item = ImageItem()
        self.sim_stm_plot.addItem(self.sim_stm_image_item)
        sim_fft_widget = GraphicsLayoutWidget()
        self.sim_fft_plot = sim_fft_widget.addPlot(title="Simulated FFT")
        self.sim_fft_plot.setAspectLocked(True)
        self.sim_fft_image_item = ImageItem()
        self.sim_fft_plot.addItem(self.sim_fft_image_item)
        sim_fft_histogram = pg.HistogramLUTItem()
        sim_fft_histogram.setImageItem(self.sim_fft_image_item)
        sim_fft_widget.addItem(sim_fft_histogram)
        exp_fft_widget = GraphicsLayoutWidget()
        self.exp_fft_plot = exp_fft_widget.addPlot(title="Experimental FFT")
        self.exp_fft_plot.setAspectLocked(True)
        self.exp_fft_image_item = ImageItem()
        self.exp_fft_plot.addItem(self.exp_fft_image_item)
        exp_fft_histogram = pg.HistogramLUTItem()
        exp_fft_histogram.setImageItem(self.exp_fft_image_item)
        exp_fft_widget.addItem(exp_fft_histogram)
        
        bottom_splitter.addWidget(sim_stm_widget)
        bottom_splitter.addWidget(sim_fft_widget)
        bottom_splitter.addWidget(exp_fft_widget)
        
        self.sim_fft_plot.setXLink(self.exp_fft_plot)
        self.sim_fft_plot.setYLink(self.exp_fft_plot)
        main_layout.addWidget(bottom_splitter, 1)

        # Przyciski na samym dole
        button_box = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        self.load_to_lfa_button = QPushButton("Load Simulation to Main Window")
        button_box.addButton(self.load_to_lfa_button, QDialogButtonBox.ButtonRole.ActionRole)
        main_layout.addWidget(button_box)
        self.button_box = button_box

    def _create_simulation_controls(self, parent_group: QGroupBox):
        """Helper that instantiates every simulation control widget."""
        controls_main_layout = QHBoxLayout(parent_group)
        
        lattice_layout = QVBoxLayout()
        lattice_group = QGroupBox("Lattice")
        lattice_form = QFormLayout(lattice_group)
        self.substrate_combo = QComboBox()
        self.substrate_combo.addItems(KNOWN_LATTICES.keys())
        self.adsorbate_combo = QComboBox()
        self.adsorbate_combo.addItems(["None", "Iodine (predefined)", "<Custom Define...>"])
        lattice_form.addRow("Substrate:", self.substrate_combo)
        lattice_form.addRow("Adsorbate:", self.adsorbate_combo)
        lattice_layout.addWidget(lattice_group)

        sim_results_group = QGroupBox("Simulation Analysis")
        self.sim_results_layout = QFormLayout(sim_results_group)
        lattice_layout.addWidget(sim_results_group)

        lattice_layout.addStretch()
        
        domain_layout = QVBoxLayout()
        domain_group = QGroupBox("Domain & Misfit")
        domain_form = QFormLayout(domain_group)
        self.compression_slider, self.compression_label = self._create_slider("Compression:", 50, 150, int(self.compression*100))
        self.stripe_width_slider, self.stripe_width_label = self._create_slider("Stripe Width (nm):", 10, 2000, int(self.stripe_width*100))
        self.relax_width_slider, self.relax_width_label = self._create_slider("Relax Width (nm):", 1, 1000, int(self.relax_width*100))
        self.domain_type_combo = QComboBox(); self.domain_type_combo.addItems(['Heavy','Super Heavy','Light','Super Light'])
        self.symmetry_combo = QComboBox(); self.symmetry_combo.addItems(['Striped','Hexagonal'])
        domain_form.addRow(self.compression_label, self.compression_slider)
        domain_form.addRow(self.stripe_width_label, self.stripe_width_slider)
        domain_form.addRow(self.relax_width_label, self.relax_width_slider)
        domain_form.addRow("Domain Type:", self.domain_type_combo)
        domain_form.addRow("Symmetry:", self.symmetry_combo)
        domain_layout.addWidget(domain_group)
        domain_layout.addStretch()

        vis_fft_layout = QVBoxLayout()
        vis_group = QGroupBox("Visualization")
        vis_form = QFormLayout(vis_group)
        self.sub_size_slider, self.sub_size_label = self._create_slider("Sub. Atom Size:", 10, 200, self.atom_size_sub)
        self.ads_size_slider, self.ads_size_label = self._create_slider("Ads. Atom Size:", 10, 400, self.atom_size_ads)
        vis_form.addRow(self.sub_size_label, self.sub_size_slider)
        vis_form.addRow(self.ads_size_label, self.ads_size_slider)
        self.show_substrate_checkbox = QCheckBox("Show Substrate Atoms")
        self.show_substrate_checkbox.setChecked(True) # Checked by default
        self.show_substrate_checkbox.setToolTip("Toggles the visibility of the substrate atoms in the simulation.")
        vis_form.addRow(self.show_substrate_checkbox)
        vis_fft_layout.addWidget(vis_group)
        fft_group = QGroupBox("FFT Settings")
        fft_form = QFormLayout(fft_group)
        self.fft_window_combo = QComboBox()
        self.fft_window_combo.addItems(['None', 'hann', 'hamming'])
        fft_form.addRow("Window:", self.fft_window_combo)
        vis_fft_layout.addWidget(fft_group)
        vis_fft_layout.addStretch()

        # sim_results_group = QGroupBox("Simulation Analysis")
        # self.sim_results_layout = QFormLayout(sim_results_group)
        # vis_fft_layout.addWidget(sim_results_group)

        self.resolution_multiplier_combo = QComboBox()
        self.resolution_multiplier_combo.addItems(["1x (Match Experiment)", "2x", "4x"])
        self.resolution_multiplier_combo.setToolTip("Increase simulation grid density for higher visual quality.\n1x matches experimental FFT grid for direct comparison.")
        vis_form.addRow("Resolution Multiplier:", self.resolution_multiplier_combo)

        auto_analysis_group = QGroupBox("Automated Analysis")
        auto_analysis_layout=QVBoxLayout(auto_analysis_group)
        self.analyze_sim_button = QPushButton("Analyze Simulated Peaks")
        auto_analysis_layout.addWidget(self.analyze_sim_button)
        self.auto_main_label = QLabel("Main Peak: -")
        self.auto_sat_label = QLabel("Satellite Peak: -")
        auto_analysis_layout.addWidget(self.auto_main_label)
        auto_analysis_layout.addWidget(self.auto_sat_label)
        self.auto_ratio_label = QLabel("Ratios (Sat/Main): -")
        auto_analysis_layout.addWidget(self.auto_ratio_label)
        vis_fft_layout.addWidget(auto_analysis_group)

        controls_main_layout.addLayout(lattice_layout)
        controls_main_layout.addLayout(domain_layout)
        controls_main_layout.addLayout(vis_fft_layout)


    # def _init_ui(self):
    #     main_layout = QVBoxLayout(self)
        
    #     # --- Top panel with three views ---
    #     top_splitter = QSplitter(Qt.Orientation.Horizontal)

    #     controls_widget = QWidget()
    #     controls_layout = QHBoxLayout(controls_widget)

    #     exp_group = QGroupBox("Experimental Parameters (Reference)")
    #     self.exp_layout = QVBoxLayout(exp_group)
    #     controls_layout.addWidget(exp_group, 1) # Stretchable splitter section
        
    #     # 1. Panel: Symulowany STM
    #     sim_stm_widget = GraphicsLayoutWidget()
    #     self.sim_stm_plot = sim_stm_widget.addPlot(title="Simulated STM")
    #     self.sim_stm_plot.setAspectLocked(True)
    #     self.sim_stm_image_item = ImageItem()
    #     self.sim_stm_plot.addItem(self.sim_stm_image_item)
    #     top_splitter.addWidget(sim_stm_widget)

    #     # 2. Panel: Symulowany FFT
    #     sim_fft_widget = GraphicsLayoutWidget()
    #     self.sim_fft_plot = sim_fft_widget.addPlot(title="Simulated FFT")
    #     self.sim_fft_plot.setAspectLocked(True)
    #     self.sim_fft_image_item = ImageItem()
    #     self.sim_fft_plot.addItem(self.sim_fft_image_item)
    #     top_splitter.addWidget(sim_fft_widget)
        
    #     # 3. Panel: Eksperymentalny FFT
    #     exp_fft_widget = GraphicsLayoutWidget()
    #     self.exp_fft_plot = exp_fft_widget.addPlot(title="Experimental FFT (Reference)")
    #     self.exp_fft_plot.setAspectLocked(True)
    #     self.exp_fft_image_item = ImageItem()
    #     self.exp_fft_plot.addItem(self.exp_fft_image_item)
    #     top_splitter.addWidget(exp_fft_widget)
        
    #     # Synchronize FFT views
    #     self.sim_fft_plot.setXLink(self.exp_fft_plot)
    #     self.sim_fft_plot.setYLink(self.exp_fft_plot)

    #     main_layout.addWidget(top_splitter)

    #     main_layout.addWidget(controls_widget)
        
    #     # --- Dolny panel z kontrolkami ---
    #     controls_group = QGroupBox("Simulation Controls")
    #     controls_main_layout = QHBoxLayout(controls_group)
        
    #     # Kolumna 1: Definicja sieci
    #     lattice_layout = QVBoxLayout()
    #     lattice_group = QGroupBox("Lattice Definition")
    #     lattice_form = QFormLayout(lattice_group)
    #     self.substrate_combo = QComboBox(); self.substrate_combo.addItems(KNOWN_LATTICES.keys())
    #     self.adsorbate_combo = QComboBox(); self.adsorbate_combo.addItems(["None", "Iodine (predefined)", "<Custom Define...>"])
    #     lattice_form.addRow("Substrate:", self.substrate_combo)
    #     lattice_form.addRow("Adsorbate:", self.adsorbate_combo)
    #     lattice_layout.addWidget(lattice_group)
    #     lattice_layout.addStretch()
    #     controls_main_layout.addLayout(lattice_layout)
        
    #     # Kolumna 2: Parametry domen i niedopasowania
    #     domain_layout = QVBoxLayout()
    #     domain_group = QGroupBox("Domain & Misfit Parameters")
    #     domain_form = QFormLayout(domain_group)
    #     self.compression_slider, self.compression_label = self._create_slider("Compression:", 50, 150, int(self.compression*100))
    #     self.stripe_width_slider, self.stripe_width_label = self._create_slider("Stripe Width (nm):", 100, 2000, int(self.stripe_width*100))
    #     self.relax_width_slider, self.relax_width_label = self._create_slider("Relax Width (nm):", 1, 1000, int(self.relax_width*100))
    #     self.domain_type_combo = QComboBox(); self.domain_type_combo.addItems(['Heavy','Super Heavy','Light','Super Light'])
    #     self.symmetry_combo = QComboBox(); self.symmetry_combo.addItems(['Striped','Hexagonal'])
    #     domain_form.addRow(self.compression_label, self.compression_slider)
    #     domain_form.addRow(self.stripe_width_label, self.stripe_width_slider)
    #     domain_form.addRow(self.relax_width_label, self.relax_width_slider)
    #     domain_form.addRow("Domain Type:", self.domain_type_combo)
    #     domain_form.addRow("Symmetry:", self.symmetry_combo)
    #     domain_layout.addWidget(domain_group)
    #     domain_layout.addStretch()
    #     controls_main_layout.addLayout(domain_layout)

    #     # Kolumna 3: Wizualizacja i FFT
    #     vis_fft_layout = QVBoxLayout()
    #     vis_group = QGroupBox("Visualization")
    #     vis_form = QFormLayout(vis_group)
    #     self.sub_size_slider, self.sub_size_label = self._create_slider("Substrate Atom Size:", 10, 200, self.atom_size_sub)
    #     self.ads_size_slider, self.ads_size_label = self._create_slider("Adsorbate Atom Size:", 10, 400, self.atom_size_ads)
    #     vis_form.addRow(self.sub_size_label, self.sub_size_slider)
    #     vis_form.addRow(self.ads_size_label, self.ads_size_slider)
    #     vis_fft_layout.addWidget(vis_group)
        
    #     fft_group = QGroupBox("FFT Settings")
    #     fft_form = QFormLayout(fft_group)
    #     self.fft_window_combo = QComboBox(); self.fft_window_combo.addItems(['None', 'hann', 'hamming'])
    #     fft_form.addRow("Window:", self.fft_window_combo)
    #     vis_fft_layout.addWidget(fft_group)
    #     vis_fft_layout.addStretch()
    #     controls_main_layout.addLayout(vis_fft_layout)

    #     main_layout.addWidget(controls_group)
        
    #     button_box = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
    #     self.load_to_lfa_button = QPushButton("Load Simulation to Main Window")
    #     button_box.addButton(self.load_to_lfa_button, QDialogButtonBox.ButtonRole.ActionRole)
    #     main_layout.addWidget(button_box)
    #     self.button_box = button_box

    # def _display_experimental_data(self):
    #     """Populate the panel with results from experimental analysis."""
    #     # Clear previous entries
    #     while self.exp_layout.count():
    #         child = self.exp_layout.takeAt(0)
    #         if child.widget():
    #             child.widget().deleteLater()

    #     if not self.experimental_data:
    #         self.exp_layout.addWidget(QLabel("No experimental data provided."))
    #         return

    #     # Use QFormLayout for nicer alignment
    #     form_layout = QFormLayout()
        
    #     # 1. Transformacja Substratu
    #     transform = self.experimental_data.get('transform_analysis')
    #     if transform:
    #         form_layout.addRow(QLabel("<b>Substrate Transform:</b>"))
    #         rot = transform.get('rotation_angle_deg', '-')
    #         stretches = transform.get('principal_stretches', ['-','-'])
    #         rmse = transform.get('rmse', '-')
    #         form_layout.addRow("Rotation:", QLabel(f"{rot:.2f}°" if isinstance(rot, float) else "-"))
    #         form_layout.addRow("Stretches:", QLabel(f"({stretches[0]:.3f}, {stretches[1]:.3f})" if isinstance(stretches, np.ndarray) else "-"))
    #         form_layout.addRow("RMSE:", QLabel(f"{rmse:.3f} px" if isinstance(rmse, float) else "-"))

    #     # 2. Parametry Adsorbatu (z QComboBox do wyboru zestawu)
    #     adsorbate_sets = self.experimental_data.get('adsorbate_real_params')
    #     if adsorbate_sets:
    #         form_layout.addRow(QWidget()) # separator
    #         form_layout.addRow(QLabel("<b>Adsorbate Sets:</b>"))
    #         self.exp_adsorbate_combo = QComboBox()
    #         self.exp_adsorbate_combo.addItems([f"Set {i+1}" for i in adsorbate_sets.keys()])
    #         self.exp_adsorbate_label = QLabel() # Etykieta na wyniki
    #         self.exp_adsorbate_combo.currentIndexChanged.connect(self._on_exp_adsorbate_set_changed)
    #         form_layout.addRow("Select Set:", self.exp_adsorbate_combo)
    #         form_layout.addRow("Parameters:", self.exp_adsorbate_label)
    #         self._on_exp_adsorbate_set_changed(0) # Display data for the first set

    #     # 3. Domain wall parameters
    #     domain_walls = self.experimental_data.get('domain_wall_params')
    #     if domain_walls:
    #         form_layout.addRow(QWidget()) # separator
    #         form_layout.addRow(QLabel("<b>Domain Wall Analysis:</b>"))
    #         dist = domain_walls.get('dist_nm_inv', '-')
    #         period = domain_walls.get('periodicity_nm', '-')
    #         i_ratio = domain_walls.get('intensity_ratio', '-')
    #         form_layout.addRow("Δg*:", QLabel(f"{dist:.4f} nm⁻¹" if isinstance(dist, float) else "-"))
    #         form_layout.addRow("Periodicity:", QLabel(f"{period:.3f} nm" if isinstance(period, float) else "-"))
    #         form_layout.addRow("Intensity Ratio:", QLabel(f"{i_ratio:.3f}" if isinstance(i_ratio, float) else "-"))

    #     self.exp_layout.addLayout(form_layout)
    #     self.exp_layout.addStretch()

    def _create_slider(self, label_text: str, min_val: int, max_val: int, initial_val: int) -> Tuple[QSlider, QLabel]:
        """Helper that builds a horizontal slider paired with a label."""
        label = QLabel(label_text)
        slider = QSlider(Qt.Orientation.Horizontal)
        slider.setRange(min_val, max_val)
        slider.setValue(initial_val)
        return slider, label

    def _display_experimental_data(self):
        """Populate the experimental-analysis panel using the new layout."""
        for layout in [self.exp_sub_layout, self.exp_transform_layout, self.exp_ads_layout, self.exp_dw_layout]:
            while layout.count():
                item = layout.takeAt(0)
                if item.widget():
                    item.widget().deleteLater()

        # --- 1. Substrate Parameters ---
        sub_params = self.experimental_data.get('substrate_real_params')
        if sub_params:
            self.exp_sub_layout.addRow("Lattice Const. |a|:", QLabel(f"{sub_params.get('a1_nm', '-'):.3f} nm"))
            self.exp_sub_layout.addRow("Angle α:", QLabel(f"{sub_params.get('alpha_deg', '-'):.2f}°"))
        else: self.exp_sub_layout.addRow(QLabel("Not calculated."))

        # --- 2. Transformation Information ---
        transform = self.experimental_data.get('transform_analysis')
        if transform:
            rot = transform.get('rotation_angle_deg', '-')
            stretches = transform.get('principal_stretches', ['-','-'])
            rmse = transform.get('rmse', '-')
            self.exp_transform_layout.addRow("Rotation:", QLabel(f"{rot:.2f}°" if isinstance(rot, float) else "-"))
            self.exp_transform_layout.addRow("Stretches:", QLabel(f"({stretches[0]:.3f}, {stretches[1]:.3f})" if isinstance(stretches, np.ndarray) else "-"))
            self.exp_transform_layout.addRow("RMSE:", QLabel(f"{rmse:.3f} px" if isinstance(rmse, float) else "-"))
        else: self.exp_transform_layout.addRow(QLabel("Not calculated."))

        # --- 3. Adsorbate Parameters (from QComboBox) ---
        adsorbate_sets = self.experimental_data.get('adsorbate_real_params')
        if adsorbate_sets and isinstance(adsorbate_sets, dict):
            self.exp_adsorbate_combo = QComboBox(); self.exp_adsorbate_combo.addItems([f"Set {i+1}" for i in adsorbate_sets.keys()])
            self.exp_adsorbate_label = QLabel()
            self.exp_adsorbate_combo.currentIndexChanged.connect(self._on_exp_adsorbate_set_changed)
            self.exp_ads_layout.addWidget(QLabel("Select Set:"))
            self.exp_ads_layout.addWidget(self.exp_adsorbate_combo)
            self.exp_ads_layout.addWidget(self.exp_adsorbate_label)
            self._on_exp_adsorbate_set_changed(0)
        else: self.exp_ads_layout.addWidget(QLabel("Not calculated."))
        
        # --- 4. Domain Wall Parameters ---
        domain_walls = self.experimental_data.get('domain_wall_params')
        if domain_walls:
            dist = domain_walls.get('dist_nm_inv', '-'); period = domain_walls.get('periodicity_nm', '-')
            i_ratio = domain_walls.get('intensity_ratio', '-')
            a_ratio = domain_walls.get('amplitude_ratio', '-')
            m_ratio = domain_walls.get('max_value_ratio', '-')
            
            self.exp_dw_layout.addRow("Δg* (nm⁻¹):", QLabel(f"{dist:.4f}" if isinstance(dist, float) else "-"))
            self.exp_dw_layout.addRow("Periodicity (nm):", QLabel(f"{period:.3f}" if isinstance(period, float) else "-"))
            self.exp_dw_layout.addRow("Intensity Ratio:", QLabel(f"{i_ratio:.3f}" if isinstance(i_ratio, float) else "-"))
            self.exp_dw_layout.addRow("Amplitude Ratio:", QLabel(f"{a_ratio:.3f}" if isinstance(a_ratio, float) else "-"))
            self.exp_dw_layout.addRow("Max Value Ratio:", QLabel(f"{m_ratio:.3f}" if isinstance(m_ratio, np.float32) else "-"))
        else: self.exp_dw_layout.addRow(QLabel("Not calculated."))
        
        # Align all groups to the top
        # self.exp_sub_layout.parentWidget().layout().addStretch()
        self.exp_ads_layout.addStretch()
        # self.exp_dw_layout.parentWidget().layout().addStretch()
        
        # self.exp_dw_layout.addStretch()

    @pyqtSlot(int)
    def _on_exp_adsorbate_set_changed(self, index: int):
        adsorbate_sets = self.experimental_data.get('adsorbate_real_params', {})
        # Dictionary keys map to indexes; derive the key from the combo box index
        key = list(adsorbate_sets.keys())[index] if index < len(adsorbate_sets) else None
        
        if key is None or key not in adsorbate_sets:
            self.exp_adsorbate_label.setText("-"); return
        
        params = adsorbate_sets[key]
        a1=params.get('a1_nm','-'); a2=params.get('a2_nm','-'); alpha=params.get('alpha_deg','-')
        text = f"|a1|={a1:.3f}nm\n|a2|={a2:.3f}nm\nα={alpha:.2f}°"
        self.exp_adsorbate_label.setText(text)

    def _connect_signals(self):
        """Connect all relevant UI controls to the simulation update slot."""
        self.button_box.accepted.connect(self.accept)
        self.button_box.rejected.connect(self.reject)

        # Connect every control to the shared update slot
        self.substrate_combo.currentTextChanged.connect(self._update_simulation)
        self.adsorbate_combo.currentTextChanged.connect(self._update_simulation)
        self.compression_slider.valueChanged.connect(self._update_simulation)
        self.stripe_width_slider.valueChanged.connect(self._update_simulation)
        self.relax_width_slider.valueChanged.connect(self._update_simulation)
        self.domain_type_combo.currentTextChanged.connect(self._update_simulation)
        self.symmetry_combo.currentTextChanged.connect(self._update_simulation)
        self.sub_size_slider.valueChanged.connect(self._update_simulation)
        self.ads_size_slider.valueChanged.connect(self._update_simulation)
        self.fft_window_combo.currentTextChanged.connect(self._update_simulation)
        self.show_substrate_checkbox.stateChanged.connect(self._update_simulation)
        self.load_to_lfa_button.clicked.connect(self._on_load_to_lfa_clicked)
        self.analyze_sim_button.clicked.connect(self._on_analyze_simulated_peaks_clicked)

        self.resolution_multiplier_combo.currentTextChanged.connect(self._update_simulation)

    @pyqtSlot()
    def _on_analyze_simulated_peaks_clicked(self):
        """Run the automated analysis against the simulated FFT image."""
        params = self._get_current_simulation_parameters()
        sim_fft_data = self._calculate_fft(self._generate_image(params), params)
        if sim_fft_data is None: return

        peak_positions = self._calculate_theoretical_peak_positions(params)
        if not peak_positions:
            QMessageBox.warning(self, "Analysis Error", "Could not calculate theoretical peak positions."); return
            
        main_peak_pos_px, satellite_peak_pos_px = peak_positions
        
        logger.info(f"Analyzing theoretical main peak at ~{main_peak_pos_px}")
        main_analysis = self._analyze_peak_at_coords(sim_fft_data, main_peak_pos_px)
        
        logger.info(f"Analyzing theoretical satellite peak at ~{satellite_peak_pos_px}")
        satellite_analysis = self._analyze_peak_at_coords(sim_fft_data, satellite_peak_pos_px)

        self._display_automated_analysis_results(main_analysis, satellite_analysis)

    def _calculate_theoretical_peak_positions(self, params: Dict[str, Any]) -> Optional[Tuple[Tuple[float, float], Tuple[float, float]]]:
        """
        Calculates the theoretical positions of the main commensurate peak and the
        first satellite peak based on simulation parameters.

        Returns:
            A tuple containing two tuples: (main_peak_coords_px, satellite_peak_coords_px),
            or (None, None) if calculation is not possible.
        """
        substrate_info = KNOWN_LATTICES.get(params.get('substrate_name'))
        if not (substrate_info and get_reciprocal_vectors):
            logger.warning("Cannot calculate theoretical positions: substrate info or get_reciprocal_vectors missing.")
            return None, None

        a_sub_nm = substrate_info['a_surf']
        sub_a1 = np.array([a_sub_nm, 0])
        sub_a2 = np.array([a_sub_nm / 2, a_sub_nm * np.sqrt(3) / 2])
        ads_ideal_a1 = sub_a1 + sub_a2
        ads_ideal_a2 = -sub_a1 + 2 * sub_a2
        
        T = np.array([[params.get('compression', 1.0), 0], [0, 1]])
        a1_prime = T @ ads_ideal_a1
        a2_prime = T @ ads_ideal_a2

        area_i = abs(np.cross(a1_prime, a2_prime))
        if area_i < 1e-9: return None, None
        
        b1_prime = 2 * np.pi * np.array([a2_prime[1], -a2_prime[0]]) / area_i
        b2_prime = 2 * np.pi * np.array([-a1_prime[1], a1_prime[0]]) / area_i

        # Position of the main peak in reciprocal space (nm⁻¹)
        main_peak_k_space = b1_prime

        l = params.get('stripe_width', 5.0)
        if l < 1e-9: return None, None
        
        # Following the cited paper, epsilon = 4*pi / (3*l)
        epsilon_magnitude = (4 * np.pi) / (3 * l)
        
        if params.get('symmetry') == 'Striped':
            # For striped patterns the inner triplet sits at epsilon/2
            split_magnitude = epsilon_magnitude / 2.0
        else: # Hexagonal
            split_magnitude = epsilon_magnitude
            
        # Splitting direction follows the reciprocal lattice vector
        direction_vector = main_peak_k_space / np.linalg.norm(main_peak_k_space)
        epsilon_vec_k_space = direction_vector * split_magnitude

        satellite_peak_k_space = main_peak_k_space + epsilon_vec_k_space

        fft_Ny, fft_Nx = params['px_y'], params['px_x']
        Lx, Ly = params['nm_x'], params['nm_y']
        
        main_kx_px = (main_peak_k_space[0] * Lx / (2*np.pi)) + fft_Nx/2
        main_ky_px = (main_peak_k_space[1] * Ly / (2*np.pi)) + fft_Ny/2
        
        sat_kx_px = (satellite_peak_k_space[0] * Lx / (2*np.pi)) + fft_Nx/2
        sat_ky_px = (satellite_peak_k_space[1] * Ly / (2*np.pi)) + fft_Ny/2

        return (main_kx_px, main_ky_px), (sat_kx_px, sat_ky_px)

    def _analyze_peak_at_coords(self, fft_image: np.ndarray, coords_px: Tuple[float, float]) -> Optional[Dict[str, float]]:
        """
        Performs a detailed analysis of a small region around the provided FFT coordinates.
        """
        center_kx, center_ky = coords_px
        roi_radius = 3 # Use a fixed 7x7 ROI for analysis
        
        if not (fit_2d_gaussian_in_roi_with_all_data and callable(fit_2d_gaussian_in_roi_with_all_data)):
            return None
            
        fit_res = fit_2d_gaussian_in_roi_with_all_data(fft_image, (int(round(center_ky)), int(round(center_kx))), roi_radius)
        if not fit_res:
            logger.warning(f"Automated Gaussian fit failed for coordinates {coords_px}.")
            # Fallback: average a 3x3 neighbourhood
            y_start, x_start = int(round(center_ky))-1, int(round(center_kx))-1
            patch = fft_image[y_start:y_start+3, x_start:x_start+3]
            return {
                "intensity_volume": np.sum(patch),
                "amplitude": np.max(patch),
                "max_value": np.max(patch)
            }

        popt_fit, (fky_abs, fkx_abs), roi_patch_used = fit_res
        
        amplitude, _, _, sigma_y, sigma_x, _, _ = popt_fit
        intensity_volume = 2 * np.pi * abs(amplitude) * abs(sigma_x) * abs(sigma_y)
        max_value = amplitude # Max value z dopasowania
        
        return {
            "intensity_volume": intensity_volume,
            "amplitude": abs(amplitude),
            "max_value": max_value
        }

    def _display_automated_analysis_results(self, main_results: Optional[Dict[str, float]], sat_results: Optional[Dict[str, float]]):
        """Display the automated analysis outcome in the dialog."""
        if not main_results:
            self.auto_main_label.setText("Main Peak: Analysis failed")
            return
        
        main_intensity = main_results['intensity_volume']
        main_amplitude = main_results['amplitude']
        main_max_val = main_results['max_value']
        self.auto_main_label.setText(f"Main Peak: I={main_intensity:.2e}, A={main_amplitude:.2e}, Max={main_max_val:.2e}")

        if not sat_results:
            self.auto_sat_label.setText("Satellite Peak: Analysis failed")
            self.auto_ratio_label.setText("Ratios (Sat/Main): N/A")
            return

        sat_intensity = sat_results['intensity_volume']
        sat_amplitude = sat_results['amplitude']
        sat_max_val = sat_results['max_value']
        self.auto_sat_label.setText(f"Satellite Peak: I={sat_intensity:.2e}, A={sat_amplitude:.2e}, Max={sat_max_val:.2e}")

        # Compute and display ratios
        intensity_ratio = sat_intensity / main_intensity if main_intensity > 1e-9 else float('inf')
        amplitude_ratio = sat_amplitude / main_amplitude if main_amplitude > 1e-9 else float('inf')
        max_value_ratio = sat_max_val / main_max_val if main_max_val > 1e-9 else float('inf')

        ratios_text = (f"Intensity: {intensity_ratio:.3f} | "
                       f"Amplitude: {amplitude_ratio:.3f} | "
                       f"Max Value: {max_value_ratio:.3f}")
        self.auto_ratio_label.setText(f"Ratios (Sat/Main): {ratios_text}")

    def _get_current_simulation_parameters(self) -> Dict[str, Any]:
        """Collect the latest parameters from the UI controls."""
        params = self.sim_params.copy() # Start with px_x, px_y, nm_x, nm_y

        multiplier_text = self.resolution_multiplier_combo.currentText() # "1x", "2x", "4x"
        multiplier = int(multiplier_text.split('x')[0])
        
        params['px_x'] = self.sim_params['px_x'] * multiplier
        params['px_y'] = self.sim_params['px_y'] * multiplier
        params['substrate_name'] = self.substrate_combo.currentText()
        params['adsorbate_name'] = self.adsorbate_combo.currentText()
        params['compression'] = self.compression_slider.value() / 100.0
        params['stripe_width'] = self.stripe_width_slider.value() / 100.0
        params['relax_width'] = self.relax_width_slider.value() / 100.0
        params['domain_type'] = self.domain_type_combo.currentText()
        params['symmetry'] = self.symmetry_combo.currentText()
        params['atom_size_sub'] = self.sub_size_slider.value()
        params['atom_size_ads'] = self.ads_size_slider.value()
        params['fft_window_type'] = self.fft_window_combo.currentText()
        params['show_substrate'] = self.show_substrate_checkbox.isChecked()
        
        return params
    
    def _generate_image(self, params: Dict[str, Any]) -> Optional[np.ndarray]:
        img_size_y, img_size_x = params['px_y'], params['px_x']
        pixel_per_nm = params['px_x'] / params['nm_x'] 

        sub_accumulator = np.zeros((img_size_y, img_size_x), dtype=np.float32)
        ads_accumulator = np.zeros((img_size_y, img_size_x), dtype=np.float32)

        def splat(coords: Optional[np.ndarray], accumulator: np.ndarray):
            if coords is None or coords.size == 0:
                return
            px = coords[:, 0] * pixel_per_nm
            py = coords[:, 1] * pixel_per_nm
            ix = np.round(px).astype(int)
            iy = np.round(py).astype(int)
            
            mask = (ix >= 0) & (ix < img_size_x) & (iy >= 0) & (iy < img_size_y)
            valid_ix = ix[mask]
            valid_iy = iy[mask]
            
            accumulator[valid_iy, valid_ix] = 1.0

        if params.get('show_substrate', True):
            sub_coords = self._get_substrate_coords(params)
            splat(sub_coords, sub_accumulator)
        
        ads_coords = self._get_adsorbate_coords(params)
        splat(ads_coords, ads_accumulator)
        
        final_image = np.zeros((img_size_y, img_size_x), dtype=np.float32)
        
        if params.get('show_substrate', True) and np.any(sub_accumulator):
            sigma_sub = max(params.get('atom_size_sub', 50) / 50.0, 0.1)
            sub_layer = gaussian_filter(sub_accumulator, sigma=sigma_sub)
            if sub_layer.max() > 0:
                final_image += sub_layer / sub_layer.max()  # Normalizuj i dodaj

        if np.any(ads_accumulator):
            sigma_ads = max(params.get('atom_size_ads', 50) / 50.0, 0.1)
            ads_layer = gaussian_filter(ads_accumulator, sigma=sigma_ads)
            if ads_layer.max() > 0:
                final_image += ads_layer / ads_layer.max()  # Normalizuj i dodaj

        if final_image.max() > 0:
            return final_image / final_image.max()
            
        return final_image
    
    # def _generate_image(self, params: Dict[str, Any]) -> Optional[np.ndarray]:
    #     """Generate STM topography based on the parameters."""
    #     img_size_y, img_size_x = params['px_y'], params['px_x']
    #     # Use a single scale to avoid distortion when pixels are non-square
    #     pixel_per_nm = params['px_x'] / params['nm_x']

    #     # Generuj koordynaty
    #     sub_coords = self._get_substrate_coords(params)
    #     ads_coords = self._get_adsorbate_coords(params)

    #     img = np.zeros((img_size_y, img_size_x), dtype=np.float32)

    #     def splat(coords: Optional[np.ndarray], intensity: float):
    #         """
    #         Helper for drawing atoms; now handles edges safely.
    #         """
    #         if coords is None or coords.size == 0:
    #             return

    #         # Skaluj pozycje w nm na piksele
    #         px = coords[:, 0] * pixel_per_nm
    #         py = coords[:, 1] * pixel_per_nm
            
    #         # Round to the nearest integer coordinates
    #         ix = np.round(px).astype(int)
    #         iy = np.round(py).astype(int)
            
    #         # Validate rounded indices fall within [0, size-1]
    #         mask = (ix >= 0) & (ix < img_size_x) & (iy >= 0) & (iy < img_size_y)
            
    #         # Use only valid indices
    #         valid_ix = ix[mask]
    #         valid_iy = iy[mask]
            
    #         # Add intensity only at valid locations
    #         img[valid_iy, valid_ix] += intensity

    #     # "Painting" the atoms
    #     # splat(sub_coords, intensity=0.5)
    #     if params.get('show_substrate', True):
    #         sub_coords = self._get_substrate_coords(params)
    #         splat(sub_coords, intensity=0.5)
    #     splat(ads_coords, intensity=1.0)
        
    #     if not img.any():
    #         return img

    #     # # Gaussian blur for a realistic look
    #     # sigma = max((params.get('atom_size_sub', 50) + params.get('atom_size_ads', 50)) / 200.0, 0.1)
    #     # if sigma > 0 and pg:
    #     #      img = pg.gaussianFilter(img, (sigma, sigma))
        
    #     # return img / img.max() if img.max() > 0 else img

    #     if params.get('show_substrate', True):
    #         # If the substrate is visible, average both atom sizes
    #         sigma = max((params.get('atom_size_sub', 50) + params.get('atom_size_ads', 50)) / 200.0, 0.1)
    #     else:
    #         # If the substrate is hidden, base blur only on adsorbate atom size
    #         # Use 100.0 instead of 200.0 to keep a similar blur scale
    #         sigma = max(params.get('atom_size_ads', 50) / 100.0, 0.1)

    #     if sigma > 0 and pg and img.any():
    #          img = pg.gaussianFilter(img, (sigma, sigma))
    #     # --------------------------------------------------------
        
    #     return img / img.max() if img.max() > 0 else img
    
    # def _generate_image(self, params: Dict[str, Any]) -> Optional[np.ndarray]:
    #     """Generate STM topography based on the parameters."""
    #     img_size = params['px_x'] # Use experimental image dimensions
    #     pixel_per_nm = params['px_x'] / params['nm_x']

    #     acc = np.zeros((img_size, img_size), dtype=float)
        
    #     def splat(coords: np.ndarray, size_param: int):
    #         """Helper for drawing atoms."""
    #         for p in coords:
    #             x, y = p[0] * pixel_per_nm, p[1] * pixel_per_nm
    #             xf, yf = int(np.floor(x)), int(np.floor(y))
    #             if 0 <= xf < img_size and 0 <= yf < img_size:
    #                 acc[yf, xf] += 1.0 # Simpler "drawing" for performance
        
    #     # Generuj koordynaty substratu i adsorbatu
    #     sub_coords = self._get_substrate_coords(params)
    #     ads_coords = self._get_adsorbate_coords(params, sub_coords)
        
    #     if sub_coords is not None: splat(sub_coords, params['atom_size_sub'])
    #     if ads_coords is not None: splat(ads_coords, params['atom_size_ads'])
        
    #     if not acc.any(): return np.zeros((img_size, img_size), dtype=np.float32)

    #     # Gaussian blur for a realistic appearance
    #     # Atom radius controls blur sigma
    #     sigma = max((params['atom_size_sub'] + params['atom_size_ads']) / 200.0, 0.1)
    #     img = pg.gaussianFilter(acc, (sigma, sigma))
        
    #     return img / img.max() if img.max() > 0 else img

    # --- Main simulation/update routine ---
    @pyqtSlot()
    def _update_simulation(self):
        """Main loop: collect parameters, generate the synthetic image, compute the FFT, and refresh the views."""
        if not self.isVisible(): return # Skip updates when the dialog is hidden
        
        params = self._get_current_simulation_parameters()
        
        # Update slider labels
        self.compression_label.setText(f"Compression: {params['compression']:.2f}")
        self.stripe_width_label.setText(f"Stripe Width: {params['stripe_width']:.2f} nm")
        self.relax_width_label.setText(f"Relax Width: {params['relax_width']:.2f} nm")
        self.sub_size_label.setText(f"Sub. Atom Size: {params['atom_size_sub']}")
        self.ads_size_label.setText(f"Ads. Atom Size: {params['atom_size_ads']}")
        
        # Generate STM image
        stm_image = self._generate_image(params)
        if stm_image is None: logger.error("STM image generation failed."); return
        
        # Compute FFT
        fft_image = self._calculate_fft(stm_image, params)
        if fft_image is None: logger.error("FFT calculation failed."); return

        exp_h, exp_w = self.experimental_fft_image.shape
        target_rect = QRectF(0, 0, exp_w, exp_h)

        if not self._initial_update_done:
            self.sim_stm_image_item.setImage(stm_image.T, autoLevels=True)
        else:
            self.sim_stm_image_item.setImage(stm_image.T, autoLevels=False)

        if not self._initial_update_done:
            self.exp_fft_image_item.setImage(self.experimental_fft_image.T, autoLevels=True)
            self.sim_fft_image_item.setImage(fft_image.T, autoLevels=True, rect=target_rect)
            
            self.sim_fft_plot.setRange(xRange=(0, exp_w), yRange=(0, exp_h))
            
            self._initial_update_done = True
        else:
            self.exp_fft_image_item.setImage(self.experimental_fft_image.T, autoLevels=False)
            self.sim_fft_image_item.setImage(fft_image.T, autoLevels=False, rect=target_rect)

        # if not self._initial_update_done:
        #     # Pierwsze uruchomienie: ustaw autoLevels i zakresy
        #     self.sim_stm_image_item.setImage(stm_image.T, autoLevels=True)
        #     self.sim_fft_image_item.setImage(fft_image.T, autoLevels=True)
            
        #     # Configure axis ranges just once
        #     px_x, px_y = params['px_x'], params['px_y']
        #     self.sim_stm_plot.setRange(xRange=(0, px_x), yRange=(0, px_y))
            
        #     fft_Ny, fft_Nx = fft_image.shape
        #     self.sim_fft_plot.setRange(xRange=(0, fft_Nx), yRange=(0, fft_Ny))

        #     self._initial_update_done = True # Mark initial update as done
        # else:
        #     # Subsequent updates swap the data but keep the view
        #     self.sim_stm_image_item.setImage(stm_image.T, autoLevels=False)
        #     self.sim_fft_image_item.setImage(fft_image.T, autoLevels=False)

        # # Display images
        # self.sim_stm_image_item.setImage(stm_image.T, autoLevels=True)
        # self.sim_fft_image_item.setImage(fft_image.T, autoLevels=True) # Poziomy kontrolowane przez histogram
        
        # # Set axis ranges for views
        # # Lx, Ly = params['nm_x'], params['nm_y']
        # px_x, px_y = params['px_x'], params['px_y']
        # self.sim_stm_plot.setRange(xRange=(0,px_x), yRange=(0,px_y))
        
        # # k_range = np.pi / (px_x/params['px_x']) # Zakres k od -k_max do k_max
        # fft_Ny, fft_Nx = fft_image.shape
        # self.sim_fft_plot.setRange(xRange=(0,fft_Nx), yRange=(0,fft_Ny))

        self._update_simulation_results_display(params)

    @pyqtSlot()
    def _on_load_to_lfa_clicked(self):
        """
        When the user clicks "Load...", the dialog finalizes data, creates a HistoryNode, and emits it.
        """
        logger.info("'Load Simulation to Main Window' clicked.")
        
        # 1. Gather current parameters and generate data
        params = self._get_current_simulation_parameters()
        stm_image = self._generate_image(params)
        if stm_image is None:
            QMessageBox.critical(self, "Error", "Failed to generate simulated STM image."); return

        # Compute final FFT (power spectrum, |F|^2)
        fft_power_spectrum = self._calculate_fft(stm_image, params)
        if fft_power_spectrum is None:
            QMessageBox.critical(self, "Error", "Failed to calculate simulated FFT."); return

        # 2. Find the parent node in history (original experimental image)
        # current_fft_node_id was provided via the constructor
        root_node = self.history_manager.get_root_node_for_node(self.current_node_id)
        if not root_node:
            QMessageBox.critical(self, "Error", "Could not find the original image in history to attach the simulation to."); return
            
        # 3. Create a new HistoryNode
        # Important: add 'scaling_mode': 'power' to the parameters
        simulation_and_fft_params = params.copy()
        simulation_and_fft_params['scaling_mode'] = 'power'

        new_node = HistoryNode(
            parent_id=root_node.node_id,
            operation_name="Simulated FFT",
            parameters=simulation_and_fft_params,
            image_data=fft_power_spectrum, # Zapisz dane |F|^2
            data_type="FFT"
        )

        # 4. Emit the new node and close the dialog
        self.simulation_accepted.emit(new_node)
        self.accept()
    
    def _update_simulation_results_display(self, params: Dict[str, Any]):
        """Compute and display lattice parameters for the simulation."""
        # Clear previous results
        while self.sim_results_layout.count():
            self.sim_results_layout.takeAt(0).widget().deleteLater()

        # Calculations for substrate
        sub_info = KNOWN_LATTICES.get(params['substrate_name'])
        if sub_info:
            a = sub_info['a_surf']
            if sub_info['type'] == 'hexagonal':
                self.sim_results_layout.addRow(QLabel("<b>Substrate (Hexagonal):</b>"))
                self.sim_results_layout.addRow(QLabel(f"  |a1|=|a2| = {a:.3f} nm, α=120°"))
            elif sub_info['type'] == 'square':
                self.sim_results_layout.addRow(QLabel("<b>Substrate (Square):</b>"))
                self.sim_results_layout.addRow(QLabel(f"  |a1|=|a2| = {a:.3f} nm, α=90°"))

        # Calculations for adsorbate
        if params['adsorbate_name'] != "None" and sub_info and sub_info['type'] == 'hexagonal':
            sub_a1 = np.array([sub_info['a_surf'], 0])
            sub_a2 = np.array([sub_info['a_surf']/2, sub_info['a_surf']*np.sqrt(3)/2])
            ads_ideal_a1 = sub_a1 + sub_a2
            ads_ideal_a2 = -sub_a1 + 2 * sub_a2
            T = np.array([[params.get('compression', 1.0), 0], [0, 1]])
            a1_prime = T @ ads_ideal_a1
            a2_prime = T @ ads_ideal_a2

            cos_alpha = np.dot(a1_prime, a2_prime) / (np.linalg.norm(a1_prime) * np.linalg.norm(a2_prime))
            alpha_deg = np.degrees(np.arccos(np.clip(cos_alpha, -1.0, 1.0)))
            
            self.sim_results_layout.addRow(QWidget()) # separator
            self.sim_results_layout.addRow(QLabel("<b>Adsorbate (Simulated):</b>"))
            self.sim_results_layout.addRow("a1' vector:", QLabel(f"({a1_prime[0]:.3f}, {a1_prime[1]:.3f}) nm"))
            self.sim_results_layout.addRow("a2' vector:", QLabel(f"({a2_prime[0]:.3f}, {a2_prime[1]:.3f}) nm"))
            self.sim_results_layout.addRow("Resulting |a1'|:", QLabel(f"{np.linalg.norm(a1_prime):.3f} nm"))
            self.sim_results_layout.addRow("Resulting |a2'|:", QLabel(f"{np.linalg.norm(a2_prime):.3f} nm"))
            self.sim_results_layout.addRow("Resulting Angle α':", QLabel(f"{alpha_deg:.2f}°"))


    def _get_substrate_coords(self, params: Dict[str, Any]) -> Optional[np.ndarray]:
        """General helper for generating substrate coordinates."""
        lattice_name = params['substrate_name']
        if lattice_name not in KNOWN_LATTICES: return None
        
        info = KNOWN_LATTICES[lattice_name]
        a_sub_nm = info['a_surf']
        l_type = info['type']
        L = params['nm_x']

        if l_type == 'hexagonal':
            a1 = np.array([a_sub_nm, 0])
            a2 = np.array([a_sub_nm/2, a_sub_nm * np.sqrt(3)/2])
        elif l_type == 'square':
            a1 = np.array([a_sub_nm, 0])
            a2 = np.array([0, a_sub_nm])
        else: return None

        pts = []
        N = int(np.ceil((L/2)/a_sub_nm)*1.5) + 5
        for i in range(-N, N+1):
            for j in range(-N, N+1):
                pts.append(i*a1 + j*a2 + np.array([L/2, L/2]))
        return np.array(pts)
        
    def _get_adsorbate_coords(self, params: Dict[str, Any]) -> Optional[np.ndarray]:
        """
        General helper for generating adsorbate coordinates, accounting for
        compression and domain-wall modelling.
        """
        if params.get('adsorbate_name') == "None":
            return None # If no adsorbate is selected, return None
        
        # Fetch substrate basis vectors used to define the adsorbate lattice
        substrate_info = KNOWN_LATTICES.get(params['substrate_name'])
        if not substrate_info:
            return None
        
        a_sub_nm = substrate_info['a_surf']
        sub_a1 = np.array([a_sub_nm, 0])
        sub_a2 = np.array([a_sub_nm/2, a_sub_nm * np.sqrt(3)/2])
        
        # --- Logic specific to "Iodine (predefined)" ---
        # Future work: extend to additional adsorbates
        if params['adsorbate_name'] == "Iodine (predefined)":
            # Define the ideal adsorbate lattice relative to the substrate
            fcc_offset = (sub_a1 + sub_a2) / 3
            ads_ideal_a1 = sub_a1 + sub_a2
            ads_ideal_a2 = -sub_a1 + 2 * sub_a2

            # Apply uniaxial compression
            T = np.array([[params['compression'], 0], [0, 1]])
            a1 = T @ ads_ideal_a1
            a2 = T @ ads_ideal_a2
            
            # Domain wall parameters
            phi_map = {'Heavy': 1/3., 'Super Heavy': 2/3., 'Light': -1/3., 'Super Light': -2/3.}
            shift = phi_map.get(params['domain_type'], 0) * a1
            
            # Generate point coordinates
            pts = []
            L = params['nm_x']
            spacing = min(np.linalg.norm(a1), np.linalg.norm(a2))
            if spacing < 1e-6: spacing = a_sub_nm
            N = int(np.ceil((L/2)/spacing)*1.5) + 15

            if params['symmetry'] == 'Striped':
                stripe_width = params.get('stripe_width', 5.0)
                relax_width = params.get('relax_width', 2.0)
                for i in range(-N, N + 1):
                    for j in range(-N, N + 1):
                        base = i*a1 + j*a2 + fcc_offset
                        p = base + np.array([L/2, L/2]) # Centrowanie
                        if stripe_width < 1e-6 or relax_width < 1e-6:
                            pts.append(p)
                            continue
                        d = int(np.floor(p[0] / stripe_width))
                        xp = p[0] - d * stripe_width
                        t = (xp - stripe_width / 2) / relax_width
                        f = 0 if t < -20 else (1 if t > 20 else (1 + np.tanh(t)) / 2)
                        pts.append(p + shift * (f if d % 2 else 1 - f))
            else: # Hexagonal symmetry
                S = int(N / 4) + 3
                M = 4
                for i in range(-S, S + 1):
                    for j in range(-S, S + 1):
                        dom = ((i + j) % 3) / 3.0 * shift
                        for u in range(-M, M + 1):
                            for v in range(-M, M + 1):
                                base = u*a1 + v*a2 + fcc_offset + dom
                                pts.append(base + np.array([L/2, L/2]))
            
            return np.array(pts)

        elif params['adsorbate_name'] == "<Custom Define...>":
            logger.warning("Custom adsorbate definition is not yet implemented.")
            return None
            
        return None
    
    def _calculate_fft(self, image_data: np.ndarray, params: Dict[str, Any]) -> Optional[np.ndarray]:
        """
        Compute the FFT from simulation parameters and return the power spectrum (|F|^2).
        """
        # Use fft_engine.calculate_fft (handles window functions)
        fft_complex = calculate_fft(
            image_data, 
            apply_window=(params['fft_window_type'] != 'None'), 
            window_type=params['fft_window_type'].lower()
        )
        
        if fft_complex is None: 
            return None
        
        # --- Change: return squared magnitude (|F|^2) ---
        # This is the power spectrum, required for intensity analysis.
        magnitude_squared = np.abs(fft_complex)**2
        
        return magnitude_squared.astype(np.float32)
