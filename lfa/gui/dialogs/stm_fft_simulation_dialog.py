# lfa/gui/dialogs/stm_fft_simulation_dialog.py
import logging
import numpy as np
from typing import Optional, Dict, Any, Tuple

from PyQt6.QtCore import Qt, pyqtSlot
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget,
    QVBoxLayout, QHBoxLayout, QGroupBox,
    QCheckBox, QLabel, QFormLayout, QPushButton,
    QSlider, QGroupBox,
    QComboBox, QDialog, QSplitter, QDialogButtonBox
)

try:
    import pyqtgraph as pg
    from pyqtgraph import GraphicsLayoutWidget, ImageItem
    PYQTGRAPH_AVAILABLE = True
except ImportError: # pragma: no cover
    pg = None; GraphicsLayoutWidget = None; ImageItem = None; PYQTGRAPH_AVAILABLE = False
    logging.error("StmFftSimulationDialog: PyQtGraph not found.")

try:
    from ...analysis.lattice import KNOWN_LATTICES
except ImportError:
    KNOWN_LATTICES = {"Error": {"type": "hexagonal", "a_surf": 0.3}}
    logging.error("StmFftSimulationDialog: Could not import KNOWN_LATTICES.")

logger = logging.getLogger(__name__)

class StmFftSimulationDialog(QDialog):
    """
    Dialog for creating simulated STM/FFT data based on user-defined parameters.
    """
    def __init__(self, 
                 experimental_fft_image: np.ndarray,
                 experimental_data: Dict[str, Any],
                 simulation_params: Dict[str, Any],
                 parent=None):
        super().__init__(parent)
        self.setWindowTitle("STM/FFT Simulation & Comparison")
        self.setMinimumSize(1400, 800)

        self.experimental_fft_image = experimental_fft_image
        self.experimental_data = experimental_data
        self.sim_params = simulation_params
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
        
        if self.exp_fft_image_item:
            self.exp_fft_image_item.setImage(self.experimental_fft_image.T)

        self._update_simulation()

    def _init_ui(self):
        main_layout = QVBoxLayout(self)
        
        # --- GÓRNA SEKCJA: KONTROLKI I WYNIKI EKSPERYMENTALNE ---
        top_widget = QWidget()
        top_layout = QHBoxLayout(top_widget)
        top_layout.setContentsMargins(0, 0, 0, 0)
        
        # Kontrolki symulacji
        sim_controls_group = QGroupBox("Simulation Controls")
        self._create_simulation_controls(sim_controls_group)
        top_layout.addWidget(sim_controls_group, 2) # Daj więcej miejsca na kontrolki

        # Wyniki eksperymentalne
        exp_group = QGroupBox("Experimental Parameters (Reference)")
        exp_main_layout = QHBoxLayout(exp_group)
        top_layout.addWidget(exp_group, 3) # Daj więcej miejsca na wyniki

        # Stworzenie pustych layoutów dla podgrup
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
        exp_fft_widget = GraphicsLayoutWidget()
        self.exp_fft_plot = exp_fft_widget.addPlot(title="Experimental FFT")
        self.exp_fft_plot.setAspectLocked(True)
        self.exp_fft_image_item = ImageItem()
        self.exp_fft_plot.addItem(self.exp_fft_image_item)
        
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
        """Metoda pomocnicza do tworzenia wszystkich kontrolek symulacji."""
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
        lattice_layout.addStretch()
        
        domain_layout = QVBoxLayout()
        domain_group = QGroupBox("Domain & Misfit")
        domain_form = QFormLayout(domain_group)
        self.compression_slider, self.compression_label = self._create_slider("Compression:", 50, 150, int(self.compression*100))
        self.stripe_width_slider, self.stripe_width_label = self._create_slider("Stripe Width (nm):", 100, 2000, int(self.stripe_width*100))
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
        vis_fft_layout.addWidget(vis_group)
        fft_group = QGroupBox("FFT Settings")
        fft_form = QFormLayout(fft_group)
        self.fft_window_combo = QComboBox()
        self.fft_window_combo.addItems(['None', 'hann', 'hamming'])
        fft_form.addRow("Window:", self.fft_window_combo)
        vis_fft_layout.addWidget(fft_group)
        vis_fft_layout.addStretch()

        controls_main_layout.addLayout(lattice_layout)
        controls_main_layout.addLayout(domain_layout)
        controls_main_layout.addLayout(vis_fft_layout)


    # def _init_ui(self):
    #     main_layout = QVBoxLayout(self)
        
    #     # --- Górny panel z trzema widokami ---
    #     top_splitter = QSplitter(Qt.Orientation.Horizontal)

    #     controls_widget = QWidget()
    #     controls_layout = QHBoxLayout(controls_widget)

    #     exp_group = QGroupBox("Experimental Parameters (Reference)")
    #     self.exp_layout = QVBoxLayout(exp_group)
    #     controls_layout.addWidget(exp_group, 1) # Rozciągliwy
        
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
        
    #     # Synchronizacja widoków FFT
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
    #     """Wypełnia panel danymi z analizy eksperymentalnej."""
    #     # Wyczyść poprzednie dane
    #     while self.exp_layout.count():
    #         child = self.exp_layout.takeAt(0)
    #         if child.widget():
    #             child.widget().deleteLater()

    #     if not self.experimental_data:
    #         self.exp_layout.addWidget(QLabel("No experimental data provided."))
    #         return

    #     # Użyj QFormLayout dla lepszego wyglądu
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
    #         self._on_exp_adsorbate_set_changed(0) # Wyświetl dane dla pierwszego zestawu

    #     # 3. Parametry Ścian Domenowych
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

    def _display_experimental_data(self):
        """Wypełnia panel danymi z analizy eksperymentalnej w nowym układzie."""
        for layout in [self.exp_sub_layout, self.exp_transform_layout, self.exp_ads_layout, self.exp_dw_layout]:
            while layout.count():
                item = layout.takeAt(0)
                if item.widget():
                    item.widget().deleteLater()

        # --- 1. Parametry Substratu ---
        sub_params = self.experimental_data.get('substrate_real_params')
        if sub_params:
            self.exp_sub_layout.addRow("Lattice Const. |a|:", QLabel(f"{sub_params.get('a1_nm', '-'):.3f} nm"))
            self.exp_sub_layout.addRow("Angle α:", QLabel(f"{sub_params.get('alpha_deg', '-'):.2f}°"))
        else: self.exp_sub_layout.addRow(QLabel("Not calculated."))

        # --- 2. Informacje o Transformacji ---
        transform = self.experimental_data.get('transform_analysis')
        if transform:
            rot = transform.get('rotation_angle_deg', '-')
            stretches = transform.get('principal_stretches', ['-','-'])
            rmse = transform.get('rmse', '-')
            self.exp_transform_layout.addRow("Rotation:", QLabel(f"{rot:.2f}°" if isinstance(rot, float) else "-"))
            self.exp_transform_layout.addRow("Stretches:", QLabel(f"({stretches[0]:.3f}, {stretches[1]:.3f})" if isinstance(stretches, np.ndarray) else "-"))
            self.exp_transform_layout.addRow("RMSE:", QLabel(f"{rmse:.3f} px" if isinstance(rmse, float) else "-"))
        else: self.exp_transform_layout.addRow(QLabel("Not calculated."))

        # --- 3. Parametry Adsorbatu (z QComboBox) ---
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
        
        # --- 4. Parametry Ścian Domenowych ---
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
        
        # Wyrównaj wszystkie grupy na górze
        # self.exp_sub_layout.parentWidget().layout().addStretch()
        self.exp_ads_layout.addStretch()
        # self.exp_dw_layout.parentWidget().layout().addStretch()
        
        # self.exp_dw_layout.addStretch()

    @pyqtSlot(int)
    def _on_exp_adsorbate_set_changed(self, index: int):
        adsorbate_sets = self.experimental_data.get('adsorbate_real_params', {})
        # Klucze słownika to indeksy, musimy pobrać klucz na podstawie indeksu comboboxa
        key = list(adsorbate_sets.keys())[index] if index < len(adsorbate_sets) else None
        
        if key is None or key not in adsorbate_sets:
            self.exp_adsorbate_label.setText("-"); return
        
        params = adsorbate_sets[key]
        a1=params.get('a1_nm','-'); a2=params.get('a2_nm','-'); alpha=params.get('alpha_deg','-')
        text = f"|a1|={a1:.3f}nm\n|a2|={a2:.3f}nm\nα={alpha:.2f}°"
        self.exp_adsorbate_label.setText(text)

    def _create_slider(self, label_text: str, min_val: int, max_val: int, initial_val: int) -> Tuple[QSlider, QLabel]:
        """Metoda pomocnicza do tworzenia suwaka z etykietą."""
        label = QLabel(label_text)
        slider = QSlider(Qt.Orientation.Horizontal)
        slider.setRange(min_val, max_val)
        slider.setValue(initial_val)
        return slider, label

    def _connect_signals(self):
        # TODO: Implementacja w kolejnym kroku
        self.button_box.accepted.connect(self.accept)
        self.button_box.rejected.connect(self.reject)
        logger.debug("StmFftSimulationDialog signals will be connected later.")

    def _update_simulation(self):
        # TODO: Implementacja w kolejnym kroku
        pass