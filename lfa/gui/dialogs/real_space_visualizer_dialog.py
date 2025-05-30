# lfa/gui/dialogs/real_space_visualizer_dialog.py
import logging
import numpy as np
from typing import Optional, List, Dict, Any, Tuple

from PyQt6.QtWidgets import (
    QDialog, QHBoxLayout, QVBoxLayout, QWidget, QGroupBox,
    QFormLayout, QCheckBox, QLabel, QComboBox, QPushButton, QSplitter, QMessageBox
)
from PyQt6.QtCore import Qt, pyqtSlot, QPointF

try:
    import pyqtgraph as pg
    from pyqtgraph import GraphicsLayoutWidget, ImageItem, PlotWidget, PlotItem, ViewBox, ScatterPlotItem, ArrowItem, TextItem
    PYQTGRAPH_AVAILABLE = True
except ImportError: # pragma: no cover
    pg = None
    GraphicsLayoutWidget = None
    ImageItem = None
    PlotWidget = None
    PlotItem = None
    ViewBox = None
    ScatterPlotItem = None
    ArrowItem = None
    TextItem = None
    PYQTGRAPH_AVAILABLE = False
    logging.error("RealSpaceFFTVisualizerDialog: PyQtGraph not found.")

# Importy z projektu (dostosuj ścieżki, jeśli są inne)
try:
    # Zakładamy, że AppController i HistoryManager są przekazywane jako argumenty,
    # więc importy tutaj są głównie dla type hinting.
    from ...logic.app_controller import AppController
    from ...logic.history_manager import HistoryManager
    from ...core.history import HistoryNode
    from ...analysis.drift_correction import apply_affine_transform # Do transformacji g* adsorbatu
    from ...analysis.lattice import LATTICE_TYPE_HEXAGONAL, LATTICE_TYPE_SQUARE # Jeśli potrzebne do interpretacji
except ImportError as e: # pragma: no cover
    AppController = None
    HistoryManager = None
    HistoryNode = None
    apply_affine_transform = None
    logging.error(f"RealSpaceFFTVisualizerDialog: Error importing project modules: {e}")

logger = logging.getLogger(__name__)

class RealSpaceFFTVisualizerDialog(QDialog):
    def __init__(self,
                 app_controller: AppController,
                 history_manager: HistoryManager,
                 current_fft_node_id: Optional[str],
                 parent=None):
        super().__init__(parent)

        self.app_controller = app_controller
        self.history_manager = history_manager
        self.current_fft_node_id = current_fft_node_id

        if not PYQTGRAPH_AVAILABLE or not self.app_controller or not self.history_manager: # pragma: no cover
            QVBoxLayout(self).addWidget(QLabel("Critical Error: PyQtGraph or App/History Controller not available."))
            self.setWindowTitle("Error")
            return

        self.setWindowTitle("Real Space & FFT Visualization")
        self.setMinimumSize(1300, 750) # Zwiększono trochę dla czytelności

        # Atrybuty dla itemów graficznych, aby można było je usuwać/aktualizować
        self.g_substrate_vector_lines: List[PlotItem] = [] # Zmieniono nazwę z arrows na lines
        self.g_adsorbate_vector_lines: List[PlotItem] = [] # Zmieniono nazwę
        self.real_space_substrate_lattice_item: Optional[ScatterPlotItem] = None
        self.real_space_substrate_vector_items: List[PlotItem] = []
        self.real_space_adsorbate_lattice_items: Dict[int, ScatterPlotItem] = {}
        self.real_space_adsorbate_vector_items: Dict[int, List[PlotItem]] = {}

        # Pobierz dane FFT na podstawie current_fft_node_id
        self.fft_data_to_display: Optional[np.ndarray] = None
        if self.current_fft_node_id and self.history_manager:
            node = self.history_manager.get_node_by_id(self.current_fft_node_id)
            if node and node.data_type == "FFT" and node.image_data is not None:
                self.fft_data_to_display = node.image_data.copy() # Zapisz dane FFT
        # --- KONIEC ---


        self._init_ui()
        self._connect_signals()
        self.update_visualizations()

        logger.debug("RealSpaceFFTVisualizerDialog initialized.")

    def _init_ui(self):
        top_level_layout = QHBoxLayout(self)
        # Główny splitter dzielący na trzy części
        main_splitter = QSplitter(Qt.Orientation.Horizontal)
        top_level_layout.addWidget(main_splitter)

        # === PANEL FFT (LEWY) ===
        # (Bez zmian w stosunku do poprzedniej wersji, ale teraz dodawany jako pierwszy do main_splitter)
        self.fft_panel_widget = GraphicsLayoutWidget()
        self.fft_view_box = self.fft_panel_widget.addViewBox(row=0, col=0, lockAspect=True, invertY=True)
        self.fft_image_item_vis = ImageItem()
        self.fft_view_box.addItem(self.fft_image_item_vis)
        self.fft_view_box.setMenuEnabled(True)
        self.fft_view_box.setMouseMode(ViewBox.PanMode)
        # TODO: Pamiętaj o zainicjalizowaniu obrazu FFT w self.fft_image_item_vis w update_visualizations()
        # if self.fft_data_to_display is not None: # Przykładowo, jeśli masz taki atrybut
        #     self.fft_image_item_vis.setImage(self.fft_data_to_display.T)
        main_splitter.addWidget(self.fft_panel_widget)


        # === PANEL PRZESTRZENI RZECZYWISTEJ (ŚRODKOWY) ===
        # (Poprzednio część prawego panelu, teraz samodzielny)
        self.real_space_plot_widget = PlotWidget()
        plot_item_rs = self.real_space_plot_widget.getPlotItem()
        if plot_item_rs:
            plot_item_rs.setAspectLocked(True)
            plot_item_rs.setTitle("Real Space Lattice Visualization")
            plot_item_rs.setLabel('left', 'Y (nm)')
            plot_item_rs.setLabel('bottom', 'X (nm)')
            plot_item_rs.showGrid(x=True, y=True, alpha=0.3)
        main_splitter.addWidget(self.real_space_plot_widget)


        # === PANEL KONTROLEK (PRAWY) ===
        # (Zawartość poprzedniego prawego panelu, oprócz real_space_plot_widget)
        controls_panel_widget = QWidget()
        controls_panel_layout = QVBoxLayout(controls_panel_widget)
        controls_panel_widget.setMinimumWidth(350) # Ustaw szerokość dla panelu kontrolek
        controls_panel_widget.setMaximumWidth(450)

        # Grupa "Display Options"
        display_options_group = QGroupBox("Display Options")
        self.display_options_form = QFormLayout(display_options_group) # Zapisz referencję
        self.cb_show_substrate_real_lattice = QCheckBox("Substrate Real Lattice")
        self.cb_show_substrate_real_lattice.setChecked(True)
        self.display_options_form.addRow(self.cb_show_substrate_real_lattice)
        
        self.adsorbate_display_checkbox_layout = QVBoxLayout() # Layout dla dynamicznych checkboxów adsorbatu
        self.display_options_form.addRow(QLabel("Adsorbate Sets (Real Space):"))
        self.display_options_form.addRow(self.adsorbate_display_checkbox_layout)
        
        self.cb_show_g_substrate_fft = QCheckBox("Substrate g* vectors (on FFT)")
        self.cb_show_g_substrate_fft.setChecked(True)
        self.display_options_form.addRow(self.cb_show_g_substrate_fft)
        
        self.cb_show_g_adsorbate_fft = QCheckBox("Adsorbate g* vectors (Current Set, on FFT)")
        self.cb_show_g_adsorbate_fft.setChecked(True)
        self.display_options_form.addRow(self.cb_show_g_adsorbate_fft)
        controls_panel_layout.addWidget(display_options_group)

        # Grupa "Substrate Transformation Info"
        transform_info_group = QGroupBox("Substrate Transformation Info")
        transform_info_layout = QFormLayout(transform_info_group)
        self.info_sub_rot_label = QLabel("-")
        self.info_sub_scale_label = QLabel("-")
        self.info_sub_rmse_label = QLabel("-")
        transform_info_layout.addRow("Rot (M->I):", self.info_sub_rot_label)
        transform_info_layout.addRow("Stretch (M->I):", self.info_sub_scale_label)
        transform_info_layout.addRow("Fit RMSE (M->I, px):", self.info_sub_rmse_label)
        controls_panel_layout.addWidget(transform_info_group)

        # Grupa "Substrate Real Space Parameters"
        sub_real_params_group = QGroupBox("Substrate Real Space Parameters")
        sub_real_params_layout = QFormLayout(sub_real_params_group)
        self.sub_real_a1_label = QLabel("- nm"); self.sub_real_a2_label = QLabel("- nm"); self.sub_real_alpha_label = QLabel("- °")
        sub_real_params_layout.addRow("|a1|:", self.sub_real_a1_label); sub_real_params_layout.addRow("|a2|:", self.sub_real_a2_label); sub_real_params_layout.addRow("Angle α:", self.sub_real_alpha_label)
        controls_panel_layout.addWidget(sub_real_params_group)
        
        # Grupa "Adsorbate Real Space Parameters (Current Set)"
        ads_real_params_group = QGroupBox("Adsorbate Real Space Parameters") # Usunięto "(Current Set)" z tytułu grupy
        ads_real_params_layout = QFormLayout(ads_real_params_group)
        self.ads_set_combo_vis = QComboBox() # ComboBox do wyboru zestawu adsorbatu
        ads_real_params_layout.addRow("Select Adsorbate Set:", self.ads_set_combo_vis)
        self.ads_real_a1_label = QLabel("- nm"); self.ads_real_a2_label = QLabel("- nm"); self.ads_real_alpha_label = QLabel("- °")
        ads_real_params_layout.addRow("|a1|:", self.ads_real_a1_label); ads_real_params_layout.addRow("|a2|:", self.ads_real_a2_label); ads_real_params_layout.addRow("Angle α:", self.ads_real_alpha_label)
        self.angle_sub_ads_label = QLabel("- °")
        ads_real_params_layout.addRow("Sub-Ads Angle:", self.angle_sub_ads_label)
        self.calculate_sub_ads_angle_button = QPushButton("Calculate Sub-Ads Angle")
        ads_real_params_layout.addRow(self.calculate_sub_ads_angle_button)
        controls_panel_layout.addWidget(ads_real_params_group)

        controls_panel_layout.addStretch(1) # Wypełniacz na dole panelu kontrolek
        
        # Przycisk Close
        self.close_button = QPushButton("Close")
        button_layout_final = QHBoxLayout() # Użyj QHBoxLayout do wyśrodkowania lub wyrównania
        button_layout_final.addStretch(1)
        button_layout_final.addWidget(self.close_button)
        controls_panel_layout.addLayout(button_layout_final) # Dodaj na dole panelu kontrolek

        main_splitter.addWidget(controls_panel_widget)

        # Ustawienie proporcji splittera (lewy-FFT, środkowy-RealSpace, prawy-Kontrolki)
        # Dostosuj te wartości do swoich preferencji
        main_splitter.setSizes([500, 400, 300]) 
        main_splitter.setStretchFactor(0, 1) # Panel FFT może się rozciągać
        main_splitter.setStretchFactor(1, 1) # Panel RealSpace może się rozciągać
        main_splitter.setStretchFactor(2, 0) # Panel kontrolek ma bardziej stałą szerokość
    

    # def _init_ui(self):
    #     top_level_layout = QHBoxLayout(self)
    #     main_splitter = QSplitter(Qt.Orientation.Horizontal)
    #     top_level_layout.addWidget(main_splitter)

    #     # === PANEL FFT (LEWY) ===
    #     self.fft_panel_widget = GraphicsLayoutWidget()
    #     self.fft_view_box = self.fft_panel_widget.addViewBox(row=0, col=0, lockAspect=True, invertY=True)
    #     self.fft_image_item_vis = ImageItem()
    #     self.fft_view_box.addItem(self.fft_image_item_vis)
    #     self.fft_view_box.setMenuEnabled(True)
    #     self.fft_view_box.setMouseMode(ViewBox.PanMode)
    #     main_splitter.addWidget(self.fft_panel_widget)

    #     # === PANEL PRZESTRZENI RZECZYWISTEJ I KONTROLEK (PRAWY) ===
    #     right_main_widget = QWidget()
    #     right_main_layout = QVBoxLayout(right_main_widget)
    #     right_splitter = QSplitter(Qt.Orientation.Vertical)
    #     right_main_layout.addWidget(right_splitter)

    #     self.real_space_plot_widget = PlotWidget()
    #     plot_item_rs = self.real_space_plot_widget.getPlotItem()
    #     if plot_item_rs:
    #         plot_item_rs.setAspectLocked(True)
    #         plot_item_rs.setTitle("Real Space Lattice Visualization")
    #         plot_item_rs.setLabel('left', 'Y (nm)')
    #         plot_item_rs.setLabel('bottom', 'X (nm)')
    #         plot_item_rs.showGrid(x=True, y=True, alpha=0.3)
    #     right_splitter.addWidget(self.real_space_plot_widget)

    #     controls_widget = QWidget()
    #     controls_layout = QVBoxLayout(controls_widget)
    #     display_options_group = QGroupBox("Display Options")
    #     display_options_form = QFormLayout(display_options_group)
    #     self.cb_show_substrate_real_lattice = QCheckBox("Substrate Real Lattice")
    #     self.cb_show_substrate_real_lattice.setChecked(True)
    #     display_options_form.addRow(self.cb_show_substrate_real_lattice)
    #     # Dynamiczne checkboxy dla adsorbatów będą w osobnym layoucie
    #     self.adsorbate_display_checkbox_layout = QVBoxLayout() # Layout dla dynamicznych checkboxów
    #     display_options_form.addRow(QLabel("Adsorbate Sets (Real Space):"))
    #     display_options_form.addRow(self.adsorbate_display_checkbox_layout) # Dodaj layout do formularza
    #     self.cb_show_g_substrate_fft = QCheckBox("Substrate g* vectors (on FFT)")
    #     self.cb_show_g_substrate_fft.setChecked(True)
    #     display_options_form.addRow(self.cb_show_g_substrate_fft)
    #     self.cb_show_g_adsorbate_fft = QCheckBox("Adsorbate g* vectors (Current Set, on FFT)")
    #     self.cb_show_g_adsorbate_fft.setChecked(True)
    #     display_options_form.addRow(self.cb_show_g_adsorbate_fft)
    #     controls_layout.addWidget(display_options_group)

    #     transform_info_group = QGroupBox("Substrate Transformation Info")
    #     transform_info_layout = QFormLayout(transform_info_group)
    #     self.info_sub_rot_label = QLabel("-")
    #     self.info_sub_scale_label = QLabel("-")
    #     self.info_sub_rmse_label = QLabel("-")
    #     transform_info_layout.addRow("Rot (M->I):", self.info_sub_rot_label)
    #     transform_info_layout.addRow("Stretch (M->I):", self.info_sub_scale_label)
    #     transform_info_layout.addRow("RMSE (M->I):", self.info_sub_rmse_label)
    #     controls_layout.addWidget(transform_info_group)

    #     sub_real_params_group = QGroupBox("Substrate Real Space Parameters")
    #     sub_real_params_layout = QFormLayout(sub_real_params_group)
    #     self.sub_real_a1_label = QLabel("- nm")
    #     self.sub_real_a2_label = QLabel("- nm")
    #     self.sub_real_alpha_label = QLabel("- °")
    #     sub_real_params_layout.addRow("|a1|:", self.sub_real_a1_label)
    #     sub_real_params_layout.addRow("|a2|:", self.sub_real_a2_label)
    #     sub_real_params_layout.addRow("Angle α:", self.sub_real_alpha_label)
    #     controls_layout.addWidget(sub_real_params_group)
        
    #     ads_real_params_group = QGroupBox("Adsorbate Real Space Parameters")
    #     ads_real_params_layout = QFormLayout(ads_real_params_group)
    #     self.ads_set_combo_vis = QComboBox()
    #     ads_real_params_layout.addRow("Select Set:", self.ads_set_combo_vis)
    #     self.ads_real_a1_label = QLabel("- nm")
    #     self.ads_real_a2_label = QLabel("- nm")
    #     self.ads_real_alpha_label = QLabel("- °")
    #     ads_real_params_layout.addRow("|a1|:", self.ads_real_a1_label)
    #     ads_real_params_layout.addRow("|a2|:", self.ads_real_a2_label)
    #     ads_real_params_layout.addRow("Angle α:", self.ads_real_alpha_label)
    #     self.angle_sub_ads_label = QLabel("- °")
    #     ads_real_params_layout.addRow("Sub-Ads Angle:", self.angle_sub_ads_label)
    #     self.calculate_sub_ads_angle_button = QPushButton("Calculate Sub-Ads Angle")
    #     ads_real_params_layout.addRow(self.calculate_sub_ads_angle_button)
    #     controls_layout.addWidget(ads_real_params_group)

    #     controls_layout.addStretch(1)
    #     right_splitter.addWidget(controls_widget)
    #     right_splitter.setSizes([350, 350])
    #     main_splitter.addWidget(right_main_widget)
    #     main_splitter.setSizes([650, 550])
    #     main_splitter.setStretchFactor(0,1)
        
    #     self.close_button = QPushButton("Close")
    #     button_layout_final = QHBoxLayout()
    #     button_layout_final.addStretch(1)
    #     button_layout_final.addWidget(self.close_button)
    #     right_main_layout.addLayout(button_layout_final)

    def _connect_signals(self):
        self.close_button.clicked.connect(self.accept)
        self.cb_show_substrate_real_lattice.stateChanged.connect(self._trigger_redraw_all_visuals)
        # Dynamiczne checkboxy adsorbatów będą podłączane w _populate_adsorbate_set_combo_vis
        self.cb_show_g_substrate_fft.stateChanged.connect(self._trigger_redraw_all_visuals)
        self.cb_show_g_adsorbate_fft.stateChanged.connect(self._trigger_redraw_all_visuals)
        self.ads_set_combo_vis.currentIndexChanged.connect(self._on_selected_adsorbate_set_changed_in_vis)
        self.calculate_sub_ads_angle_button.clicked.connect(self._on_calculate_sub_ads_angle_clicked)


    @pyqtSlot()
    def _trigger_redraw_all_visuals(self):
        logger.debug("Visualizer: Redraw all visuals requested by checkbox/combo change.")
        # Aktualizacja danych nie jest potrzebna, tylko przerysowanie
        self._redraw_fft_overlays()
        self._redraw_real_space_lattices()
        # self._update_real_space_param_labels() # Etykiety są aktualizowane w update_visualizations

    @pyqtSlot(int)
    def _on_selected_adsorbate_set_changed_in_vis(self, combo_box_index: int):
        if combo_box_index < 0: return # Brak wyboru
        set_index = self.ads_set_combo_vis.itemData(combo_box_index)
        if set_index is not None:
            logger.debug(f"Visualizer: Selected adsorbate set in combo changed to index {set_index}")
            self._update_real_space_param_labels() # Zaktualizuj etykiety dla wybranego zestawu
            self._redraw_fft_overlays() # Przerysuj g* adsorbatu dla nowego zestawu
            self._redraw_real_space_lattices() # Przerysuj sieci rzeczywiste
            self.angle_sub_ads_label.setText("- °") # Zresetuj kąt Sub-Ads
        else: # pragma: no cover
             logger.warning(f"Visualizer: No user data for combo box index {combo_box_index}")


    def update_visualizations(self):
        if not self.app_controller or not self.history_manager or not PYQTGRAPH_AVAILABLE: return
        logger.info("RealSpaceFFTVisualizerDialog: Updating all visualizations...")
        
        current_hist_node: Optional[HistoryNode] = None
        if self.current_fft_node_id:
            current_hist_node = self.history_manager.get_node_by_id(self.current_fft_node_id)
        
        if current_hist_node and current_hist_node.data_type == "FFT" and current_hist_node.image_data is not None:
            self.fft_image_item_vis.setImage(current_hist_node.image_data.T)
        else: self.fft_image_item_vis.clear()
        logger.warning("Could not display FFT image in visualizer.")

        if self.app_controller.substrate_transform_analysis_m2i:
            analysis = self.app_controller.substrate_transform_analysis_m2i
            self.info_sub_rot_label.setText(f"{analysis.get('rotation_angle_deg', 'N/A'):.2f}°")
            s_x,s_y = analysis.get('principal_stretches',[np.nan,np.nan])
            self.info_sub_scale_label.setText(f"({s_x:.3f}, {s_y:.3f})")
            self.info_sub_rmse_label.setText(f"{analysis.get('rmse', 'N/A'):.3f} px")
        else: 
            self.info_sub_rot_label.setText("-")
            self.info_sub_scale_label.setText("-")
            self.info_sub_rmse_label.setText("-")

        self._populate_adsorbate_set_combo_and_checkboxes() # Zmieniono nazwę
        self._redraw_fft_overlays()
        self._redraw_real_space_lattices()
        self._update_real_space_param_labels()
        self.angle_sub_ads_label.setText("- °") # Zresetuj kąt Sub-Ads przy pełnym odświeżeniu


    def _populate_adsorbate_set_combo_and_checkboxes(self):
        self.ads_set_combo_vis.blockSignals(True)
        self.ads_set_combo_vis.clear()
        
        # Czyszczenie starych dynamicznych checkboxów
        for i in reversed(range(self.adsorbate_display_checkbox_layout.count())): 
            widget_item = self.adsorbate_display_checkbox_layout.itemAt(i)
            if widget_item and widget_item.widget(): # type: ignore
                widget_item.widget().deleteLater() # type: ignore
        self.adsorbate_set_checkboxes = []

        if self.app_controller and self.app_controller.adsorbate_spot_sets:
            num_sets = len(self.app_controller.adsorbate_spot_sets)
            if num_sets == 0 and len(self.app_controller.corrected_adsorbate_spot_sets) > 0 : # Jeśli są tylko skorygowane
                 num_sets = len(self.app_controller.corrected_adsorbate_spot_sets)

            for i in range(num_sets):
                self.ads_set_combo_vis.addItem(f"Set {i+1}", userData=i)
                # Dynamiczne tworzenie checkboxów dla każdego zestawu adsorbatu
                cb = QCheckBox(f"Adsorbate Set {i+1} Real Lattice")
                cb.setChecked(True) # Domyślnie pokazuj
                cb.stateChanged.connect(self._trigger_redraw_all_visuals)
                self.adsorbate_display_checkbox_layout.addWidget(cb)
                self.adsorbate_set_checkboxes.append(cb)
            
            current_app_controller_set_idx = self.app_controller.current_adsorbate_set_index
            if 0 <= current_app_controller_set_idx < self.ads_set_combo_vis.count():
                 self.ads_set_combo_vis.setCurrentIndex(current_app_controller_set_idx)
            elif self.ads_set_combo_vis.count() > 0: self.ads_set_combo_vis.setCurrentIndex(0)
        
        self.ads_set_combo_vis.blockSignals(False)
        # Ręczne wywołanie slotu po wypełnieniu
        if self.ads_set_combo_vis.count() > 0 :
            self._on_selected_adsorbate_set_changed_in_vis(self.ads_set_combo_vis.currentIndex())


    def _redraw_fft_overlays(self):
        logger.debug("Visualizer: Redrawing FFT overlays...")
        
        # Czyszczenie starych wektorów g*
        for item in self.g_substrate_vector_lines: # Zmieniono nazwę
            if item.scene() is self.fft_view_box.scene(): self.fft_view_box.removeItem(item)
        self.g_substrate_vector_lines.clear()

        for item in self.g_adsorbate_vector_lines: # Zmieniono nazwę
            if item.scene() is self.fft_view_box.scene(): self.fft_view_box.removeItem(item)
        self.g_adsorbate_vector_lines.clear()

        if not self.app_controller or self.fft_data_to_display is None: return

        fft_rows_ky, fft_cols_kx = self.fft_data_to_display.shape
        center_kx_px = fft_cols_kx / 2.0
        center_ky_px = fft_rows_ky / 2.0

        # --- Rysowanie Wektorów g* Substratu (Krok 4) ---
        if self.cb_show_g_substrate_fft.isChecked() and \
           self.app_controller.substrate_real_space_results and \
           "g1_vec_px" in self.app_controller.substrate_real_space_results:
            
            g1s_px = self.app_controller.substrate_real_space_results.get("g1_vec_px")
            g2s_px = self.app_controller.substrate_real_space_results.get("g2_vec_px")
            
            if g1s_px and g2s_px:
                pen_sub = pg.mkPen(color='r', width=2.5, style=Qt.PenStyle.SolidLine)
                # Wektor g1s
                line1s = pg.PlotDataItem(
                    x=[center_kx_px, center_kx_px + g1s_px[0]], 
                    y=[center_ky_px, center_ky_px + g1s_px[1]], 
                    pen=pen_sub,
                    name="g_sub1" # Dla legendy, jeśli będzie
                )
                self.fft_view_box.addItem(line1s)
                self.g_substrate_vector_lines.append(line1s)
                # Wektor g2s
                line2s = pg.PlotDataItem(
                    x=[center_kx_px, center_kx_px + g2s_px[0]], 
                    y=[center_ky_px, center_ky_px + g2s_px[1]], 
                    pen=pen_sub,
                    name="g_sub2"
                )
                self.fft_view_box.addItem(line2s)
                self.g_substrate_vector_lines.append(line2s)
                logger.debug(f"Drew substrate g-vectors: g1_px={g1s_px}, g2_px={g2s_px}")

        # --- Rysowanie Wektorów g* Adsorbatu (dla aktywnego zestawu) (Rozszerzenie Kroku 4) ---
        current_ads_set_idx_vis = self.ads_set_combo_vis.currentData() # Pobierz int z userData
        
        if self.cb_show_g_adsorbate_fft.isChecked() and \
           current_ads_set_idx_vis is not None and \
           self.app_controller.adsorbate_real_space_results and \
           current_ads_set_idx_vis in self.app_controller.adsorbate_real_space_results:
            
            ads_params = self.app_controller.adsorbate_real_space_results.get(current_ads_set_idx_vis)
            if ads_params and "g1_vec_px_ideal_sys" in ads_params:
                g1a_ideal_px = ads_params["g1_vec_px_ideal_sys"] # Wektor w idealnym systemie substratu
                g2a_ideal_px = ads_params["g2_vec_px_ideal_sys"] # Wektor w idealnym systemie substratu

                # Transformuj wektory adsorbatu z "idealnego systemu" substratu
                # z powrotem do "zniekształconej" przestrzeni FFT, aby je poprawnie nałożyć.
                # Używamy transformacji odwrotnej do F_m2i substratu.
                if self.app_controller.substrate_F_m2i is not None and \
                   self.app_controller.substrate_t_m2i is not None and \
                   apply_affine_transform is not None:
                    try:
                        F_inv = np.linalg.inv(self.app_controller.substrate_F_m2i)
                        # t_m2i to translacja z measured do ideal.
                        # Dla wektora (który jest różnicą punktów), translacja się znosi przy transformacji liniowej.
                        # g_distorted = g_ideal @ F_inv.T
                        g1a_distorted_px = np.dot(np.array(g1a_ideal_px), F_inv.T)
                        g2a_distorted_px = np.dot(np.array(g2a_ideal_px), F_inv.T)
                        
                        pen_ads = pg.mkPen(color='b', width=2.5, style=Qt.PenStyle.DashLine)
                        line1a = pg.PlotDataItem(
                            x=[center_kx_px, center_kx_px + g1a_distorted_px[0]], 
                            y=[center_ky_px, center_ky_px + g1a_distorted_px[1]], 
                            pen=pen_ads, name=f"g_ads{current_ads_set_idx_vis+1}_1"
                        )
                        line2a = pg.PlotDataItem(
                            x=[center_kx_px, center_kx_px + g2a_distorted_px[0]], 
                            y=[center_ky_px, center_ky_px + g2a_distorted_px[1]], 
                            pen=pen_ads, name=f"g_ads{current_ads_set_idx_vis+1}_2"
                        )
                        self.fft_view_box.addItem(line1a)
                        self.fft_view_box.addItem(line2a)
                        self.g_adsorbate_vector_lines.extend([line1a, line2a])
                        logger.debug(f"Drew adsorbate set {current_ads_set_idx_vis} g-vectors (distorted for FFT view): "
                                     f"g1_dist_px={g1a_distorted_px}, g2_dist_px={g2a_distorted_px}")
                    except np.linalg.LinAlgError: # pragma: no cover
                        logger.error("LinAlgError when inverting substrate transform for adsorbate g-vector display.")
                    except Exception as e: # pragma: no cover
                        logger.error(f"Error transforming/drawing adsorbate g-vectors: {e}")
            else: # pragma: no cover
                logger.debug(f"No adsorbate g-vector data for set {current_ads_set_idx_vis} or transform missing.")
        else:
            logger.debug("Not drawing adsorbate g-vectors (checkbox off or no data).")

    def _redraw_real_space_lattices(self): # Zaimplementowano Krok 5
        logger.debug("Visualizer: Redrawing real space lattices...")
        plot_item_rs = self.real_space_plot_widget.getPlotItem()
        if not plot_item_rs or not self.app_controller: return # pragma: no cover
        plot_item_rs.clear() # Wyczyść poprzednie rysunki
        plot_item_rs.showGrid(x=True, y=True, alpha=0.3) # Przywróć siatkę, jeśli clear ją usuwa

        # Sieć substratu
        if self.cb_show_substrate_real_lattice.isChecked() and self.app_controller.substrate_real_space_results:
            sub_params = self.app_controller.substrate_real_space_results
            if "a1_vec_nm" in sub_params and "a2_vec_nm" in sub_params:
                self._draw_single_real_space_lattice(
                    plot_item_rs, 
                    np.array(sub_params["a1_vec_nm"]), 
                    np.array(sub_params["a2_vec_nm"]),
                    pen_color='r', symbol='o', symbol_size=8, symbol_color='darkred', label_text="S"
                )
        
        # Sieci adsorbatu (iteruj po wszystkich dynamicznych checkboxach)
        for i, cb_ads_set in enumerate(self.adsorbate_set_checkboxes):
            if cb_ads_set.isChecked():
                set_index = self.ads_set_combo_vis.itemData(i) # Lub bezpośrednio i, jeśli ComboBox i checkboxy są synchronizowane
                if set_index is not None:
                    ads_params = self.app_controller.adsorbate_real_space_results.get(set_index)
                    if ads_params and "a1_vec_nm" in ads_params and "a2_vec_nm" in ads_params:
                        # Użyj różnych kolorów/symboli dla różnych zestawów
                        colors = ['b', 'g', 'purple', 'orange']
                        symbols = ['s', 't', 'd', 'star']
                        color = colors[i % len(colors)]
                        symbol = symbols[i % len(symbols)]
                        self._draw_single_real_space_lattice(
                            plot_item_rs,
                            np.array(ads_params["a1_vec_nm"]),
                            np.array(ads_params["a2_vec_nm"]),
                            pen_color=color, symbol=symbol, symbol_size=7, symbol_color=color, label_text=f"A{i+1}",
                            # offset_factor=0.05 * (i + 1) # Małe przesunięcie dla każdego zestawu
                            offset_factor=0.0
                        )
        plot_item_rs.autoRange()


    def _draw_single_real_space_lattice(self, plot_item: PlotItem, 
                                        a1_vec: np.ndarray, a2_vec: np.ndarray, 
                                        pen_color='k', symbol='o', symbol_size=8, symbol_color='k', label_text="",
                                        offset_factor=0.0, n_cells: int = 2):
        if a1_vec is None or a2_vec is None or len(a1_vec) != 2 or len(a2_vec) != 2: return

        points_data = []
        offset = (a1_vec + a2_vec) * offset_factor
        origin_real = offset # Początek dla wektorów
        
        for m in range(-n_cells, n_cells + 1):
            for n in range(-n_cells, n_cells + 1):
                pt = m * a1_vec + n * a2_vec + offset
                points_data.append({'pos': pt, 'symbol': symbol, 'size': symbol_size, 'pen': None, 'brush': pg.mkBrush(symbol_color)})
        
        if points_data:
            scatter = ScatterPlotItem()
            scatter.setData(spots=points_data)
            plot_item.addItem(scatter)

        pen = pg.mkPen(pen_color, width=2)
        # a1 vector from origin_real
        plot_item.plot([origin_real[0], origin_real[0] + a1_vec[0]], [origin_real[1], origin_real[1] + a1_vec[1]], pen=pen)
        # a2 vector from origin_real
        plot_item.plot([origin_real[0], origin_real[0] + a2_vec[0]], [origin_real[1], origin_real[1] + a2_vec[1]], pen=pen)

        # Etykiety dla wektorów
        if label_text:
            text_a1 = pg.TextItem(f"{label_text}-a1", color=pg.mkColor(pen_color), anchor=(0.5, 1.2))
            text_a1.setPos(origin_real[0] + a1_vec[0]*0.5, origin_real[1] + a1_vec[1]*0.5)
            plot_item.addItem(text_a1)
            text_a2 = pg.TextItem(f"{label_text}-a2", color=pg.mkColor(pen_color), anchor=(0.5, 1.2))
            text_a2.setPos(origin_real[0] + a2_vec[0]*0.5, origin_real[1] + a2_vec[1]*0.5)
            plot_item.addItem(text_a2)

        # Oznaczenie kąta (uproszczone)
        norm_a1 = np.linalg.norm(a1_vec)
        norm_a2 = np.linalg.norm(a2_vec)
        if norm_a1 > 1e-6 and norm_a2 > 1e-6:
            angle_line_end1 = origin_real + a1_vec * 0.3 / norm_a1 * min(norm_a1, norm_a2)
            angle_line_end2 = origin_real + a2_vec * 0.3 / norm_a2 * min(norm_a1, norm_a2)
            plot_item.plot([angle_line_end1[0], origin_real[0], angle_line_end2[0]],
                           [angle_line_end1[1], origin_real[1], angle_line_end2[1]], 
                           pen=pg.mkPen(color=pen_color, style=Qt.PenStyle.DotLine, width=1))


    def _update_real_space_param_labels(self):
        logger.debug("Visualizer: Updating real space parameter labels...")
        # Substrat
        if self.app_controller and self.app_controller.substrate_real_space_results:
            params = self.app_controller.substrate_real_space_results
            self.sub_real_a1_label.setText(f"{params.get('a1_nm', '-'):.3f} nm")
            self.sub_real_a2_label.setText(f"{params.get('a2_nm', '-'):.3f} nm")
            self.sub_real_alpha_label.setText(f"{params.get('alpha_deg', '-'):.2f} °")
        else: 
            self.sub_real_a1_label.setText("- nm")
            self.sub_real_a2_label.setText("- nm")
            self.sub_real_alpha_label.setText("- °")

        # Adsorbat (dla wybranego zestawu w ComboBoxie tego dialogu)
        current_ads_set_idx_vis = self.ads_set_combo_vis.currentData()
        if self.app_controller and current_ads_set_idx_vis is not None and \
           current_ads_set_idx_vis in self.app_controller.adsorbate_real_space_results:
            params = self.app_controller.adsorbate_real_space_results[current_ads_set_idx_vis]
            self.ads_real_a1_label.setText(f"{params.get('a1_nm', '-'):.3f} nm")
            self.ads_real_a2_label.setText(f"{params.get('a2_nm', '-'):.3f} nm")
            self.ads_real_alpha_label.setText(f"{params.get('alpha_deg', '-'):.2f} °")
        else: 
            self.ads_real_a1_label.setText("- nm")
            self.ads_real_a2_label.setText("- nm")
            self.ads_real_alpha_label.setText("- °")
        
        # Zresetuj kąt Sub-Ads, bo zależy od wybranego zestawu i obliczeń substratu
        self.angle_sub_ads_label.setText("- °")
        self.calculate_sub_ads_angle_button.setEnabled(bool(
            self.app_controller and self.app_controller.substrate_real_space_results and
            current_ads_set_idx_vis is not None and
            current_ads_set_idx_vis in self.app_controller.adsorbate_real_space_results and
            self.app_controller.adsorbate_real_space_results[current_ads_set_idx_vis] # Upewnij się, że słownik nie jest pusty/błędem
        ))
        
    @pyqtSlot()
    def _on_calculate_sub_ads_angle_clicked(self): # Krok 6
        logger.debug("Visualizer: Calculate Substrate-Adsorbate Angle clicked.")
        if not self.app_controller: return

        current_ads_set_idx_vis = self.ads_set_combo_vis.currentData()
        if current_ads_set_idx_vis is None:
            QMessageBox.information(self, "Info", "Please select an adsorbate set.")
            return

        sub_params = self.app_controller.substrate_real_space_results
        ads_params = self.app_controller.adsorbate_real_space_results.get(current_ads_set_idx_vis)

        if not (sub_params and "a1_vec_nm" in sub_params and ads_params and "a1_vec_nm" in ads_params):
            self.angle_sub_ads_label.setText("N/A (Params missing)")
            QMessageBox.warning(self, "Data Missing", "Substrate or adsorbate real space parameters not calculated yet.")
            return

        try:
            a1_s_vec = np.array(sub_params["a1_vec_nm"])
            a1_a_vec = np.array(ads_params["a1_vec_nm"])

            norm_s = np.linalg.norm(a1_s_vec)
            norm_a = np.linalg.norm(a1_a_vec)

            if norm_s < 1e-9 or norm_a < 1e-9: # pragma: no cover
                self.angle_sub_ads_label.setText("N/A (Zero vector)")
                return
            
            dot_product = np.dot(a1_s_vec, a1_a_vec)
            cos_theta = np.clip(dot_product / (norm_s * norm_a), -1.0, 1.0)
            angle_rad = np.arccos(cos_theta)
            angle_deg = np.degrees(angle_rad)
            
            self.angle_sub_ads_label.setText(f"{angle_deg:.2f} °")
            logger.info(f"Calculated angle between substrate a1 and adsorbate set {current_ads_set_idx_vis} a1: {angle_deg:.2f}°")

        except Exception as e: # pragma: no cover
            logger.error(f"Error calculating substrate-adsorbate angle: {e}")
            self.angle_sub_ads_label.setText("Error")
            QMessageBox.critical(self, "Calculation Error", f"Could not calculate angle: {e}")


    def get_dialog_results(self) -> Dict[str, Any]: 
        return {}

    def accept(self): 
        logger.info("RealSpaceFFTVisualizerDialog closed by OK/Close.")
        super().accept()

    def reject(self): 
        logger.info("RealSpaceFFTVisualizerDialog rejected/closed.")
        super().reject()
        
    def closeEvent(self, event):
        logger.debug("RealSpaceFFTVisualizerDialog closeEvent. Cleaning up GL items.")
        if hasattr(self, 'gl_roi_view_widget') and self.gl_roi_view_widget:
            if hasattr(self, 'gl_roi_surface_plot_item') and self.gl_roi_surface_plot_item : 
                self.gl_roi_view_widget.removeItem(self.gl_roi_surface_plot_item)
            self.gl_roi_surface_plot_item = None
            self.gl_roi_view_widget.setParent(None)
            self.gl_roi_view_widget.deleteLater()
        if hasattr(self, 'gl_gauss_view_widget') and self.gl_gauss_view_widget:
            if hasattr(self, 'gl_gauss_surface_plot_item') and self.gl_gauss_surface_plot_item : 
                self.gl_gauss_view_widget.removeItem(self.gl_gauss_surface_plot_item)
                self.gl_gauss_surface_plot_item = None
            self.gl_gauss_view_widget.setParent(None)
            self.gl_gauss_view_widget.deleteLater()
        super().closeEvent(event)