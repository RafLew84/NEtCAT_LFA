# lfa/gui/dialogs/real_space_visualizer_dialog.py
from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple

import numpy as np
from PyQt6.QtCore import Qt, pyqtSlot
from PyQt6.QtWidgets import (
    QCheckBox,
    QDialog,
    QLabel,
    QMessageBox,
    QVBoxLayout,
)

try:
    import pyqtgraph as pg
    from pyqtgraph import (
        ArrowItem,
        GraphicsLayoutWidget,
        ImageItem,
        PlotItem,
        PlotWidget,
        ScatterPlotItem,
        TextItem,
        ViewBox,
    )
    PYQTGRAPH_AVAILABLE = True
except ImportError:
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

if TYPE_CHECKING:
    from ...core.history import HistoryNode
    from ...logic.app_controller import AppController
    from ...logic.history_manager import HistoryManager

from ..utils.display import (
    format_float,
    format_pair_with_sigma,
    format_value_with_sigma,
)
from ..visualizers.real_space_pyvista_adapter import RealSpacePyVistaAdapter, RealSpaceSceneConfig
from ..visualizers.real_space_state import RealSpaceVisualizerState
from ..visualizers.real_space_view import (
    RealSpaceVisualizerWidgets,
    build_real_space_visualizer_ui,
)
from .presenters import RealSpaceVisualizerPresenter

logger = logging.getLogger(__name__)

class RealSpaceFFTVisualizerDialog(QDialog):
    """
    A dialog window for visualizing real space and FFT (Fourier Transform) data.
    
    This dialog provides a comprehensive visualization interface with three main panels:
    - FFT Panel: Displays the Fourier Transform of the image
    - Real Space Panel: Shows the real space lattice visualization
    - Controls Panel: Contains display options and parameter information
    
    Features:
    - Side-by-side view of FFT and real space data
    - Interactive ROI selection and manipulation
    - Display of substrate and adsorbate lattice vectors
    - Real-time parameter updates and visualization
    - Support for multiple adsorbate sets
    
    Attributes:
        app_controller (AppController): Main application controller
        history_manager (HistoryManager): Manages operation history
        current_fft_node_id (Optional[str]): ID of the current FFT node
        fft_data_to_display (Optional[np.ndarray]): FFT data to be displayed
        g_substrate_vector_lines (List[PlotItem]): Lines representing substrate g* vectors
        g_adsorbate_vector_lines (List[PlotItem]): Lines representing adsorbate g* vectors
        real_space_substrate_lattice_item (Optional[ScatterPlotItem]): Substrate lattice visualization
        real_space_adsorbate_lattice_items (Dict[int, ScatterPlotItem]): Adsorbate lattice visualizations
    """

    def __init__(self,
                 app_controller: AppController,
                 history_manager: HistoryManager,
                 current_fft_node_id: Optional[str],
                 parent=None):
        """
        Initialize the Real Space and FFT Visualizer dialog.
        
        Args:
            app_controller (AppController): Main application controller
            history_manager (HistoryManager): Manager for operation history
            current_fft_node_id (Optional[str]): ID of the current FFT node
            parent (Optional[QWidget]): Parent widget for the dialog
            
        Note:
            The dialog requires both PyQtGraph and application controllers to be available.
            If either is missing, an error message will be displayed.
        """
        super().__init__(parent)

        self.app_controller = app_controller
        self.history_manager = history_manager
        self.current_fft_node_id = current_fft_node_id
        self.visual_alignment_angle_rad = 0.0

        if not PYQTGRAPH_AVAILABLE or not self.app_controller or not self.history_manager:
            QVBoxLayout(self).addWidget(QLabel("Critical Error: PyQtGraph or App/History Controller not available."))
            self.setWindowTitle("Error")
            return

        self._presenter = RealSpaceVisualizerPresenter(self.app_controller, logger=logger)
        self._offset_state = RealSpaceVisualizerState(self.app_controller, logger)
        self._pyvista_adapter = RealSpacePyVistaAdapter(logger=logger)

        self.setWindowTitle("Real Space & FFT Visualization")
        self.setMinimumSize(1300, 750)
        current_flags=self.windowFlags()
        self.setWindowFlags(current_flags | Qt.WindowType.WindowMinimizeButtonHint | Qt.WindowType.WindowMaximizeButtonHint)


        self.g_substrate_vector_lines: List[PlotItem] = []
        self.g_adsorbate_vector_lines: List[PlotItem] = []
        self.real_space_substrate_lattice_item: Optional[ScatterPlotItem] = None
        self.real_space_substrate_vector_items: List[PlotItem] = []
        self.real_space_adsorbate_lattice_items: Dict[int, ScatterPlotItem] = {}
        self.real_space_adsorbate_vector_items: Dict[int, List[PlotItem]] = {}
        self.custom_adsorbate_definition: Optional[Dict[str, Any]] = None
        self.real_space_view_box: Optional[ViewBox] = None
        self._real_space_last_view_range: Optional[Tuple[Tuple[float, float], Tuple[float, float]]] = None
        self._real_space_force_autorange: bool = True

        self.fft_data_to_display: Optional[np.ndarray] = None
        if self.current_fft_node_id and self.history_manager:
            node = self.history_manager.get_node_by_id(self.current_fft_node_id)
            if node and node.data_type == "FFT" and node.image_data is not None:
                self.fft_data_to_display = node.image_data.copy()

        self._widgets: Optional[RealSpaceVisualizerWidgets] = None
        self._init_ui()
        self._connect_signals()
        self.update_visualizations()

        logger.debug("RealSpaceFFTVisualizerDialog initialized.")

    def _init_ui(self):
        """
        Initialize the user interface components.
        
        Creates and arranges:
        - FFT visualization panel
        - Real space visualization panel
        - Control panel with display options and parameters
        - ROI selection and manipulation tools
        """
        widgets = build_real_space_visualizer_ui(
            self,
            on_real_space_view_range_changed=self._on_real_space_view_range_changed if PYQTGRAPH_AVAILABLE else None,
        )
        self._widgets = widgets

        for attr_name, value in widgets.__dict__.items():
            if attr_name == "extra_references":
                for extra_name, extra_value in value.items():
                    setattr(self, extra_name, extra_value)
                continue
            setattr(self, attr_name, value)

        self._update_offset_controls_from_state()
        return

    def _connect_signals(self):
        """
        Connect UI element signals to their respective slots.
        
        Handles:
        - Checkbox state changes
        - ROI changes
        - Parameter updates
        - Dialog button actions
        """
        self.close_button.clicked.connect(self.accept)
        self.cb_show_substrate_real_lattice.stateChanged.connect(self._trigger_redraw_all_visuals)
        self.cb_visual_align.stateChanged.connect(self._trigger_redraw_all_visuals)
        self.cb_show_g_substrate_fft.stateChanged.connect(self._trigger_redraw_all_visuals)
        self.cb_show_g_adsorbate_fft.stateChanged.connect(self._trigger_redraw_all_visuals)
        self.ads_set_combo_vis.currentIndexChanged.connect(self._on_selected_adsorbate_set_changed_in_vis)
        self.calculate_sub_ads_angle_button.clicked.connect(self._on_calculate_sub_ads_angle_clicked)
        self.substrate_lattice_cells_spin.valueChanged.connect(lambda _: self._trigger_redraw_all_visuals())
        self.adsorbate_lattice_cells_spin.valueChanged.connect(lambda _: self._trigger_redraw_all_visuals())
        self.substrate_atom_size_spin.valueChanged.connect(lambda _: self._trigger_redraw_all_visuals())
        self.adsorbate_atom_size_spin.valueChanged.connect(lambda _: self._trigger_redraw_all_visuals())
        self.custom_adsorbate_visibility_checkbox.stateChanged.connect(self._on_custom_adsorbate_visibility_changed)
        self.custom_adsorbate_apply_button.clicked.connect(self._on_custom_adsorbate_apply_clicked)
        self.custom_adsorbate_clear_button.clicked.connect(self._on_custom_adsorbate_clear_clicked)
        self.supercell_size_spinbox.valueChanged.connect(self._on_3d_settings_changed)
        self.custom_length_convert_button.clicked.connect(self._on_custom_length_convert_clicked)
        self.custom_symbol_combo.currentTextChanged.connect(self._on_custom_symbol_changed)
        self.launch_3d_button.clicked.connect(self._launch_3d_viewer)
        self.substrate_offset_x_spin.valueChanged.connect(self._on_substrate_offset_value_changed)
        self.substrate_offset_y_spin.valueChanged.connect(self._on_substrate_offset_value_changed)
        self.adsorbate_offset_x_spin.valueChanged.connect(self._on_adsorbate_offset_value_changed)
        self.adsorbate_offset_y_spin.valueChanged.connect(self._on_adsorbate_offset_value_changed)

    @pyqtSlot()
    def _on_3d_settings_changed(self):
        """
        Slot called when a 3D visualization parameter (like supercell size) changes.
        If the 3D viewer is already open, it triggers a redraw.
        """
        self._refresh_3d_plotter_if_open()

    def _refresh_3d_plotter_if_open(self):
        """Redraw the 3D viewer if it is currently visible."""
        if self._pyvista_adapter.is_open():
            logger.debug("Refreshing existing PyVista plotter.")
            self._launch_3d_viewer()

    @pyqtSlot(int)
    def _on_custom_adsorbate_visibility_changed(self, _state: int):
        if self.custom_adsorbate_definition is None and self.custom_adsorbate_visibility_checkbox.isChecked():
            QMessageBox.information(
                self,
                "Custom Adsorbate",
                "Please define custom lattice vectors before enabling its display."
            )
            self.custom_adsorbate_visibility_checkbox.blockSignals(True)
            self.custom_adsorbate_visibility_checkbox.setChecked(False)
            self.custom_adsorbate_visibility_checkbox.blockSignals(False)
            return
        self._trigger_redraw_all_visuals()

    @pyqtSlot()
    def _on_custom_adsorbate_apply_clicked(self):
        a1_vec = np.array([self.custom_a1_x_spin.value(), self.custom_a1_y_spin.value()], dtype=float)
        a2_vec = np.array([self.custom_a2_x_spin.value(), self.custom_a2_y_spin.value()], dtype=float)
        offset_vec = np.array([self.custom_offset_x_spin.value(), self.custom_offset_y_spin.value()], dtype=float)

        a1_norm = np.linalg.norm(a1_vec)
        a2_norm = np.linalg.norm(a2_vec)
        if a1_norm < 1e-9 or a2_norm < 1e-9:
            QMessageBox.warning(self, "Invalid Custom Adsorbate", "Lattice vectors must be non-zero.")
            return

        determinant = a1_vec[0] * a2_vec[1] - a1_vec[1] * a2_vec[0]
        if abs(determinant) < 1e-9:
            QMessageBox.warning(self, "Invalid Custom Adsorbate", "Lattice vectors must be linearly independent.")
            return

        self.custom_adsorbate_definition = {
            "a1_vec_nm": a1_vec,
            "a2_vec_nm": a2_vec,
            "offset_nm": offset_vec,
            "symbol": self.custom_symbol_combo.currentText(),
        }
        if not self.custom_adsorbate_visibility_checkbox.isChecked():
            self.custom_adsorbate_visibility_checkbox.blockSignals(True)
            self.custom_adsorbate_visibility_checkbox.setChecked(True)
            self.custom_adsorbate_visibility_checkbox.blockSignals(False)

        self._real_space_force_autorange = True
        self._trigger_redraw_all_visuals()

    @pyqtSlot()
    def _on_custom_adsorbate_clear_clicked(self):
        self.custom_adsorbate_definition = None
        self.custom_a1_x_spin.setValue(1.0)
        self.custom_a1_y_spin.setValue(0.0)
        self.custom_a2_x_spin.setValue(0.0)
        self.custom_a2_y_spin.setValue(1.0)
        self.custom_offset_x_spin.setValue(0.0)
        self.custom_offset_y_spin.setValue(0.0)
        self.custom_symbol_combo.setCurrentText("star")
        self.custom_a1_length_spin.setValue(1.0)
        self.custom_a2_length_spin.setValue(1.0)
        self.custom_angle_a1_spin.setValue(0.0)
        self.custom_angle_between_spin.setValue(60.0)

        self.custom_adsorbate_visibility_checkbox.blockSignals(True)
        self.custom_adsorbate_visibility_checkbox.setChecked(False)
        self.custom_adsorbate_visibility_checkbox.blockSignals(False)

        self._real_space_force_autorange = True
        self._trigger_redraw_all_visuals()

    @pyqtSlot()
    def _on_custom_length_convert_clicked(self):
        a1_length = self.custom_a1_length_spin.value()
        a2_length = self.custom_a2_length_spin.value()
        angle_a1_deg = self.custom_angle_a1_spin.value()
        angle_between_deg = self.custom_angle_between_spin.value()

        if a1_length <= 0.0 or a2_length <= 0.0:
            QMessageBox.warning(self, "Invalid Length", "Vector lengths must be positive.")
            return

        angle_a1_rad = np.deg2rad(angle_a1_deg)
        angle_a2_rad = np.deg2rad(angle_a1_deg + angle_between_deg)

        a1_x = a1_length * np.cos(angle_a1_rad)
        a1_y = a1_length * np.sin(angle_a1_rad)
        a2_x = a2_length * np.cos(angle_a2_rad)
        a2_y = a2_length * np.sin(angle_a2_rad)

        for spin, value in (
            (self.custom_a1_x_spin, a1_x),
            (self.custom_a1_y_spin, a1_y),
            (self.custom_a2_x_spin, a2_x),
            (self.custom_a2_y_spin, a2_y),
        ):
            spin.blockSignals(True)
            spin.setValue(value)
            spin.blockSignals(False)

        if self.custom_adsorbate_definition is not None:
            self._real_space_force_autorange = False
            self._trigger_redraw_all_visuals()

    @pyqtSlot(str)
    def _on_custom_symbol_changed(self, new_symbol: str):
        if self.custom_adsorbate_definition is not None:
            self.custom_adsorbate_definition["symbol"] = new_symbol
            self._real_space_force_autorange = False
            self._trigger_redraw_all_visuals()

    @pyqtSlot()
    def _launch_3d_viewer(self):
        """
        Launches or updates a separate, interactive 3D window using BackgroundPlotter.
        """
        if not self._pyvista_adapter.is_available():
            QMessageBox.warning(self, "Dependency Error", "PyVista is not installed. 3D visualization is unavailable.")
            return

        current_set_idx = self.ads_set_combo_vis.currentData()
        adsorbate_params = None
        if hasattr(self.app_controller, "adsorbate_real_space_results"):
            adsorbate_params = self.app_controller.adsorbate_real_space_results.get(current_set_idx)

        config = RealSpaceSceneConfig(
            supercell_size=self.supercell_size_spinbox.value(),
            align_adsorbate=self.cb_visual_align.isChecked(),
            alignment_angle_rad=self.visual_alignment_angle_rad,
            substrate_params=getattr(self.app_controller, "substrate_real_space_results", None),
            adsorbate_params=adsorbate_params,
            substrate_offset_nm=self._offset_state.get_substrate_offset(),
            adsorbate_offset_nm=self._offset_state.get_adsorbate_offset(current_set_idx),
        )

        try:
            self._pyvista_adapter.refresh_scene(config)
        except Exception as exc:
            logger.exception("Failed to render 3D lattice viewer: %s", exc)
            QMessageBox.critical(self, "3D Viewer Error", f"Could not render 3D scene:\n{exc}")

    @pyqtSlot()
    def _trigger_redraw_all_visuals(self):
        """
        Trigger redrawing of all visual elements.
        
        This method is called when display settings change and updates:
        - FFT overlays
        - Real space lattice visualizations
        """
        logger.debug("Visualizer: Redraw all visuals requested by checkbox/combo change.")
        preserve_force_flag = self._real_space_force_autorange
        self._redraw_fft_overlays()
        if not preserve_force_flag:
            self._real_space_force_autorange = False
        self._redraw_real_space_lattices()

    @pyqtSlot(int)
    def _on_selected_adsorbate_set_changed_in_vis(self, combo_box_index: int):
        """
        Handle changes in the selected adsorbate set.
        
        Args:
            combo_box_index (int): Index of the newly selected adsorbate set
            
        Note:
            Updates parameter labels and redraws visualizations for the selected set.
            Resets the substrate-adsorbate angle display.
        """
        if combo_box_index < 0: return
        set_index = self.ads_set_combo_vis.itemData(combo_box_index)
        if set_index is not None:
            logger.debug(f"Visualizer: Selected adsorbate set in combo changed to index {set_index}")
            self._update_offset_controls_from_state()
            self._update_real_space_param_labels()
            self._redraw_fft_overlays()
            self._real_space_force_autorange = True
            self._redraw_real_space_lattices()
            self.angle_sub_ads_label.setText("- deg")
        else:
             logger.warning(f"Visualizer: No user data for combo box index {combo_box_index}")


    def update_visualizations(self):
        """
        Update all visualizations based on current data and settings.
        
        Updates:
        - FFT image display
        - Substrate transformation information
        - Real space lattice visualizations
        - Parameter labels and values
        """
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
            rotation_display = format_value_with_sigma(
                analysis.get("rotation_angle_deg"),
                analysis.get("rotation_angle_deg_sigma"),
                "deg",
                value_precision=2,
                sigma_precision=2,
            )
            scale_text = format_pair_with_sigma(
                analysis.get("principal_stretches"),
                analysis.get("principal_stretches_sigma"),
                precision=3,
                sigma_precision=3,
            )
            rmse_text = format_float(analysis.get("rmse"), 3)

            self.info_sub_rot_label.setText(rotation_display if rotation_display != "-" else "-")
            self.info_sub_scale_label.setText(scale_text if scale_text != "-" else "-")
            self.info_sub_rmse_label.setText(f"{rmse_text} px" if rmse_text != "-" else "-")
        else:
            self.info_sub_rot_label.setText("-")
            self.info_sub_scale_label.setText("-")
            self.info_sub_rmse_label.setText("-")

        self._populate_adsorbate_set_combo_and_checkboxes()
        self._redraw_fft_overlays()
        self._real_space_force_autorange = True
        self._redraw_real_space_lattices()
        self._update_real_space_param_labels()
        self.angle_sub_ads_label.setText("- deg")


    def _populate_adsorbate_set_combo_and_checkboxes(self):
        """
        Populate the adsorbate set selection combo box and create display checkboxes.
        
        Creates:
        - Combo box entries for each adsorbate set
        - Checkboxes for toggling individual set visibility
        - Updates UI to reflect current selection
        """
        self.ads_set_combo_vis.blockSignals(True)
        self.ads_set_combo_vis.clear()

        for i in reversed(range(self.adsorbate_sets_checkbox_layout.count())):
            widget_item = self.adsorbate_sets_checkbox_layout.itemAt(i)
            if widget_item and widget_item.widget():
                widget_item.widget().deleteLater()
        self.adsorbate_set_checkboxes = []

        summary = self._presenter.get_adsorbate_sets_summary()
        for info in summary.sets:
            self.ads_set_combo_vis.addItem(info.label, userData=info.index)
            cb = QCheckBox(f"Adsorbate Set {info.index + 1} Real Lattice")
            cb.setChecked(True)
            cb.stateChanged.connect(self._trigger_redraw_all_visuals)
            self.adsorbate_sets_checkbox_layout.addWidget(cb)
            self.adsorbate_set_checkboxes.append(cb)

        if summary.selected_index is not None:
            combo_row = self.ads_set_combo_vis.findData(summary.selected_index)
            if combo_row != -1:
                self.ads_set_combo_vis.setCurrentIndex(combo_row)
            elif self.ads_set_combo_vis.count() > 0:
                self.ads_set_combo_vis.setCurrentIndex(0)
        self.ads_set_combo_vis.blockSignals(False)

        if self.ads_set_combo_vis.count() > 0:
            self._on_selected_adsorbate_set_changed_in_vis(self.ads_set_combo_vis.currentIndex())
        else:
            self._update_offset_controls_from_state()
    
    @pyqtSlot()
    def _redraw_fft_overlays(self):
        """
        Draw overlays on the FFT, including substrate g* vectors
        and vectors to corrected adsorbate peak positions.
        """
        logger.debug("Visualizer: Redrawing FFT overlays...")
        plot_item = self.fft_view_box

        for item in self.g_substrate_vector_lines: plot_item.removeItem(item)
        self.g_substrate_vector_lines.clear()
        for item in self.g_adsorbate_vector_lines: plot_item.removeItem(item)
        self.g_adsorbate_vector_lines.clear()

        if not self.app_controller or self.fft_data_to_display is None: return

        fft_rows_ky, fft_cols_kx = self.fft_data_to_display.shape
        center_kx_px = fft_cols_kx / 2.0
        center_ky_px = fft_rows_ky / 2.0

        if self.cb_show_g_substrate_fft.isChecked() and self.app_controller.substrate_real_space_results:
            sub_params = self.app_controller.substrate_real_space_results
            g1s_px = sub_params.get("g1_vec_px")
            g2s_px = sub_params.get("g2_vec_px")
            if g1s_px and g2s_px:
                pen_sub = pg.mkPen(color='r', width=2.5, style=Qt.PenStyle.SolidLine)
                line1s = pg.PlotDataItem(x=[center_kx_px, center_kx_px + g1s_px[0]], y=[center_ky_px, center_ky_px + g1s_px[1]], pen=pen_sub)
                line2s = pg.PlotDataItem(x=[center_kx_px, center_kx_px + g2s_px[0]], y=[center_ky_px, center_ky_px + g2s_px[1]], pen=pen_sub)
                plot_item.addItem(line1s); plot_item.addItem(line2s)
                self.g_substrate_vector_lines.extend([line1s, line2s])

        current_ads_set_idx_vis = self.ads_set_combo_vis.currentData()
        if self.cb_show_g_adsorbate_fft.isChecked() and current_ads_set_idx_vis is not None:
            
            if 0 <= current_ads_set_idx_vis < len(self.app_controller.corrected_adsorbate_spot_sets):
                corrected_spots = self.app_controller.corrected_adsorbate_spot_sets[current_ads_set_idx_vis]
                
                if corrected_spots:
                    pen_ads = pg.mkPen(color='b', width=2.5, style=Qt.PenStyle.DashLine)
                    
                    for spot_kx, spot_ky in corrected_spots:
                        line_ads = pg.PlotDataItem(
                            x=[center_kx_px, spot_kx], 
                            y=[center_ky_px, spot_ky], 
                            pen=pen_ads
                        )
                        plot_item.addItem(line_ads)
                        self.g_adsorbate_vector_lines.append(line_ads)
                    
                    logger.debug(f"Drew {len(corrected_spots)} vectors to corrected adsorbate positions.")



    def _on_real_space_view_range_changed(self, view_box: ViewBox, view_range):
        """
        Track the current view range of the real-space plot so it can be restored after redraws.
        """
        if not view_range or len(view_range) != 2:
            return
        try:
            x_range = tuple(float(v) for v in view_range[0])
            y_range = tuple(float(v) for v in view_range[1])
        except (TypeError, ValueError):
            return
        self._real_space_last_view_range = (x_range, y_range)

    def _redraw_real_space_lattices(self):
        """
        Redraw real space lattice visualizations.
        
        Creates:
        - Substrate lattice visualization
        - Adsorbate lattice visualizations for each active set
        - Vector representations with proper scaling and orientation
        """
        logger.debug("Visualizer: Redrawing real space lattices...")
        plot_item_rs = self.real_space_plot_widget.getPlotItem()
        if not plot_item_rs or not self.app_controller: return # pragma: no cover

        view_box = self.real_space_view_box or plot_item_rs.getViewBox()
        previous_range = self._real_space_last_view_range
        force_autorange = self._real_space_force_autorange or previous_range is None

        plot_item_rs.clear()
        plot_item_rs.showGrid(x=True, y=True, alpha=0.3)

        drew_any_lattice = False

        if self.cb_show_substrate_real_lattice.isChecked() and self.app_controller.substrate_real_space_results:
            sub_params = self.app_controller.substrate_real_space_results
            if "a1_vec_nm" in sub_params and "a2_vec_nm" in sub_params:
                self._draw_single_real_space_lattice(
                    plot_item_rs, 
                    np.array(sub_params["a1_vec_nm"]), 
                    np.array(sub_params["a2_vec_nm"]),
                    pen_color='r',
                    symbol='o',
                    symbol_size=float(self.substrate_atom_size_spin.value()),
                    symbol_color='darkred',
                    label_text="S",
                    n_cells=int(self.substrate_lattice_cells_spin.value())
                )
                drew_any_lattice = True

        for i, cb_ads_set in enumerate(self.adsorbate_set_checkboxes):
            if cb_ads_set.isChecked():
                set_index = self.ads_set_combo_vis.itemData(i)
                if set_index is not None:
                    ads_params = self.app_controller.adsorbate_real_space_results.get(set_index)
                    if ads_params and "a1_vec_nm" in ads_params and "a2_vec_nm" in ads_params:
                        a1_ads_vec = np.array(ads_params["a1_vec_nm"])
                        a2_ads_vec = np.array(ads_params["a2_vec_nm"])
                        if self.cb_visual_align.isChecked():
                            theta = self.visual_alignment_angle_rad
                            rotation_matrix = np.array([[np.cos(theta), -np.sin(theta)],
                                                        [np.sin(theta),  np.cos(theta)]])
                            a1_ads_vec = np.dot(rotation_matrix, a1_ads_vec)
                            a2_ads_vec = np.dot(rotation_matrix, a2_ads_vec)
                        colors = ['b', 'g', 'purple', 'orange']
                        symbols = ['s', 't', 'd', 'star']
                        color = colors[i % len(colors)]
                        symbol = symbols[i % len(symbols)]
                        self._draw_single_real_space_lattice(
                            plot_item_rs,
                            a1_ads_vec,
                            a2_ads_vec,
                            pen_color=color,
                            symbol=symbol,
                            symbol_size=float(self.adsorbate_atom_size_spin.value()),
                            symbol_color=color,
                            label_text=f"A{i+1}",
                            offset_factor=0.0,
                            n_cells=int(self.adsorbate_lattice_cells_spin.value())
                        )
                        drew_any_lattice = True

        if self.custom_adsorbate_definition and self.custom_adsorbate_visibility_checkbox.isChecked():
            custom_a1 = np.array(self.custom_adsorbate_definition["a1_vec_nm"], dtype=float)
            custom_a2 = np.array(self.custom_adsorbate_definition["a2_vec_nm"], dtype=float)
            custom_offset = np.array(self.custom_adsorbate_definition.get("offset_nm", np.zeros(2)), dtype=float)

            if self.cb_visual_align.isChecked() and self.app_controller.substrate_real_space_results:
                theta = self.visual_alignment_angle_rad
                rotation_matrix = np.array([[np.cos(theta), -np.sin(theta)],
                                            [np.sin(theta),  np.cos(theta)]])
                custom_a1 = np.dot(rotation_matrix, custom_a1)
                custom_a2 = np.dot(rotation_matrix, custom_a2)
                custom_offset = np.dot(rotation_matrix, custom_offset)

            custom_symbol = self.custom_adsorbate_definition.get("symbol", "star")

            self._draw_single_real_space_lattice(
                plot_item_rs,
                custom_a1,
                custom_a2,
                pen_color='orange',
                symbol=custom_symbol,
                symbol_size=float(self.adsorbate_atom_size_spin.value()),
                symbol_color='orange',
                label_text="Custom",
                offset_factor=0.0,
                n_cells=int(self.adsorbate_lattice_cells_spin.value()),
                absolute_offset=custom_offset
            )
            drew_any_lattice = True

        if view_box and drew_any_lattice:
            if force_autorange:
                view_box.autoRange(padding=0.02)
            elif previous_range:
                view_box.setRange(xRange=list(previous_range[0]), yRange=list(previous_range[1]), padding=0)

        self._real_space_force_autorange = False


    def _draw_single_real_space_lattice(self,
                                        plot_item: PlotItem,
                                        a1_vec: np.ndarray,
                                        a2_vec: np.ndarray,
                                        pen_color='k',
                                        symbol='o',
                                        symbol_size: float = 8.0,
                                        symbol_color='k',
                                        label_text: str = "",
                                        offset_factor: float = 0.0,
                                        n_cells: int = 10,
                                        absolute_offset: Optional[np.ndarray] = None):
        """
        Draw a single real space lattice visualization.
        
        Args:
            plot_item (PlotItem): The plot item to draw on
            a1_vec (np.ndarray): First lattice vector
            a2_vec (np.ndarray): Second lattice vector
            pen_color (str): Color for vector lines
            symbol (str): Symbol for lattice points
            symbol_size (int): Size of lattice point symbols
            symbol_color (str): Color of lattice points
            label_text (str): Text label for the lattice
            offset_factor (float): Offset factor for lattice position
            n_cells (int): Number of cells to draw in each direction
            
        Note:
            The lattice is drawn with proper scaling and includes:
            - Lattice points
            - Basis vectors
            - Angle indicators
            - Vector labels
        """
        if a1_vec is None or a2_vec is None or len(a1_vec) != 2 or len(a2_vec) != 2: return

        points_data = []
        offset = (a1_vec + a2_vec) * offset_factor
        if absolute_offset is not None:
            absolute_offset = np.asarray(absolute_offset, dtype=float)
            if absolute_offset.shape == (2,):
                offset = offset + absolute_offset
        origin_real = offset
        
        for m in range(-n_cells, n_cells + 1):
            for n in range(-n_cells, n_cells + 1):
                pt = m * a1_vec + n * a2_vec + offset
                points_data.append({'pos': pt, 'symbol': symbol, 'size': symbol_size, 'pen': None, 'brush': pg.mkBrush(symbol_color)})
        
        if points_data:
            scatter = ScatterPlotItem()
            scatter.setData(spots=points_data)
            plot_item.addItem(scatter)

        pen = pg.mkPen(pen_color, width=2)
        plot_item.plot([origin_real[0], origin_real[0] + a1_vec[0]], [origin_real[1], origin_real[1] + a1_vec[1]], pen=pen)
        plot_item.plot([origin_real[0], origin_real[0] + a2_vec[0]], [origin_real[1], origin_real[1] + a2_vec[1]], pen=pen)

        if label_text:
            text_a1 = pg.TextItem(f"{label_text}-a1", color=pg.mkColor(pen_color), anchor=(0.5, 1.2))
            text_a1.setPos(origin_real[0] + a1_vec[0]*0.5, origin_real[1] + a1_vec[1]*0.5)
            plot_item.addItem(text_a1)
            text_a2 = pg.TextItem(f"{label_text}-a2", color=pg.mkColor(pen_color), anchor=(0.5, 1.2))
            text_a2.setPos(origin_real[0] + a2_vec[0]*0.5, origin_real[1] + a2_vec[1]*0.5)
            plot_item.addItem(text_a2)

        norm_a1 = np.linalg.norm(a1_vec)
        norm_a2 = np.linalg.norm(a2_vec)
        if norm_a1 > 1e-6 and norm_a2 > 1e-6:
            angle_line_end1 = origin_real + a1_vec * 0.3 / norm_a1 * min(norm_a1, norm_a2)
            angle_line_end2 = origin_real + a2_vec * 0.3 / norm_a2 * min(norm_a1, norm_a2)
            plot_item.plot([angle_line_end1[0], origin_real[0], angle_line_end2[0]],
                           [angle_line_end1[1], origin_real[1], angle_line_end2[1]], 
                           pen=pg.mkPen(color=pen_color, style=Qt.PenStyle.DotLine, width=1))


    def _update_real_space_param_labels(self):
        """Update real space parameter labels with current values."""
        logger.debug("Visualizer: Updating real space parameter labels...")

        def apply_display(label: QLabel, display) -> None:
            label.setText(display.text)
            label.setToolTip(display.tooltip)

        current_ads_set_idx_vis = self.ads_set_combo_vis.currentData()
        bundle = self._presenter.build_real_space_label_bundle(current_ads_set_idx_vis)

        apply_display(self.sub_real_a1_label, bundle.substrate_a1)
        apply_display(self.sub_real_a2_label, bundle.substrate_a2)
        apply_display(self.sub_real_alpha_label, bundle.substrate_alpha)

        apply_display(self.ads_real_a1_label, bundle.adsorbate_a1)
        apply_display(self.ads_real_a2_label, bundle.adsorbate_a2)
        apply_display(self.ads_real_alpha_label, bundle.adsorbate_alpha)

        self.calibration_sigma_label.setText(bundle.calibration_text)
        self.angle_sub_ads_label.setText("- deg")
        self.calculate_sub_ads_angle_button.setEnabled(bundle.angle_button_enabled)

    @pyqtSlot()
    def _on_calculate_sub_ads_angle_clicked(self):
        """
        Performs two calculations:
        1. Calculates and displays the absolute angle between default a1 vectors.
        2. Computes in the background the minimal angle needed for visual alignment.
        """
        logger.debug("Visualizer: Calculate Sub-Ads Angle button clicked.")
        if not self.app_controller:
            return

        current_ads_set_idx_vis = self.ads_set_combo_vis.currentData()
        if current_ads_set_idx_vis is None:
            QMessageBox.information(self, "Info", "Please select an adsorbate set.")
            return

        result = self._presenter.calculate_sub_ads_angle(current_ads_set_idx_vis)
        self.angle_sub_ads_label.setText(result.display_text)

        if result.error_message:
            QMessageBox.critical(self, "Calculation Error", f"Could not calculate angle: {result.error_message}")
            return

        if result.alignment_angle_rad is not None:
            self.visual_alignment_angle_rad = result.alignment_angle_rad
            if logger:
                logger.info(
                    "Stored visual alignment angle (background): %.3fdeg",
                    np.degrees(self.visual_alignment_angle_rad),
                )
            if self.cb_visual_align.isChecked():
                self._trigger_redraw_all_visuals()
        else:
            self.visual_alignment_angle_rad = 0.0

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
        if hasattr(self, "_pyvista_adapter"):
            self._pyvista_adapter.close()
        super().closeEvent(event)

    @pyqtSlot(float)
    def _on_substrate_offset_value_changed(self, _value: float):
        new_offset = (self.substrate_offset_x_spin.value(), self.substrate_offset_y_spin.value())
        if self._offset_state.set_substrate_offset(new_offset):
            logger.debug("Updated substrate visual offset to %s.", new_offset)
            self._refresh_3d_plotter_if_open()

    @pyqtSlot(float)
    def _on_adsorbate_offset_value_changed(self, _value: float):
        current_combo_index = self.ads_set_combo_vis.currentIndex()
        set_index = self.ads_set_combo_vis.itemData(current_combo_index)
        if set_index is None:
            return
        new_offset = (self.adsorbate_offset_x_spin.value(), self.adsorbate_offset_y_spin.value())
        if self._offset_state.set_adsorbate_offset(set_index, new_offset):
            logger.debug("Updated adsorbate set %s visual offset to %s.", set_index, new_offset)
            self._refresh_3d_plotter_if_open()

    def _update_offset_controls_from_state(self):
        """Sync spin boxes with controller-stored offsets."""
        sub_offset = self._offset_state.get_substrate_offset()
        for spin, value in (
            (self.substrate_offset_x_spin, sub_offset[0]),
            (self.substrate_offset_y_spin, sub_offset[1]),
        ):
            spin.blockSignals(True)
            spin.setValue(value)
            spin.blockSignals(False)

        current_combo_index = self.ads_set_combo_vis.currentIndex()
        set_index = self.ads_set_combo_vis.itemData(current_combo_index)
        offset = self._offset_state.get_adsorbate_offset(set_index)

        for spin, value in (
            (self.adsorbate_offset_x_spin, offset[0]),
            (self.adsorbate_offset_y_spin, offset[1]),
        ):
            spin.blockSignals(True)
            spin.setValue(value)
            spin.blockSignals(False)

