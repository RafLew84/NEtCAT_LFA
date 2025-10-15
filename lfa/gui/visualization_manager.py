# lfa/gui/visualization_manager.py
import logging
import numpy as np
from typing import Optional, List, Tuple, Union, Dict, Any
from PyQt6.QtCore import QObject, pyqtSignal, QPointF, Qt 

try:
    import pyqtgraph as pg
except ImportError: # pragma: no cover
    pg = None
    logging.critical("VisualizationManager: PyQtGraph is not available! Visualizations will not work.")

try:
    from ..core.history import HistoryNode
    from ..logic.history_manager import HistoryManager # Zakładając, że history_manager jest w logic
    from ..analysis.lattice import get_reciprocal_points, KNOWN_LATTICES
except ImportError as e: # pragma: no cover
    logging.error(f"VisualizationManager: Error importing project modules: {e}")
    HistoryNode = None
    HistoryManager = None
    KNOWN_LATTICES = False
    def get_reciprocal_points(*args, **kwargs): return None


logger = logging.getLogger(__name__)

class VisualizationManager(QObject):
    """
    Manages the display of images (STM, FFT) and graphical overlays
    (ideal lattice, selected spots) in the main ImageView.
    """
    fft_view_clicked = pyqtSignal(QPointF)

    def __init__(self,
                 image_view: pg.ImageView,
                 history_manager: HistoryManager,
                 parent: Optional[QObject] = None):
        super().__init__(parent)

        if not pg or image_view is None: # pragma: no cover
            logger.critical("VisualizationManager: PyQtGraph or ImageView is not available during initialization!")
            self.image_view = None
            self.view_box = None
            self.image_item = None
            self._is_initialized_correctly = False
            return

        self._is_initialized_correctly = True
        self.image_view = image_view
        self.view_box = self.image_view.getView()
        self.image_item = self.image_view.getImageItem()
        self.history_manager = history_manager

        self.ideal_lattice_overlay_item: Optional[pg.ScatterPlotItem] = None
        self.substrate_spot_markers: Optional[pg.ScatterPlotItem] = None
        self.adsorbate_spot_set_markers: List[pg.ScatterPlotItem] = []

        self._current_fft_mouse_click_connection = None 

        logger.info("VisualizationManager initialized successfully.")

    def update_view(self,
                    current_node: Optional[HistoryNode],
                    show_ideal_lattice: bool,
                    selected_substrate_name: str, 
                    custom_lattice_definition: Optional[Dict[str, Any]], 
                    panel_custom_option_text: str,
                    substrate_spots_data: List[Tuple[float, float]],
                    show_substrate_markers: bool,
                    adsorbate_spot_sets_data: List[List[Tuple[float, float]]],
                    show_adsorbate_markers: bool
                    ) -> None:
        """
        Main method updating the entire image view (ImageItem) with graphical overlays.
        Replaces a large part of the logic from the original MainWindow.display_image_data() method.

        Args:
            current_node: History node to display.
            show_ideal_lattice: Whether to show the ideal lattice overlay.
            selected_substrate_name: Name of the selected substrate from the panel.
            custom_lattice_definition: Custom lattice definition (if exists).
            panel_custom_option_text: Text of the "<Custom Define...>" option from the panel.
            substrate_spots_data: List of substrate spot coordinates.
            show_substrate_markers: Whether to show substrate spot markers.
            adsorbate_spot_sets_data: List of lists of adsorbate spot coordinates.
            show_adsorbate_markers: Whether to show adsorbate spot markers.
        """
        if not self._is_initialized_correctly or self.view_box is None or self.image_item is None:
            logger.error("VisualizationManager not properly initialized, cannot update view.")
            return

        logger.debug("VisualizationManager: Updating view...")

        self._clear_all_graphic_items()      
        self._disconnect_fft_click_handler() 

        if current_node and current_node.image_data is not None:
            display_data = current_node.image_data
            data_type = current_node.data_type

            self._set_image_display(display_data, data_type)

            if data_type == "FFT":
                self._connect_fft_click_handler() 

                if KNOWN_LATTICES and show_ideal_lattice:
                    self._draw_ideal_lattice_overlay(
                        fft_image_data=display_data,
                        current_history_node=current_node, 
                        selected_substrate_name=selected_substrate_name,
                        custom_lattice_definition=custom_lattice_definition,
                        panel_custom_option_text=panel_custom_option_text
                    )
            
            if data_type == "FFT":
                self._draw_spot_markers(
                    substrate_spots_data, show_substrate_markers,
                    adsorbate_spot_sets_data, show_adsorbate_markers
                )
            
            self.view_box.autoRange() 
            logger.debug(f"VisualizationManager: View updated for node '{current_node.operation_name}'.")
        else:
            self.image_item.clear()
            logger.debug("VisualizationManager: No node to display or node has no data. View cleared.")

    def _clear_all_graphic_items(self):
        """Internal method for clearing all managed graphic items (overlays, markers)."""
        if not self.view_box: return

        if self.ideal_lattice_overlay_item:
            try: self.view_box.removeItem(self.ideal_lattice_overlay_item)
            except RuntimeError: pass
            self.ideal_lattice_overlay_item = None
        
        if self.substrate_spot_markers:
            try: self.view_box.removeItem(self.substrate_spot_markers)
            except RuntimeError: pass
            self.substrate_spot_markers = None

        for marker_set in self.adsorbate_spot_set_markers:
            if marker_set:
                try: self.view_box.removeItem(marker_set)
                except RuntimeError: pass
        self.adsorbate_spot_set_markers = []

    def _set_image_display(self, image_data: np.ndarray, data_type: str):
        """Sets the image data in the ImageItem with the appropriate orientation and scaling."""
        if not self.image_item or not self.view_box: return

        if data_type == "STM":
            self.view_box.invertY(True)
            self.image_item.setImage(image_data.astype(np.float32).T)
        elif data_type == "FFT":
            self.view_box.invertY(True)
            self.image_item.setImage(image_data.astype(np.float32).T)
        else: # pragma: no cover
            logger.warning(f"VisualizationManager: Unknown data type '{data_type}', displaying like STM.")
            self.view_box.invertY(True)
            self.image_item.setImage(image_data.astype(np.float32).T, autoLevels=True)

    def _connect_fft_click_handler(self):
        """Connects the internal slot _handle_fft_view_mouse_click to the ImageItem scene click signal."""
        if not self._is_initialized_correctly or not self.image_item:
            logger.warning("VisualizationManager: Cannot connect FFT click handler - ImageItem not available.")
            return
        
        # Upewnij się, że nie ma już aktywnego połączenia
        if self._current_fft_mouse_click_connection is not None:
            logger.debug("VisualizationManager: FFT click handler already connected or not properly disconnected previously.")
            self._disconnect_fft_click_handler() # Spróbuj odłączyć na wszelki wypadek

        scene = getattr(self.image_item, 'scene', lambda: None)()
        if scene and hasattr(scene, 'sigMouseClicked'):
            try:
                self._current_fft_mouse_click_connection = scene.sigMouseClicked.connect(self._handle_fft_view_mouse_click)
                logger.debug("VisualizationManager: FFT mouse click handler successfully connected.")
            except Exception as e: # pragma: no cover
                logger.error(f"VisualizationManager: Failed to connect FFT mouse click handler: {e}")
        elif not scene: # pragma: no cover
             logger.error("VisualizationManager: Cannot connect FFT click handler, ImageItem scene is None.")
        elif not hasattr(scene, 'sigMouseClicked'): # pragma: no cover
             logger.error("VisualizationManager: Scene object does not have sigMouseClicked signal.")


    def _disconnect_fft_click_handler(self):
        """Disconnects the internal slot from the ImageItem scene click signal."""
        if self._current_fft_mouse_click_connection is not None:
            if self.image_item: # Tylko jeśli image_item istnieje
                scene = getattr(self.image_item, 'scene', lambda: None)()
                if scene and hasattr(scene, 'sigMouseClicked'):
                    try:
                        scene.sigMouseClicked.disconnect(self._current_fft_mouse_click_connection)
                        logger.debug("VisualizationManager: FFT mouse click handler disconnected.")
                    except (TypeError, RuntimeError): # pragma: no cover
                        logger.debug("VisualizationManager: Could not disconnect FFT mouse click (normal if connection was already broken or scene changed).")
            self._current_fft_mouse_click_connection = None
            
    def _handle_fft_view_mouse_click(self, event):
        """
        Internal slot handling mouse clicks on the FFT image.
        Maps the click coordinates and emits the `fft_view_clicked` signal.
        """
        if not self._is_initialized_correctly or not self.image_item or not event or not hasattr(event, 'button'):
            if hasattr(event, 'ignore'): event.ignore() # pragma: no cover
            return

        if event.button() == Qt.MouseButton.LeftButton:
            pos_in_item_coords = self.image_item.mapFromScene(event.scenePos())

            mapped_pos_data_coords = self.image_item.mapToData(pos_in_item_coords)

            if mapped_pos_data_coords is not None:
                
                self.fft_view_clicked.emit(QPointF(mapped_pos_data_coords.x(), mapped_pos_data_coords.y()))
                logger.debug(f"VisualizationManager: FFT view clicked. Emitted data coords (original kx, original ky): ({mapped_pos_data_coords.x():.2f}, {mapped_pos_data_coords.y():.2f})")
                if hasattr(event, 'accept'): event.accept()
            else: # pragma: no cover
                logger.debug("VisualizationManager: FFT view click outside image data bounds for mapToData.")
                if hasattr(event, 'ignore'): event.ignore()
        else:
            if hasattr(event, 'ignore'): event.ignore()


    def _draw_ideal_lattice_overlay(self,
                                    fft_image_data: np.ndarray,
                                    current_history_node: HistoryNode, 
                                    selected_substrate_name: Union[str, Dict[str, Any], None],
                                    custom_lattice_definition: Optional[Dict[str, Any]],
                                    panel_custom_option_text: str):
        """Draws the ideal lattice overlay on the FFT image."""
        if not self.view_box or not KNOWN_LATTICES: return

        lattice_info_to_use: Optional[Union[str, Dict[str, Any]]] = None
        if selected_substrate_name == panel_custom_option_text and custom_lattice_definition:
            lattice_info_to_use = custom_lattice_definition
        elif isinstance(selected_substrate_name, str) and \
             selected_substrate_name != "None" and \
             selected_substrate_name != panel_custom_option_text:
            lattice_info_to_use = selected_substrate_name
        
        if not lattice_info_to_use:
            logger.debug("VisualizationManager: No valid lattice selected for overlay.")
            return

        root_node = self.history_manager.get_root_node_for_node(current_history_node.node_id)
        if not (root_node and root_node.operation_name == "Original"): # pragma: no cover
            logger.warning("VisualizationManager: Could not trace back to Original node for lattice calibration.")
            return

        orig_params = root_node.parameters
        Lx = orig_params.get("size_nm_x")
        Ly = orig_params.get("size_nm_y")
        
        fft_data_rows_ky, fft_data_cols_kx = fft_image_data.shape

        if not (Lx and Ly and Lx > 0 and Ly > 0 and fft_data_cols_kx > 0 and fft_data_rows_ky > 0): # pragma: no cover
            logger.warning("VisualizationManager: Missing calibration data (Lx, Ly) or invalid FFT shape for lattice overlay.")
            return

        ideal_points_g_nm_inv = get_reciprocal_points(lattice_info_to_use, max_hk=2) # Gx, Gy w nm^-1
        if not ideal_points_g_nm_inv:
            logger.warning("VisualizationManager: Could not get ideal reciprocal points.")
            return

        pixel_coords_for_scatter = []
        center_display_x = fft_data_rows_ky / 2.0
        center_display_y = fft_data_cols_kx / 2.0

        for Gx_nm_inv, Gy_nm_inv in ideal_points_g_nm_inv:
            display_x_px = center_display_x + (Gy_nm_inv * Ly)
            display_y_px = center_display_y + (Gx_nm_inv * Lx)
            pixel_coords_for_scatter.append({
                'pos': (display_x_px, display_y_px),
                'symbol': 'o', 'size': 7,
                'pen': pg.mkPen('r', width=1.5), 'brush': pg.mkBrush(None)
            })
        
        if pixel_coords_for_scatter:
            self.ideal_lattice_overlay_item = pg.ScatterPlotItem()
            self.ideal_lattice_overlay_item.setData(spots=pixel_coords_for_scatter)
            self.view_box.addItem(self.ideal_lattice_overlay_item)
            display_name = selected_substrate_name if isinstance(selected_substrate_name, str) else selected_substrate_name.get("name", "Custom")
            logger.info(f"VisualizationManager: Displayed ideal lattice overlay for '{display_name}'.")


    def _draw_spot_markers(self,
                           substrate_spots: List[Tuple[float, float]], show_substrate: bool,
                           adsorbate_sets: List[List[Tuple[float, float]]], show_adsorbate: bool):
        """Draws markers for selected substrate and adsorbate spots."""
        if not self.view_box: return

        # --- Substrate Spots ---
        if show_substrate and substrate_spots:
            display_substrate_spots = [(kx, ky) for kx, ky in substrate_spots]
            try:
                self.substrate_spot_markers = pg.ScatterPlotItem(
                    pos=np.array(display_substrate_spots), symbol='o', size=10,
                    pen=pg.mkPen('g', width=2), brush=pg.mkBrush(None)
                )
                self.view_box.addItem(self.substrate_spot_markers)
                logger.debug(f"VisualizationManager: Redrew {len(substrate_spots)} substrate spots.")
            except Exception as e: # pragma: no cover
                logger.exception(f"Error creating/adding substrate spot markers: {e}")

        if show_adsorbate and adsorbate_sets:
            adsorbate_colors = ['b', 'c', 'm', (255, 165, 0)] # Orange
            new_markers_list = []
            for i, spot_set in enumerate(adsorbate_sets):
                if spot_set:
                    display_spot_set = [(kx, ky) for kx, ky in spot_set]
                    color = adsorbate_colors[i % len(adsorbate_colors)]
                    try:
                        markers = pg.ScatterPlotItem(
                            pos=np.array(display_spot_set), symbol='s', size=10,
                            pen=pg.mkPen(color, width=2), brush=pg.mkBrush(None)
                        )
                        self.view_box.addItem(markers)
                        new_markers_list.append(markers)
                    except Exception as e: # pragma: no cover
                        logger.exception(f"Error creating/adding adsorbate spot markers for set {i}: {e}")
            self.adsorbate_spot_set_markers = new_markers_list
            if self.adsorbate_spot_set_markers:
                logger.debug(f"VisualizationManager: Redrew adsorbate spots for {len(self.adsorbate_spot_set_markers)} sets.")

    def redraw_spot_markers(self, 
                            substrate_spots_data: List[Tuple[float, float]], 
                            show_substrate: bool,
                            adsorbate_spot_sets_data: List[List[Tuple[float, float]]], 
                            show_adsorbate: bool):
        """Redraws the spot markers."""
        if not self._is_initialized_correctly: return
        logger.debug("VisualizationManager: Redrawing spot markers.")
        if self.substrate_spot_markers and self.view_box:
            try: self.view_box.removeItem(self.substrate_spot_markers)
            except RuntimeError: pass
            self.substrate_spot_markers = None
        if self.view_box:
            for marker_set in self.adsorbate_spot_set_markers:
                if marker_set: 
                    try: 
                        self.view_box.removeItem(marker_set)
                    except RuntimeError: 
                        pass
        
        self.adsorbate_spot_set_markers = []
        self._draw_spot_markers(substrate_spots_data, show_substrate, adsorbate_spot_sets_data, show_adsorbate)

    def _clear_spot_markers_only(self):
        """Clears the spot markers."""
        if not self.view_box: return
        if self.substrate_spot_markers:
            try: self.view_box.removeItem(self.substrate_spot_markers)
            except RuntimeError: pass
            self.substrate_spot_markers = None
        for marker_set in self.adsorbate_spot_set_markers:
            if marker_set:
                try: self.view_box.removeItem(marker_set)
                except RuntimeError: pass
        self.adsorbate_spot_set_markers = []