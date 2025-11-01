# lfa/gui/visualization_manager.py
import logging
import numpy as np
from typing import Optional, List, Tuple, Union, Dict, Any
from PyQt6.QtCore import QObject, pyqtSignal, QPointF, Qt 

from .utils.display import sanitize_numeric_array

try:
    import pyqtgraph as pg
except ImportError:
    pg = None
    logging.critical("VisualizationManager: PyQtGraph is not available! Visualizations will not work.")

try:
    from ..core.history import HistoryNode
    from ..logic.history_manager import HistoryManager
    from ..analysis.lattice import get_reciprocal_points, KNOWN_LATTICES
except ImportError as e: 
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

        if not pg or image_view is None:
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

        self.substrate_raw_markers: Optional[pg.ScatterPlotItem] = None
        self.substrate_transformed_markers: Optional[pg.ScatterPlotItem] = None
        self.substrate_pair_lines: List[pg.PlotDataItem] = []
        self._substrate_raw_visible: bool = True
        self._substrate_transformed_visible: bool = True

        self.adsorbate_raw_markers: Dict[int, pg.ScatterPlotItem] = {}
        self.adsorbate_transformed_markers: Dict[int, pg.ScatterPlotItem] = {}
        self.adsorbate_pair_lines: Dict[int, List[pg.PlotDataItem]] = {}
        self._adsorbate_raw_visible: bool = True
        self._adsorbate_transformed_visible: bool = True
        self._adsorbate_raw_visibility: Dict[int, bool] = {}
        self._adsorbate_transformed_visibility: Dict[int, bool] = {}

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
            
            self.view_box.autoRange() 
            logger.debug(f"VisualizationManager: View updated for node '{current_node.operation_name}'.")
        else:
            self.image_item.clear()
            logger.debug("VisualizationManager: No node to display or node has no data. View cleared.")

    def _remove_item_from_view(self, item):
        """Safely removes a graphic item from the view box."""
        if not item or not self.view_box:
            return
        try:
            self.view_box.removeItem(item)
        except RuntimeError:
            pass

    def _clear_all_graphic_items(self):
        """Internal method for clearing all managed graphic items (overlays, markers)."""
        if not self.view_box: return

        if self.ideal_lattice_overlay_item:
            self._remove_item_from_view(self.ideal_lattice_overlay_item)
            self.ideal_lattice_overlay_item = None

        if self.substrate_raw_markers:
            self._remove_item_from_view(self.substrate_raw_markers)
            self.substrate_raw_markers = None

        if self.substrate_transformed_markers:
            self._remove_item_from_view(self.substrate_transformed_markers)
            self.substrate_transformed_markers = None

        for line in self.substrate_pair_lines:
            self._remove_item_from_view(line)
        self.substrate_pair_lines = []

        for marker in self.adsorbate_raw_markers.values():
            self._remove_item_from_view(marker)
        self.adsorbate_raw_markers = {}

        for marker in self.adsorbate_transformed_markers.values():
            self._remove_item_from_view(marker)
        self.adsorbate_transformed_markers = {}

        for line_list in self.adsorbate_pair_lines.values():
            for line in line_list:
                self._remove_item_from_view(line)
        self.adsorbate_pair_lines = {}

    def update_substrate_spot_pairs(
        self,
        spot_pairs: List[Tuple[Tuple[float, float], Tuple[float, float]]]
    ) -> None:
        """Render both raw and transformed substrate spots together with pair links."""
        raw_coords = [raw for raw, _ in spot_pairs if raw is not None]
        transformed_coords = [transformed for _, transformed in spot_pairs if transformed is not None]
        self.update_substrate_raw_spots(raw_coords)
        self.update_substrate_transformed_spots(transformed_coords)
        self.update_substrate_pair_lines(spot_pairs)

    def update_substrate_raw_spots(self, raw_points: List[Tuple[float, float]]) -> None:
        """Render raw substrate spot markers."""
        if not self.view_box:
            return

        if self.substrate_raw_markers:
            self._remove_item_from_view(self.substrate_raw_markers)
            self.substrate_raw_markers = None

        points_array = sanitize_numeric_array(raw_points) if raw_points else None
        if points_array is None or points_array.size == 0:
            return

        try:
            marker = pg.ScatterPlotItem(
                pos=points_array,
                symbol='x',
                size=11,
                pen=pg.mkPen((255, 215, 0), width=2),
                brush=pg.mkBrush(None)
            )
            marker.setVisible(self._substrate_raw_visible)
            self.view_box.addItem(marker)
            self.substrate_raw_markers = marker
        except Exception as exc:
            logger.exception("VisualizationManager: Failed to draw raw substrate spots: %s", exc)

    def update_substrate_transformed_spots(self, transformed_points: List[Tuple[float, float]]) -> None:
        """Render transformed substrate spot markers."""
        if not self.view_box:
            return

        if self.substrate_transformed_markers:
            self._remove_item_from_view(self.substrate_transformed_markers)
            self.substrate_transformed_markers = None

        points_array = sanitize_numeric_array(transformed_points) if transformed_points else None
        if points_array is None or points_array.size == 0:
            return

        try:
            marker = pg.ScatterPlotItem(
                pos=points_array,
                symbol='o',
                size=11,
                pen=pg.mkPen((0, 200, 140), width=2),
                brush=pg.mkBrush(None)
            )
            marker.setVisible(self._substrate_transformed_visible)
            self.view_box.addItem(marker)
            self.substrate_transformed_markers = marker
        except Exception as exc:
            logger.exception("VisualizationManager: Failed to draw transformed substrate spots: %s", exc)

    def update_substrate_pair_lines(
        self,
        spot_pairs: List[Tuple[Tuple[float, float], Tuple[float, float]]]
    ) -> None:
        """Draw connector lines between raw and transformed substrate spots."""
        if not self.view_box:
            return

        for line in self.substrate_pair_lines:
            self._remove_item_from_view(line)
        self.substrate_pair_lines = []

        if not spot_pairs:
            return

        new_lines: List[pg.PlotDataItem] = []
        for raw_point, transformed_point in spot_pairs:
            raw_array = sanitize_numeric_array(raw_point) if raw_point is not None else None
            transformed_array = sanitize_numeric_array(transformed_point) if transformed_point is not None else None
            if raw_array is None or transformed_array is None or raw_array.size < 2 or transformed_array.size < 2:
                continue
            try:
                line_item = pg.PlotDataItem(
                    x=[raw_array[0], transformed_array[0]],
                    y=[raw_array[1], transformed_array[1]],
                    pen=pg.mkPen((180, 180, 180, 180), width=1, style=Qt.PenStyle.DashLine)
                )
                line_item.setVisible(self._substrate_raw_visible and self._substrate_transformed_visible)
                self.view_box.addItem(line_item)
                new_lines.append(line_item)
            except Exception as exc:
                logger.exception("VisualizationManager: Failed to draw substrate pair line: %s", exc)
        self.substrate_pair_lines = new_lines

    def set_substrate_raw_visible(self, visible: bool) -> None:
        """Toggle visibility of raw substrate markers."""
        self._substrate_raw_visible = visible
        if self.substrate_raw_markers:
            self.substrate_raw_markers.setVisible(visible)
        self._update_substrate_pair_visibility()

    def set_substrate_transformed_visible(self, visible: bool) -> None:
        """Toggle visibility of transformed substrate markers."""
        self._substrate_transformed_visible = visible
        if self.substrate_transformed_markers:
            self.substrate_transformed_markers.setVisible(visible)
        self._update_substrate_pair_visibility()

    def _update_substrate_pair_visibility(self) -> None:
        """Ensure connector lines follow combined visibility state."""
        should_show = self._substrate_raw_visible and self._substrate_transformed_visible
        for line in self.substrate_pair_lines:
            line.setVisible(should_show)

    def update_adsorbate_spot_pairs(
        self,
        set_id: int,
        spot_pairs: List[Tuple[Tuple[float, float], Tuple[float, float]]]
    ) -> None:
        """Render raw and transformed adsorbate spots plus connector lines for a given set."""
        raw_coords = [raw for raw, _ in spot_pairs if raw is not None]
        transformed_coords = [transformed for _, transformed in spot_pairs if transformed is not None]
        self.update_adsorbate_raw_spots(set_id, raw_coords)
        self.update_adsorbate_transformed_spots(set_id, transformed_coords)
        self.update_adsorbate_pair_lines(set_id, spot_pairs)
        self._update_adsorbate_pair_visibility_for_set(set_id)

    def update_adsorbate_raw_spots(self, set_id: int, raw_points: List[Tuple[float, float]]) -> None:
        """Render raw adsorbate spot markers for a given set."""
        if not self.view_box:
            return

        existing_marker = self.adsorbate_raw_markers.pop(set_id, None)
        if existing_marker:
            self._remove_item_from_view(existing_marker)

        points_array = sanitize_numeric_array(raw_points) if raw_points else None
        if points_array is None or points_array.size == 0:
            return

        try:
            marker = pg.ScatterPlotItem(
                pos=points_array,
                symbol='x',
                size=10,
                pen=pg.mkPen((255, 140, 0), width=2),
                brush=pg.mkBrush(None)
            )
            marker.setVisible(self._is_adsorbate_raw_visible(set_id))
            self.view_box.addItem(marker)
            self.adsorbate_raw_markers[set_id] = marker
        except Exception as exc:
            logger.exception("VisualizationManager: Failed to draw raw adsorbate spots for set %s: %s", set_id, exc)

    def update_adsorbate_transformed_spots(self, set_id: int, transformed_points: List[Tuple[float, float]]) -> None:
        """Render transformed adsorbate spot markers for a given set."""
        if not self.view_box:
            return

        existing_marker = self.adsorbate_transformed_markers.pop(set_id, None)
        if existing_marker:
            self._remove_item_from_view(existing_marker)

        points_array = sanitize_numeric_array(transformed_points) if transformed_points else None
        if points_array is None or points_array.size == 0:
            return

        try:
            marker = pg.ScatterPlotItem(
                pos=points_array,
                symbol='s',
                size=10,
                pen=pg.mkPen((70, 130, 180), width=2),
                brush=pg.mkBrush(None)
            )
            marker.setVisible(self._is_adsorbate_transformed_visible(set_id))
            self.view_box.addItem(marker)
            self.adsorbate_transformed_markers[set_id] = marker
        except Exception as exc:
            logger.exception("VisualizationManager: Failed to draw transformed adsorbate spots for set %s: %s", set_id, exc)

    def update_adsorbate_pair_lines(
        self,
        set_id: int,
        spot_pairs: List[Tuple[Tuple[float, float], Tuple[float, float]]]
    ) -> None:
        """Draw connector lines between raw and transformed adsorbate spots for a given set."""
        if not self.view_box:
            return

        existing_lines = self.adsorbate_pair_lines.pop(set_id, [])
        for line in existing_lines:
            self._remove_item_from_view(line)

        if not spot_pairs:
            return

        new_lines: List[pg.PlotDataItem] = []
        for raw_point, transformed_point in spot_pairs:
            raw_array = sanitize_numeric_array(raw_point) if raw_point is not None else None
            transformed_array = sanitize_numeric_array(transformed_point) if transformed_point is not None else None
            if raw_array is None or transformed_array is None or raw_array.size < 2 or transformed_array.size < 2:
                continue
            try:
                line_item = pg.PlotDataItem(
                    x=[raw_array[0], transformed_array[0]],
                    y=[raw_array[1], transformed_array[1]],
                    pen=pg.mkPen((255, 140, 0, 160), width=1, style=Qt.PenStyle.DashLine)
                )
                line_item.setVisible(self._adsorbate_raw_visible and self._adsorbate_transformed_visible)
                self.view_box.addItem(line_item)
                new_lines.append(line_item)
            except Exception as exc:
                logger.exception("VisualizationManager: Failed to draw adsorbate pair line for set %s: %s", set_id, exc)

        if new_lines:
            self.adsorbate_pair_lines[set_id] = new_lines
        self._update_adsorbate_pair_visibility_for_set(set_id)

    def set_adsorbate_raw_visible(self, visible: bool) -> None:
        """Toggle visibility of raw adsorbate markers."""
        self._adsorbate_raw_visible = visible
        self._adsorbate_raw_visibility.clear()
        for set_id, marker in self.adsorbate_raw_markers.items():
            marker.setVisible(visible)
        self._update_adsorbate_pair_visibility()

    def set_adsorbate_raw_visible_for_set(self, set_id: int, visible: bool) -> None:
        self._adsorbate_raw_visibility[set_id] = visible
        marker = self.adsorbate_raw_markers.get(set_id)
        if marker:
            marker.setVisible(visible)
        self._update_adsorbate_pair_visibility_for_set(set_id)

    def set_adsorbate_transformed_visible(self, visible: bool) -> None:
        """Toggle visibility of transformed adsorbate markers."""
        self._adsorbate_transformed_visible = visible
        self._adsorbate_transformed_visibility.clear()
        for set_id, marker in self.adsorbate_transformed_markers.items():
            marker.setVisible(visible)
        self._update_adsorbate_pair_visibility()

    def set_adsorbate_transformed_visible_for_set(self, set_id: int, visible: bool) -> None:
        self._adsorbate_transformed_visibility[set_id] = visible
        marker = self.adsorbate_transformed_markers.get(set_id)
        if marker:
            marker.setVisible(visible)
        self._update_adsorbate_pair_visibility_for_set(set_id)

    def _is_adsorbate_raw_visible(self, set_id: int) -> bool:
        return self._adsorbate_raw_visibility.get(set_id, self._adsorbate_raw_visible)

    def _is_adsorbate_transformed_visible(self, set_id: int) -> bool:
        return self._adsorbate_transformed_visibility.get(set_id, self._adsorbate_transformed_visible)

    def _update_adsorbate_pair_visibility_for_set(self, set_id: int) -> None:
        should_show = self._is_adsorbate_raw_visible(set_id) and self._is_adsorbate_transformed_visible(set_id)
        for line in self.adsorbate_pair_lines.get(set_id, []):
            line.setVisible(should_show)

    def _update_adsorbate_pair_visibility(self) -> None:
        """Ensure adsorbate connector lines respect combined visibility flags."""
        for set_id in list(self.adsorbate_pair_lines.keys()):
            self._update_adsorbate_pair_visibility_for_set(set_id)

    def clear_adsorbate_layers_for_set(self, set_id: int) -> None:
        """Remove all overlay items associated with an adsorbate set."""
        marker = self.adsorbate_raw_markers.pop(set_id, None)
        if marker:
            self._remove_item_from_view(marker)

        marker = self.adsorbate_transformed_markers.pop(set_id, None)
        if marker:
            self._remove_item_from_view(marker)

        line_items = self.adsorbate_pair_lines.pop(set_id, [])
        for line in line_items:
            self._remove_item_from_view(line)
        self._adsorbate_raw_visibility.pop(set_id, None)
        self._adsorbate_transformed_visibility.pop(set_id, None)

    def clear_adsorbate_layers(self) -> None:
        """Remove overlays for all adsorbate sets."""
        all_ids = set(self.adsorbate_raw_markers.keys()) | set(self.adsorbate_transformed_markers.keys()) | set(self.adsorbate_pair_lines.keys())
        for set_id in all_ids:
            self.clear_adsorbate_layers_for_set(set_id)

    def _set_image_display(self, image_data: np.ndarray, data_type: str):
        """Sets the image data in the ImageItem with the appropriate orientation and scaling."""
        if not self.image_item or not self.view_box: return

        if data_type == "STM":
            self.view_box.invertY(True)
            self.image_item.setImage(image_data.astype(np.float32).T)
        elif data_type == "FFT":
            self.view_box.invertY(True)
            self.image_item.setImage(image_data.astype(np.float32).T)
        else: 
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
            except Exception as e:
                logger.error(f"VisualizationManager: Failed to connect FFT mouse click handler: {e}")
        elif not scene: 
             logger.error("VisualizationManager: Cannot connect FFT click handler, ImageItem scene is None.")
        elif not hasattr(scene, 'sigMouseClicked'):
             logger.error("VisualizationManager: Scene object does not have sigMouseClicked signal.")


    def _disconnect_fft_click_handler(self):
        """Disconnects the internal slot from the ImageItem scene click signal."""
        if self._current_fft_mouse_click_connection is not None:
            if self.image_item: 
                scene = getattr(self.image_item, 'scene', lambda: None)()
                if scene and hasattr(scene, 'sigMouseClicked'):
                    try:
                        scene.sigMouseClicked.disconnect(self._current_fft_mouse_click_connection)
                        logger.debug("VisualizationManager: FFT mouse click handler disconnected.")
                    except (TypeError, RuntimeError):
                        logger.debug("VisualizationManager: Could not disconnect FFT mouse click (normal if connection was already broken or scene changed).")
            self._current_fft_mouse_click_connection = None
            
    def _handle_fft_view_mouse_click(self, event):
        """
        Internal slot handling mouse clicks on the FFT image.
        Maps the click coordinates and emits the ``fft_view_clicked`` signal.
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
        if custom_lattice_definition and (
            selected_substrate_name == panel_custom_option_text or
            (isinstance(selected_substrate_name, str) and
             selected_substrate_name not in ("None", panel_custom_option_text) and
             selected_substrate_name not in KNOWN_LATTICES)
        ):
            lattice_info_to_use = custom_lattice_definition
        elif isinstance(selected_substrate_name, str) and \
             selected_substrate_name != "None" and \
             selected_substrate_name in KNOWN_LATTICES:
            lattice_info_to_use = selected_substrate_name
        
        if not lattice_info_to_use:
            logger.debug("VisualizationManager: No valid lattice selected for overlay.")
            return

        root_node = self.history_manager.get_root_node_for_node(current_history_node.node_id)
        if not (root_node and root_node.operation_name == "Original"):
            logger.warning("VisualizationManager: Could not trace back to Original node for lattice calibration.")
            return

        orig_params = root_node.parameters
        Lx = orig_params.get("size_nm_x")
        Ly = orig_params.get("size_nm_y")
        
        fft_data_rows_ky, fft_data_cols_kx = fft_image_data.shape

        if not (Lx and Ly and Lx > 0 and Ly > 0 and fft_data_cols_kx > 0 and fft_data_rows_ky > 0):
            logger.warning("VisualizationManager: Missing calibration data (Lx, Ly) or invalid FFT shape for lattice overlay.")
            return

        ideal_points_g_nm_inv = get_reciprocal_points(lattice_info_to_use, max_hk=2)
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


