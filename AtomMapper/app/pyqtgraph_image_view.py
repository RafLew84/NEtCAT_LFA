"""pyqtgraph ImageView-based STM viewport aligned with the LFA main-image control."""

from __future__ import annotations

from typing import Optional, Sequence

import numpy as np

from PyQt6.QtCore import QRectF, Qt, pyqtSignal
from PyQt6.QtWidgets import QLabel, QStackedWidget, QVBoxLayout, QWidget

from .models import AtomPoint, AtomRow, LoadedImage, ROIState
from .row_geometry import RowGeometry

try:
    import pyqtgraph as pg
except ImportError:  # pragma: no cover - exercised only in missing-dependency environments
    pg = None


class PyQtGraphSTMViewport(QWidget):
    """STM viewport backed by pyqtgraph.ImageView, mirroring the LFA display control."""

    roi_state_edited = pyqtSignal(object)
    point_selected = pyqtSignal(object)
    point_move_requested = pyqtSignal(object)

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.current_loaded_image: Optional[LoadedImage] = None
        self.current_roi_state: Optional[ROIState] = None
        self.current_atom_rows: tuple[AtomRow, ...] = ()
        self.current_active_row_id: Optional[str] = None
        self.current_active_image_id: Optional[str] = None
        self.current_active_point_id: Optional[str] = None
        self.current_row_geometry: Optional[RowGeometry] = None
        self.current_disturbance_markers: tuple[dict[str, object], ...] = ()
        self.backend_available: bool = pg is not None
        self._syncing_roi_overlay = False
        self._syncing_active_point_target = False

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(8)

        self.title_label = QLabel("STM image")
        self.title_label.setStyleSheet("font-size: 16px; font-weight: 600;")

        self.stack = QStackedWidget(self)

        self.placeholder_label = QLabel("No image selected")
        self.placeholder_label.setObjectName("atommapper_pg_view_placeholder")
        self.placeholder_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.placeholder_label.setWordWrap(True)
        self.placeholder_label.setStyleSheet(
            "border: 1px solid palette(mid); color: palette(mid); padding: 24px;"
        )

        self.image_view = None
        self.plot_widget = None
        self.image_item = None
        self.histogram_widget = None
        self.roi_item = None
        self.row_axis_item = None
        self.row_disturbance_scatter_item = None
        self.point_scatter_item = None
        self.active_point_target = None
        if self.backend_available:
            self.image_view = pg.ImageView(self)
            self.image_view.setObjectName("atommapper_pg_image_view")
            self.plot_widget = self.image_view
            self.image_item = self.image_view.getImageItem()
            self.histogram_widget = self.image_view.getHistogramWidget()
            self.view_box.invertY(True)

            self.row_axis_item = pg.PlotCurveItem(
                pen=pg.mkPen(color=(255, 120, 60, 230), width=2.2, style=Qt.PenStyle.DashLine),
            )
            self.row_axis_item.hide()
            self.view_box.addItem(self.row_axis_item)

            self.row_disturbance_scatter_item = pg.ScatterPlotItem(
                size=14,
                pen=pg.mkPen(color=(40, 20, 20, 245), width=1.5),
                brush=pg.mkBrush(255, 70, 70, 235),
                pxMode=True,
            )
            self.row_disturbance_scatter_item.hide()
            self.view_box.addItem(self.row_disturbance_scatter_item)

            self.point_scatter_item = pg.ScatterPlotItem(
                size=11,
                pen=pg.mkPen(color=(20, 20, 20, 220), width=1.5),
                brush=pg.mkBrush(255, 210, 60, 220),
                hoverable=True,
                pxMode=True,
            )
            self.point_scatter_item.sigClicked.connect(self._on_point_scatter_clicked)
            self.view_box.addItem(self.point_scatter_item)

            self.active_point_target = pg.TargetItem(
                pos=(0.0, 0.0),
                size=14,
                symbol="crosshair",
                pen=pg.mkPen(color=(30, 30, 30, 255), width=2.2),
                brush=pg.mkBrush(110, 255, 140, 120),
                hoverPen=pg.mkPen(color=(255, 255, 255, 255), width=2.4),
                hoverBrush=pg.mkBrush(110, 255, 140, 180),
                movable=True,
            )
            self.active_point_target.hide()
            self.active_point_target.sigPositionChangeFinished.connect(
                self._on_active_point_target_move_finished
            )
            self.view_box.addItem(self.active_point_target)

            self.roi_item = pg.RectROI(
                [0.0, 0.0],
                [4.0, 4.0],
                pen=pg.mkPen(color=(255, 180, 40), width=2),
                movable=True,
                rotatable=False,
                resizable=True,
            )
            self.roi_item.hide()
            self.roi_item.sigRegionChanged.connect(self._on_roi_item_changed)
            self.view_box.addItem(self.roi_item)
            self.stack.addWidget(self.image_view)

        self.info_label = QLabel(
            "pyqtgraph ImageView viewport ready. Histogram/LUT controls match the LFA-style image display."
        )
        self.info_label.setWordWrap(True)
        self.info_label.setAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter)
        self.info_label.setStyleSheet("font-size: 12px; color: palette(mid);")

        self.stack.addWidget(self.placeholder_label)
        self._show_placeholder("No image selected")

        layout.addWidget(self.title_label)
        layout.addWidget(self.stack, 1)
        layout.addWidget(self.info_label)

    @property
    def view_box(self):
        """Return the active pyqtgraph view box when the backend is available."""

        if self.image_view is None:
            return None
        return self.image_view.getView()

    def set_loaded_image(self, loaded_image: Optional[LoadedImage]) -> None:
        """Store the active image and toggle between placeholder and backend canvas."""

        self.current_loaded_image = loaded_image
        if loaded_image is None:
            self._clear_image_view()
            self.current_roi_state = None
            self._show_placeholder("No image selected")
            return

        if not self.backend_available or self.image_view is None or self.image_item is None:
            self._show_placeholder("pyqtgraph backend is not available.")
            return

        image_array = np.asarray(loaded_image.image_data, dtype=np.float32)
        if image_array.ndim != 2:
            raise ValueError(f"Expected 2D image data, got shape {image_array.shape!r}.")

        self.image_item.setImage(image_array.T, autoLevels=True)
        self.reset_view()
        self._update_roi_overlay_from_state()
        self._update_point_overlay()
        self._update_active_point_target()
        self._update_row_geometry_overlay()
        self.stack.setCurrentWidget(self.image_view)
        self.info_label.setText(
            f"{loaded_image.display_name} | "
            f"{loaded_image.pixels_x}x{loaded_image.pixels_y} px | "
            "rendered via pyqtgraph.ImageView using the same image-display flow as LFA."
        )

    def set_roi_state(self, roi_state: Optional[ROIState]) -> None:
        """Update the ROI overlay shown on top of the image."""

        self.current_roi_state = roi_state
        self._update_roi_overlay_from_state()

    def set_atom_rows(
        self,
        atom_rows: Sequence[AtomRow],
        *,
        active_row_id: Optional[str] = None,
        active_image_id: Optional[str] = None,
        active_point_id: Optional[str] = None,
    ) -> None:
        """Render saved-point markers for the current image family."""

        self.current_atom_rows = tuple(atom_rows)
        self.current_active_row_id = active_row_id
        self.current_active_image_id = active_image_id
        self.current_active_point_id = active_point_id
        self._update_point_overlay()
        self._update_active_point_target()

    def set_row_geometry_overlay(
        self,
        row_geometry: Optional[RowGeometry],
        *,
        disturbance_markers: Sequence[dict[str, object]] | None = None,
    ) -> None:
        """Render the fitted active-row axis and optional disturbance markers."""

        self.current_row_geometry = row_geometry
        self.current_disturbance_markers = tuple(
            dict(marker) for marker in (disturbance_markers or ())
        )
        self._update_row_geometry_overlay()

    def reset_view(self) -> None:
        """Reset the visible range to the full image bounds."""

        if self.current_loaded_image is None or self.view_box is None:
            return
        self.view_box.autoRange()

    def _show_placeholder(self, message: str) -> None:
        self.placeholder_label.setText(message)
        self.stack.setCurrentWidget(self.placeholder_label)
        self.info_label.setText(
            "pyqtgraph ImageView viewport ready. Histogram/LUT controls match the LFA-style image display."
        )

    def _update_roi_overlay_from_state(self) -> None:
        if self.roi_item is None:
            return

        image = self.current_loaded_image
        roi = self.current_roi_state
        if image is None or roi is None:
            self.roi_item.hide()
            return

        clamped_roi = roi.clamped(image_width=image.pixels_x, image_height=image.pixels_y)
        if clamped_roi != roi:
            self.current_roi_state = clamped_roi

        self._syncing_roi_overlay = True
        try:
            self.roi_item.setPos((float(clamped_roi.x), float(clamped_roi.y)), update=False)
            self.roi_item.setSize((float(clamped_roi.width), float(clamped_roi.height)), update=False)
            self.roi_item.show()
        finally:
            self._syncing_roi_overlay = False

    def _on_roi_item_changed(self) -> None:
        if self._syncing_roi_overlay or self.roi_item is None or self.current_loaded_image is None:
            return

        image = self.current_loaded_image
        x_value = float(self.roi_item.pos().x())
        y_value = float(self.roi_item.pos().y())
        width_value = float(self.roi_item.size().x())
        height_value = float(self.roi_item.size().y())

        roi_state = ROIState(
            x=int(round(x_value)),
            y=int(round(y_value)),
            width=int(round(width_value)),
            height=int(round(height_value)),
        ).clamped(image_width=image.pixels_x, image_height=image.pixels_y)

        self.current_roi_state = roi_state
        self._update_roi_overlay_from_state()
        self.roi_state_edited.emit(roi_state)

    def _update_point_overlay(self) -> None:
        if self.point_scatter_item is None:
            return

        image = self.current_loaded_image
        if image is None:
            self.point_scatter_item.setData([], [])
            return

        points_payload: list[dict[str, object]] = []
        for row in self.current_atom_rows:
            for point in row.points:
                if point.source_group_id != image.source_group_id:
                    continue

                is_active_row = row.row_id == self.current_active_row_id
                is_active_image = point.image_id == self.current_active_image_id
                is_active_point = point.point_id == self.current_active_point_id

                if is_active_point:
                    brush = pg.mkBrush(110, 255, 140, 250)
                    pen = pg.mkPen(color=(20, 20, 20, 250), width=2.2)
                    size = 15
                elif is_active_row and is_active_image:
                    brush = pg.mkBrush(255, 210, 60, 240)
                    pen = pg.mkPen(color=(30, 30, 30, 240), width=1.8)
                    size = 13
                elif is_active_row:
                    brush = pg.mkBrush(255, 120, 120, 220)
                    pen = pg.mkPen(color=(30, 30, 30, 220), width=1.5)
                    size = 11
                elif is_active_image:
                    brush = pg.mkBrush(120, 220, 255, 210)
                    pen = pg.mkPen(color=(30, 30, 30, 210), width=1.4)
                    size = 10
                else:
                    brush = pg.mkBrush(180, 180, 180, 180)
                    pen = pg.mkPen(color=(30, 30, 30, 180), width=1.2)
                    size = 9

                points_payload.append(
                    {
                        "pos": (float(point.x_px), float(point.y_px)),
                        "data": {
                            "row_id": row.row_id,
                            "point_id": point.point_id,
                            "image_id": point.image_id,
                        },
                        "size": size,
                        "brush": brush,
                        "pen": pen,
                    }
                )

        self.point_scatter_item.setData(points_payload)

    def _update_row_geometry_overlay(self) -> None:
        if self.row_axis_item is None or self.row_disturbance_scatter_item is None:
            return

        if self.current_loaded_image is None or self.current_row_geometry is None:
            self.row_axis_item.setData([], [])
            self.row_axis_item.hide()
            self.row_disturbance_scatter_item.setData([])
            self.row_disturbance_scatter_item.hide()
            return

        geometry = self.current_row_geometry
        half_span = max(float(geometry.span_length_px) * 0.5, 0.5)
        delta_x = float(geometry.direction_x_px) * half_span
        delta_y = float(geometry.direction_y_px) * half_span
        self.row_axis_item.setData(
            [
                float(geometry.reference_x_px) - delta_x,
                float(geometry.reference_x_px) + delta_x,
            ],
            [
                float(geometry.reference_y_px) - delta_y,
                float(geometry.reference_y_px) + delta_y,
            ],
        )
        self.row_axis_item.show()

        disturbance_payload: list[dict[str, object]] = []
        for marker in self.current_disturbance_markers:
            x_value = marker.get("x_px")
            y_value = marker.get("y_px")
            if x_value is None or y_value is None:
                continue
            score = float(marker.get("score", 1.0))
            disturbance_payload.append(
                {
                    "pos": (float(x_value), float(y_value)),
                    "size": 12.0 + min(max(score, 0.0), 4.0) * 2.0,
                    "symbol": "star",
                    "brush": pg.mkBrush(255, 80, 80, 235),
                    "pen": pg.mkPen(color=(40, 20, 20, 245), width=1.4),
                    "data": dict(marker),
                }
            )

        self.row_disturbance_scatter_item.setData(disturbance_payload)
        if disturbance_payload:
            self.row_disturbance_scatter_item.show()
        else:
            self.row_disturbance_scatter_item.hide()

    def _on_point_scatter_clicked(self, _scatter_item, points, _event) -> None:
        if not points:
            return
        payload = points[0].data()
        if not isinstance(payload, dict):
            return
        self.point_selected.emit(dict(payload))

    def _update_active_point_target(self) -> None:
        if self.active_point_target is None:
            return

        active_point = self._find_active_point()
        if active_point is None:
            self.active_point_target.hide()
            return

        self._syncing_active_point_target = True
        try:
            self.active_point_target.setPos((float(active_point.x_px), float(active_point.y_px)))
            self.active_point_target.show()
        finally:
            self._syncing_active_point_target = False

    def _find_active_point(self) -> Optional[AtomPoint]:
        active_point_id = self.current_active_point_id
        if active_point_id is None:
            return None
        for row in self.current_atom_rows:
            for point in row.points:
                if point.point_id == active_point_id:
                    return point
        return None

    def _on_active_point_target_move_finished(self) -> None:
        if self._syncing_active_point_target or self.active_point_target is None:
            return

        active_point = self._find_active_point()
        if active_point is None:
            return

        position = self.active_point_target.pos()
        self.point_move_requested.emit(
            {
                "row_id": active_point.row_id,
                "point_id": active_point.point_id,
                "image_id": active_point.image_id,
                "x_px": float(position.x()),
                "y_px": float(position.y()),
                "source": "drag",
            }
        )

    def image_bounding_rect(self) -> Optional[QRectF]:
        """Return the current image bounding rectangle in image coordinates."""

        if self.image_item is None:
            return None
        rect = self.image_item.boundingRect()
        if rect.isNull():
            return None
        return rect

    def closeEvent(self, event) -> None:  # pragma: no cover - exercised indirectly in GUI teardown
        self._clear_image_view(detach=True)
        super().closeEvent(event)

    def _clear_image_view(self, detach: bool = False) -> None:
        """Clear image/ROI state and optionally detach graphics items during teardown."""

        if self.row_axis_item is not None:
            try:
                self.row_axis_item.setData([], [])
                self.row_axis_item.hide()
            except Exception:
                pass
            if detach:
                try:
                    self.row_axis_item.setParentItem(None)
                except Exception:
                    pass
        if self.row_disturbance_scatter_item is not None:
            try:
                self.row_disturbance_scatter_item.setData([])
                self.row_disturbance_scatter_item.hide()
            except Exception:
                pass
            if detach:
                try:
                    self.row_disturbance_scatter_item.setParentItem(None)
                except Exception:
                    pass
        if self.point_scatter_item is not None:
            try:
                self.point_scatter_item.setData([], [])
            except Exception:
                pass
            if detach:
                try:
                    self.point_scatter_item.setParentItem(None)
                except Exception:
                    pass
        if self.active_point_target is not None:
            try:
                self.active_point_target.hide()
            except Exception:
                pass
            if detach:
                try:
                    self.active_point_target.setParentItem(None)
                except Exception:
                    pass
        if self.roi_item is not None:
            try:
                self.roi_item.hide()
            except Exception:
                pass
            if detach:
                try:
                    self.roi_item.setParentItem(None)
                except Exception:
                    pass
        if self.image_view is not None:
            try:
                self.image_view.clear()
            except Exception:
                pass
            if detach:
                try:
                    self.image_view.close()
                except Exception:
                    pass
