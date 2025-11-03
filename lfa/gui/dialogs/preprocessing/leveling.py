"""Plane leveling preprocessing dialog."""

import logging
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from .base import (
    BasePreprocessingDialog,
    ImageItem,
    QCheckBox,
    QDialogButtonBox,
    QFrame,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QPushButton,
    QRadioButton,
    Qt,
    QVBoxLayout,
    QWidget,
    RectROI,
    pg,
    pyqtSlot,
)

try:
    from lfa.preprocessing.leveling import fit_plane, fit_plane_3pts
except ImportError:  # pragma: no cover
    logging.error("Could not import leveling helpers (fit_plane, fit_plane_3pts).")

    def fit_plane(*args, **kwargs):
        return None

    def fit_plane_3pts(*args, **kwargs):
        return None

logger = logging.getLogger(__name__)


class PlaneLevelingDialog(BasePreprocessingDialog):
    """Dialog for plane leveling with whole, ROI, and 3-point modes."""

    def __init__(self, original_data: np.ndarray, parent=None):
        if original_data is None:
            raise ValueError("Original data cannot be None")
        super().__init__("Plane Leveling", parent)
        self.original_data = original_data.astype(np.float32)
        self.preview_data = self.original_data.copy()

        self._is_selecting_points = False
        self._selected_points: List[Tuple[int, int]] = []
        self._mouse_click_connection = None

        self.setWindowTitle(f"{self.operation_name} Settings")
        self.setMinimumSize(900, 550)
        current_flags = self.windowFlags()
        self.setWindowFlags(
            current_flags
            | Qt.WindowType.WindowMinimizeButtonHint
            | Qt.WindowType.WindowMaximizeButtonHint
        )

        main_layout = QVBoxLayout(self)
        top_layout = QHBoxLayout()
        controls_area_layout = QVBoxLayout()
        bottom_layout = QHBoxLayout()

        pg.setConfigOption("background", "w")
        pg.setConfigOption("foreground", "k")
        self.win = pg.GraphicsLayoutWidget()
        self.plot_original = self.win.addPlot(row=0, col=0, title="Original", name="plot_orig")
        self.img_original = ImageItem()
        self.plot_original.addItem(self.img_original)
        self.plot_original.hideAxis("left")
        self.plot_original.hideAxis("bottom")
        self.plot_original.setAspectLocked(True)

        self.plot_processed = self.win.addPlot(row=0, col=1, title="Preview", name="plot_proc")
        self.img_processed = ImageItem()
        self.plot_processed.addItem(self.img_processed)
        self.plot_processed.hideAxis("left")
        self.plot_processed.hideAxis("bottom")
        self.plot_processed.setAspectLocked(True)
        self.plot_processed.vb.setXLink(self.plot_original.vb)
        self.plot_processed.vb.setYLink(self.plot_original.vb)
        self.plot_original.vb.invertY(True)
        self.plot_processed.vb.invertY(True)
        top_layout.addWidget(self.win, stretch=3)

        controls_panel = QWidget()
        controls_panel.setMaximumWidth(260)
        controls_panel.setLayout(controls_area_layout)

        parameter_widget_container = QWidget()
        parameter_layout = QVBoxLayout(parameter_widget_container)
        parameter_layout.setContentsMargins(0, 0, 0, 0)
        self._create_parameter_controls(parameter_layout)
        controls_area_layout.addWidget(parameter_widget_container)

        controls_area_layout.addWidget(
            QFrame(frameShape=QFrame.Shape.HLine, frameShadow=QFrame.Shadow.Sunken)
        )

        self.apply_to_roi_only_checkbox = QCheckBox("Apply plane only to ROI area")
        self.apply_to_roi_only_checkbox.setChecked(False)
        controls_area_layout.addWidget(self.apply_to_roi_only_checkbox)

        self.live_preview_checkbox = QCheckBox("Live Preview")
        self.live_preview_checkbox.setChecked(True)
        controls_area_layout.addWidget(self.live_preview_checkbox)

        controls_area_layout.addWidget(
            QFrame(frameShape=QFrame.Shape.HLine, frameShadow=QFrame.Shadow.Sunken)
        )

        self.roi_info_label = QLabel("ROI: Not selected")
        controls_area_layout.addWidget(self.roi_info_label)

        height, width = self.original_data.shape
        roi_w, roi_h = width // 4, height // 4
        roi_x = width // 2 - roi_w // 2
        roi_y = height // 2 - roi_h // 2
        self.roi = RectROI(
            pos=(roi_x, roi_y),
            size=(roi_w, roi_h),
            pen=pg.mkPen("g", width=2),
            translateSnap=True,
            scaleSnap=True,
        )
        self.plot_original.addItem(self.roi)

        controls_area_layout.addStretch()
        top_layout.addWidget(controls_panel, stretch=1)

        self.button_box = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        self.button_box.button(QDialogButtonBox.StandardButton.Ok).setText("Apply Changes")
        bottom_layout.addWidget(self.button_box)

        main_layout.addLayout(top_layout)
        main_layout.addLayout(bottom_layout)

        self._manage_roi_with_checkbox = False
        self._initialize_common_behavior()
        self._update_mode_ui()

    def _create_parameter_controls(self, layout: QVBoxLayout) -> None:
        mode_groupbox = QGroupBox("Leveling Mode")
        mode_layout = QVBoxLayout(mode_groupbox)

        self.rb_whole = QRadioButton("Whole image")
        self.rb_roi = QRadioButton("Fit plane to ROI")
        self.rb_3pt = QRadioButton("Fit plane through 3 points")
        self.rb_whole.setChecked(True)

        for rb in (self.rb_whole, self.rb_roi, self.rb_3pt):
            mode_layout.addWidget(rb)
            rb.toggled.connect(self._on_mode_changed)

        layout.addWidget(mode_groupbox)

        self.points_groupbox = QGroupBox("3-point selection")
        points_layout = QVBoxLayout(self.points_groupbox)
        self.point_coords_labels: List[QLabel] = []
        for idx in range(3):
            label = QLabel(f"P{idx + 1}: -")
            points_layout.addWidget(label)
            self.point_coords_labels.append(label)

        buttons_row = QHBoxLayout()
        self.select_points_button = QPushButton("Select")
        self.select_points_button.clicked.connect(self._toggle_point_selection)
        buttons_row.addWidget(self.select_points_button)

        self.clear_points_button = QPushButton("Clear")
        self.clear_points_button.setEnabled(False)
        self.clear_points_button.clicked.connect(self._clear_points)
        buttons_row.addWidget(self.clear_points_button)

        points_layout.addLayout(buttons_row)
        layout.addWidget(self.points_groupbox)

    def _update_mode_ui(self) -> None:
        is_roi_mode = self.rb_roi.isChecked()
        is_3pt_mode = self.rb_3pt.isChecked()

        self.roi.setVisible(is_roi_mode)
        self.roi_info_label.setVisible(is_roi_mode)
        self.apply_to_roi_only_checkbox.setVisible(is_roi_mode)
        if not is_roi_mode:
            self.apply_to_roi_only_checkbox.setChecked(False)

        self.points_groupbox.setVisible(is_3pt_mode)
        if not is_3pt_mode:
            self._toggle_point_selection(force_off=True)
            self._clear_points(trigger_preview=False)

    @pyqtSlot()
    def _on_mode_changed(self) -> None:
        self._update_mode_ui()
        if self.live_preview_checkbox.isChecked():
            self._update_preview()

    def _update_preview_slot(self) -> None:
        if self.live_preview_checkbox.isChecked():
            self._update_preview()

    def _toggle_point_selection(self, force_off: bool = False) -> None:
        enable = not force_off and not self._is_selecting_points
        if enable:
            self._is_selecting_points = True
            self.select_points_button.setText("Stop selecting")
            vb = self.plot_original.getViewBox()
            scene = vb.scene() if vb is not None else None
            if scene is not None and hasattr(scene, "sigMouseClicked"):
                self._mouse_click_connection = scene.sigMouseClicked.connect(self._handle_mouse_click)
        else:
            if self._mouse_click_connection is not None:
                try:
                    vb = self.plot_original.getViewBox()
                    scene = vb.scene() if vb is not None else None
                    if scene is not None and hasattr(scene, "sigMouseClicked"):
                        scene.sigMouseClicked.disconnect(self._mouse_click_connection)
                except Exception as exc:  # pragma: no cover
                    logger.warning("PlaneLeveling: failed to disconnect point selection: %s", exc)
                self._mouse_click_connection = None
            self._is_selecting_points = False
            self.select_points_button.setText("Select")

    def _clear_points(self, trigger_preview: bool = True) -> None:
        self._selected_points.clear()
        for idx, label in enumerate(self.point_coords_labels, start=1):
            label.setText(f"P{idx}: -")
        self.clear_points_button.setEnabled(False)
        if trigger_preview and self.live_preview_checkbox.isChecked():
            self._update_preview()

    def _handle_mouse_click(self, event) -> None:
        if not self._is_selecting_points or len(self._selected_points) >= 3:
            return
        vb = self.plot_original.getViewBox()
        if vb is None:
            return
        pos_data = vb.mapSceneToView(event.scenePos())
        x = int(round(pos_data.x()))
        y = int(round(pos_data.y()))
        height, width = self.original_data.shape
        if 0 <= x < width and 0 <= y < height:
            self._selected_points.append((x, y))
            idx = len(self._selected_points)
            self.point_coords_labels[idx - 1].setText(f"P{idx}: ({x}, {y})")
            self.clear_points_button.setEnabled(True)
            logger.info("PlaneLeveling: point %s added at (%s, %s)", idx, x, y)
            if len(self._selected_points) == 3:
                self._toggle_point_selection(force_off=True)
                if self.live_preview_checkbox.isChecked():
                    self._update_preview()
        else:
            logger.warning("PlaneLeveling: click (%s, %s) outside bounds", x, y)

    def _get_current_parameters(self) -> Dict[str, Any]:
        mode = "whole"
        if self.rb_roi.isChecked():
            mode = "roi"
        elif self.rb_3pt.isChecked():
            mode = "3pt"

        params: Dict[str, Any] = {"mode": mode, "apply_roi_only": False}
        if mode == "roi":
            params["apply_roi_only"] = self.apply_to_roi_only_checkbox.isChecked()
        elif mode == "3pt":
            params["points"] = self._selected_points.copy()
        return params

    def _calculate_leveled_image(self, image_in: np.ndarray, params: Dict[str, Any]) -> Optional[np.ndarray]:
        mode = params.get("mode", "whole")
        apply_roi_only = params.get("apply_roi_only", False)
        roi_slice = None

        try:
            if mode == "whole":
                fitted_plane = fit_plane(image_in, roi_slice=None)
            elif mode == "roi":
                roi_slice = self._get_roi_slice()
                if roi_slice is None:
                    logger.warning("PlaneLeveling: ROI mode selected but ROI invalid.")
                    return image_in
                fitted_plane = fit_plane(image_in, roi_slice=roi_slice)
            elif mode == "3pt":
                points = params.get("points", [])
                if len(points) != 3:
                    logger.warning("PlaneLeveling: 3pt mode without 3 points.")
                    return image_in
                fitted_plane = fit_plane_3pts(image_in, points)
            else:
                logger.error("PlaneLeveling: unknown mode %s", mode)
                return image_in
        except Exception as exc:  # pragma: no cover
            logger.exception("PlaneLeveling: plane fitting failed", exc_info=exc)
            return None

        if fitted_plane is None:
            return None

        if mode == "roi" and apply_roi_only and roi_slice is not None:
            leveled = image_in.copy()
            leveled[roi_slice] = image_in[roi_slice] - fitted_plane[roi_slice]
            return leveled

        return image_in - fitted_plane

    def _apply_operation(self, image: np.ndarray, params: Dict[str, Any]) -> Optional[np.ndarray]:
        return self._calculate_leveled_image(image, params)

    def accept(self) -> None:
        params = self._get_current_parameters()
        if params.get("mode") == "3pt" and len(params.get("points", [])) != 3:
            QMessageBox.warning(self, "Missing Points", "Please select exactly 3 points.")
            return
        result = self._calculate_leveled_image(self.original_data, params)
        if result is None:
            QMessageBox.critical(self, "Error", "Plane leveling failed. See logs for details.")
            self._final_processed_data = None
            self._final_is_roi_applied_only = False
            super().reject()
            return
        if np.allclose(result, self.original_data):
            logger.info("PlaneLeveling: data unchanged; rejecting dialog.")
            self._final_processed_data = None
            super().reject()
            return
        self._final_processed_data = result
        self._final_params = params
        self._final_is_roi_applied_only = params.get("apply_roi_only", False)
        super().accept()

    def reject(self) -> None:
        self._toggle_point_selection(force_off=True)
        super().reject()

    def _on_roi_changed(self) -> None:
        super()._on_roi_changed()
        if self.rb_roi.isChecked() and self.live_preview_checkbox.isChecked():
            self._update_preview()


__all__ = ["PlaneLevelingDialog"]
