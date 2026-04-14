"""Bridge that connects the pyqtgraph STM viewport with fit-preview widgets."""

from __future__ import annotations

from typing import Optional

from PyQt6.QtCore import QObject, pyqtSignal

from .fit_settings import FitSettingsState
from .fit_models import LocalFitModelType, LocalFitRequest
from .gaussian_fit import fit_local_peak
from .gaussian_preview import GaussianFitPreviewWidget
from .image_utils import extract_roi_patch
from .models import LoadedImage, ROIState
from .polygon_mask import PolygonMaskState, build_polygon_mask_for_roi
from .pyqtgraph_image_view import PyQtGraphSTMViewport
from .roi_preview import ROIPreviewWidget


class PyQtGraphPreviewBridge(QObject):
    """Synchronize the pyqtgraph viewport with optional ROI and fit previews."""

    roi_state_edited = pyqtSignal(object)

    def __init__(
        self,
        viewport: PyQtGraphSTMViewport,
        roi_preview: ROIPreviewWidget | None,
        gaussian_preview: GaussianFitPreviewWidget,
        parent: Optional[QObject] = None,
    ) -> None:
        super().__init__(parent)
        self.viewport = viewport
        self.roi_preview = roi_preview
        self.gaussian_preview = gaussian_preview
        self.current_loaded_image: Optional[LoadedImage] = None
        self.current_roi_state: Optional[ROIState] = None
        self.current_roi_patch_data = None
        self.current_polygon_mask_state: Optional[PolygonMaskState] = None
        self.current_fit_settings_state = FitSettingsState()

        self.viewport.roi_state_edited.connect(self._on_viewport_roi_state_edited)

    def set_loaded_image(self, loaded_image: Optional[LoadedImage]) -> None:
        """Push a new active image through the viewport and both previews."""

        image_changed = (
            loaded_image is None
            or self.current_loaded_image is None
            or loaded_image.image_id != self.current_loaded_image.image_id
        )
        self.current_loaded_image = loaded_image
        if image_changed:
            self.current_polygon_mask_state = None
        self.viewport.set_loaded_image(loaded_image)
        if self.roi_preview is not None:
            self.roi_preview.set_loaded_image(loaded_image)
        self._refresh_gaussian_preview()

    def set_roi_state(self, roi_state: Optional[ROIState]) -> None:
        """Push a new ROI through the viewport and both previews."""

        self.current_roi_state = roi_state
        self.viewport.set_roi_state(roi_state)
        if self.roi_preview is not None:
            self.roi_preview.set_roi_state(roi_state)
        self._refresh_gaussian_preview()

    def set_fit_settings_state(self, fit_settings_state: FitSettingsState) -> None:
        """Update the active fit-settings state used by preview and point capture."""

        self.current_fit_settings_state = fit_settings_state.normalized()
        self._refresh_gaussian_preview()

    def set_polygon_mask_state(self, polygon_mask_state: Optional[PolygonMaskState]) -> None:
        """Update the active polygon fit-mask state used by local fitting."""

        self.current_polygon_mask_state = None if polygon_mask_state is None else polygon_mask_state.normalized()
        self.viewport.set_polygon_mask_state(self.current_polygon_mask_state)
        self._refresh_gaussian_preview()

    def _on_viewport_roi_state_edited(self, roi_state: ROIState) -> None:
        self.current_roi_state = roi_state
        if self.roi_preview is not None:
            self.roi_preview.set_roi_state(roi_state)
        self._refresh_gaussian_preview()
        self.roi_state_edited.emit(roi_state)

    def compute_current_fit_result(self):
        """Compute a fresh Gaussian-fit result for the current image/ROI state."""

        image = self.current_loaded_image
        roi = self.current_roi_state
        if image is None or roi is None:
            return None

        patch = extract_roi_patch(image.image_data, roi)
        self.current_roi_patch_data = patch
        if patch is None:
            return None

        fit_mask = build_polygon_mask_for_roi(roi, self.current_polygon_mask_state)

        return fit_local_peak(
            LocalFitRequest(
                model=self.current_fit_settings_state.model,
                roi_patch=patch,
                roi_origin_yx=(roi.y, roi.x),
                compute_uncertainty=self.current_fit_settings_state.common.compute_uncertainty,
                fit_mask=fit_mask,
                fit_settings_state=self.current_fit_settings_state,
            )
        )

    def _refresh_gaussian_preview(self) -> None:
        if self.current_loaded_image is None or self.current_roi_state is None:
            self.current_roi_patch_data = None
        self.gaussian_preview.set_fit_result(self.compute_current_fit_result())
