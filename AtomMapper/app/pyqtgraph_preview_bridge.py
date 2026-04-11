"""Bridge that connects the pyqtgraph STM viewport with existing preview widgets."""

from __future__ import annotations

from typing import Optional

from PyQt6.QtCore import QObject, pyqtSignal

from .gaussian_fit import fit_gaussian_to_roi_patch
from .gaussian_preview import GaussianFitPreviewWidget
from .image_utils import extract_roi_patch
from .models import LoadedImage, ROIState
from .pyqtgraph_image_view import PyQtGraphSTMViewport
from .roi_preview import ROIPreviewWidget


class PyQtGraphPreviewBridge(QObject):
    """Synchronize the pyqtgraph viewport with ROI and Gaussian-fit previews."""

    roi_state_edited = pyqtSignal(object)

    def __init__(
        self,
        viewport: PyQtGraphSTMViewport,
        roi_preview: ROIPreviewWidget,
        gaussian_preview: GaussianFitPreviewWidget,
        parent: Optional[QObject] = None,
    ) -> None:
        super().__init__(parent)
        self.viewport = viewport
        self.roi_preview = roi_preview
        self.gaussian_preview = gaussian_preview
        self.current_loaded_image: Optional[LoadedImage] = None
        self.current_roi_state: Optional[ROIState] = None

        self.viewport.roi_state_edited.connect(self._on_viewport_roi_state_edited)

    def set_loaded_image(self, loaded_image: Optional[LoadedImage]) -> None:
        """Push a new active image through the viewport and both previews."""

        self.current_loaded_image = loaded_image
        self.viewport.set_loaded_image(loaded_image)
        self.roi_preview.set_loaded_image(loaded_image)
        self._refresh_gaussian_preview()

    def set_roi_state(self, roi_state: Optional[ROIState]) -> None:
        """Push a new ROI through the viewport and both previews."""

        self.current_roi_state = roi_state
        self.viewport.set_roi_state(roi_state)
        self.roi_preview.set_roi_state(roi_state)
        self._refresh_gaussian_preview()

    def _on_viewport_roi_state_edited(self, roi_state: ROIState) -> None:
        self.current_roi_state = roi_state
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
        if patch is None:
            return None

        return fit_gaussian_to_roi_patch(
            patch,
            roi_origin_yx=(roi.y, roi.x),
            compute_uncertainty=False,
        )

    def _refresh_gaussian_preview(self) -> None:
        self.gaussian_preview.set_fit_result(self.compute_current_fit_result())
