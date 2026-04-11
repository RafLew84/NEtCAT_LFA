"""Application state and controller logic for AtomMapper."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Iterable, Optional, Sequence

from PyQt6.QtCore import QObject, pyqtSignal

from .io import load_loaded_image
from .models import LoadedImage, ROIState

logger = logging.getLogger(__name__)


class AtomMapperController(QObject):
    """Owns the loaded-image list and active image selection."""

    loaded_images_changed = pyqtSignal()
    active_image_changed = pyqtSignal(object)
    roi_state_changed = pyqtSignal(object)

    def __init__(self, parent: Optional[QObject] = None) -> None:
        super().__init__(parent)
        self._loaded_images: list[LoadedImage] = []
        self._active_image_index: Optional[int] = None
        self._roi_states_by_source: dict[str, ROIState] = {}

    @property
    def loaded_images(self) -> tuple[LoadedImage, ...]:
        """Return the loaded images as an immutable snapshot."""

        return tuple(self._loaded_images)

    @property
    def active_image_index(self) -> Optional[int]:
        """Return the index of the active image when one is selected."""

        return self._active_image_index

    @property
    def active_image(self) -> Optional[LoadedImage]:
        """Return the currently active image."""

        if self._active_image_index is None:
            return None
        if not (0 <= self._active_image_index < len(self._loaded_images)):
            return None
        return self._loaded_images[self._active_image_index]

    @property
    def active_roi_state(self) -> Optional[ROIState]:
        """Return the ROI assigned to the current active image."""

        active = self.active_image
        if active is None:
            return None
        return self._roi_states_by_source.get(active.source_path)

    def set_loaded_images(self, images: Sequence[LoadedImage]) -> None:
        """Replace the loaded-image collection and reset the active selection."""

        self._loaded_images = list(images)
        self._active_image_index = 0 if self._loaded_images else None
        self._roi_states_by_source = {
            image.source_path: self._build_default_roi_state(image) for image in self._loaded_images
        }
        logger.info("AtomMapperController: loaded image collection replaced. Count=%d", len(images))
        self.loaded_images_changed.emit()
        self.active_image_changed.emit(self.active_image)
        self.roi_state_changed.emit(self.active_roi_state)

    def add_loaded_image(self, image: LoadedImage) -> None:
        """Append an image to the collection."""

        self._loaded_images.append(image)
        self._roi_states_by_source[image.source_path] = self._build_default_roi_state(image)
        if self._active_image_index is None:
            self._active_image_index = 0
        logger.info("AtomMapperController: image appended '%s'.", image.display_name)
        self.loaded_images_changed.emit()
        self.active_image_changed.emit(self.active_image)
        self.roi_state_changed.emit(self.active_roi_state)

    def load_files(self, file_paths: Iterable[str | Path]) -> list[LoadedImage]:
        """Load a batch of STM files and append them to the controller state."""

        loaded: list[LoadedImage] = []
        for file_path in file_paths:
            logger.info("AtomMapperController: loading '%s'.", file_path)
            image = load_loaded_image(file_path)
            self._loaded_images.append(image)
            loaded.append(image)

        if loaded and self._active_image_index is None:
            self._active_image_index = 0

        if loaded:
            for image in loaded:
                self._roi_states_by_source[image.source_path] = self._build_default_roi_state(image)
            logger.info("AtomMapperController: loaded %d file(s).", len(loaded))
            self.loaded_images_changed.emit()
            self.active_image_changed.emit(self.active_image)
            self.roi_state_changed.emit(self.active_roi_state)
        return loaded

    def select_image(self, index: int) -> Optional[LoadedImage]:
        """Set the active image by index."""

        if not (0 <= index < len(self._loaded_images)):
            raise IndexError(f"Image index {index} is out of range.")

        if self._active_image_index == index:
            return self.active_image

        self._active_image_index = index
        active = self.active_image
        if active is not None:
            logger.info("AtomMapperController: active image changed to '%s'.", active.display_name)
        self.active_image_changed.emit(active)
        self.roi_state_changed.emit(self.active_roi_state)
        return active

    def update_active_roi_state(self, roi_state: ROIState) -> ROIState:
        """Persist a new ROI for the currently active image."""

        active = self.active_image
        if active is None:
            raise ValueError("Cannot update ROI without an active image.")

        clamped_roi = roi_state.clamped(active.pixels_x, active.pixels_y)
        self._roi_states_by_source[active.source_path] = clamped_roi
        logger.info(
            "AtomMapperController: ROI updated for '%s' to x=%d y=%d w=%d h=%d.",
            active.display_name,
            clamped_roi.x,
            clamped_roi.y,
            clamped_roi.width,
            clamped_roi.height,
        )
        self.roi_state_changed.emit(clamped_roi)
        return clamped_roi

    @staticmethod
    def _build_default_roi_state(image: LoadedImage) -> ROIState:
        shorter_side = max(1, min(image.pixels_x, image.pixels_y))
        roi_size = min(shorter_side, max(12, int(round(shorter_side * 0.2))))
        x = max(0, (image.pixels_x - roi_size) // 2)
        y = max(0, (image.pixels_y - roi_size) // 2)
        return ROIState(x=x, y=y, width=roi_size, height=roi_size)
