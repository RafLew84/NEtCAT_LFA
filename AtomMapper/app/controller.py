"""Application state and controller logic for AtomMapper."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Iterable, Optional, Sequence

from PyQt6.QtCore import QObject, pyqtSignal

from .io import load_loaded_image
from .models import LoadedImage, ROIState
from .preprocessing import (
    apply_bm3d,
    apply_blur,
    apply_non_local_means,
    build_bm3d_metadata,
    build_blur_metadata,
    build_nlm_metadata,
)

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
        self._roi_states_by_image_id: dict[str, ROIState] = {}

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
    def active_source_group_id(self) -> Optional[str]:
        """Return the source-group identifier of the active image."""

        active = self.active_image
        if active is None:
            return None
        return active.source_group_id

    @property
    def active_roi_state(self) -> Optional[ROIState]:
        """Return the ROI assigned to the current active image."""

        active = self.active_image
        if active is None:
            return None
        return self._roi_states_by_image_id.get(active.image_id)

    @property
    def original_images(self) -> tuple[LoadedImage, ...]:
        """Return only source/original images."""

        return tuple(image for image in self._loaded_images if image.is_original)

    def images_for_source_group(self, source_group_id: str) -> tuple[LoadedImage, ...]:
        """Return all images belonging to the same source family."""

        return tuple(
            image for image in self._loaded_images if image.source_group_id == source_group_id
        )

    def variant_images_for_source_group(self, source_group_id: str) -> tuple[LoadedImage, ...]:
        """Return non-original variants for a given source family."""

        return tuple(
            image
            for image in self._loaded_images
            if image.source_group_id == source_group_id and not image.is_original
        )

    def set_loaded_images(self, images: Sequence[LoadedImage]) -> None:
        """Replace the loaded-image collection and reset the active selection."""

        self._ensure_unique_image_ids(images)
        self._loaded_images = list(images)
        self._active_image_index = 0 if self._loaded_images else None
        self._roi_states_by_image_id = {
            image.image_id: self._build_default_roi_state(image) for image in self._loaded_images
        }
        logger.info("AtomMapperController: loaded image collection replaced. Count=%d", len(images))
        self.loaded_images_changed.emit()
        self.active_image_changed.emit(self.active_image)
        self.roi_state_changed.emit(self.active_roi_state)

    def add_loaded_image(self, image: LoadedImage) -> None:
        """Append an image to the collection."""

        self._ensure_image_id_not_present(image.image_id)
        self._loaded_images.append(image)
        self._roi_states_by_image_id[image.image_id] = self._build_default_roi_state(image)
        if self._active_image_index is None:
            self._active_image_index = 0
        logger.info("AtomMapperController: image appended '%s'.", image.display_name)
        self.loaded_images_changed.emit()
        self.active_image_changed.emit(self.active_image)
        self.roi_state_changed.emit(self.active_roi_state)

    def add_loaded_variant(self, image: LoadedImage, *, make_active: bool = True) -> None:
        """Append a derived image variant linked to an existing parent/original image."""

        if image.parent_image_id is None:
            raise ValueError("Derived image must define parent_image_id.")

        parent = self._find_image_by_id(image.parent_image_id)
        if parent is None:
            raise ValueError(f"Derived image parent '{image.parent_image_id}' is not loaded.")

        if image.source_group_id != parent.source_group_id:
            raise ValueError("Derived image must share source_group_id with its parent image.")

        self.add_loaded_image(image)
        if make_active:
            self.select_image(len(self._loaded_images) - 1)

    def create_blur_variant_for_active_image(
        self,
        *,
        sigma_px: float = 1.0,
        make_active: bool = True,
    ) -> LoadedImage:
        """Create, append, and optionally activate a blurred variant of the active image."""

        active = self.active_image
        if active is None:
            raise ValueError("Cannot create blur variant without an active image.")

        blurred_data = apply_blur(active.image_data, sigma_px=sigma_px)
        variant = active.derive_variant(
            variant_name="blur",
            image_data=blurred_data,
            metadata_updates=build_blur_metadata(sigma_px=sigma_px),
        )
        self.add_loaded_variant(variant, make_active=make_active)
        return variant

    def create_nlm_variant_for_active_image(
        self,
        *,
        h: float = 0.1,
        patch_size: int = 5,
        patch_distance: int = 6,
        fast_mode: bool = True,
        make_active: bool = True,
    ) -> LoadedImage:
        """Create, append, and optionally activate an NLM-denoised variant."""

        active = self.active_image
        if active is None:
            raise ValueError("Cannot create NLM variant without an active image.")

        denoised_data = apply_non_local_means(
            active.image_data,
            h=h,
            patch_size=patch_size,
            patch_distance=patch_distance,
            fast_mode=fast_mode,
        )
        variant = active.derive_variant(
            variant_name="nlm",
            image_data=denoised_data,
            metadata_updates=build_nlm_metadata(
                h=h,
                patch_size=patch_size,
                patch_distance=patch_distance,
                fast_mode=fast_mode,
            ),
        )
        self.add_loaded_variant(variant, make_active=make_active)
        return variant

    def create_bm3d_variant_for_active_image(
        self,
        *,
        sigma_psd: float = 0.1,
        stage: str = "all_stages",
        make_active: bool = True,
    ) -> LoadedImage:
        """Create, append, and optionally activate a BM3D-denoised variant."""

        active = self.active_image
        if active is None:
            raise ValueError("Cannot create BM3D variant without an active image.")

        denoised_data = apply_bm3d(active.image_data, sigma_psd=sigma_psd)
        variant = active.derive_variant(
            variant_name="bm3d",
            image_data=denoised_data,
            metadata_updates=build_bm3d_metadata(
                sigma_psd=sigma_psd,
                stage=stage,
            ),
        )
        self.add_loaded_variant(variant, make_active=make_active)
        return variant

    def load_files(self, file_paths: Iterable[str | Path]) -> list[LoadedImage]:
        """Load a batch of STM files and append them to the controller state."""

        loaded: list[LoadedImage] = []
        for file_path in file_paths:
            logger.info("AtomMapperController: loading '%s'.", file_path)
            image = load_loaded_image(file_path)
            self._ensure_image_id_not_present(image.image_id)
            self._loaded_images.append(image)
            loaded.append(image)

        if loaded and self._active_image_index is None:
            self._active_image_index = 0

        if loaded:
            for image in loaded:
                self._roi_states_by_image_id[image.image_id] = self._build_default_roi_state(image)
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
        self._roi_states_by_image_id[active.image_id] = clamped_roi
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

    def _find_image_by_id(self, image_id: str) -> Optional[LoadedImage]:
        for image in self._loaded_images:
            if image.image_id == image_id:
                return image
        return None

    def _ensure_image_id_not_present(self, image_id: str) -> None:
        if self._find_image_by_id(image_id) is not None:
            raise ValueError(f"Image id '{image_id}' is already loaded.")

    @staticmethod
    def _ensure_unique_image_ids(images: Sequence[LoadedImage]) -> None:
        seen: set[str] = set()
        for image in images:
            if image.image_id in seen:
                raise ValueError(f"Duplicate image id '{image.image_id}' in image collection.")
            seen.add(image.image_id)
