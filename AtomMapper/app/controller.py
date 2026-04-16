"""Application state and controller logic for AtomMapper."""

from __future__ import annotations

from dataclasses import replace
import logging
from pathlib import Path
from typing import Iterable, Optional, Sequence

from PyQt6.QtCore import QObject, pyqtSignal

from .io import load_loaded_image
from .models import AtomPoint, AtomRow, LoadedImage, ROIState
from .preprocessing import (
    apply_bm3d,
    apply_blur,
    apply_flip,
    apply_non_local_means,
    apply_rotation,
    build_bm3d_metadata,
    build_blur_metadata,
    build_flip_metadata,
    build_nlm_metadata,
    build_rotation_metadata,
)
from .session_model import AtomMapperSession

logger = logging.getLogger(__name__)


class AtomMapperController(QObject):
    """Owns the loaded-image list and active image selection."""

    loaded_images_changed = pyqtSignal()
    active_image_changed = pyqtSignal(object)
    roi_state_changed = pyqtSignal(object)
    rows_changed = pyqtSignal()
    active_row_changed = pyqtSignal(object)
    row_points_changed = pyqtSignal(object)

    def __init__(self, parent: Optional[QObject] = None) -> None:
        super().__init__(parent)
        self._loaded_images: list[LoadedImage] = []
        self._active_image_index: Optional[int] = None
        self._roi_states_by_image_id: dict[str, ROIState] = {}
        self._rows: list[AtomRow] = []
        self._active_row_id_by_source_group: dict[str, str] = {}

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
    def roi_states_by_image_id(self) -> dict[str, ROIState]:
        """Return a shallow copy of the ROI state mapped by image id."""

        return dict(self._roi_states_by_image_id)

    @property
    def atom_rows(self) -> tuple[AtomRow, ...]:
        """Return the known atom rows as an immutable snapshot."""

        return tuple(self._rows)

    @property
    def active_row(self) -> Optional[AtomRow]:
        """Return the active row for the currently active image family."""

        active_group = self.active_source_group_id
        if active_group is None:
            return None
        row_id = self._active_row_id_by_source_group.get(active_group)
        if row_id is None:
            return None
        return self._find_row_by_id(row_id)

    @property
    def active_row_id(self) -> Optional[str]:
        """Return the identifier of the currently active row."""

        active_row = self.active_row
        if active_row is None:
            return None
        return active_row.row_id

    @property
    def active_row_id_by_source_group(self) -> dict[str, str]:
        """Return the active-row mapping for all loaded source groups."""

        return dict(self._active_row_id_by_source_group)

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

    def rows_for_source_group(self, source_group_id: str) -> tuple[AtomRow, ...]:
        """Return all rows bound to the same source family."""

        return tuple(row for row in self._rows if row.source_group_id == source_group_id)

    def set_loaded_images(self, images: Sequence[LoadedImage]) -> None:
        """Replace the loaded-image collection and reset the active selection."""

        self._ensure_unique_image_ids(images)
        self._loaded_images = list(images)
        self._active_image_index = 0 if self._loaded_images else None
        self._roi_states_by_image_id = {
            image.image_id: self._build_default_roi_state(image) for image in self._loaded_images
        }
        rows_pruned = self._prune_rows_to_loaded_source_groups()
        logger.info("AtomMapperController: loaded image collection replaced. Count=%d", len(images))
        self.loaded_images_changed.emit()
        if rows_pruned:
            self.rows_changed.emit()
        self.active_image_changed.emit(self.active_image)
        self.roi_state_changed.emit(self.active_roi_state)
        self.active_row_changed.emit(self.active_row)

    def restore_from_session(self, session: AtomMapperSession) -> None:
        """Replace the runtime state with a validated session snapshot."""

        images = list(session.loaded_images)
        rows = list(session.rows)

        self._ensure_unique_image_ids(images)
        self._ensure_unique_row_ids(rows)
        self._ensure_unique_point_ids(rows)

        images_by_id = {image.image_id: image for image in images}
        allowed_groups = {image.source_group_id for image in images}
        for row in rows:
            if row.source_group_id not in allowed_groups:
                raise ValueError(
                    f"Row source_group_id '{row.source_group_id}' is not present in loaded images."
                )
            for point in row.points:
                image = images_by_id.get(point.image_id)
                if image is None:
                    raise ValueError(
                        f"Point image_id '{point.image_id}' is not present in loaded images."
                    )
                if image.source_group_id != row.source_group_id:
                    raise ValueError(
                        "Point image source_group_id must match the row/source family."
                    )
                if point.source_group_id != row.source_group_id:
                    raise ValueError(
                        "Point source_group_id must match the row/source family."
                    )

        self._loaded_images = images
        self._active_image_index = None
        if session.active_image_id is not None:
            for index, image in enumerate(self._loaded_images):
                if image.image_id == session.active_image_id:
                    self._active_image_index = index
                    break

        self._roi_states_by_image_id = {
            image.image_id: session.roi_states_by_image_id.get(
                image.image_id,
                self._build_default_roi_state(image),
            ).clamped(image.pixels_x, image.pixels_y)
            for image in self._loaded_images
        }
        self._rows = rows
        self._active_row_id_by_source_group = dict(session.active_row_id_by_source_group)

        logger.info(
            "AtomMapperController: restored session with %d image(s), %d row(s).",
            len(self._loaded_images),
            len(self._rows),
        )
        self.loaded_images_changed.emit()
        self.rows_changed.emit()
        self.active_image_changed.emit(self.active_image)
        self.roi_state_changed.emit(self.active_roi_state)
        self.active_row_changed.emit(self.active_row)

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
        self.active_row_changed.emit(self.active_row)

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

    def create_rotate_variant_for_active_image(
        self,
        *,
        quarter_turns: int = 1,
        make_active: bool = True,
    ) -> LoadedImage:
        """Create, append, and optionally activate a 90-degree rotation variant."""

        active = self.active_image
        if active is None:
            raise ValueError("Cannot create rotation variant without an active image.")

        normalized_turns = int(quarter_turns) % 4
        if normalized_turns == 0:
            raise ValueError("quarter_turns must not be a multiple of 4.")

        rotated_data = apply_rotation(active.image_data, quarter_turns=normalized_turns)
        swap_geometry = normalized_turns % 2 == 1
        variant = active.derive_variant(
            variant_name=f"rotate-{normalized_turns * 90}",
            image_data=rotated_data,
            metadata_updates=build_rotation_metadata(quarter_turns=normalized_turns),
            pixels_x=active.pixels_y if swap_geometry else active.pixels_x,
            pixels_y=active.pixels_x if swap_geometry else active.pixels_y,
            size_nm_x=active.size_nm_y if swap_geometry else active.size_nm_x,
            size_nm_y=active.size_nm_x if swap_geometry else active.size_nm_y,
        )
        self.add_loaded_variant(variant, make_active=make_active)
        return variant

    def create_flip_variant_for_active_image(
        self,
        *,
        flip_x: bool = True,
        flip_y: bool = False,
        make_active: bool = True,
    ) -> LoadedImage:
        """Create, append, and optionally activate a flipped image variant."""

        active = self.active_image
        if active is None:
            raise ValueError("Cannot create flip variant without an active image.")

        flipped_data = apply_flip(active.image_data, flip_x=flip_x, flip_y=flip_y)
        variant_suffix = "xy" if flip_x and flip_y else ("x" if flip_x else "y")
        variant = active.derive_variant(
            variant_name=f"flip-{variant_suffix}",
            image_data=flipped_data,
            metadata_updates=build_flip_metadata(flip_x=flip_x, flip_y=flip_y),
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
        self.active_row_changed.emit(self.active_row)
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

    def set_atom_rows(self, rows: Sequence[AtomRow]) -> None:
        """Replace the atom-row collection after validating identities and loaded families."""

        allowed_groups = {image.source_group_id for image in self._loaded_images}
        self._ensure_unique_row_ids(rows)
        self._ensure_unique_point_ids(rows)
        for row in rows:
            if row.source_group_id not in allowed_groups:
                raise ValueError(
                    f"Row source_group_id '{row.source_group_id}' is not present in loaded images."
                )

        self._rows = list(rows)
        self._active_row_id_by_source_group = {}
        for row in self._rows:
            self._active_row_id_by_source_group.setdefault(row.source_group_id, row.row_id)

        self.rows_changed.emit()
        self.active_row_changed.emit(self.active_row)

    def add_row(self, row: AtomRow, *, make_active: bool = True) -> AtomRow:
        """Append a new row to the controller state."""

        if row.source_group_id not in {image.source_group_id for image in self._loaded_images}:
            raise ValueError(
                f"Row source_group_id '{row.source_group_id}' is not present in loaded images."
            )
        self._ensure_row_id_not_present(row.row_id)
        self._ensure_point_ids_not_present(row.points)

        self._rows.append(row)
        if row.source_group_id not in self._active_row_id_by_source_group or make_active:
            self._active_row_id_by_source_group[row.source_group_id] = row.row_id

        self.rows_changed.emit()
        self.active_row_changed.emit(self.active_row)
        return row

    def create_row_for_active_source_group(
        self,
        *,
        display_name: str = "",
        color_hex: Optional[str] = None,
        make_active: bool = True,
    ) -> AtomRow:
        """Create and register a new row for the active image family."""

        active_group = self.active_source_group_id
        if active_group is None:
            raise ValueError("Cannot create a row without an active image/source group.")

        row = AtomRow(
            source_group_id=active_group,
            display_name=display_name,
            color_hex=color_hex,
        )
        return self.add_row(row, make_active=make_active)

    def select_row(self, row_id: str) -> Optional[AtomRow]:
        """Mark a row as active for its source-group family."""

        row = self._find_row_by_id(row_id)
        if row is None:
            raise ValueError(f"Unknown row_id '{row_id}'.")

        self._active_row_id_by_source_group[row.source_group_id] = row.row_id
        if self.active_source_group_id == row.source_group_id:
            self.active_row_changed.emit(row)
        return row

    def remove_row(self, row_id: str) -> Optional[AtomRow]:
        """Remove a row from the controller state and repair active-row mapping."""

        row = self._find_row_by_id(row_id)
        if row is None:
            return None

        self._rows = [existing for existing in self._rows if existing.row_id != row.row_id]
        if self._active_row_id_by_source_group.get(row.source_group_id) == row.row_id:
            replacement = next(
                (existing.row_id for existing in self._rows if existing.source_group_id == row.source_group_id),
                None,
            )
            if replacement is None:
                self._active_row_id_by_source_group.pop(row.source_group_id, None)
            else:
                self._active_row_id_by_source_group[row.source_group_id] = replacement

        self.rows_changed.emit()
        self.active_row_changed.emit(self.active_row)
        return row

    def add_point_to_row(self, point: AtomPoint, *, insert_index: int | None = None) -> AtomRow:
        """Store a point inside the selected row after validating image-family consistency."""

        row = self._find_row_by_id(point.row_id)
        if row is None:
            raise ValueError(f"Unknown row_id '{point.row_id}'.")

        image = self._find_image_by_id(point.image_id)
        if image is None:
            raise ValueError(f"Unknown image_id '{point.image_id}'.")
        if image.source_group_id != row.source_group_id:
            raise ValueError("Point image_id belongs to a different source_group_id than the row.")
        if point.source_group_id != row.source_group_id:
            raise ValueError("Point source_group_id must match the row/source family.")
        self._ensure_point_id_not_present(point.point_id, ignore_row_id=row.row_id)

        normalized_point = self._point_with_physical_coordinates(point, image=image)
        updated_row = row.with_inserted_point(normalized_point, insert_index=insert_index)
        self._replace_row(updated_row)
        self.row_points_changed.emit(updated_row)
        if self.active_source_group_id == updated_row.source_group_id:
            self._active_row_id_by_source_group[updated_row.source_group_id] = updated_row.row_id
            self.active_row_changed.emit(updated_row)
        return updated_row

    def replace_point_in_row(self, point: AtomPoint) -> AtomRow:
        """Replace an existing point inside a row after validating identities."""

        row = self._find_row_by_id(point.row_id)
        if row is None:
            raise ValueError(f"Unknown row_id '{point.row_id}'.")

        existing_point = next((item for item in row.points if item.point_id == point.point_id), None)
        if existing_point is None:
            raise ValueError(
                f"Point id '{point.point_id}' is not present in row '{point.row_id}'."
            )

        image = self._find_image_by_id(point.image_id)
        if image is None:
            raise ValueError(f"Unknown image_id '{point.image_id}'.")
        if image.source_group_id != row.source_group_id:
            raise ValueError("Point image_id belongs to a different source_group_id than the row.")
        if point.source_group_id != row.source_group_id:
            raise ValueError("Point source_group_id must match the row/source family.")
        if existing_point.point_id != point.point_id:
            raise ValueError("Point replacement must preserve point_id.")

        updated_row = row.with_point(point)
        self._replace_row(updated_row)
        self.row_points_changed.emit(updated_row)
        if self.active_source_group_id == updated_row.source_group_id:
            self._active_row_id_by_source_group[updated_row.source_group_id] = updated_row.row_id
            self.active_row_changed.emit(updated_row)
        return updated_row

    def remove_point_from_row(self, row_id: str, point_id: str) -> AtomRow:
        """Remove a single point from a row while preserving the row itself."""

        row = self._find_row_by_id(row_id)
        if row is None:
            raise ValueError(f"Unknown row_id '{row_id}'.")

        existing_point = next(
            (point for point in row.points if point.point_id == str(point_id).strip()),
            None,
        )
        if existing_point is None:
            raise ValueError(f"Point id '{point_id}' is not present in row '{row_id}'.")

        updated_row = row.without_point(existing_point.point_id)
        self._replace_row(updated_row)
        self.row_points_changed.emit(updated_row)
        if self.active_source_group_id == updated_row.source_group_id:
            self._active_row_id_by_source_group[updated_row.source_group_id] = updated_row.row_id
            self.active_row_changed.emit(updated_row)
        return updated_row

    def move_point_in_row(
        self,
        *,
        row_id: str,
        point_id: str,
        x_px: float,
        y_px: float,
        x_nm: Optional[float] = None,
        y_nm: Optional[float] = None,
        source: str = "manual",
    ) -> AtomRow:
        """Apply a manual coordinate correction to a point already stored in a row."""

        row = self._find_row_by_id(row_id)
        if row is None:
            raise ValueError(f"Unknown row_id '{row_id}'.")

        point = next((item for item in row.points if item.point_id == str(point_id).strip()), None)
        if point is None:
            raise ValueError(f"Point id '{point_id}' is not present in row '{row_id}'.")

        image = self._find_image_by_id(point.image_id)
        if image is None:
            raise ValueError(f"Unknown image_id '{point.image_id}'.")

        if x_nm is None or y_nm is None:
            calibration = image.physical_calibration
            if calibration is not None:
                calibrated_x_nm, calibrated_y_nm = calibration.point_px_to_nm(x_px, y_px)
                if x_nm is None:
                    x_nm = calibrated_x_nm
                if y_nm is None:
                    y_nm = calibrated_y_nm

        corrected_point = point.with_manual_position(
            x_px=x_px,
            y_px=y_px,
            x_nm=x_nm,
            y_nm=y_nm,
            source=source,
        )
        return self.replace_point_in_row(corrected_point)

    def reorder_point_in_row(
        self,
        *,
        row_id: str,
        point_id: str,
        target_index: int,
    ) -> AtomRow:
        """Move a stored point to a new position inside its row and reindex the row."""

        row = self._find_row_by_id(row_id)
        if row is None:
            raise ValueError(f"Unknown row_id '{row_id}'.")

        updated_row = row.with_reordered_point(point_id, target_index=target_index)
        self._replace_row(updated_row)
        self.row_points_changed.emit(updated_row)
        if self.active_source_group_id == updated_row.source_group_id:
            self._active_row_id_by_source_group[updated_row.source_group_id] = updated_row.row_id
            self.active_row_changed.emit(updated_row)
        return updated_row

    @staticmethod
    def _point_with_physical_coordinates(point: AtomPoint, *, image: LoadedImage) -> AtomPoint:
        """Return a point with `x_nm`/`y_nm` populated from image calibration when possible."""

        calibration = image.physical_calibration
        if calibration is None:
            return point

        needs_update = point.x_nm is None or point.y_nm is None
        if not needs_update:
            return point

        x_nm, y_nm = calibration.point_px_to_nm(point.x_px, point.y_px)
        return replace(point, x_nm=x_nm, y_nm=y_nm)

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

    def _find_row_by_id(self, row_id: str) -> Optional[AtomRow]:
        for row in self._rows:
            if row.row_id == row_id:
                return row
        return None

    def _replace_row(self, updated_row: AtomRow) -> None:
        for index, row in enumerate(self._rows):
            if row.row_id == updated_row.row_id:
                self._rows[index] = updated_row
                return
        raise ValueError(f"Unknown row_id '{updated_row.row_id}'.")

    def _prune_rows_to_loaded_source_groups(self) -> bool:
        allowed_groups = {image.source_group_id for image in self._loaded_images}
        before_count = len(self._rows)
        self._rows = [row for row in self._rows if row.source_group_id in allowed_groups]
        self._active_row_id_by_source_group = {
            group_id: row_id
            for group_id, row_id in self._active_row_id_by_source_group.items()
            if group_id in allowed_groups and self._find_row_by_id(row_id) is not None
        }
        return len(self._rows) != before_count

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

    def _ensure_row_id_not_present(self, row_id: str) -> None:
        if self._find_row_by_id(row_id) is not None:
            raise ValueError(f"Row id '{row_id}' is already present.")

    @staticmethod
    def _ensure_unique_row_ids(rows: Sequence[AtomRow]) -> None:
        seen: set[str] = set()
        for row in rows:
            if row.row_id in seen:
                raise ValueError(f"Duplicate row id '{row.row_id}' in row collection.")
            seen.add(row.row_id)

    @staticmethod
    def _ensure_unique_point_ids(rows: Sequence[AtomRow]) -> None:
        seen: set[str] = set()
        for row in rows:
            for point in row.points:
                if point.point_id in seen:
                    raise ValueError(f"Duplicate point id '{point.point_id}' in row collection.")
                seen.add(point.point_id)

    def _ensure_point_ids_not_present(
        self,
        points: Sequence[AtomPoint],
        *,
        ignore_row_id: Optional[str] = None,
    ) -> None:
        existing_ids = {
            point.point_id
            for row in self._rows
            if row.row_id != ignore_row_id
            for point in row.points
        }
        for point in points:
            if point.point_id in existing_ids:
                raise ValueError(f"Point id '{point.point_id}' is already present.")

    def _ensure_point_id_not_present(self, point_id: str, *, ignore_row_id: Optional[str] = None) -> None:
        self._ensure_point_ids_not_present(
            [AtomPoint(
                point_id=point_id,
                row_id="__check__",
                image_id="__check__",
                source_group_id="__check__",
                point_index=0,
                x_px=0.0,
                y_px=0.0,
            )],
            ignore_row_id=ignore_row_id,
        )
