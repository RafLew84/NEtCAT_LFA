"""Data models used by the AtomMapper application."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional
from uuid import uuid4

import numpy as np


@dataclass(frozen=True)
class ROIState:
    """Image-space rectangular ROI stored in pixel coordinates."""

    x: int
    y: int
    width: int
    height: int

    def clamped(self, image_width: int, image_height: int, minimum_size: int = 4) -> "ROIState":
        """Return an ROI clamped to image bounds."""

        max_width = max(1, int(image_width))
        max_height = max(1, int(image_height))
        min_size = max(1, int(minimum_size))

        width = min(max_width, max(min_size, int(round(self.width))))
        height = min(max_height, max(min_size, int(round(self.height))))
        x = min(max(0, int(round(self.x))), max(0, max_width - width))
        y = min(max(0, int(round(self.y))), max(0, max_height - height))
        return ROIState(x=x, y=y, width=width, height=height)


@dataclass(frozen=True)
class LoadedImage:
    """Normalized STM image payload used by AtomMapper."""

    source_path: str
    display_name: str
    file_extension: str
    image_data: np.ndarray = field(repr=False)
    pixels_x: int
    pixels_y: int
    size_nm_x: float
    size_nm_y: float
    image_id: str = ""
    source_group_id: str = ""
    parent_image_id: Optional[str] = None
    variant_name: str = "original"
    metadata: Dict[str, Any] = field(default_factory=dict, repr=False)
    raw_metadata: Dict[str, Any] = field(default_factory=dict, repr=False)

    def __post_init__(self) -> None:
        """Normalize variant identity fields for later session/preprocessing work."""

        image_id = str(self.image_id).strip() or uuid4().hex
        variant_name = str(self.variant_name).strip().lower() or "original"
        source_group_id = str(self.source_group_id).strip() or image_id

        object.__setattr__(self, "image_id", image_id)
        object.__setattr__(self, "source_group_id", source_group_id)
        object.__setattr__(self, "variant_name", variant_name)

    @property
    def path(self) -> Path:
        """Return the source path as a ``Path`` object."""

        return Path(self.source_path)

    @property
    def is_original(self) -> bool:
        """Return ``True`` when the image is the source/original variant."""

        return self.parent_image_id is None and self.variant_name == "original"

    @property
    def pixel_size_nm_x(self) -> Optional[float]:
        """Return the physical pixel size along X when available."""

        if self.pixels_x <= 0 or self.size_nm_x <= 0.0:
            return None
        return self.size_nm_x / self.pixels_x

    @property
    def pixel_size_nm_y(self) -> Optional[float]:
        """Return the physical pixel size along Y when available."""

        if self.pixels_y <= 0 or self.size_nm_y <= 0.0:
            return None
        return self.size_nm_y / self.pixels_y

    def derive_variant(
        self,
        *,
        variant_name: str,
        image_data: np.ndarray,
        display_name: Optional[str] = None,
        metadata_updates: Optional[Dict[str, Any]] = None,
        raw_metadata_updates: Optional[Dict[str, Any]] = None,
    ) -> "LoadedImage":
        """Create a derived image variant that stays in the same source group."""

        updated_metadata = dict(self.metadata)
        if metadata_updates:
            updated_metadata.update(metadata_updates)

        updated_raw_metadata = dict(self.raw_metadata)
        if raw_metadata_updates:
            updated_raw_metadata.update(raw_metadata_updates)

        normalized_variant = str(variant_name).strip().lower()
        if not normalized_variant:
            raise ValueError("variant_name must be a non-empty string.")

        variant_display_name = display_name or f"{self.path.stem} [{normalized_variant}]{self.path.suffix}"

        return LoadedImage(
            image_id=uuid4().hex,
            source_group_id=self.source_group_id,
            parent_image_id=self.image_id,
            variant_name=normalized_variant,
            source_path=self.source_path,
            display_name=variant_display_name,
            file_extension=self.file_extension,
            image_data=np.asarray(image_data, dtype=float),
            pixels_x=self.pixels_x,
            pixels_y=self.pixels_y,
            size_nm_x=self.size_nm_x,
            size_nm_y=self.size_nm_y,
            metadata=updated_metadata,
            raw_metadata=updated_raw_metadata,
        )
