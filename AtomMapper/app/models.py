"""Data models used by the AtomMapper application."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np


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
    metadata: Dict[str, Any] = field(default_factory=dict, repr=False)
    raw_metadata: Dict[str, Any] = field(default_factory=dict, repr=False)

    @property
    def path(self) -> Path:
        """Return the source path as a ``Path`` object."""

        return Path(self.source_path)

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

