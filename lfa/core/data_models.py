# lfa/core/data_models.py
"""
Defines the core data structure for holding STM image data and metadata.
This module provides the STMImage class which encapsulates both the image data
and its associated metadata in a standardized format.
"""

import uuid
from dataclasses import dataclass, field
from typing import Any, Dict, Optional

import numpy as np


@dataclass
class OriginalImageRecord:
    """
    .. :no-index:

    Represents a single original STM image loaded into the session.

    The dataclass fields mirror the metadata retained by the history manager;
    consult the annotations for the precise schema.
    """
    image_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    display_name: str = "Original Image"
    stm_image: Optional["STMImage"] = field(repr=False, default=None)
    source_path: Optional[str] = None
    extra_metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class STMImage:
    """
    .. :no-index:

    Represents STM image data and associated metadata.

    Field descriptions are provided via type hints; refer to the dataclass
    annotations for the full metadata schema, which covers physical dimensions,
    scan parameters, offsets, and instrument settings.
    """
    file_name: str
    raw_header: Dict[str, Any] = field(repr=False) # Store original header for reference
    data: np.ndarray = field(repr=False) # The actual image data (e.g., height, current)

    pixels_x: int = 0
    pixels_y: int = 0
    size_nm_x: float = 0.0
    size_nm_y: float = 0.0
    offset_nm_x: float = 0.0
    offset_nm_y: float = 0.0
    scan_angle_deg: float = 0.0
    bias_v: float = 0.0 
    setpoint_a: float | None = None
    scan_speed_nm_s: float | None = None
    z_nm_per_raw: float | None = None
    image_type: str = "Unknown"

    def __post_init__(self):
        """
        Calculate pixel dimensions from data array shape if not provided.
        
        This method is automatically called after initialization to ensure
        pixel dimensions are properly set based on the data array shape.
        It handles various cases where dimensions might be missing or
        potentially transposed.
        """
        if self.data is not None:
            if self.pixels_y == 0 and self.pixels_x == 0:
                 self.pixels_y, self.pixels_x = self.data.shape
            elif self.pixels_x == 0: 
                 if self.data.shape[0] == self.pixels_y:
                      self.pixels_x = self.data.shape[1]
                 elif self.data.shape[1] == self.pixels_y:
                      self.pixels_x = self.data.shape[0]
            elif self.pixels_y == 0: 
                 if self.data.shape[1] == self.pixels_x:
                      self.pixels_y = self.data.shape[0]
                 elif self.data.shape[0] == self.pixels_x: 
                      self.pixels_y = self.data.shape[1]


    def get_pixel_size_nm(self) -> tuple[float | None, float | None]:
        """
        Calculates the pixel size in nanometers for x and y directions.

        This method computes the physical size of each pixel by dividing
        the total image size by the number of pixels in each direction.
        Returns None for directions where the calculation is not possible
        (e.g., if size or pixel count is zero or negative).

        Returns:
            tuple[float | None, float | None]: Pixel size in nm (x, y), or None if calculation is not possible.
        """
        px_x = self.size_nm_x / self.pixels_x if self.pixels_x > 0 and self.size_nm_x > 0 else None
        px_y = self.size_nm_y / self.pixels_y if self.pixels_y > 0 and self.size_nm_y > 0 else None
        return px_x, px_y
