# lfa/core/data_models.py
"""
Defines the core data structure for holding STM image data and metadata.
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Dict, Any

@dataclass
class STMImage:
    """
    Represents STM image data and associated metadata.

    Attributes:
        file_name (str): Path to the original data file.
        raw_header (Dict): Dictionary holding the raw header information as read from the file.
                           Excluded from standard repr.
        data (np.ndarray): 2D NumPy array containing the image data (e.g., height, current).
                           Excluded from standard repr.
        pixels_x (int): Number of pixels along the x-axis.
        pixels_y (int): Number of pixels along the y-axis.
        size_nm_x (float): Physical size of the image along the x-axis in nanometers.
        size_nm_y (float): Physical size of the image along the y-axis in nanometers.
        offset_nm_x (float): Physical offset of the image center along the x-axis in nanometers.
        offset_nm_y (float): Physical offset of the image center along the y-axis in nanometers.
        scan_angle_deg (float): Scan angle in degrees relative to the physical x/y axes.
        bias_v (float): Bias voltage applied during scanning, in Volts.
        setpoint_a (float | None): Tunneling current setpoint in Amperes (if available).
        scan_speed_nm_s (float | None): Scan speed in the fast direction in nm/s (if available).
        z_nm_per_raw (float | None): Conversion factor for raw z-axis data to nm (if applicable, e.g., for S94 topography).
        image_type (str): Type of the image data (e.g., "Topography", "Current", "Unknown").
    """
    file_name: str
    raw_header: Dict[str, Any] = field(repr=False) # Store original header for reference
    data: np.ndarray = field(repr=False) # The actual image data (e.g., height, current)

    # Standardized metadata fields
    pixels_x: int = 0
    pixels_y: int = 0
    size_nm_x: float = 0.0
    size_nm_y: float = 0.0
    offset_nm_x: float = 0.0
    offset_nm_y: float = 0.0
    scan_angle_deg: float = 0.0
    bias_v: float = 0.0 # Bias Voltage in Volts
    setpoint_a: float | None = None # Setpoint current in Amperes (if available)
    scan_speed_nm_s: float | None = None # Scan speed in nm/s (if available)
    z_nm_per_raw: float | None = None # Conversion factor for z-axis (if applicable)
    image_type: str = "Unknown" # e.g., "Topography", "Current"

    def __post_init__(self):
        """Calculate pixel dimensions from data array shape if not provided."""
        if self.data is not None:
            if self.pixels_y == 0 and self.pixels_x == 0:
                 # Assume shape is (rows, columns) -> (y, x)
                 self.pixels_y, self.pixels_x = self.data.shape
            elif self.pixels_x == 0: # Only y is set
                 # This case is less common, might indicate swapped dimensions earlier
                 if self.data.shape[0] == self.pixels_y:
                      self.pixels_x = self.data.shape[1]
                 elif self.data.shape[1] == self.pixels_y: # Check if shape seems transposed
                      self.pixels_x = self.data.shape[0]
            elif self.pixels_y == 0: # Only x is set
                 if self.data.shape[1] == self.pixels_x:
                      self.pixels_y = self.data.shape[0]
                 elif self.data.shape[0] == self.pixels_x: # Check if shape seems transposed
                      self.pixels_y = self.data.shape[1]


    def get_pixel_size_nm(self) -> tuple[float | None, float | None]:
        """
        Calculates the pixel size in nanometers for x and y directions.

        Returns:
            tuple[float | None, float | None]: Pixel size in nm (x, y), or None if calculation is not possible.
        """
        px_x = self.size_nm_x / self.pixels_x if self.pixels_x > 0 and self.size_nm_x > 0 else None
        px_y = self.size_nm_y / self.pixels_y if self.pixels_y > 0 and self.size_nm_y > 0 else None
        return px_x, px_y