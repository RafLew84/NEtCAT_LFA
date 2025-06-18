# lfa/core/history.py
"""
History tracking and management for LFA operations.
This module provides functionality for tracking the history of image processing operations,
including FFT transformations and ROI-based operations, with support for parameter tracking
and data type management.
"""
import numpy as np
from dataclasses import dataclass, field
from typing import Optional, Dict, Any, Literal, Tuple, List
import time
import uuid # For generating unique IDs
import logging

logger = logging.getLogger(__name__)

@dataclass
class HistoryNode:
    """
    Represents a single state in the processing history tree.
    
    This class stores information about a single processing operation, including
    the operation type, parameters, timestamp, and the resulting image data.
    It supports both STM and FFT data types, and can track ROI-based operations.

    Attributes:
        node_id (str): Unique identifier for this history node.
        parent_id (Optional[str]): ID of the parent node in the history tree.
        operation_name (str): Name of the operation performed.
        parameters (Dict[str, Any]): Dictionary of operation parameters.
        timestamp (float): Time when the operation was performed.
        image_data (Optional[np.ndarray]): The processed image data.
        data_type (Literal["STM", "FFT"]): Type of data stored in image_data.
        source_roi_slice (Optional[Tuple[slice, slice]]): ROI slice if operation was ROI-based.
    """
    node_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    parent_id: Optional[str] = None
    operation_name: str = "Original"
    parameters: Dict[str, Any] = field(default_factory=dict) # Will store scaling_mode for FFT
    timestamp: float = field(default_factory=time.time)
    # Stores the actual data: float32 for STM, scaled magnitude float32 for FFT
    image_data: Optional[np.ndarray] = field(repr=False, default=None)
    data_type: Literal["STM", "FFT"] = "STM" # Type of data in image_data
    complex_fft_data: Optional[np.ndarray] = field(repr=False, default=None)
    # Stores the source ROI slice if operation resulted from ROI
    source_roi_slice: Optional[Tuple[slice, slice]] = field(repr=False, default=None)

    def get_display_text(self) -> str:
        """
        Generates a human-readable text representation for display in lists.
        
        This method creates a formatted string that includes:
        - Operation name
        - Data type (FFT if applicable)
        - ROI information if applicable
        - Formatted parameter list (truncated if too long)
        
        Returns:
            str: Formatted display text for the history node.
        """
        base_name = self.operation_name
        suffix = ""

        if self.operation_name == "Original":
            return "Original Image"

        if self.data_type == "FFT":
            base_name += " (FFT)"

        # Check if source_roi_slice exists to determine if it came from ROI
        if self.source_roi_slice is not None:
            suffix += " from ROI"

        if not self.parameters:
            param_str = "No parameters"
        else:
            max_param_len = 35 # Maximum length for parameter string
            ellipsis = "..."
            fitting_param_parts = []
            current_len = 0
            params_truncated = False
            # Sort parameters for consistent display (e.g., scaling_mode first)
            param_items = list(self.parameters.items())
            param_items.sort(key=lambda item: (
                item[0] != 'scaling_mode', # scaling_mode first
                item[0] != 'mode',         # then mode (for leveling)
                item[0]                    # then alphabetically
            ))

            for i, (k, v) in enumerate(param_items):
                # Skip internal flags unless necessary
                if k == 'apply_roi_only': continue

                # Format common parameters nicely
                if k == 'scaling_mode': part = f"Scale:{v}"
                elif k == 'window_type' and v is None: part = "NoWin"
                elif k == 'window_type': part = f"Win:{v}"
                elif k == 'mode': part = f"Mode:{v}" # For leveling
                # Shorten points list
                elif k == 'points' and isinstance(v, list) and len(v) == 3:
                    part = f"Pts:[{v[0]}..]"
                # Format floats nicely
                elif isinstance(v, float): part = f"{k}={v:.2g}" # General float format
                else: part = f"{k}={v}" # Default format

                part_len = len(part)
                separator_len = len(", ") if i > 0 else 0
                if current_len + separator_len + part_len > max_param_len:
                    params_truncated = True
                    break # Stop adding parameters

                if i > 0:
                    fitting_param_parts.append(", ")
                    current_len += separator_len
                fitting_param_parts.append(part)
                current_len += part_len

            param_str = "".join(fitting_param_parts)
            if params_truncated:
                param_str += ellipsis

            if not param_str and params_truncated: # Handle case where first param was too long
                 param_str = ellipsis
            elif not param_str and not params_truncated and self.parameters:
                 param_str = "..." # Params exist but didn't fit or were skipped

        if param_str == "No parameters":
             return f"{base_name}{suffix}" # E.g., "FFT from ROI"
        else:
             return f"{base_name}{suffix} ({param_str})" # E.g., "FFT from ROI (Scale:log, Win:hann)"


    def __post_init__(self):
        """
        Validates the initialization parameters after object creation.
        
        This method checks:
        - image_data is a NumPy array or None
        - data_type is either 'STM' or 'FFT'
        - source_roi_slice is properly formatted if provided
        
        Raises:
            TypeError: If image_data or source_roi_slice have invalid types
            ValueError: If data_type is invalid
        """
        # Input validation
        if self.image_data is not None and not isinstance(self.image_data, np.ndarray):
            raise TypeError("image_data must be a NumPy array or None")
        if self.data_type not in ("STM", "FFT"):
            raise ValueError("data_type must be 'STM' or 'FFT'")
        if self.source_roi_slice is not None and not (
            isinstance(self.source_roi_slice, tuple) and
            len(self.source_roi_slice) == 2 and
            isinstance(self.source_roi_slice[0], slice) and
            isinstance(self.source_roi_slice[1], slice)
            ):
             raise TypeError("source_roi_slice must be None or Tuple[slice, slice]")

