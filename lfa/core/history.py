# lfa/core/history.py
"""
History tracking and management for LFA operations.
This module provides functionality for tracking the history of image processing operations,
including FFT transformations and ROI-based operations, with support for parameter tracking
and data type management.
"""
import logging
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Dict, Literal, Optional, Tuple

import numpy as np

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
        original_image_id (Optional[str]): Identifier linking the node to its originating image.
    """
    node_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    parent_id: Optional[str] = None
    operation_name: str = "Original"
    parameters: Dict[str, Any] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)
    image_data: Optional[np.ndarray] = field(repr=False, default=None)
    data_type: Literal["STM", "FFT"] = "STM"
    complex_fft_data: Optional[np.ndarray] = field(repr=False, default=None)
    source_roi_slice: Optional[Tuple[slice, slice]] = field(repr=False, default=None)
    original_image_id: Optional[str] = None

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
            return self.parameters.get("original_label", "Original Image")

        source_label = self.parameters.get("source_image_label") or self.parameters.get("original_label")
        if source_label:
            base_name += f" [{source_label}]"

        if self.data_type == "FFT":
            base_name += " (FFT)"

        if self.source_roi_slice is not None:
            suffix += " from ROI"

        if not self.parameters:
            param_str = "No parameters"
        else:
            max_param_len = 35
            ellipsis = "..."
            fitting_param_parts = []
            current_len = 0
            params_truncated = False
            param_items = list(self.parameters.items())
            param_items.sort(key=lambda item: (
                item[0] != 'scaling_mode',
                item[0] != 'mode',        
                item[0]                   
            ))

            for i, (k, v) in enumerate(param_items):
                if k in ('apply_roi_only', 'source_image_id', 'source_image_label', 'original_label'):
                    continue

                if k == 'scaling_mode': part = f"Scale:{v}"
                elif k == 'window_type' and v is None: part = "NoWin"
                elif k == 'window_type': part = f"Win:{v}"
                elif k == 'mode': part = f"Mode:{v}"
                elif k == 'points' and isinstance(v, list) and len(v) == 3:
                    part = f"Pts:[{v[0]}..]"
                elif isinstance(v, float): part = f"{k}={v:.2g}"
                else: part = f"{k}={v}"

                part_len = len(part)
                separator_len = len(", ") if i > 0 else 0
                if current_len + separator_len + part_len > max_param_len:
                    params_truncated = True
                    break

                if i > 0:
                    fitting_param_parts.append(", ")
                    current_len += separator_len
                fitting_param_parts.append(part)
                current_len += part_len

            param_str = "".join(fitting_param_parts)
            if params_truncated:
                param_str += ellipsis

            if not param_str and params_truncated:
                 param_str = ellipsis
            elif not param_str and not params_truncated and self.parameters:
                 param_str = "..."

        if param_str == "No parameters":
             return f"{base_name}{suffix}"
        else:
             return f"{base_name}{suffix} ({param_str})"


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

