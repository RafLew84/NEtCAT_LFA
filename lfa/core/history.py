# lfa/core/history.py
"""
Data structures for managing processing history.
"""
import numpy as np
from dataclasses import dataclass, field
from typing import Optional, Dict, Any
import time
import uuid # For generating unique IDs

@dataclass
class HistoryNode:
    """Represents a single state in the processing history tree."""
    node_id: str = field(default_factory=lambda: str(uuid.uuid4())) # Unique ID
    parent_id: Optional[str] = None # ID of the parent node, None for root
    operation_name: str = "Original" # e.g., "Original", "Gaussian Blur"
    parameters: Dict[str, Any] = field(default_factory=dict) # Parameters used
    timestamp: float = field(default_factory=time.time)
    is_roi_applied: bool = False
    # Store the actual image data for this state (simpler approach first)
    image_data: Optional[np.ndarray] = field(repr=False, default=None)

    def get_display_text(self) -> str:
        """
        Generates text representation for display in lists.
        Truncates parameter list intelligently if too long.
        """
        if self.operation_name == "Original":
            return "Original Image"
        
        base_name = self.operation_name
        
        if self.is_roi_applied:
            base_name += " (ROI)"

        if not self.parameters:
            return f"{self.operation_name} (No parameters)"

        max_param_len = 40 # Max length for the parameter part inside parentheses
        ellipsis = "..."
        # Store only the parameter strings (e.g., "sigma=1.0")
        fitting_param_parts = []
        current_len = 0
        params_truncated = False

        param_items = list(self.parameters.items())
        for i, (k, v) in enumerate(param_items):
            part = f"{k}={v}"
            part_len = len(part)
            # Length includes comma and space if not the first item
            check_len = current_len + (len(", ") if i > 0 else 0) + part_len

            # Check if adding this part would exceed the limit
            if check_len > max_param_len:
                params_truncated = True
                # If even the *first* item is too long, we need special handling
                if i == 0 and part_len > max_param_len:
                     # Truncate the first item itself if it's too long
                     fitting_param_parts.append(part[:max(0, max_param_len - len(ellipsis))] + ellipsis)
                     current_len = max_param_len # Mark as full
                break # Stop adding parameters

            # Add the part and update current length
            fitting_param_parts.append(part)
            current_len = check_len # Update length based on check_len

        # Join the parts that fit with ", "
        param_str = ", ".join(fitting_param_parts)

        # Append ellipsis *after joining* if truncation occurred AND we didn't already add it
        # (the case where the first item itself was truncated and ellipsis added)
        if params_truncated and not param_str.endswith(ellipsis):
            # We need to ensure adding ellipsis doesn't exceed max length,
            # recalculate the length accurately or rely on the check within the loop.
            # The loop ensures that param_str length <= max_param_len.
            # If we add ellipsis, it might exceed. Let's re-check.
            if len(param_str) + len(ellipsis) <= max_param_len + len(", "): # Allow slight overflow for ellipsis
                 param_str += ellipsis
            else:
                 # If adding ellipsis makes it too long, truncate param_str first
                 param_str = param_str[:max(0, max_param_len - len(ellipsis))] + ellipsis


        # Final check for empty parameters after potential truncation
        if not param_str and params_truncated:
             param_str = ellipsis # Should happen only if max_param_len is tiny

        return f"{self.operation_name} ({param_str})"

    def __post_init__(self):
        # Ensure image_data is None or a NumPy array
        if self.image_data is not None and not isinstance(self.image_data, np.ndarray):
            raise TypeError("image_data must be a NumPy array or None")