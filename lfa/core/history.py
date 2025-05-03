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
    # Store the actual image data for this state (simpler approach first)
    image_data: Optional[np.ndarray] = field(repr=False, default=None)

    def get_display_text(self) -> str:
        """Generates text representation for display in lists."""
        if self.operation_name == "Original":
            return "Original Image"
        # Create a string from parameters, limit length if necessary
        param_str = ", ".join(f"{k}={v}" for k, v in self.parameters.items())
        if len(param_str) > 40:
             param_str = param_str[:37] + "..."
        return f"{self.operation_name} ({param_str})"

    def __post_init__(self):
        # Ensure image_data is None or a NumPy array
        if self.image_data is not None and not isinstance(self.image_data, np.ndarray):
            raise TypeError("image_data must be a NumPy array or None")