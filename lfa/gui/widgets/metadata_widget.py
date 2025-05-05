# lfa/gui/widgets/metadata_widget.py
"""
QWidget for displaying metadata associated with a HistoryNode.
"""
import logging
import os
from typing import Optional, Dict, Any

try:
    from PyQt6.QtWidgets import QWidget, QFormLayout, QLabel
    from PyQt6.QtCore import Qt
except ImportError:
    logging.critical("Failed to import necessary PyQt6 modules for MetadataWidget.")
    # Define dummy classes if Qt is not available (for headless testing?)
    class QWidget: pass
    class QFormLayout: pass
    class QLabel: pass
    Qt = None

# Use relative import to get HistoryNode
try:
    from ...core.history import HistoryNode # Relative import: .. goes up to lfa/, then core.history
except ImportError:
    logging.error("Could not import HistoryNode in metadata_widget. Metadata display might fail.")
    HistoryNode = None # Define as None if import fails


logger = logging.getLogger(__name__)

class MetadataWidget(QWidget):
    """Widget to display metadata from a HistoryNode."""
    def __init__(self, parent=None):
        """Initializes the metadata display widget."""
        super().__init__(parent)
        self.layout = QFormLayout(self)
        self.layout.setContentsMargins(10, 10, 10, 10)
        self.layout.setSpacing(5)
        # Set alignment for labels to be on top for word wrapping
        self.layout.setLabelAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignTop)
        self.layout.setFormAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignTop)


        # --- Labels for displaying metadata ---
        # Original File Info Section
        self.filename_label = QLabel("-")
        self.orig_dims_px_label = QLabel("-")
        self.orig_dims_nm_label = QLabel("-")
        self.orig_bias_label = QLabel("-")
        self.orig_setpoint_label = QLabel("-")
        self.orig_angle_label = QLabel("-")

        # Current Node Info Section
        self.node_op_label = QLabel("-")
        self.node_type_label = QLabel("-")
        self.node_params_label = QLabel("-")
        self.node_params_label.setWordWrap(True) # Allow wrapping
        self.node_roi_label = QLabel("-")
        self.node_shape_label = QLabel("-")

        # --- Add rows to layout ---
        # Use QLabel directly for section headers for easier styling if needed
        self.layout.addRow(QLabel("<b>Original File:</b>"))
        self.layout.addRow("Filename:", self.filename_label)
        self.layout.addRow("Dimensions (px):", self.orig_dims_px_label)
        self.layout.addRow("Size (nm):", self.orig_dims_nm_label)
        self.layout.addRow("Bias (V):", self.orig_bias_label)
        self.layout.addRow("Setpoint (A):", self.orig_setpoint_label)
        self.layout.addRow("Scan Angle (°):", self.orig_angle_label)

        # Add spacing or a line between sections
        self.layout.addRow(QWidget()) # Empty row for spacing

        self.layout.addRow(QLabel("<b>Current State:</b>"))
        self.layout.addRow("Operation:", self.node_op_label)
        self.layout.addRow("Data Type:", self.node_type_label)
        self.layout.addRow("Shape:", self.node_shape_label)
        self.layout.addRow("Source ROI:", self.node_roi_label)
        self.layout.addRow("Parameters:", self.node_params_label)

    def clear_labels(self):
        """Clears all metadata labels."""
        self.filename_label.setText("-")
        self.orig_dims_px_label.setText("-")
        self.orig_dims_nm_label.setText("-")
        self.orig_bias_label.setText("-")
        self.orig_setpoint_label.setText("-")
        self.orig_angle_label.setText("-")
        self.node_op_label.setText("-")
        self.node_type_label.setText("-")
        self.node_params_label.setText("-")
        self.node_roi_label.setText("-")
        self.node_shape_label.setText("-")

    def update_metadata(self, node: Optional['HistoryNode'], history: Dict[str, 'HistoryNode']):
        """Updates the labels with information from the given node and history."""
        if node is None or history is None or HistoryNode is None:
            self.clear_labels()
            return

        # --- Display Current Node Info ---
        self.node_op_label.setText(f"<i>{node.operation_name}</i>")
        self.node_type_label.setText(node.data_type)
        shape_str = str(node.image_data.shape) if node.image_data is not None else "N/A"
        self.node_shape_label.setText(shape_str)

        # Format ROI slice
        if node.source_roi_slice:
            rs, cs = node.source_roi_slice
            # Ensure start/stop are not None before accessing attributes
            r_start = rs.start if rs.start is not None else '0'
            r_stop = rs.stop if rs.stop is not None else 'end'
            c_start = cs.start if cs.start is not None else '0'
            c_stop = cs.stop if cs.stop is not None else 'end'
            roi_str = f"Rows [{r_start}:{r_stop}], Cols [{c_start}:{c_stop}]"
        else:
            roi_str = "N/A (Whole Image)"
        self.node_roi_label.setText(roi_str)

        # Format parameters
        # Filter out internal flags unless needed, format nicely
        params_to_show = {k: v for k, v in node.parameters.items() if k not in ['apply_roi_only']}
        param_str = ", ".join(f"{k}={v:.3g}" if isinstance(v, float) else f"{k}={v}"
                             for k, v in params_to_show.items())
        if not param_str:
             param_str = "-"
        self.node_params_label.setText(param_str)


        # --- Find Root Node and Display Original Metadata ---
        root_node = node
        visited = {node.node_id}
        # Iterate max 100 steps back to prevent infinite loop on error
        for _ in range(100):
            if not root_node.parent_id or root_node.parent_id not in history or root_node.parent_id in visited:
                break # Reached root or cycle or missing parent
            visited.add(root_node.parent_id)
            root_node = history[root_node.parent_id]

        if root_node.operation_name == "Original":
            orig_params = root_node.parameters # Metadata stored here
            self.filename_label.setText(orig_params.get("filename", "-"))
            px_x = orig_params.get("pixels_x", 0); px_y = orig_params.get("pixels_y", 0)
            self.orig_dims_px_label.setText(f"{px_x} x {px_y}" if px_x and px_y else "-")
            nm_x = orig_params.get("size_nm_x", 0.0); nm_y = orig_params.get("size_nm_y", 0.0)
            self.orig_dims_nm_label.setText(f"{nm_x:.2f} x {nm_y:.2f}" if nm_x and nm_y else "-")
            bias = orig_params.get("bias_v", None)
            self.orig_bias_label.setText(f"{bias:.4f}" if bias is not None else "-")
            setpoint = orig_params.get("setpoint_a", None)
            self.orig_setpoint_label.setText(f"{setpoint:.3e}" if setpoint is not None else "-")
            angle = orig_params.get("scan_angle_deg", None)
            self.orig_angle_label.setText(f"{angle:.1f}" if angle is not None else "-")
        else:
            # Could not find root node? Clear original info
            logger.warning(f"Could not trace back to root node from node {node.node_id}")
            self.filename_label.setText("?")
            self.orig_dims_px_label.setText("?")
            self.orig_dims_nm_label.setText("?")
            self.orig_bias_label.setText("?")
            self.orig_setpoint_label.setText("?")
            self.orig_angle_label.setText("?")