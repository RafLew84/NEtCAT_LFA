# lfa/gui/widgets/metadata_widget.py
"""
QWidget for displaying metadata associated with a HistoryNode.
"""
import logging
from typing import Dict, Optional

try:
    from PyQt6.QtCore import Qt
    from PyQt6.QtWidgets import QFormLayout, QLabel, QWidget
except ImportError:
    logging.critical("Failed to import necessary PyQt6 modules for MetadataWidget.")
    class QWidget: pass
    class QFormLayout: pass
    class QLabel: pass
    Qt = None

try:
    from ...core.history import HistoryNode
except ImportError:
    logging.error("Could not import HistoryNode in metadata_widget. Metadata display might fail.")
    HistoryNode = None


logger = logging.getLogger(__name__)

class MetadataWidget(QWidget):
    """Widget to display metadata from a HistoryNode."""
    def __init__(self, parent=None):
        """Initializes the metadata display widget."""
        super().__init__(parent)
        self.layout = QFormLayout(self)
        self.layout.setContentsMargins(10, 10, 10, 10)
        self.layout.setSpacing(5)
        self.layout.setLabelAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignTop)
        self.layout.setFormAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignTop)

        self.filename_label = QLabel("-")
        self.orig_label_label = QLabel("-")
        self.orig_dims_px_label = QLabel("-")
        self.orig_dims_nm_label = QLabel("-")
        self.orig_bias_label = QLabel("-")
        self.orig_setpoint_label = QLabel("-")
        self.orig_angle_label = QLabel("-")

        self.node_op_label = QLabel("-")
        self.node_type_label = QLabel("-")
        self.node_params_label = QLabel("-")
        self.node_params_label.setWordWrap(True)
        self.node_roi_label = QLabel("-")
        self.node_shape_label = QLabel("-")
        self.node_source_label = QLabel("-")

        self.layout.addRow(QLabel("<b>Original File:</b>"))
        self.layout.addRow("Filename:", self.filename_label)
        self.layout.addRow("Image Label:", self.orig_label_label)
        self.layout.addRow("Dimensions (px):", self.orig_dims_px_label)
        self.layout.addRow("Size (nm):", self.orig_dims_nm_label)
        self.layout.addRow("Bias (V):", self.orig_bias_label)
        self.layout.addRow("Setpoint (A):", self.orig_setpoint_label)
        self.layout.addRow("Scan Angle (°):", self.orig_angle_label)

        self.layout.addRow(QWidget())

        self.layout.addRow(QLabel("<b>Current State:</b>"))
        self.layout.addRow("Operation:", self.node_op_label)
        self.layout.addRow("Data Type:", self.node_type_label)
        self.layout.addRow("Shape:", self.node_shape_label)
        self.layout.addRow("Source Image:", self.node_source_label)
        self.layout.addRow("Source ROI:", self.node_roi_label)
        self.layout.addRow("Parameters:", self.node_params_label)

    def clear_labels(self):
        """Clears all metadata labels."""
        self.filename_label.setText("-")
        self.orig_label_label.setText("-")
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
        self.node_source_label.setText("-")

    def update_metadata(self, node: Optional['HistoryNode'], history: Dict[str, 'HistoryNode']):
        """Updates the labels with information from the given node and history."""
        history_dict: Optional[Dict[str, HistoryNode]] = None
        root_node: Optional[HistoryNode] = None
        history_mgr = None

        if isinstance(history, dict):
            history_dict = history
            if node:
                temp_node = node
                visited = {node.node_id}
                for _ in range(100):
                    if not temp_node.parent_id or temp_node.parent_id not in history_dict or temp_node.parent_id in visited:
                        root_node = temp_node
                        break
                    visited.add(temp_node.parent_id)
                    temp_node = history_dict[temp_node.parent_id]
                if not root_node: root_node = temp_node # Fallback
        else:
            history_mgr = history
            if node and history_mgr:
                 root_node = history_mgr.get_root_node_for_node(node.node_id)
                 history_dict = history_mgr.history

        if node is None or history_dict is None:
            self.clear_labels()
            return

        self.node_op_label.setText(f"<i>{node.operation_name}</i>")
        self.node_type_label.setText(node.data_type)
        shape_str = str(node.image_data.shape) if node.image_data is not None else "N/A"
        self.node_shape_label.setText(shape_str)

        if node.source_roi_slice:
            rs, cs = node.source_roi_slice
            r_start = rs.start if rs.start is not None else '0'
            r_stop = rs.stop if rs.stop is not None else 'end'
            c_start = cs.start if cs.start is not None else '0'
            c_stop = cs.stop if cs.stop is not None else 'end'
            roi_str = f"Rows [{r_start}:{r_stop}], Cols [{c_start}:{c_stop}]"
        else:
            roi_str = "N/A (Whole Image)"
        self.node_roi_label.setText(roi_str)

        source_label = node.parameters.get("source_image_label")
        if not source_label and node.original_image_id and history_mgr:
            record = history_mgr.get_original_image_record(node.original_image_id)
            if record:
                source_label = record.display_name
        if not source_label and root_node and root_node.operation_name == "Original":
            source_label = root_node.parameters.get("original_label")
        self.node_source_label.setText(source_label or "-")

        params_to_show = {
            k: v for k, v in node.parameters.items()
            if k not in ['apply_roi_only', 'source_image_id', 'source_image_label', 'original_label']
        }
        param_str = ", ".join(f"{k}={v:.3g}" if isinstance(v, float) else f"{k}={v}"
                             for k, v in params_to_show.items())
        if not param_str:
             param_str = "-"
        self.node_params_label.setText(param_str)


        if root_node and root_node.operation_name == "Original":
            orig_params = root_node.parameters 
            self.filename_label.setText(orig_params.get("filename", "-"))
            self.orig_label_label.setText(orig_params.get("original_label", "-"))
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
            logger.warning(f"Could not trace back to a valid 'Original' root node from node {node.node_id if node else 'None'}. Root found: {root_node.operation_name if root_node else 'None'}")
            self.filename_label.setText("?")
            self.orig_label_label.setText("?")
            self.orig_dims_px_label.setText("?")
            self.orig_dims_nm_label.setText("?")
            self.orig_bias_label.setText("?")
            self.orig_setpoint_label.setText("?")
            self.orig_angle_label.setText("?")
            self.node_source_label.setText("?")
