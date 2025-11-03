# tests/gui/widgets/test_metadata_widget.py
"""
Unit tests for the MetadataWidget using pytest-qt.
"""

from typing import Dict, Tuple

import numpy as np
import pytest

# Import the widget and the data structure it uses
try:
    from lfa.gui.widgets.metadata_widget import MetadataWidget
except ImportError:
    pytest.fail("Could not import MetadataWidget from lfa.gui.widgets.metadata_widget", pytrace=False)

try:
    from lfa.core.history import HistoryNode
except ImportError:
    pytest.fail("Could not import HistoryNode from lfa.core.history", pytrace=False)

# Requires PyQt6 for widget instantiation
pytest.importorskip("PyQt6", reason="PyQt6 not found, skipping GUI widget tests")
# Requires pytest-qt for qtbot fixture
pytest.importorskip("pytestqt", reason="pytest-qt not found, skipping GUI widget tests")

# --- Fixtures ---

@pytest.fixture
def sample_history() -> Tuple[Dict[str, HistoryNode], str, str, str]:
    """Creates a sample history tree for testing."""
    # Original STM node
    root_id = "root-001"
    root_label = "Original Image 1"
    root_params = {
        "filename": "test_image.stp", "pixels_x": 128, "pixels_y": 128,
        "size_nm_x": 50.5, "size_nm_y": 51.2, "bias_v": 0.75,
        "setpoint_a": 1.5e-10, "scan_angle_deg": 15.0,
        "original_label": root_label,
        "source_image_label": root_label,
    }
    root_node = HistoryNode(
        node_id=root_id,
        operation_name="Original",
        parameters=root_params,
        image_data=np.zeros((128, 128), dtype=np.float32), # Dummy data
        data_type="STM"
    )

    # Gaussian Blur node
    blur_id = "blur-002"
    blur_params = {"sigma": 1.5, 'apply_roi_only': False, "source_image_label": root_label}
    blur_node = HistoryNode(
        node_id=blur_id,
        parent_id=root_id,
        operation_name="Gaussian Blur",
        parameters=blur_params,
        image_data=np.zeros((128, 128), dtype=np.float32), # Dummy data
        data_type="STM" # Result of blur on STM is STM
    )

    # FFT from ROI node
    fft_id = "fft-003"
    fft_params = {'apply_window': True, 'window_type': 'hann', 'scaling_mode': 'log', "source_image_label": root_label}
    fft_roi_slice = (slice(32, 96), slice(32, 96))
    fft_node = HistoryNode(
        node_id=fft_id,
        parent_id=blur_id, # Parent is the blurred image
        operation_name="FFT",
        parameters=fft_params,
        image_data=np.zeros((64, 64), dtype=np.float32), # FFT result (scaled magnitude)
        data_type="FFT",
        source_roi_slice=fft_roi_slice
    )

    history = {
        root_id: root_node,
        blur_id: blur_node,
        fft_id: fft_node
    }
    return history, root_id, blur_id, fft_id

# --- Test Functions ---

def test_metadata_widget_init(qtbot):
    """Test if the MetadataWidget initializes correctly."""
    widget = MetadataWidget()
    qtbot.addWidget(widget) # Register widget with qtbot for cleanup
    assert widget is not None
    # Check if labels exist (sanity check)
    assert hasattr(widget, 'filename_label')
    assert hasattr(widget, 'orig_label_label')
    assert hasattr(widget, 'node_source_label')
    assert hasattr(widget, 'node_op_label')
    assert widget.filename_label.text() == "-" # Check default text
    assert widget.orig_label_label.text() == "-"
    assert widget.node_source_label.text() == "-"

def test_metadata_widget_clear(qtbot, sample_history):
    """Test if update_metadata with None node clears labels."""
    history, root_id, _, _ = sample_history
    widget = MetadataWidget()
    qtbot.addWidget(widget)

    # Set some initial text (optional, but verifies clearing works)
    widget.filename_label.setText("Some Text")
    widget.node_op_label.setText("Some Op")

    # Call update with None node
    widget.update_metadata(None, history)

    # Assert labels are cleared
    assert widget.filename_label.text() == "-"
    assert widget.orig_dims_px_label.text() == "-"
    assert widget.orig_dims_nm_label.text() == "-"
    assert widget.orig_bias_label.text() == "-"
    assert widget.orig_setpoint_label.text() == "-"
    assert widget.orig_angle_label.text() == "-"
    assert widget.orig_label_label.text() == "-"
    assert widget.node_op_label.text() == "-"
    assert widget.node_type_label.text() == "-"
    assert widget.node_shape_label.text() == "-"
    assert widget.node_roi_label.text() == "-"
    assert widget.node_params_label.text() == "-"
    assert widget.node_source_label.text() == "-"

def test_metadata_widget_root_node(qtbot, sample_history):
    """Test displaying metadata for the root (Original Image) node."""
    history, root_id, _, _ = sample_history
    root_node = history[root_id]
    widget = MetadataWidget()
    qtbot.addWidget(widget)

    widget.update_metadata(root_node, history)

    # Check Original File Info (should come from root_node.parameters)
    assert widget.filename_label.text() == "test_image.stp"
    assert widget.orig_dims_px_label.text() == "128 x 128"
    assert widget.orig_dims_nm_label.text() == "50.50 x 51.20"
    assert widget.orig_bias_label.text() == "0.7500"
    assert "1.500e-10" in widget.orig_setpoint_label.text() # Check scientific notation
    assert widget.orig_angle_label.text() == "15.0"
    assert widget.orig_label_label.text() == "Original Image 1"

    # Check Current State Info (should also reflect the root node)
    assert "<i>Original</i>" in widget.node_op_label.text() # Check italic tag
    assert widget.node_type_label.text() == "STM"
    assert widget.node_shape_label.text() == str(root_node.image_data.shape)
    assert widget.node_roi_label.text() == "N/A (Whole Image)" # No source ROI for original
    assert widget.node_source_label.text() == "Original Image 1"
    # Check parameters (original metadata stored here) - might be long
    assert "filename=test_image.stp" in widget.node_params_label.text()
    assert "pixels_x=128" in widget.node_params_label.text()

def test_metadata_widget_child_node(qtbot, sample_history):
    """Test displaying metadata for a child node (Gaussian Blur)."""
    history, root_id, blur_id, _ = sample_history
    blur_node = history[blur_id]
    widget = MetadataWidget()
    qtbot.addWidget(widget)

    widget.update_metadata(blur_node, history)

    # Check Original File Info (should still come from root node)
    assert widget.filename_label.text() == "test_image.stp"
    assert widget.orig_dims_px_label.text() == "128 x 128"
    assert widget.orig_dims_nm_label.text() == "50.50 x 51.20"
    assert widget.orig_bias_label.text() == "0.7500"
    assert "1.500e-10" in widget.orig_setpoint_label.text()
    assert widget.orig_angle_label.text() == "15.0"
    assert widget.orig_label_label.text() == "Original Image 1"

    # Check Current State Info (should reflect the blur node)
    assert "<i>Gaussian Blur</i>" in widget.node_op_label.text()
    assert widget.node_type_label.text() == "STM" # Blur on STM gives STM
    assert widget.node_shape_label.text() == str(blur_node.image_data.shape)
    assert widget.node_roi_label.text() == "N/A (Whole Image)" # Blur was not ROI-only
    assert widget.node_source_label.text() == "Original Image 1"
    # Check parameters (sigma should be present)
    assert "sigma=1.5" in widget.node_params_label.text()

def test_metadata_widget_fft_node(qtbot, sample_history):
    """Test displaying metadata for an FFT node created from an ROI."""
    history, _, _, fft_id = sample_history
    fft_node = history[fft_id] # fft_node.parameters = {'apply_window': True, 'window_type': 'hann', 'scaling_mode': 'log', 'apply_roi_only': False}
    widget = MetadataWidget()
    qtbot.addWidget(widget)

    widget.update_metadata(fft_node, history)

    # Check Original File Info (should trace back to root node)
    assert widget.filename_label.text() == "test_image.stp"
    assert widget.orig_dims_px_label.text() == "128 x 128"
    # ... (check other original fields if needed)

    # Check Current State Info (should reflect the FFT node)
    assert "<i>FFT</i>" in widget.node_op_label.text()
    assert widget.node_type_label.text() == "FFT"
    assert widget.node_shape_label.text() == str(fft_node.image_data.shape) # Shape of FFT result
    # Check formatted ROI string
    assert "Rows [32:96], Cols [32:96]" in widget.node_roi_label.text()
    assert widget.node_source_label.text() == "Original Image 1"

    # --- POPRAWIONE ASERCJE dla formatu parametrów z MetadataWidget ---
    params_text = widget.node_params_label.text()
    # Sprawdzamy, czy kluczowe parametry są obecne w stringu w formacie k=v
    # apply_roi_only jest filtrowane w MetadataWidget.update_metadata, więc go nie sprawdzamy
    assert "apply_window=True" in params_text
    assert "window_type=hann" in params_text
    assert "scaling_mode=log" in params_text
