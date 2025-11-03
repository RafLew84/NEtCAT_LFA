# tests/core/test_history.py
"""
Unit tests for the HistoryNode class in lfa.core.history.
"""
import time

import numpy as np
import pytest  # Import pytest

# Import the class to be tested
try:
    from lfa.core.history import HistoryNode
except ImportError:
     pytest.fail("Could not import HistoryNode from lfa.core.history", pytrace=False)

# --- Test Functions ---

def test_history_node_default_initialization():
    """Test default values when creating a HistoryNode."""
    node = HistoryNode()
    assert isinstance(node.node_id, str) and len(node.node_id) > 0, "node_id should be a non-empty string"
    assert node.parent_id is None, "Default parent_id should be None"
    assert node.operation_name == "Original", "Default operation_name should be 'Original'"
    assert node.parameters == {}, "Default parameters should be an empty dict"
    assert node.image_data is None, "Default image_data should be None"
    # Check timestamp is recent (within last 5 seconds)
    assert time.time() - node.timestamp < 5, "Timestamp seems incorrect"

def test_history_node_custom_initialization():
    """Test creating a HistoryNode with custom values."""
    parent_id = "parent_node_123"
    op_name = "Gaussian Blur"
    params = {"sigma": 1.5, "mode": "reflect"}
    # Create dummy data
    data = np.array([[1, 2], [3, 4]], dtype=np.float32)

    node = HistoryNode(
        parent_id=parent_id,
        operation_name=op_name,
        parameters=params,
        image_data=data
    )

    assert isinstance(node.node_id, str) # ID should still be generated
    assert node.parent_id == parent_id
    assert node.operation_name == op_name
    assert node.parameters == params
    assert node.image_data is data # Check if it's the same object (initially)
    assert np.array_equal(node.image_data, data) # Check if data content is correct

def test_history_node_unique_ids():
    """Test if subsequently created nodes have unique IDs."""
    node1 = HistoryNode()
    node2 = HistoryNode()
    assert node1.node_id != node2.node_id, "Node IDs are not unique"

def test_history_node_get_display_text_original():
    """Test display text for the original/root node."""
    node = HistoryNode() # Defaults to "Original"
    assert node.get_display_text() == "Original Image"

def test_history_node_get_display_text_operation():
    """Test display text for a node representing an operation."""
    params = {"sigma": 2.0, "mode": "nearest"}
    node = HistoryNode(operation_name="Gaussian Blur", parameters=params, data_type="STM")
    expected_text = "Gaussian Blur (Mode:nearest, sigma=2)"
    assert node.get_display_text() == expected_text

def test_history_node_get_display_text_long_params():
    """Test display text truncation for long parameter strings."""
    params = {"alpha": 0.1, "beta": 0.2, "gamma": 0.3, "delta": 0.4, "epsilon": 0.5, "zeta": 0.6}
    node = HistoryNode(operation_name="Complex Filter", parameters=params)
    text = node.get_display_text()
    assert text.startswith("Complex Filter (alpha=0.1, beta=0.2, delta=0.4")
    assert text.endswith("...)")
    assert len(text) <= len("Complex Filter ()") + 35 + len("...")


def test_history_node_invalid_image_data_type():
    """Test that initializing with incorrect image_data type raises TypeError."""
    with pytest.raises(TypeError):
        HistoryNode(image_data=[1, 2, 3]) # Pass a list instead of ndarray
