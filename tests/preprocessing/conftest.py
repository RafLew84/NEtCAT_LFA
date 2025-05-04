# tests/preprocessing/conftest.py
"""
Shared fixtures for preprocessing tests.
"""
import pytest
import numpy as np
from typing import Tuple 

@pytest.fixture(scope="module") 
def original_image_nl() -> np.ndarray:
    """A simple image with varying regions, shared across preprocessing tests."""
    img = np.zeros((30, 30), dtype=np.float32)
    img[5:15, 5:15] = 0.5  # First square
    img[15:25, 15:25] = 1.0 # Second square (brighter)
    # Add a small gradient
    gradient = np.linspace(0, 0.2, 30)
    img += gradient[:, np.newaxis]
    return img