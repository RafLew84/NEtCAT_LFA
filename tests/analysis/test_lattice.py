# tests/analysis/test_lattice.py
"""
Unit tests for lattice definition and reciprocal space calculations
in lfa.analysis.lattice.
"""
import pytest
import numpy as np
from typing import Dict, List, Tuple, Optional

# Import functions and data to test
try:
    from lfa.analysis.lattice import (
        KNOWN_LATTICES,
        get_reciprocal_vectors,
        get_reciprocal_points
    )
    LATTICE_MODULE_AVAILABLE = True
except ImportError:
    pytest.fail("Could not import from lfa.analysis.lattice. Does the file and __init__.py exist?", pytrace=False)
    LATTICE_MODULE_AVAILABLE = False # Redundant but for clarity

# Helper for floating point comparisons
TOL = 1e-6

@pytest.mark.skipif(not LATTICE_MODULE_AVAILABLE, reason="Lattice module not available")
class TestLatticeCalculations:

    def test_known_lattices_exist(self):
        """Test that KNOWN_LATTICES is populated."""
        assert KNOWN_LATTICES is not None
        assert len(KNOWN_LATTICES) > 0
        assert "Au(111)" in KNOWN_LATTICES
        assert "Graphene" in KNOWN_LATTICES

    def test_get_reciprocal_vectors_hexagonal(self):
        """Test reciprocal vector calculation for a hexagonal lattice."""
        # Using Graphene as an example a_surf = 0.246 nm
        graphene_info = KNOWN_LATTICES["Graphene"]
        b_vectors = get_reciprocal_vectors(graphene_info)
        assert b_vectors is not None
        b1_star, b2_star = b_vectors

        # Expected magnitude for hexagonal (without 2pi): |b*| = (1/a_surf) * (2/sqrt(3))
        a_surf = graphene_info["a_surf"]
        expected_b_mag = (1.0 / a_surf) * (2.0 / np.sqrt(3))

        assert np.allclose(np.linalg.norm(b1_star), expected_b_mag, atol=TOL)
        assert np.allclose(np.linalg.norm(b2_star), expected_b_mag, atol=TOL)

        # Check angle between b1_star and b2_star (should be 60 degrees or pi/3)
        # b1_star is [mag, 0]
        # b2_star is [mag*cos(60), mag*sin(60)]
        assert np.allclose(b1_star, [expected_b_mag, 0.0], atol=TOL)
        assert np.allclose(b2_star, [expected_b_mag * 0.5, expected_b_mag * np.sqrt(3)/2.0], atol=TOL)
        # Angle check via dot product
        cos_angle = np.dot(b1_star, b2_star) / (np.linalg.norm(b1_star) * np.linalg.norm(b2_star))
        assert np.allclose(cos_angle, np.cos(np.pi / 3), atol=TOL) # 60 degrees

    def test_get_reciprocal_vectors_square(self):
        """Test reciprocal vector calculation for a square lattice."""
        # Create a dummy square lattice info
        square_info = {"type": "square", "a_surf": 0.3} # a_surf = 0.3 nm
        b_vectors = get_reciprocal_vectors(square_info)
        assert b_vectors is not None
        b1_star, b2_star = b_vectors

        # Expected magnitude for square (without 2pi): |b*| = 1/a_surf
        a_surf = square_info["a_surf"]
        expected_b_mag = 1.0 / a_surf

        assert np.allclose(np.linalg.norm(b1_star), expected_b_mag, atol=TOL)
        assert np.allclose(np.linalg.norm(b2_star), expected_b_mag, atol=TOL)
        # b1_star along x, b2_star along y
        assert np.allclose(b1_star, [expected_b_mag, 0.0], atol=TOL)
        assert np.allclose(b2_star, [0.0, expected_b_mag], atol=TOL)
        # Angle check (should be 90 degrees, so dot product is 0)
        assert np.allclose(np.dot(b1_star, b2_star), 0.0, atol=TOL)

    def test_get_reciprocal_vectors_invalid_type(self):
        """Test with an unsupported lattice type."""
        invalid_info = {"type": "triangle", "a_surf": 0.3}
        assert get_reciprocal_vectors(invalid_info) is None

    def test_get_reciprocal_points_known_lattice(self):
        """Test generation of reciprocal points for a known lattice."""
        points_au1 = get_reciprocal_points("Au(111)", max_hk=1)
        assert points_au1 is not None
        # For max_hk=1, (h,k) range from -1 to 1. (0,0) is excluded.
        # (-1,-1) to (1,1) is 3x3 = 9 pairs, minus (0,0) gives 8 points.
        assert len(points_au1) == (2*1+1)**2 - 1 # (2*max_hk+1)^2 - 1

        points_au2 = get_reciprocal_points("Au(111)", max_hk=2)
        assert points_au2 is not None
        assert len(points_au2) == (2*2+1)**2 - 1 # 5x5 - 1 = 24 points

    def test_get_reciprocal_points_custom_dict(self):
        """Test generation of points using a custom lattice dictionary."""
        custom_info = {"name": "MySquare", "type": "square", "a_surf": 0.25}
        points = get_reciprocal_points(custom_info, max_hk=1)
        assert points is not None
        assert len(points) == 8

        # Check one specific point for square a=0.25 -> |b*|=4
        # e.g., h=1, k=0 -> G = 1 * [4,0] + 0 * [0,4] = (4,0)
        assert (4.0, 0.0) in [(round(p[0],6), round(p[1],6)) for p in points] # Round for float comparison

    def test_get_reciprocal_points_unknown_name(self):
        """Test with an unknown lattice name."""
        assert get_reciprocal_points("UnknownSubstrate", max_hk=1) is None

    def test_get_reciprocal_points_invalid_info(self):
        """Test with invalid lattice_name_or_info type."""
        assert get_reciprocal_points(123, max_hk=1) is None # Pass an int
        assert get_reciprocal_points({"type": "square"}, max_hk=1) is None # Missing a_surf
        assert get_reciprocal_points({"a_surf": 0.3}, max_hk=1) is None # Missing type