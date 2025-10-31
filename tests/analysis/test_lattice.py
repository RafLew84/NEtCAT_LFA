# tests/analysis/test_lattice.py
"""
Unit tests for lattice definition and reciprocal/real space calculations
in lfa.analysis.lattice.
"""
import pytest
import numpy as np
from typing import Dict, List, Tuple, Optional, Union
from unittest.mock import patch

# Import functions and data to test
try:
    from lfa.analysis.lattice import (
        KNOWN_LATTICES,
        get_reciprocal_vectors,
        get_reciprocal_points,
        get_nearest_reciprocal_points,
        select_reciprocal_lattice_basis_vectors, # Nowa do testowania
        convert_g_vector_px_to_nm_inv,         # Nowa do testowania
        calculate_real_space_vectors_from_g,   # Nowa do testowania
        get_real_space_lattice_parameters,     # Nowa do testowania
        LATTICE_TYPE_HEXAGONAL,                # Stała typu
        LATTICE_TYPE_SQUARE                  # Stała typu
    )
    LATTICE_MODULE_AVAILABLE = True
except ImportError as e: # pragma: no cover
    pytest.fail(f"Could not import from lfa.analysis.lattice: {e}", pytrace=False)
    LATTICE_MODULE_AVAILABLE = False

# Helper for floating point comparisons
TOL = 1e-6

@pytest.mark.skipif(not LATTICE_MODULE_AVAILABLE, reason="Lattice module not available")
class TestLatticeCalculations:

    # --- Istniejące testy (test_known_lattices_exist, test_get_reciprocal_vectors_*, 
    #      test_get_reciprocal_points_*, test_get_nearest_reciprocal_points_*)
    #      pozostają bez zmian. Poniżej tylko ich placeholder, aby było wiadomo, że są. ---

    def test_known_lattices_exist(self):
        assert KNOWN_LATTICES is not None and len(KNOWN_LATTICES) > 0

    def test_get_reciprocal_vectors_hexagonal(self):
        info = {"type": LATTICE_TYPE_HEXAGONAL, "a_surf": 0.288} # Au(111) a_surf
        b_mag_expected = (1.0 / 0.288) * (2.0 / np.sqrt(3))
        b1, b2 = get_reciprocal_vectors(info)
        assert b1 is not None and b2 is not None
        assert np.allclose(b1, [b_mag_expected, 0.0], atol=TOL)
        assert np.allclose(b2, [b_mag_expected * 0.5, b_mag_expected * np.sqrt(3)/2.0], atol=TOL)

    def test_get_reciprocal_vectors_square(self):
        info = {"type": LATTICE_TYPE_SQUARE, "a_surf": 0.250}
        b_mag_expected = 1.0 / 0.250
        b1, b2 = get_reciprocal_vectors(info)
        assert b1 is not None and b2 is not None
        assert np.allclose(b1, [b_mag_expected, 0.0], atol=TOL)
        assert np.allclose(b2, [0.0, b_mag_expected], atol=TOL)

    def test_get_reciprocal_points_known_lattice(self):
        points_au1 = get_reciprocal_points("Au(111)", max_hk=1)
        assert points_au1 is not None; assert len(points_au1) == 8

    def test_get_nearest_reciprocal_points_hex(self):
        points = get_nearest_reciprocal_points(KNOWN_LATTICES["Au(111)"])
        assert points is not None; assert len(points) == 6

    def test_get_nearest_reciprocal_points_square(self):
        points = get_nearest_reciprocal_points(KNOWN_LATTICES["Cu(100)"])
        assert points is not None; assert len(points) == 4


    # --- NOWE TESTY DLA select_reciprocal_lattice_basis_vectors ---
    def test_select_basis_vectors_hexagonal_ideal(self):
        """Test selection for ideal hexagonal g-vectors."""
        # 6 idealnych wektorów g* dla sieci heksagonalnej o długości b_mag, pierwszy wzdłuż osi x
        b_mag = 10.0
        g_vecs = [
            (b_mag, 0.0),
            (b_mag * np.cos(np.pi/3), b_mag * np.sin(np.pi/3)),
            (b_mag * np.cos(2*np.pi/3), b_mag * np.sin(2*np.pi/3)),
            (-b_mag, 0.0),
            (b_mag * np.cos(4*np.pi/3), b_mag * np.sin(4*np.pi/3)),
            (b_mag * np.cos(5*np.pi/3), b_mag * np.sin(5*np.pi/3)),
        ]
        # Przemieszajmy je, aby sprawdzić sortowanie w funkcji
        import random
        random.shuffle(g_vecs)

        g1, g2 = select_reciprocal_lattice_basis_vectors(g_vecs, LATTICE_TYPE_HEXAGONAL)
        assert g1 is not None and g2 is not None
        # Sprawdź długości
        assert np.isclose(np.linalg.norm(g1), b_mag, atol=TOL)
        assert np.isclose(np.linalg.norm(g2), b_mag, atol=TOL)
        # Sprawdź kąt (powinien być 60 stopni)
        cos_angle = np.dot(g1, g2) / (np.linalg.norm(g1) * np.linalg.norm(g2))
        assert np.isclose(cos_angle, np.cos(np.pi/3), atol=TOL) # cos(60 deg) = 0.5

    def test_select_basis_vectors_square_ideal(self):
        """Test selection for ideal square g-vectors."""
        b_mag = 10.0
        g_vecs = [
            (b_mag, 0.0), (0.0, b_mag), (-b_mag, 0.0), (0.0, -b_mag)
        ]
        import random
        random.shuffle(g_vecs)
        
        g1, g2 = select_reciprocal_lattice_basis_vectors(g_vecs, LATTICE_TYPE_SQUARE)
        assert g1 is not None and g2 is not None
        # Sprawdź długości (mogą być uśrednione)
        assert np.isclose(np.linalg.norm(g1), b_mag, atol=TOL)
        assert np.isclose(np.linalg.norm(g2), b_mag, atol=TOL)
        # Sprawdź kąt (powinien być 90 stopni -> cos_angle = 0)
        cos_angle = np.dot(g1, g2) / (np.linalg.norm(g1) * np.linalg.norm(g2))
        assert np.isclose(cos_angle, 0.0, atol=TOL)


    # --- NOWE TESTY DLA convert_g_vector_px_to_nm_inv ---
    def test_convert_g_vector_px_to_nm_inv(self):
        g_px = (50.0, 25.0) # (dkx_px, dky_px)
        Lx_nm, Ly_nm = 100.0, 50.0 # Rozmiary rzeczywiste obrazu
        # fft_cols_kx, fft_rows_ky - nie są używane w tej funkcji, ale przekazywane
        # do get_real_space_lattice_parameters
        
        # Wzór: g_nm_inv = g_px / L_nm
        expected_g_nm_inv = (50.0 / 100.0, 25.0 / 50.0) # (0.5, 0.5)
        
        result = convert_g_vector_px_to_nm_inv(g_px, Lx_nm, Ly_nm, 256, 256) # Kształt FFT nieistotny dla tej funkcji
        assert result is not None
        assert np.allclose(result, expected_g_nm_inv, atol=TOL)

    def test_convert_g_vector_invalid_calibration(self):
        assert convert_g_vector_px_to_nm_inv((10,10), 0, 10, 100, 100) is None
        assert convert_g_vector_px_to_nm_inv((10,10), 10, 0, 100, 100) is None


    # --- NOWE TESTY DLA calculate_real_space_vectors_from_g ---
    def test_calculate_real_space_vectors_square(self):
        # Dla sieci kwadratowej, g1* = (1/a, 0), g2* = (0, 1/a)
        # => a1 = (a, 0), a2 = (0, a)
        a_val = 0.25
        g_mag = 1.0 / a_val # = 4.0
        g1_nm_inv = (g_mag, 0.0)
        g2_nm_inv = (0.0, g_mag)
        
        a1, a2 = calculate_real_space_vectors_from_g(g1_nm_inv, g2_nm_inv)
        assert a1 is not None and a2 is not None
        assert np.allclose(a1, (a_val, 0.0), atol=TOL)
        assert np.allclose(a2, (0.0, a_val), atol=TOL)

    def test_calculate_real_space_vectors_collinear(self):
        g1 = (1.0, 1.0)
        g2 = (2.0, 2.0) # Współliniowe
        assert calculate_real_space_vectors_from_g(g1, g2) is None


    # --- NOWE TESTY DLA get_real_space_lattice_parameters (głównej funkcji) ---
    @patch('lfa.analysis.lattice.select_reciprocal_lattice_basis_vectors')
    @patch('lfa.analysis.lattice.convert_g_vector_px_to_nm_inv')
    @patch('lfa.analysis.lattice.calculate_real_space_vectors_from_g')
    def test_get_real_space_lattice_parameters_flow(self, mock_calc_real_vecs, mock_convert_g, mock_select_basis, mocker):
        """Testuje przepływ danych przez get_real_space_lattice_parameters."""
        selected_g_px = [(10.0,0.0), (0.0,10.0)] # Uproszczone, funkcja select_basis powinna dostać więcej
        lattice_type = LATTICE_TYPE_SQUARE
        Lx, Ly = 100.0, 100.0
        fft_shape = (256, 256)

        # Mockowanie zwracanych wartości przez funkcje pomocnicze
        mock_select_basis.return_value = ((10.0, 0.0), (0.0, 10.0)) # g1_px, g2_px
        mock_convert_g.side_effect = [
            (0.1, 0.0),  # g1_nm_inv = (10/100, 0/100)
            (0.0, 0.1)   # g2_nm_inv = (0/100, 10/100)
        ]
        # D = 0.1*0.1 - 0*0 = 0.01
        # a1 = (1/0.01) * [0.1, 0] = (10, 0)
        # a2 = (1/0.01) * [0, 0.1] = (0, 10)
        mock_calc_real_vecs.return_value = ((10.0, 0.0), (0.0, 10.0)) # a1_nm, a2_nm

        results = get_real_space_lattice_parameters(
            selected_g_vectors_relative_px=selected_g_px * 2, # Przekaż 4 dla square
            lattice_type=lattice_type,
            Lx_nm=Lx, Ly_nm=Ly,
            fft_shape_cols_kx=fft_shape[1], fft_shape_rows_ky=fft_shape[0]
        )

        assert results is not None
        mock_select_basis.assert_called_once_with(selected_g_px * 2, lattice_type, return_details=False)
        assert mock_convert_g.call_count == 2
        mock_calc_real_vecs.assert_called_once_with((0.1,0.0), (0.0,0.1))
        
        assert np.isclose(results["a1_nm"], 10.0, atol=TOL)
        assert np.isclose(results["a2_nm"], 10.0, atol=TOL)
        assert np.isclose(results["alpha_deg"], 90.0, atol=TOL)
        assert np.allclose(results["a1_vec_nm"], (10.0, 0.0), atol=TOL)

    def test_get_real_space_lattice_parameters_with_covariance(self):
        selected_g_px = [
            (1.0, 0.0),
            (0.0, 1.0),
            (-1.0, 0.0),
            (0.0, -1.0),
        ]
        lattice_type = LATTICE_TYPE_SQUARE
        Lx = Ly = 10.0
        fft_shape = (128, 128)
        sigma_px = 0.01
        cov_matrix = np.diag([sigma_px ** 2, sigma_px ** 2])
        cov_list = [cov_matrix for _ in selected_g_px]

        results = get_real_space_lattice_parameters(
            selected_g_vectors_relative_px=selected_g_px,
            lattice_type=lattice_type,
            Lx_nm=Lx,
            Ly_nm=Ly,
            fft_shape_cols_kx=fft_shape[1],
            fft_shape_rows_ky=fft_shape[0],
            selected_g_vector_covariances_px=cov_list,
        )

        assert results is not None
        assert "a1_nm_sigma" in results
        assert results["a1_nm_sigma"] > 0.0
        assert "real_space_metric_covariance" in results
        assert results["real_space_metric_covariance"].shape == (3, 3)

    def test_get_real_space_parameters_invalid_input(self):
        assert get_real_space_lattice_parameters([], LATTICE_TYPE_SQUARE, 10, 10, 100, 100) is None
        assert get_real_space_lattice_parameters([(1,1),(2,2)], LATTICE_TYPE_SQUARE, 0, 10, 100, 100) is None

    def test_pixel_calibration_uncertainty_contributes_to_covariance(self):
        g_vectors = [(10.0, 0.0), (0.0, 12.0)]
        result = get_real_space_lattice_parameters(
            selected_g_vectors_relative_px=g_vectors,
            lattice_type=LATTICE_TYPE_SQUARE,
            Lx_nm=20.0,
            Ly_nm=22.0,
            fft_shape_cols_kx=128,
            fft_shape_rows_ky=128,
            selected_g_vector_covariances_px=None,
            Lx_sigma_nm=0.5,
            Ly_sigma_nm=0.4,
        )
        assert result is not None
        g1_cov = result.get("g1_vec_cov_nm_inv")
        assert g1_cov is not None
        expected_var_x = ((10.0 / (20.0 ** 2)) ** 2) * (0.5 ** 2)
        assert np.isclose(g1_cov[0][0], expected_var_x)
