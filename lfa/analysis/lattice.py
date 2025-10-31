# lfa/analysis/lattice.py
"""
Functions and data related to crystal lattices and reciprocal space.
Provides functionality for analyzing crystal structures, calculating reciprocal lattice vectors,
and determining real-space parameters from FFT data.
"""
import logging
import numpy as np
from typing import Dict, Tuple, List, Optional, Union, Any

from ase import Atoms
from ase.build import make_supercell

from .uncertainty import (
    propagate_linear,
    propagate_monte_carlo,
    PropagationResult,
)
from ..core.constants import (
    ADSORBATE_LATTICE_TYPE_HEXAGONAL,
    ADSORBATE_LATTICE_TYPE_SQUARE,
    ADSORBATE_LATTICE_TYPE_UNKNOWN,
    LATTICE_TYPE_CUSTOM,
    LATTICE_TYPE_HEXAGONAL,
    LATTICE_TYPE_SQUARE,
)

LATTICE_TYPE_UNKNOWN = "Unknown"

logger = logging.getLogger(__name__)

def augment_covariance_with_calibration(
    g_vector_px: Tuple[float, float],
    covariance_nm_inv: Optional[np.ndarray],
    Lx_nm: float,
    Ly_nm: float,
    sigma_Lx_nm: float,
    sigma_Ly_nm: float,
) -> Optional[np.ndarray]:
    """
    Add contributions from pixel-size calibration uncertainty to a covariance matrix.
    """
    has_existing = covariance_nm_inv is not None
    if covariance_nm_inv is not None:
        cov = np.array(covariance_nm_inv, dtype=float)
        if cov.shape != (2, 2):
            cov = np.zeros((2, 2), dtype=float)
            has_existing = False
    else:
        cov = np.zeros((2, 2), dtype=float)

    added = False
    gx_px = float(g_vector_px[0]) if g_vector_px is not None else 0.0
    gy_px = float(g_vector_px[1]) if g_vector_px is not None else 0.0

    if sigma_Lx_nm and sigma_Lx_nm > 0.0 and abs(Lx_nm) > 1e-12:
        deriv_x = gx_px / (Lx_nm ** 2)
        cov[0, 0] += (deriv_x * sigma_Lx_nm) ** 2
        added = True

    if sigma_Ly_nm and sigma_Ly_nm > 0.0 and abs(Ly_nm) > 1e-12:
        deriv_y = gy_px / (Ly_nm ** 2)
        cov[1, 1] += (deriv_y * sigma_Ly_nm) ** 2
        added = True

    if not has_existing and not added:
        return None
    return cov

# Lattice Definitions
# Store real-space lattice constant 'a' (nearest neighbor or conventional cell param) in nm.
# Store type ('hexagonal', 'square') to determine reciprocal lattice calculation.
KNOWN_LATTICES: Dict[str, Dict] = {
    "Au(111)": {
        "type": LATTICE_TYPE_HEXAGONAL,
        "a_bulk": 0.408, # nm
        "a_surf": 0.408 / np.sqrt(2), # nm (~0.288)
        "source": "Approx. bulk value"
    },
    "Ag(111)": {
        "type": LATTICE_TYPE_HEXAGONAL,
        "a_bulk": 0.409,
        "a_surf": 0.409 / np.sqrt(2), # ~0.289 nm
        "source": "Approx. bulk value"
    },
    "Cu(111)": {
        "type": LATTICE_TYPE_HEXAGONAL,
        "a_bulk": 0.361,
        "a_surf": 0.361 / np.sqrt(2), # ~0.255 nm
        "source": "Approx. bulk value"
    },
     "Cu(100)": {
        "type": LATTICE_TYPE_SQUARE,
        "a_bulk": 0.361,
        "a_surf": 0.361 / np.sqrt(2), # ~0.255 nm (side length of surface unit cell)
        "source": "Approx. bulk value"
    },
     "Ag(100)": {
        "type": LATTICE_TYPE_SQUARE,
        "a_bulk": 0.409,
        "a_surf": 0.409 / np.sqrt(2), # ~0.289 nm
        "source": "Approx. bulk value"
    },
    "Graphene": {
        "type": LATTICE_TYPE_HEXAGONAL,
        "a_surf": 0.246, # nm
        "source": "Typical value"
    },
    "HOPG": { # Often approximated as graphene for surface studies
        "type": LATTICE_TYPE_HEXAGONAL,
        "a_surf": 0.246, # nm
        "source": "Typical value"
    },
    "Au(100)": {
        "type": LATTICE_TYPE_SQUARE,
        "a_bulk": 0.408,  # nm (lattice constant)
        "a_surf": 0.408 / np.sqrt(2),  # ~0.288 nm
        "source": "Approx. bulk value"
    },
    # Platinum (Pt) - FCC
    "Pt(111)": {
        "type": LATTICE_TYPE_HEXAGONAL,
        "a_bulk": 0.392,  # nm
        "a_surf": 0.392 / np.sqrt(2),  # ~0.277 nm
        "source": "Approx. bulk value"
    },
    "Pt(100)": {
        "type": LATTICE_TYPE_SQUARE,
        "a_bulk": 0.392,
        "a_surf": 0.392 / np.sqrt(2),  # ~0.277 nm
        "source": "Approx. bulk value"
    },
    # Nickel (Ni) - FCC
    "Ni(111)": {
        "type": LATTICE_TYPE_HEXAGONAL,
        "a_bulk": 0.352,  # nm
        "a_surf": 0.352 / np.sqrt(2),  # ~0.249 nm
        "source": "Approx. bulk value"
    },
    "Ni(100)": {
        "type": LATTICE_TYPE_SQUARE,
        "a_bulk": 0.352,
        "a_surf": 0.352 / np.sqrt(2),  # ~0.249 nm
        "source": "Approx. bulk value"
    },
    # Add more lattices here
}

# --- Reciprocal Lattice Calculation ---

def _compute_reciprocal_from_direct_vectors(
    a_vec_nm: np.ndarray,
    b_vec_nm: np.ndarray
) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    """
    Builds reciprocal basis vectors (without 2pi factor) from two direct vectors.
    """
    if a_vec_nm.shape != (2,) or b_vec_nm.shape != (2,):
        logger.error("Direct lattice vectors must be 2D arrays.")
        return None

    area = a_vec_nm[0] * b_vec_nm[1] - a_vec_nm[1] * b_vec_nm[0]
    if abs(area) < 1e-9:
        logger.error("Direct lattice vectors are collinear; reciprocal basis undefined.")
        return None

    b1_star = np.array([b_vec_nm[1], -b_vec_nm[0]]) / area
    b2_star = np.array([-a_vec_nm[1], a_vec_nm[0]]) / area
    return b1_star, b2_star

def _build_direct_vectors_from_lengths(
    a_length_nm: Optional[float],
    b_length_nm: Optional[float],
    gamma_deg: Optional[float]
) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    """
    Returns 2D direct lattice vectors from lengths and enclosed angle.
    """
    if not (a_length_nm and b_length_nm and gamma_deg is not None):
        logger.error("Manual lattice definition needs a, b, and gamma.")
        return None
    if a_length_nm <= 0 or b_length_nm <= 0:
        logger.error("Manual lattice vector lengths must be positive.")
        return None

    gamma_rad = np.deg2rad(gamma_deg)
    if abs(np.sin(gamma_rad)) < 1e-6:
        logger.error("Manual lattice angle produces nearly collinear vectors.")
        return None

    a_vec = np.array([a_length_nm, 0.0], dtype=float)
    b_vec = np.array([b_length_nm * np.cos(gamma_rad),
                      b_length_nm * np.sin(gamma_rad)], dtype=float)
    return a_vec, b_vec

def get_reciprocal_vectors(lattice_info: Dict) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    """
    Calculates the primary reciprocal lattice vectors b1*, b2* (without 2pi).
    
    Args:
        lattice_info (Dict): Dictionary containing lattice type and surface lattice constant.
        
    Returns:
        Optional[Tuple[np.ndarray, np.ndarray]]: Tuple of (b1*, b2*) vectors or None on error.
    """
    l_type = lattice_info.get("type")

    if lattice_info.get("a_vec_nm") is not None and lattice_info.get("b_vec_nm") is not None:
        try:
            a_vec = np.array(lattice_info["a_vec_nm"], dtype=float)
            b_vec = np.array(lattice_info["b_vec_nm"], dtype=float)
        except Exception:  # pragma: no cover
            logger.exception("Failed parsing manual lattice vectors.")
            return None
        return _compute_reciprocal_from_direct_vectors(a_vec, b_vec)

    if l_type == LATTICE_TYPE_CUSTOM:
        direct_vectors = _build_direct_vectors_from_lengths(
            lattice_info.get("a_length_nm"),
            lattice_info.get("b_length_nm"),
            lattice_info.get("gamma_deg")
        )
        if direct_vectors is None:
            return None
        a_vec, b_vec = direct_vectors
        return _compute_reciprocal_from_direct_vectors(a_vec, b_vec)

    a = lattice_info.get("a_surf")  # Use surface lattice constant
    if not l_type or not a:
        logger.error("Lattice info missing 'type' or 'a_surf'.")
        return None

    if l_type == LATTICE_TYPE_HEXAGONAL:
        b_mag = (1.0 / a) * (2.0 / np.sqrt(3))
        b1_star = np.array([b_mag, 0.0])
        b2_star = np.array([b_mag * np.cos(np.pi / 3), b_mag * np.sin(np.pi / 3)])
        return b1_star, b2_star
    if l_type == LATTICE_TYPE_SQUARE:
        b_mag = 1.0 / a
        b1_star = np.array([b_mag, 0.0])
        b2_star = np.array([0.0, b_mag])
        return b1_star, b2_star

    logger.error(f"Unsupported lattice type: {l_type}")
    return None

def get_reciprocal_points(lattice_name_or_info: Union[str, Dict],
                           max_hk: int = 2) -> Optional[List[Tuple[float, float]]]:
    """
    Generates reciprocal lattice points G* = h*b1* + k*b2*.
    Can accept a lattice name (from KNOWN_LATTICES) or a direct lattice info dictionary.
    
    Args:
        lattice_name_or_info: Name of a known lattice or a lattice info dictionary.
        max_hk: Maximum order of reciprocal lattice points to generate.
        
    Returns:
        Optional[List[Tuple[float, float]]]: List of (Gx*, Gy*) coordinates or None on error.
    """
    lattice_info: Optional[Dict] = None
    display_name = ""

    if isinstance(lattice_name_or_info, str):
        display_name = lattice_name_or_info
        if display_name not in KNOWN_LATTICES:
            logger.error(f"Unknown lattice name: {display_name}")
            return None
        lattice_info = KNOWN_LATTICES[display_name]
    elif isinstance(lattice_name_or_info, dict):
        lattice_info = lattice_name_or_info
        display_name = lattice_info.get("name", "Custom") 
    else:
        logger.error("Invalid argument for lattice_name_or_info. Must be str or dict.")
        return None

    if not lattice_info: # Dodatkowe sprawdzenie
        logger.error(f"Lattice info is empty for '{display_name}'.")
        return None

    vectors = get_reciprocal_vectors(lattice_info)
    if vectors is None:
        return None
    b1_star, b2_star = vectors

    points = []
    for h in range(-max_hk, max_hk + 1):
        for k in range(-max_hk, max_hk + 1):
            if h == 0 and k == 0: continue
            g_star = h * b1_star + k * b2_star
            points.append(tuple(g_star))

    logger.info(f"Generated {len(points)} reciprocal points for '{display_name}' up to order {max_hk}.")
    return points

def get_nearest_reciprocal_points(
    lattice_name_or_info: Union[str, Dict],
    num_points_hex: int = 6, # Default 6 for hexagonal
    num_points_square: int = 4 # Default 4 for square
) -> Optional[List[Tuple[float, float]]]:
    """
    Generates a specific number of nearest non-zero reciprocal lattice points
    (G* = h*b1* + k*b2*) around the center (0,0).

    Args:
        lattice_name_or_info: Name of a known lattice or a lattice info dictionary.
        num_points_hex: Number of nearest points to return for a hexagonal lattice.
        num_points_square: Number of nearest points to return for a square lattice.

    Returns:
        Optional[List[Tuple[float, float]]]: A list containing the (Gx, Gy) coordinates
                                             of the nearest reciprocal points, sorted by
                                             distance from origin, or None on error.
    """
    lattice_info_dict: Optional[Dict] = None
    display_name = "" # Dla logowania

    if isinstance(lattice_name_or_info, str):
        display_name = lattice_name_or_info
        if display_name not in KNOWN_LATTICES:
            logger.error(f"get_nearest_reciprocal_points: Unknown lattice name: {display_name}")
            return None
        lattice_info_dict = KNOWN_LATTICES[display_name]
    elif isinstance(lattice_name_or_info, dict):
        lattice_info_dict = lattice_name_or_info
        display_name = lattice_info_dict.get("name", "Custom")
    else:
        logger.error("get_nearest_reciprocal_points: Invalid argument for lattice_name_or_info. Must be str or dict.")
        return None

    if not lattice_info_dict: # pragma: no cover
        logger.error(f"get_nearest_reciprocal_points: Lattice info is empty for '{display_name}'.")
        return None

    lattice_type = lattice_info_dict.get("type")
    if not lattice_type:
        if lattice_info_dict.get("a_vec_nm") is not None and lattice_info_dict.get("b_vec_nm") is not None:
            lattice_type = LATTICE_TYPE_CUSTOM
        else: # pragma: no cover
            logger.error(f"get_nearest_reciprocal_points: Lattice type not specified in info for '{display_name}'.")
            return None

    candidate_points = get_reciprocal_points(lattice_info_dict, max_hk=1)

    if not candidate_points:
        logger.error(f"get_nearest_reciprocal_points: Could not generate candidate points for '{display_name}'.")
        return None

    sorted_points = sorted(candidate_points, key=lambda p: p[0]**2 + p[1]**2)

    preferred_count = lattice_info_dict.get("preferred_point_count")
    if lattice_type == LATTICE_TYPE_HEXAGONAL:
        num_points_to_return = num_points_hex
    elif lattice_type == LATTICE_TYPE_SQUARE:
        num_points_to_return = num_points_square
    else:
        num_points_to_return = preferred_count if isinstance(preferred_count, int) and preferred_count > 0 else max(num_points_hex, num_points_square)


    if len(sorted_points) < num_points_to_return: # pragma: no cover
        logger.warning(f"get_nearest_reciprocal_points: Generated fewer candidate points ({len(sorted_points)}) "
                       f"than requested ({num_points_to_return}) for '{display_name}' with max_hk=1. "
                       f"Returning all found non-zero points.")
        return sorted_points
    
    logger.info(f"Returning {num_points_to_return} nearest reciprocal points for '{display_name}' ({lattice_type}).")
    return sorted_points[:num_points_to_return]

def select_reciprocal_lattice_basis_vectors(
    g_vectors_relative_px: List[Tuple[float, float]],
    lattice_type: str,
    *,
    return_details: bool = False,
) -> Optional[Tuple[Tuple[float, float], Tuple[float, float]] | Tuple[
    Tuple[float, float],
    Tuple[float, float],
    Dict[str, Any],
]]:
    """
    Selects two primary reciprocal lattice basis vectors (g1*, g2*)
    from a list of g-vectors (in pixels, relative to FFT center).

    Args:
        g_vectors_relative_px: List of (dkx_px, dky_px) vectors.
                               Expected 6 for hexagonal, 4 for square.
        lattice_type: LATTICE_TYPE_HEXAGONAL or LATTICE_TYPE_SQUARE.

    Returns:
        A tuple containing two basis vectors (g1_px, g2_px), or None if selection fails.
        Each vector is a tuple (dkx_px, dky_px).
    """
    if not g_vectors_relative_px:
        logger.warning("select_reciprocal_lattice_basis_vectors: No g-vectors provided.")
        return None

    details: Dict[str, Any] = {}

    if lattice_type == LATTICE_TYPE_HEXAGONAL:
        if len(g_vectors_relative_px) < 6:
            logger.warning(f"Hexagonal lattice: expected 6 spots for reliable basis vector selection, got {len(g_vectors_relative_px)}.")
            if len(g_vectors_relative_px) >= 2: 
                 logger.warning("Fallback: Taking first two available spots for hexagonal basis, may be incorrect.")
                 g1 = tuple(g_vectors_relative_px[0])
                 g2 = tuple(g_vectors_relative_px[1])
                 if return_details:
                     details = {
                         "g1": {"source_index": 0, "scale_factor": 1.0},
                         "g2": {"type": "direct", "source_index": 1, "scale_factor": 1.0},
                     }
                     return g1, g2, details
                 return g1, g2
            return None

        indexed_vectors = [(idx, np.array(vec, dtype=float)) for idx, vec in enumerate(g_vectors_relative_px)]
        sorted_pairs = sorted(indexed_vectors[:6], key=lambda item: np.arctan2(item[1][1], item[1][0]))

        lengths = [np.linalg.norm(vec) for _, vec in sorted_pairs]
        if not lengths or any(l < 1e-6 for l in lengths): 
            logger.warning("Hexagonal g-vectors have zero or near-zero length(s). Cannot determine basis.")
            return None
        avg_len_px = np.mean(lengths)

        g1_index, g1_ref_vec = sorted_pairs[0]
        g1_ref_len = np.linalg.norm(g1_ref_vec)
        if g1_ref_len < 1e-9:
            logger.warning("Primary hexagonal vector has near-zero length.")
            return None
        scale = avg_len_px / g1_ref_len
        g1_vec = g1_ref_vec * scale

        cos60, sin60 = np.cos(np.pi/3), np.sin(np.pi/3)
        rotation_matrix = np.array([[cos60, -sin60], [sin60, cos60]], dtype=float)
        g2_vec = rotation_matrix @ g1_vec

        g1_tuple = (float(g1_vec[0]), float(g1_vec[1]))
        g2_tuple = (float(g2_vec[0]), float(g2_vec[1]))

        logger.info(f"Hexagonal basis: g1_px={g1_tuple}, g2_px={g2_tuple} (avg_len={avg_len_px:.2f}px)")
        if return_details:
            details = {
                "g1": {"source_index": g1_index, "scale_factor": scale},
                "g2": {
                    "type": "rotation",
                    "rotation_matrix": rotation_matrix,
                    "source": "g1",
                },
            }
            return g1_tuple, g2_tuple, details
        return g1_tuple, g2_tuple

    elif lattice_type == LATTICE_TYPE_SQUARE:
        if len(g_vectors_relative_px) < 4:
            logger.warning(f"Square lattice: expected 4 spots for reliable basis vector selection, got {len(g_vectors_relative_px)}.")
            if len(g_vectors_relative_px) >= 2: # Simple fallback
                 logger.warning("Fallback: Taking first two available spots for square basis, may be incorrect.")
                 # If available, check whether the first two vectors are roughly orthogonal
                 v1 = np.array(g_vectors_relative_px[0])
                 v2 = np.array(g_vectors_relative_px[1])
                 norm_v1, norm_v2 = np.linalg.norm(v1), np.linalg.norm(v2)
                 if norm_v1 > 1e-6 and norm_v2 > 1e-6:
                    if abs(np.dot(v1,v2) / (norm_v1 * norm_v2)) < 0.5: # Angle close to 90 deg (+/-30 deg tolerance)
                        g1_tuple = (float(v1[0]), float(v1[1]))
                        g2_tuple = (float(v2[0]), float(v2[1]))
                        if return_details:
                            details = {
                                "g1": {"source_index": 0, "scale_factor": 1.0},
                                "g2": {"type": "direct", "source_index": 1, "scale_factor": 1.0},
                            }
                            return g1_tuple, g2_tuple, details
                        return g1_tuple, g2_tuple
            return None

        # Average the lengths of the first four candidate vectors
        indexed_vectors = [(idx, np.array(vec, dtype=float)) for idx, vec in enumerate(g_vectors_relative_px[:4])]
        lengths = [np.linalg.norm(vec) for _, vec in indexed_vectors]
        if not lengths or any(l < 1e-6 for l in lengths):
            logger.warning("Square g-vectors have zero or near-zero length(s). Cannot determine basis.")
            return None
        avg_len_px = np.mean(lengths)

        # Select two orthogonal vectors with the averaged length.
        # We assume one aligns roughly with the kx axis and the other with the ky axis.
        # This is an approximation; a more robust approach would search over vector pairs.
        # It also assumes the ideal spots are already oriented sensibly
        # (e.g. produced by get_nearest_reciprocal_points) so the first two can form a basis.
        
        # Find the vectors closest to the kx and ky axes
        # Vector closest to the kx axis (large |x|, small |y|)
        sorted_by_angle_to_x = sorted(
            indexed_vectors,
            key=lambda item: abs(np.arctan2(item[1][1], item[1][0]))
        )
        g1_index, g1_candidate_dir = sorted_by_angle_to_x[0]
        g1_len = np.linalg.norm(g1_candidate_dir)
        if g1_len < 1e-6:
            logger.warning("g1 candidate zero length for square")
            return None
        g1_scale = avg_len_px / g1_len
        g1_vec = g1_candidate_dir * g1_scale

        # Choose the vector that is most orthogonal to g1_vec
        g2_candidate_dir = None
        min_dot_product_abs = float('inf')
        g2_index = None
        for idx, v_cand_np in indexed_vectors:
            if np.allclose(v_cand_np, g1_candidate_dir):
                continue
            dot_prod_abs = abs(np.dot(g1_vec, v_cand_np))
            if dot_prod_abs < min_dot_product_abs:
                min_dot_product_abs = dot_prod_abs
                g2_candidate_dir = v_cand_np
                g2_index = idx

        if g2_candidate_dir is None:
            logger.warning("Could not find a suitable g2 candidate for square.")
            return None

        g2_len = np.linalg.norm(g2_candidate_dir)
        if g2_len < 1e-6:
            logger.warning("g2 candidate zero length for square")
            return None
        g2_scale = avg_len_px / g2_len
        g2_vec = g2_candidate_dir * g2_scale
        
        # Final orthogonality check (within tolerance)
        cos_angle_g1_g2 = np.dot(g1_vec, g2_vec) / (np.linalg.norm(g1_vec) * np.linalg.norm(g2_vec))
        if abs(cos_angle_g1_g2) > 0.2: # Larger than ~cos(80 deg) or cos(100 deg) -> not very perpendicular
            logger.warning(f"Selected square basis vectors are not orthogonal enough (cos_angle={cos_angle_g1_g2:.2f}). Fallback.")
            # Fallback: rotate g1 by 90 degrees to synthesize g2
            rotation_matrix = np.array([[0.0, -1.0], [1.0, 0.0]], dtype=float)
            g2_vec = rotation_matrix @ g1_vec
            details_g2 = {
                "type": "rotation",
                "rotation_matrix": rotation_matrix,
                "source": "g1",
            }
        else:
            details_g2 = {
                "type": "scaled",
                "source_index": g2_index,
                "scale_factor": g2_scale,
            }

        g1_tuple = (float(g1_vec[0]), float(g1_vec[1]))
        g2_tuple = (float(g2_vec[0]), float(g2_vec[1]))
        logger.info(f"Square basis: g1_px={g1_tuple}, g2_px={g2_tuple} (avg_len={avg_len_px:.2f}px)")
        if return_details:
            details = {
                "g1": {"source_index": g1_index, "scale_factor": g1_scale},
                "g2": details_g2,
            }
            return g1_tuple, g2_tuple, details
        return g1_tuple, g2_tuple
    else:
        if len(g_vectors_relative_px) < 2:
            logger.warning("Custom lattice requires at least two reciprocal vectors for basis selection.")
            return None
        for i in range(len(g_vectors_relative_px)):
            for j in range(i + 1, len(g_vectors_relative_px)):
                g1 = g_vectors_relative_px[i]
                g2 = g_vectors_relative_px[j]
                cross = g1[0] * g2[1] - g1[1] * g2[0]
                if abs(cross) > 1e-6:
                    logger.info(f"Custom lattice basis chosen using vectors at indices {i} and {j}.")
                    g1_tuple = (float(g1[0]), float(g1[1]))
                    g2_tuple = (float(g2[0]), float(g2[1]))
                    if return_details:
                        details = {
                            "g1": {"source_index": i, "scale_factor": 1.0},
                            "g2": {"type": "direct", "source_index": j, "scale_factor": 1.0},
                        }
                        return g1_tuple, g2_tuple, details
                    return g1_tuple, g2_tuple
        logger.warning("Custom lattice: provided reciprocal vectors are collinear; cannot form a basis.")
        return None

    if return_details:
        return None

def convert_g_vector_px_to_nm_inv(
    g_vector_px: Tuple[float, float],
    Lx_nm: float,
    Ly_nm: float,
    fft_shape_cols_kx: int, # N_kx
    fft_shape_rows_ky: int  # N_ky
) -> Optional[Tuple[float, float]]:
    """
    Converts a reciprocal lattice vector from FFT pixel coordinates (relative to center)
    to physical units (nm^-1).

    Args:
        g_vector_px: (g_kx_px, g_ky_px) components of the g-vector in pixels,
                     measured from the FFT center.
        Lx_nm: Real-space calibration length corresponding to the kx direction (FFT columns).
        Ly_nm: Real-space calibration length corresponding to the ky direction (FFT rows).
        fft_shape_cols_kx: Total number of columns in the FFT (N_kx).
        fft_shape_rows_ky: Total number of rows in the FFT (N_ky).

    Returns:
        (g_kx_nm_inv, g_ky_nm_inv) in nm^-1, or None if inputs are invalid.
    """
    if not (Lx_nm > 0 and Ly_nm > 0 and fft_shape_cols_kx > 0 and fft_shape_rows_ky > 0):
        logger.error("Invalid calibration data or FFT shape for g-vector conversion.")
        return None

    g_kx_px, g_ky_px = g_vector_px

    # Convert pixel coordinates to physical units (nm^-1)
    g_kx_nm_inv = g_kx_px * (1.0 / Lx_nm)
    g_ky_nm_inv = g_ky_px * (1.0 / Ly_nm)
    
    return g_kx_nm_inv, g_ky_nm_inv

def calculate_real_space_vectors_from_g(
    g1_star_nm_inv: Tuple[float, float],
    g2_star_nm_inv: Tuple[float, float]
) -> Optional[Tuple[Tuple[float, float], Tuple[float, float]]]:
    """
    Calculates real-space lattice vectors (a1, a2) from two reciprocal
    basis vectors (g1*, g2*) given in nm^-1. (Assumes 2D, no 2pi factor)

    Args:
        g1_star_nm_inv: First reciprocal basis vector (g1x*, g1y*) in nm^-1.
        g2_star_nm_inv: Second reciprocal basis vector (g2x*, g2y*) in nm^-1.

    Returns:
        Tuple containing two real-space vectors (a1_nm, a2_nm), or None if g-vectors are collinear.
        Each vector is a tuple (ax_nm, ay_nm).
    """
    g1x, g1y = g1_star_nm_inv
    g2x, g2y = g2_star_nm_inv

    # Calculate determinant D = g1x*g2y - g1y*g2x
    # This is equivalent to |g1* x g2*|_z / (2pi)^2 if 2pi was included,
    # or Area_reciprocal_cell / (2pi)^2.
    # Without 2pi, D is Area_reciprocal_cell.
    determinant = g1x * g2y - g1y * g2x

    if abs(determinant) < 1e-9: # Check for collinearity (determinant is zero)
        logger.error("Reciprocal vectors are collinear; cannot calculate real space vectors.")
        return None

    # Calculate real space basis vectors (without 2pi factor in formulas):
    # a1 = (1/D) * [g2y, -g2x]
    # a2 = (1/D) * [-g1y, g1x]
    a1_nm = ( (1.0 / determinant) * g2y,
              (1.0 / determinant) * (-g2x) )
    
    a2_nm = ( (1.0 / determinant) * (-g1y),
              (1.0 / determinant) * g1x )
              
    return a1_nm, a2_nm


def _real_space_metrics_from_reciprocal(flat_g: np.ndarray) -> np.ndarray:
    """
    Helper mapping flattened reciprocal vectors to real-space magnitudes/angle.

    Parameters
    ----------
    flat_g:
        Array-like of length 4: [g1x, g1y, g2x, g2y] in nm^-1.
    """
    if flat_g.size != 4:
        raise ValueError("Expected flattened reciprocal vector array of length 4.")

    g1 = (float(flat_g[0]), float(flat_g[1]))
    g2 = (float(flat_g[2]), float(flat_g[3]))
    real_space_vecs = calculate_real_space_vectors_from_g(g1, g2)
    if real_space_vecs is None:
        raise ValueError("Real-space vectors undefined for provided reciprocal basis.")

    a1_vec_nm, a2_vec_nm = real_space_vecs
    a1_mag_nm = float(np.linalg.norm(a1_vec_nm))
    a2_mag_nm = float(np.linalg.norm(a2_vec_nm))
    if a1_mag_nm < 1e-12 or a2_mag_nm < 1e-12:
        raise ValueError("Real-space vector magnitude too small for stable propagation.")

    cos_alpha = float(np.dot(a1_vec_nm, a2_vec_nm) / (a1_mag_nm * a2_mag_nm))
    cos_alpha = float(np.clip(cos_alpha, -1.0, 1.0))
    alpha_deg = float(np.degrees(np.arccos(cos_alpha)))

    return np.array([a1_mag_nm, a2_mag_nm, alpha_deg], dtype=float)


def compute_real_space_metric_uncertainty(
    g1_nm_inv: Tuple[float, float],
    g2_nm_inv: Tuple[float, float],
    combined_covariance_nm_inv: np.ndarray,
    *,
    use_monte_carlo_on_failure: bool = True,
    monte_carlo_samples: int = 1024,
) -> Optional[PropagationResult]:
    """
    Estimate uncertainties for (a1, a2, alpha) given reciprocal vector covariance.

    Parameters
    ----------
    g1_nm_inv, g2_nm_inv:
        Reciprocal basis vectors in nm^-1.
    combined_covariance_nm_inv:
        4x4 covariance matrix for [g1x, g1y, g2x, g2y] (nm^-1 units).
    """
    cov = np.asarray(combined_covariance_nm_inv, dtype=float)
    if cov.shape != (4, 4):
        raise ValueError("Reciprocal vector covariance must be a 4x4 matrix.")

    x0 = np.array([g1_nm_inv[0], g1_nm_inv[1], g2_nm_inv[0], g2_nm_inv[1]], dtype=float)

    try:
        return propagate_linear(
            lambda vec: _real_space_metrics_from_reciprocal(vec),
            x0,
            cov,
        )
    except ValueError:
        if not use_monte_carlo_on_failure:
            raise
        try:
            return propagate_monte_carlo(
                lambda vec: _real_space_metrics_from_reciprocal(vec),
                x0,
                cov,
                samples=monte_carlo_samples,
            )
        except Exception as exc:  # pragma: no cover - defensive
            logger.warning("Monte Carlo propagation failed: %s", exc)
            return None

def get_real_space_lattice_parameters(
    selected_g_vectors_relative_px: List[Tuple[float, float]],
    lattice_type: str,
    Lx_nm: float,                 # Real-space size corresponding to kx-direction (FFT columns)
    Ly_nm: float,                 # Real-space size corresponding to ky-direction (FFT rows)
    fft_shape_cols_kx: int,       # Total columns in FFT (N_kx)
    fft_shape_rows_ky: int,       # Total rows in FFT (N_ky)
    selected_g_vector_covariances_px: Optional[List[Optional[np.ndarray]]] = None,
    Lx_sigma_nm: float = 0.0,
    Ly_sigma_nm: float = 0.0,
) -> Optional[Dict[str, Any]]:
    """
    Calculates real-space lattice parameters (a1, a2 magnitudes, angle alpha)
    for a given substrate definition and its selected reciprocal lattice spots.

    Args:
        selected_g_vectors_relative_px: List of 6 (hex) or 4 (square) g-vectors
                                       (dkx_px, dky_px) relative to FFT center.
        lattice_type: LATTICE_TYPE_HEXAGONAL or LATTICE_TYPE_SQUARE.
        Lx_nm: Real-space calibration length for kx.
        Ly_nm: Real-space calibration length for ky.
        fft_shape_cols_kx: Number of columns in the FFT.
        fft_shape_rows_ky: Number of rows in the FFT.

    Returns:
        A dictionary with "a1_nm", "a2_nm", "alpha_deg", "a1_vec_nm", "a2_vec_nm",
        "g1_vec_px", "g2_vec_px", "g1_vec_nm_inv", "g2_vec_nm_inv", or None on failure.
    """
    if not (Lx_nm > 0 and Ly_nm > 0 and fft_shape_cols_kx > 0 and fft_shape_rows_ky > 0):
        logger.error("Invalid calibration data (Lx, Ly) or FFT shape.")
        return None

    want_details = selected_g_vector_covariances_px is not None
    if want_details:
        selection = select_reciprocal_lattice_basis_vectors(
            selected_g_vectors_relative_px,
            lattice_type,
            return_details=True,
        )
        if selection is None:
            logger.error("Failed to select basis reciprocal vectors in pixels.")
            return None
        g1_s_px, g2_s_px, selection_details = selection
    else:
        selection = select_reciprocal_lattice_basis_vectors(
            selected_g_vectors_relative_px,
            lattice_type,
            return_details=False,
        )
        if selection is None:
            logger.error("Failed to select basis reciprocal vectors in pixels.")
            return None
        g1_s_px, g2_s_px = selection
        selection_details = {}

    g1_cov_px: Optional[np.ndarray] = None
    g2_cov_px: Optional[np.ndarray] = None
    if want_details and selected_g_vector_covariances_px:
        cov_list = [
            np.array(cov, dtype=float) if cov is not None else None
            for cov in selected_g_vector_covariances_px
        ]

        g1_info = selection_details.get("g1", {})
        g1_index = g1_info.get("source_index")
        g1_scale = g1_info.get("scale_factor", 1.0)
        if g1_index is not None and 0 <= g1_index < len(cov_list):
            base_cov = cov_list[g1_index]
            if base_cov is not None:
                g1_cov_px = (g1_scale ** 2) * base_cov

        g2_info = selection_details.get("g2", {})
        g2_type = g2_info.get("type")
        if g2_type == "rotation":
            rotation_matrix = np.asarray(g2_info.get("rotation_matrix"), dtype=float)
            if g1_cov_px is not None and rotation_matrix.shape == (2, 2):
                g2_cov_px = rotation_matrix @ g1_cov_px @ rotation_matrix.T
        elif g2_type == "scaled":
            g2_index = g2_info.get("source_index")
            g2_scale = g2_info.get("scale_factor", 1.0)
            if g2_index is not None and 0 <= g2_index < len(cov_list):
                base_cov = cov_list[g2_index]
                if base_cov is not None:
                    g2_cov_px = (g2_scale ** 2) * base_cov
        elif g2_type == "direct":
            g2_index = g2_info.get("source_index")
            if g2_index is not None and 0 <= g2_index < len(cov_list):
                g2_cov_px = cov_list[g2_index]

    # 2. Convert basis g-vectors to physical units (nm^-1)
    g1_s_nm_inv = convert_g_vector_px_to_nm_inv(g1_s_px, Lx_nm, Ly_nm, fft_shape_cols_kx, fft_shape_rows_ky)
    g2_s_nm_inv = convert_g_vector_px_to_nm_inv(g2_s_px, Lx_nm, Ly_nm, fft_shape_cols_kx, fft_shape_rows_ky)

    if g1_s_nm_inv is None or g2_s_nm_inv is None:
        logger.error("Failed to convert basis g-vectors to physical units.")
        return None

    # 3. Calculate real-space vectors (a1_nm, a2_nm)
    real_space_vecs = calculate_real_space_vectors_from_g(g1_s_nm_inv, g2_s_nm_inv)
    if real_space_vecs is None:
        logger.error("Failed to calculate real space vectors (g-vectors might be collinear).")
        return None
    a1_s_vec_nm, a2_s_vec_nm = real_space_vecs

    # 4. Calculate magnitudes and angle
    a1_s_mag_nm = np.linalg.norm(a1_s_vec_nm)
    a2_s_mag_nm = np.linalg.norm(a2_s_vec_nm)

    if a1_s_mag_nm < 1e-9 or a2_s_mag_nm < 1e-9:
        logger.error("Calculated real space vectors have zero or near-zero magnitude.")
        return None
        
    # Calculate angle between vectors
    dot_product = np.dot(a1_s_vec_nm, a2_s_vec_nm)
    cos_alpha_s = dot_product / (a1_s_mag_nm * a2_s_mag_nm)
    cos_alpha_s = np.clip(cos_alpha_s, -1.0, 1.0) # Ensure value is in arccos domain
    alpha_s_rad = np.arccos(cos_alpha_s)
    alpha_s_deg = np.degrees(alpha_s_rad)

    scale_matrix = np.array([[1.0 / Lx_nm, 0.0], [0.0, 1.0 / Ly_nm]], dtype=float)
    g1_cov_nm_inv: Optional[np.ndarray] = None
    g2_cov_nm_inv: Optional[np.ndarray] = None
    if g1_cov_px is not None:
        g1_cov_nm_inv = scale_matrix @ g1_cov_px @ scale_matrix.T
    if g2_cov_px is not None:
        g2_cov_nm_inv = scale_matrix @ g2_cov_px @ scale_matrix.T

    Lx_sigma_nm = float(Lx_sigma_nm or 0.0)
    Ly_sigma_nm = float(Ly_sigma_nm or 0.0)

    g1_cov_nm_inv = augment_covariance_with_calibration(
        g1_s_px,
        g1_cov_nm_inv,
        Lx_nm,
        Ly_nm,
        Lx_sigma_nm,
        Ly_sigma_nm,
    )
    g2_cov_nm_inv = augment_covariance_with_calibration(
        g2_s_px,
        g2_cov_nm_inv,
        Lx_nm,
        Ly_nm,
        Lx_sigma_nm,
        Ly_sigma_nm,
    )

    metrics_uncertainty: Optional[PropagationResult] = None
    if g1_cov_nm_inv is not None and g2_cov_nm_inv is not None:
        combined_covariance = np.zeros((4, 4), dtype=float)
        combined_covariance[:2, :2] = g1_cov_nm_inv
        combined_covariance[2:, 2:] = g2_cov_nm_inv
        try:
            metrics_uncertainty = compute_real_space_metric_uncertainty(
                g1_s_nm_inv,
                g2_s_nm_inv,
                combined_covariance,
            )
        except ValueError as exc:  # pragma: no cover - defensive
            logger.warning("Unable to propagate lattice parameter uncertainties: %s", exc)

    results = {
        "a1_nm": a1_s_mag_nm,
        "a2_nm": a2_s_mag_nm,
        "alpha_deg": alpha_s_deg,
        "a1_vec_nm": a1_s_vec_nm, # Tuple (ax, ay)
        "a2_vec_nm": a2_s_vec_nm, # Tuple (ax, ay)
        "g1_vec_px": g1_s_px,     # Tuple (gkx, gky) in pixels
        "g2_vec_px": g2_s_px,
        "g1_vec_nm_inv": g1_s_nm_inv, # Tuple (gkx, gky) in nm^-1
        "g2_vec_nm_inv": g2_s_nm_inv,
    }
    if g1_cov_px is not None:
        results["g1_vec_cov_px"] = g1_cov_px
    if g2_cov_px is not None:
        results["g2_vec_cov_px"] = g2_cov_px
    if g1_cov_nm_inv is not None:
        results["g1_vec_cov_nm_inv"] = g1_cov_nm_inv
    if g2_cov_nm_inv is not None:
        results["g2_vec_cov_nm_inv"] = g2_cov_nm_inv
    if metrics_uncertainty is not None:
        results["real_space_metric_covariance"] = metrics_uncertainty.covariance
        diag_entries = np.clip(np.diag(metrics_uncertainty.covariance), 0.0, None)
        if diag_entries.size >= 3:
            results["a1_nm_sigma"] = float(np.sqrt(diag_entries[0]))
            results["a2_nm_sigma"] = float(np.sqrt(diag_entries[1]))
            results["alpha_deg_sigma"] = float(np.sqrt(diag_entries[2]))
    results["pixel_calibration_sigma_nm"] = (float(Lx_sigma_nm), float(Ly_sigma_nm))
    logger.info(f"Calculated real space params: a1={a1_s_mag_nm:.4f}nm, a2={a2_s_mag_nm:.4f}nm, alpha={alpha_s_deg:.2f}deg")
    return results

def select_adsorbate_reciprocal_basis_vectors_px(
    corrected_g_vectors_relative_px: List[Tuple[float, float]], # g* vectors of adsorbate from ideal system center
    expected_lattice_type: str,
) -> Optional[Tuple[Tuple[float, float], Tuple[float, float]]]:
    """
    Selects two primary reciprocal lattice basis vectors (g1_A*, g2_A*) for an adsorbate
    from a list of its corrected g-vectors (in pixels, relative to FFT center of the ideal system),
    considering the expected lattice type and the number of provided spots.

    Args:
        corrected_g_vectors_relative_px: List of (dkx_px, dky_px) g-vectors for the adsorbate,
                                         relative to the center of the ideal FFT system.
        expected_lattice_type: String indicating the expected lattice type
                               ("Hexagonal", "Square", "Unknown").

    Returns:
        A tuple containing two basis vectors (g1_A_px, g2_A_px), or None if selection fails.
        Each vector is a tuple (dkx_px, dky_px).
    """
    num_spots_selected = len(corrected_g_vectors_relative_px)
    logger.info(f"Selecting adsorbate basis g-vectors. Expected type: '{expected_lattice_type}', Spots provided: {num_spots_selected}")

    if not corrected_g_vectors_relative_px or num_spots_selected < 2:
        logger.warning("Not enough adsorbate g-vectors provided (need at least 2) to define a basis.")
        return None

    # Scenario 1: User selected exactly 2 spots (treated as g1* and g2*)
    if num_spots_selected == 2:
        g1_px = corrected_g_vectors_relative_px[0]
        g2_px = corrected_g_vectors_relative_px[1]
        
        # Check if vectors are non-zero and non-collinear
        norm_g1 = np.linalg.norm(g1_px)
        norm_g2 = np.linalg.norm(g2_px)
        if norm_g1 < 1e-6 or norm_g2 < 1e-6:
            logger.warning("Adsorbate (2 spots): One or both selected g-vectors are zero length.")
            return None
        
        # Cross product (z-component) to check collinearity
        cross_product_z = g1_px[0] * g2_px[1] - g1_px[1] * g2_px[0]
        if abs(cross_product_z) / (norm_g1 * norm_g2) < 1e-3: # Check sine of angle
            logger.warning("Adsorbate (2 spots): Selected g-vectors are collinear.")
            return None
            
        logger.info(f"Adsorbate (2 spots defined): Using g1*={g1_px}, g2*={g2_px}")
        return g1_px, g2_px

    # Scenario 2: Expected type "Hexagonal"
    elif expected_lattice_type == LATTICE_TYPE_HEXAGONAL.capitalize(): # Compare with capital letter as in ComboBox
        if num_spots_selected < 2: # Need at least 2 to define, ideal is 6
             logger.warning(f"Hexagonal Adsorbate: Not enough spots (need >= 2, got {num_spots_selected}).")
             return None
        if num_spots_selected < 6:
            logger.warning(f"Hexagonal Adsorbate: {num_spots_selected} spots provided, less than ideal 6. "
                           "Attempting to find basis. Result may be less accurate.")
        
        # Sort vectors by length (shortest first) - assume first-order peaks are strongest/nearest
        # Then by angle for consistency
        sorted_g_vecs_by_len = sorted(corrected_g_vectors_relative_px, key=lambda v: np.linalg.norm(v))
        
        # Take 'num_to_consider' shortest vectors (e.g., 6 if available, or all)
        num_to_consider = min(num_spots_selected, 6)
        candidate_vecs = sorted_g_vecs_by_len[:num_to_consider]
        
        # If we have 6 spots, apply averaging method for anisotropy
        if num_spots_selected >= 6: # Use >= 6 to handle case when user clicked more
            # Sort 6 shortest by angle
            sorted_angularly = sorted(candidate_vecs, key=lambda v: np.arctan2(v[1], v[0]))

            # Axis vectors (averaging opposite pairs)
            # Assume sorted_angularly[0] and sorted_angularly[3] are a pair, etc.
            d1 = (np.array(sorted_angularly[0]) - np.array(sorted_angularly[3])) / 2.0
            d2 = (np.array(sorted_angularly[1]) - np.array(sorted_angularly[4])) / 2.0
            d3 = (np.array(sorted_angularly[2]) - np.array(sorted_angularly[5])) / 2.0
            
            # List of non-zero axis vectors
            axis_vectors = [v for v in [d1, d2, d3] if np.linalg.norm(v) > 1e-6]
            if len(axis_vectors) < 2:
                logger.warning("Hexagonal Adsorbate (6 spots): Could not determine at least two distinct axes.")
                return None
            
            # Choose d1 as g1_A* (first non-zero axis vector)
            g1_px = tuple(axis_vectors[0])
            
            # Choose as g2_A* the axis vector that forms angle closest to 60 degrees with g1_px
            best_g2_candidate = None
            min_angle_diff_to_60 = float('inf')
            target_angle_rad = np.pi / 3 # 60 degrees
            norm_g1 = np.linalg.norm(g1_px)

            for v_cand_np in axis_vectors[1:]: # Check remaining axis vectors
                norm_v_cand = np.linalg.norm(v_cand_np)
                if norm_v_cand < 1e-6: continue

                dot_product = np.dot(g1_px, v_cand_np)
                cos_theta = np.clip(dot_product / (norm_g1 * norm_v_cand), -1.0, 1.0)
                angle_rad = np.arccos(cos_theta)
                
                angle_diff = abs(angle_rad - target_angle_rad) # Difference from 60 degrees
                # Could also check difference from 120 degrees if that makes sense for basis selection
                # angle_diff_120 = abs(angle_rad - 2*target_angle_rad) 
                # current_min_diff = min(angle_diff, angle_diff_120)

                if angle_diff < min_angle_diff_to_60:
                    min_angle_diff_to_60 = angle_diff
                    best_g2_candidate = v_cand_np
            
            if best_g2_candidate is not None:
                g2_px = tuple(best_g2_candidate)
                logger.info(f"Adsorbate Hexagonal (6 spots aniso): Basis g1*={g1_px}, g2*={g2_px}")
                return g1_px, g2_px
            else:
                logger.warning("Adsorbate Hexagonal (6 spots aniso): Could not determine a suitable second basis vector from axes.")
                return None

        elif num_spots_selected >= 2: # Less than 6, but at least 2 for Hexagonal
            logger.warning("Hexagonal Adsorbate: Less than 6 spots.")
            # Use logic for "Unknown" - find two shortest, linearly independent vectors
            return None

    # Scenario 3: Expected type "Square"
    elif expected_lattice_type == LATTICE_TYPE_SQUARE.capitalize():
        if num_spots_selected < 2: # Need at least 2, ideal is 4
            logger.warning(f"Square Adsorbate: Not enough spots (need >= 2, got {num_spots_selected}).")
            return None
        if num_spots_selected < 4:
            logger.warning(f"Square Adsorbate: {num_spots_selected} spots provided, less than ideal 4. "
                           "Attempting to find basis. Result may be less accurate.")

        sorted_g_vecs_by_len = sorted(corrected_g_vectors_relative_px, key=lambda v: np.linalg.norm(v))
        num_to_consider = min(num_spots_selected, 4)
        candidate_vecs = sorted_g_vecs_by_len[:num_to_consider]

        if num_spots_selected >= 4: # Use >= 4
            # Sort 4 shortest by angle
            sorted_angularly = sorted(candidate_vecs, key=lambda v: np.arctan2(v[1], v[0]))
            # Axis vectors
            d1 = (np.array(sorted_angularly[0]) - np.array(sorted_angularly[2])) / 2.0
            d2 = (np.array(sorted_angularly[1]) - np.array(sorted_angularly[3])) / 2.0

            norm_d1 = np.linalg.norm(d1); norm_d2 = np.linalg.norm(d2)
            if norm_d1 < 1e-6 or norm_d2 < 1e-6:
                logger.warning("Square Adsorbate (4 spots): Axis vectors have zero length.")
                return None

            # Check perpendicularity of d1 and d2
            cos_angle_d1_d2 = np.dot(d1, d2) / (norm_d1 * norm_d2)
            if abs(cos_angle_d1_d2) > 0.2: # Angle significantly different from 90 degrees (cos(90)=0, cos(78)~0.2)
                logger.warning(f"Square Adsorbate (4 spots): Axis vectors d1, d2 are not orthogonal (cos_angle={cos_angle_d1_d2:.2f}). "
                               "Consider re-selecting spots or using 'Unknown' type.")
                # Could try selecting different pair or return error, or fall back to "Unknown" logic
                # For now, if not perpendicular, use them anyway to describe parallelogram
            
            g1_px = tuple(d1)
            g2_px = tuple(d2)
            logger.info(f"Adsorbate Square/Rect (4 spots): Basis g1*={g1_px}, g2*={g2_px}")
            return g1_px, g2_px

    # Scenario 4: Expected type "Unknown" (or fallback from other types)
    elif expected_lattice_type == ADSORBATE_LATTICE_TYPE_UNKNOWN: # num_spots_selected >=2 already checked at start
        # Find two shortest, linearly independent vectors
        if len(corrected_g_vectors_relative_px) < 2:
             return None 
        
        # Sort all available vectors by length
        sorted_g_vecs_by_len = sorted(corrected_g_vectors_relative_px, key=lambda v: np.linalg.norm(v))
        
        g1_px_cand = np.array(sorted_g_vecs_by_len[0])
        if np.linalg.norm(g1_px_cand) < 1e-6:
            logger.warning("Adsorbate (Unknown): Shortest g-vector is zero length.")
            return None

        g2_px_cand = None
        for i in range(1, len(sorted_g_vecs_by_len)):
            current_g2_cand = np.array(sorted_g_vecs_by_len[i])
            if np.linalg.norm(current_g2_cand) < 1e-6: continue # Skip zero vectors

            # Check collinearity with g1_px_cand
            cross_product_z = g1_px_cand[0] * current_g2_cand[1] - g1_px_cand[1] * current_g2_cand[0]
            # Check sine of angle to avoid normalization issues for very short vectors
            if abs(cross_product_z) / (np.linalg.norm(g1_px_cand) * np.linalg.norm(current_g2_cand) + 1e-9) > 1e-3: # Not collinear
                g2_px_cand = current_g2_cand
                break
        
        if g2_px_cand is None:
            logger.warning("Adsorbate (Unknown): Could not find two non-collinear g-vectors.")
            return None
            
        g1_px = tuple(g1_px_cand)
        g2_px = tuple(g2_px_cand)
        logger.info(f"Adsorbate (Unknown type, {num_spots_selected} spots): Selected basis g1*={g1_px}, g2*={g2_px}")
        return g1_px, g2_px

    else:
        logger.error(f"Unsupported expected_lattice_type '{expected_lattice_type}' or insufficient spots for selection.")
        return None
    
def calculate_d_spacing_from_ideal_spot(
    spot_corrected_ideal_px: Tuple[float, float],
    fft_shape: Tuple[int, int],
    lx_nm: float,
    ly_nm: float
) -> Optional[float]:
    """
    Calculates the real-space d-spacing for a single corrected spot.

    Args:
        spot_corrected_ideal_px: Corrected (kx, ky) coordinates in the ideal FFT system.
        fft_shape: The shape of the FFT data (rows_ky, cols_kx).
        lx_nm: Real-space calibration size in the x-direction (nm).
        ly_nm: Real-space calibration size in the y-direction (nm).

    Returns:
        The calculated d-spacing in nm, or None on error.
    """
    if not all([spot_corrected_ideal_px, fft_shape, lx_nm, ly_nm]):
        logger.warning("d-spacing calc: Missing input data.")
        return None
        
    try:
        fft_rows_ky, fft_cols_kx = fft_shape
        center_kx_ideal = fft_cols_kx / 2.0
        center_ky_ideal = fft_rows_ky / 2.0
        
        g_vector_ideal_px = (spot_corrected_ideal_px[0] - center_kx_ideal, 
                             spot_corrected_ideal_px[1] - center_ky_ideal)
        
        g_vector_nm_inv = convert_g_vector_px_to_nm_inv(g_vector_ideal_px, lx_nm, ly_nm, fft_cols_kx, fft_rows_ky)
        if g_vector_nm_inv is None:
            raise ValueError("k-space vector conversion failed.")
            
        g_mag_nm_inv = np.linalg.norm(g_vector_nm_inv)
        return 1.0 / g_mag_nm_inv if g_mag_nm_inv > 1e-9 else float('inf')
        
    except Exception as e:
        logger.error(f"Error in calculate_d_spacing_from_ideal_spot: {e}")
        return None

def calculate_superstructure_periodicity_parameters(
    main_peak_data: Dict[str, Any],
    satellite_peak_data: Dict[str, Any],
    fft_shape: Tuple[int, int],
    lx_nm: float,
    ly_nm: float
) -> Optional[Dict[str, float]]:
    """
    Calculates superstructure periodicity parameters based on a main and satellite peak.

    Args:
        main_peak_data: Dictionary containing data for the main peak.
                        Must include 'corrected', 'intensity', 'amplitude', 'max_value'.
        satellite_peak_data: Dictionary for the satellite peak.
        fft_shape, lx_nm, ly_nm: Calibration data.

    Returns:
        A dictionary with calculated parameters, or None on error.
    """
    try:
        # Validate input data
        required_keys = ['corrected', 'intensity', 'amplitude', 'max_value']
        if not all(k in main_peak_data and main_peak_data[k] is not None for k in required_keys) or \
           not all(k in satellite_peak_data and satellite_peak_data[k] is not None for k in required_keys):
            logger.warning("Superstructure periodicity parameter calculation failed: Incomplete peak data.")
            return None

        main_corr_px = main_peak_data['corrected']
        sat_corr_px = satellite_peak_data['corrected']

        # Calculate difference vector
        delta_g_vec_ideal_px = (sat_corr_px[0] - main_corr_px[0], sat_corr_px[1] - main_corr_px[1])
        dist_fft_px = np.linalg.norm(delta_g_vec_ideal_px)
        
        # Convert and calculate distance
        fft_rows_ky, fft_cols_kx = fft_shape
        delta_g_vec_nm_inv = convert_g_vector_px_to_nm_inv(delta_g_vec_ideal_px, lx_nm, ly_nm, fft_cols_kx, fft_rows_ky)
        if delta_g_vec_nm_inv is None: raise ValueError("k-space conversion failed for delta_g.")
        dist_nm_inv = np.linalg.norm(delta_g_vec_nm_inv)
        periodicity_nm = 1.0 / dist_nm_inv if dist_nm_inv > 1e-9 else float('inf')
        
        # Calculate ratios
        intensity_ratio = satellite_peak_data['intensity'] / main_peak_data['intensity'] if main_peak_data['intensity'] > 1e-9 else float('inf')
        amplitude_ratio = satellite_peak_data['amplitude'] / main_peak_data['amplitude'] if main_peak_data['amplitude'] > 1e-9 else float('inf')
        max_value_ratio = satellite_peak_data['max_value'] / main_peak_data['max_value'] if main_peak_data['max_value'] > 1e-9 else float('inf')

        return {
            "dist_px": dist_fft_px,
            "dist_nm_inv": dist_nm_inv,
            "periodicity_nm": periodicity_nm,
            "intensity_ratio": intensity_ratio,
            "amplitude_ratio": amplitude_ratio,
            "max_value_ratio": max_value_ratio
        }
    except Exception as e:
        logger.error(f"Error in calculate_superstructure_periodicity_parameters: {e}")
        return None

# Backward compatibility alias; remove once legacy code is updated.
calculate_domain_wall_parameters = calculate_superstructure_periodicity_parameters
    
def create_ase_supercell_from_2d_vectors(
    a1_vec_nm: np.ndarray, 
    a2_vec_nm: np.ndarray, 
    atom_symbol: str = 'Au', 
    size: Tuple[int, int] = (21, 21),
    offset_fractional: Tuple[float, float] = (0.0, 0.0),
    z_height_nm: float = 1.0
) -> Optional[Atoms]:
    """
    Creates a 3D ASE Atoms object representing a 2D surface supercell.

    Args:
        a1_vec_nm (np.ndarray): 2D real-space lattice vector a1 in nm.
        a2_vec_nm (np.ndarray): 2D real-space lattice vector a2 in nm.
        atom_symbol (str): Chemical symbol of the atom for the primitive cell.
        size (Tuple[int, int]): The (N, M) size of the supercell to create.

    Returns:
        Optional[ase.Atoms]: The resulting ASE Atoms object for the supercell,
                             or None on error.
    """
    if a1_vec_nm is None or a2_vec_nm is None:
        return None

    try:
        # Lift 2D vectors into 3D by adding a zero z-component
        a1_3d = np.append(a1_vec_nm, 0)
        a2_3d = np.append(a2_vec_nm, 0)

        # Define a third vector perpendicular to the plane to introduce vacuum (2 nm ~ 20 Angstrom)
        a3_3d = np.array([0, 0, 2.0])

        # Assemble the unit cell matrix
        cell_3d = np.array([a1_3d, a2_3d, a3_3d])

        cell_height_nm = cell_3d[2, 2]
        z_fractional = z_height_nm / cell_height_nm

        # Create a primitive cell with a single atom placed mid-vacuum (z=0.5)
        primitive_cell = Atoms(
            symbols=[atom_symbol],
            scaled_positions=[(offset_fractional[0], offset_fractional[1], z_fractional)],
            cell=cell_3d,
            pbc=[True, True, False]  # Periodic only within the XY plane
        )

        # Build the supercell by repeating the primitive cell
        # Transformation matrix P for the supercell replication
        P = np.array([[size[0], 0, 0],
                      [0, size[1], 0],
                      [0, 0, 1]])
        supercell = make_supercell(primitive_cell, P)
        return supercell
    except Exception as e:
        logger.error(f"Failed to create ASE structure: {e}")
        return None
