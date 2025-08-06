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

LATTICE_TYPE_HEXAGONAL = "hexagonal"
LATTICE_TYPE_SQUARE = "square"
LATTICE_TYPE_UNKNOWN = "Unknown"

ADSORBATE_LATTICE_TYPE_UNKNOWN = "Unknown"
ADSORBATE_LATTICE_TYPE_HEXAGONAL = "Hexagonal"
ADSORBATE_LATTICE_TYPE_SQUARE = "Square"

logger = logging.getLogger(__name__)

# Lattice Definitions
# Store real-space lattice constant 'a' (nearest neighbor or conventional cell param) in nm.
# Store type ('hexagonal', 'square') to determine reciprocal lattice calculation.
KNOWN_LATTICES: Dict[str, Dict] = {
    "Au(111)": {
        "type": "hexagonal",
        "a_bulk": 0.408, # nm
        "a_surf": 0.408 / np.sqrt(2), # nm (~0.288)
        "source": "Approx. bulk value"
    },
    "Ag(111)": {
        "type": "hexagonal",
        "a_bulk": 0.409,
        "a_surf": 0.409 / np.sqrt(2), # ~0.289 nm
        "source": "Approx. bulk value"
    },
    "Cu(111)": {
        "type": "hexagonal",
        "a_bulk": 0.361,
        "a_surf": 0.361 / np.sqrt(2), # ~0.255 nm
        "source": "Approx. bulk value"
    },
     "Cu(100)": {
        "type": "square",
        "a_bulk": 0.361,
        "a_surf": 0.361 / np.sqrt(2), # ~0.255 nm (side length of surface unit cell)
        "source": "Approx. bulk value"
    },
     "Ag(100)": {
        "type": "square",
        "a_bulk": 0.409,
        "a_surf": 0.409 / np.sqrt(2), # ~0.289 nm
        "source": "Approx. bulk value"
    },
    "Graphene": {
        "type": "hexagonal",
        "a_surf": 0.246, # nm
        "source": "Typical value"
    },
    "HOPG": { # Often approximated as graphene for surface studies
        "type": "hexagonal",
        "a_surf": 0.246, # nm
        "source": "Typical value"
    },
    "Au(100)": {
        "type": "square",
        "a_bulk": 0.408,  # nm (stała sieciowa)
        "a_surf": 0.408 / np.sqrt(2),  # ~0.288 nm
        "source": "Approx. bulk value"
    },
    # Platinum (Pt) - FCC
    "Pt(111)": {
        "type": "hexagonal",
        "a_bulk": 0.392,  # nm
        "a_surf": 0.392 / np.sqrt(2),  # ~0.277 nm
        "source": "Approx. bulk value"
    },
    "Pt(100)": {
        "type": "square",
        "a_bulk": 0.392,
        "a_surf": 0.392 / np.sqrt(2),  # ~0.277 nm
        "source": "Approx. bulk value"
    },
    # Nickel (Ni) - FCC
    "Ni(111)": {
        "type": "hexagonal",
        "a_bulk": 0.352,  # nm
        "a_surf": 0.352 / np.sqrt(2),  # ~0.249 nm
        "source": "Approx. bulk value"
    },
    "Ni(100)": {
        "type": "square",
        "a_bulk": 0.352,
        "a_surf": 0.352 / np.sqrt(2),  # ~0.249 nm
        "source": "Approx. bulk value"
    },
    # Add more lattices here
}

# --- Reciprocal Lattice Calculation ---

def get_reciprocal_vectors(lattice_info: Dict) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    """
    Calculates the primary reciprocal lattice vectors b1*, b2* (without 2pi).
    
    Args:
        lattice_info (Dict): Dictionary containing lattice type and surface lattice constant.
        
    Returns:
        Optional[Tuple[np.ndarray, np.ndarray]]: Tuple of (b1*, b2*) vectors or None on error.
    """
    l_type = lattice_info.get("type")
    a = lattice_info.get("a_surf") # Use surface lattice constant

    if not l_type or not a:
        logger.error("Lattice info missing 'type' or 'a_surf'.")
        return None

    if l_type == "hexagonal":
        b_mag = (1.0 / a) * (2.0 / np.sqrt(3))
        # Define b1* along x-axis
        b1_star = np.array([b_mag, 0.0])
        # b2* is rotated by 60 degrees
        b2_star = np.array([b_mag * np.cos(np.pi / 3), b_mag * np.sin(np.pi / 3)])
        return b1_star, b2_star
    elif l_type == "square":
        b_mag = 1.0 / a
        b1_star = np.array([b_mag, 0.0])
        b2_star = np.array([0.0, b_mag])
        return b1_star, b2_star
    else:
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
    if not lattice_type: # pragma: no cover
        logger.error(f"get_nearest_reciprocal_points: Lattice type not specified in info for '{display_name}'.")
        return None

    candidate_points = get_reciprocal_points(lattice_info_dict, max_hk=1)

    if not candidate_points:
        logger.error(f"get_nearest_reciprocal_points: Could not generate candidate points for '{display_name}'.")
        return None

    sorted_points = sorted(candidate_points, key=lambda p: p[0]**2 + p[1]**2)

    num_points_to_return = 0
    if lattice_type == LATTICE_TYPE_HEXAGONAL:
        num_points_to_return = num_points_hex
    elif lattice_type == LATTICE_TYPE_SQUARE:
        num_points_to_return = num_points_square
    else: # pragma: no cover
        logger.warning(f"get_nearest_reciprocal_points: Unsupported lattice type '{lattice_type}' for specific point count. Returning all from max_hk=1.")
        return sorted_points[:min(len(sorted_points), 6)]


    if len(sorted_points) < num_points_to_return: # pragma: no cover
        logger.warning(f"get_nearest_reciprocal_points: Generated fewer candidate points ({len(sorted_points)}) "
                       f"than requested ({num_points_to_return}) for '{display_name}' with max_hk=1. "
                       f"Returning all found non-zero points.")
        return sorted_points
    
    logger.info(f"Returning {num_points_to_return} nearest reciprocal points for '{display_name}' ({lattice_type}).")
    return sorted_points[:num_points_to_return]

def select_reciprocal_lattice_basis_vectors(
    g_vectors_relative_px: List[Tuple[float, float]],
    lattice_type: str
) -> Optional[Tuple[Tuple[float, float], Tuple[float, float]]]:
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

    if lattice_type == LATTICE_TYPE_HEXAGONAL:
        if len(g_vectors_relative_px) < 6:
            logger.warning(f"Hexagonal lattice: expected 6 spots for reliable basis vector selection, got {len(g_vectors_relative_px)}.")
            if len(g_vectors_relative_px) >= 2: 
                 logger.warning("Fallback: Taking first two available spots for hexagonal basis, may be incorrect.")
                 return g_vectors_relative_px[0], g_vectors_relative_px[1]
            return None

        sorted_g_vecs = sorted(g_vectors_relative_px, key=lambda v: np.arctan2(v[1], v[0]))
        
        lengths = [np.sqrt(v[0]**2 + v[1]**2) for v in sorted_g_vecs[:6]]
        if not lengths or any(l < 1e-6 for l in lengths): 
            logger.warning("Hexagonal g-vectors have zero or near-zero length(s). Cannot determine basis.")
            return None
        avg_len_px = np.mean(lengths)

        g1_ref_dir = sorted_g_vecs[0]
        g1_ref_len = lengths[0]
        
        g1_px = (g1_ref_dir[0] * avg_len_px / g1_ref_len, 
                 g1_ref_dir[1] * avg_len_px / g1_ref_len)

        cos60, sin60 = np.cos(np.pi/3), np.sin(np.pi/3)
        g2_px = (g1_px[0] * cos60 - g1_px[1] * sin60,
                 g1_px[0] * sin60 + g1_px[1] * cos60)
        
        logger.info(f"Hexagonal basis: g1_px={g1_px}, g2_px={g2_px} (avg_len={avg_len_px:.2f}px)")
        return g1_px, g2_px

    elif lattice_type == LATTICE_TYPE_SQUARE:
        if len(g_vectors_relative_px) < 4:
            logger.warning(f"Square lattice: expected 4 spots for reliable basis vector selection, got {len(g_vectors_relative_px)}.")
            if len(g_vectors_relative_px) >= 2: # Prosty fallback
                 logger.warning("Fallback: Taking first two available spots for square basis, may be incorrect.")
                 # Sprawdź, czy są mniej więcej prostopadłe
                 v1 = np.array(g_vectors_relative_px[0]); v2 = np.array(g_vectors_relative_px[1])
                 norm_v1, norm_v2 = np.linalg.norm(v1), np.linalg.norm(v2)
                 if norm_v1 > 1e-6 and norm_v2 > 1e-6:
                    if abs(np.dot(v1,v2) / (norm_v1 * norm_v2)) < 0.5: # Kąt ~90deg (+-30)
                        return g_vectors_relative_px[0], g_vectors_relative_px[1]
            return None

        # Uśrednij długości pierwszych 4 wektorów
        lengths = [np.sqrt(v[0]**2 + v[1]**2) for v in g_vectors_relative_px[:4]]
        if not lengths or any(l < 1e-6 for l in lengths):
            logger.warning("Square g-vectors have zero or near-zero length(s). Cannot determine basis.")
            return None
        avg_len_px = np.mean(lengths)

        # Wybierz dwa ortogonalne wektory o uśrednionej długości.
        # Możemy przyjąć, że jeden jest wzdłuż osi kx, a drugi wzdłuż osi ky.
        # To jest uproszczenie; bardziej robustne byłoby znalezienie par i uśrednienie.
        # Załóżmy, że idealne piki są już w dobrej orientacji (np. z get_nearest_reciprocal_points)
        # i pierwsze dwa po sortowaniu (lub po odpowiednim filtrowaniu) są bazowe.
        
        # Spróbuj znaleźć wektor najbliższy osi kx i najbliższy osi ky
        g_vecs_np = np.array(g_vectors_relative_px[:4])
        
        # Wektor najbliższy osi kx (największa składowa x, mała y)
        # Sortuj po malejącej wartości abs(x) i rosnącej abs(y)
        # lub prostsze: wybierz ten z najmniejszym kątem do osi X
        sorted_by_angle_to_x = sorted(g_vecs_np, key=lambda v: abs(np.arctan2(v[1], v[0])))
        g1_candidate_dir = sorted_by_angle_to_x[0]
        g1_len = np.linalg.norm(g1_candidate_dir)
        if g1_len < 1e-6: logger.warning("g1 candidate zero length for square"); return None
        g1_px = tuple(g1_candidate_dir * avg_len_px / g1_len)

        # Znajdź wektor najbardziej prostopadły do g1_px
        g2_candidate_dir = None
        min_dot_product_abs = float('inf')
        for v_cand_np in g_vecs_np:
            if np.allclose(v_cand_np, g1_candidate_dir): continue # Pomiń ten sam wektor
            # Iloczyn skalarny bliski zeru oznacza prostopadłość
            dot_prod_abs = abs(np.dot(g1_px, v_cand_np))
            if dot_prod_abs < min_dot_product_abs:
                min_dot_product_abs = dot_prod_abs
                g2_candidate_dir = v_cand_np
        
        if g2_candidate_dir is None: logger.warning("Could not find a suitable g2 candidate for square."); return None
        
        g2_len = np.linalg.norm(g2_candidate_dir)
        if g2_len < 1e-6: logger.warning("g2 candidate zero length for square"); return None
        g2_px = tuple(g2_candidate_dir * avg_len_px / g2_len)
        
        # Sprawdź, czy są rzeczywiście (w przybliżeniu) prostopadłe
        cos_angle_g1_g2 = np.dot(g1_px, g2_px) / (np.linalg.norm(g1_px) * np.linalg.norm(g2_px))
        if abs(cos_angle_g1_g2) > 0.2: # Większe niż ~cos(80deg) lub cos(100deg) - mało prostopadłe
            logger.warning(f"Selected square basis vectors are not orthogonal enough (cos_angle={cos_angle_g1_g2:.2f}). Fallback.")
            # Fallback: utwórz g2 jako obrót g1 o 90 stopni
            g2_px = (-g1_px[1], g1_px[0]) # Obrót o +90 stopni

        logger.info(f"Square basis: g1_px={g1_px}, g2_px={g2_px} (avg_len={avg_len_px:.2f}px)")
        return g1_px, g2_px
    else:
        logger.error(f"Unsupported lattice type for basis vector selection: {lattice_type}")
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

def get_real_space_lattice_parameters(
    selected_g_vectors_relative_px: List[Tuple[float, float]],
    lattice_type: str,
    Lx_nm: float,                 # Real-space size corresponding to kx-direction (FFT columns)
    Ly_nm: float,                 # Real-space size corresponding to ky-direction (FFT rows)
    fft_shape_cols_kx: int,       # Total columns in FFT (N_kx)
    fft_shape_rows_ky: int        # Total rows in FFT (N_ky)
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

    # 1. Select basis g-vectors in pixels
    basis_g_px = select_reciprocal_lattice_basis_vectors(selected_g_vectors_relative_px, lattice_type)
    if basis_g_px is None:
        logger.error("Failed to select basis reciprocal vectors in pixels.")
        return None
    g1_s_px, g2_s_px = basis_g_px

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

    results = {
        "a1_nm": a1_s_mag_nm,
        "a2_nm": a2_s_mag_nm,
        "alpha_deg": alpha_s_deg,
        "a1_vec_nm": a1_s_vec_nm, # Tuple (ax, ay)
        "a2_vec_nm": a2_s_vec_nm, # Tuple (ax, ay)
        "g1_vec_px": g1_s_px,     # Tuple (gkx, gky) in pixels
        "g2_vec_px": g2_s_px,
        "g1_vec_nm_inv": g1_s_nm_inv, # Tuple (gkx, gky) in nm^-1
        "g2_vec_nm_inv": g2_s_nm_inv
    }
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

def calculate_domain_wall_parameters(
    main_peak_data: Dict[str, Any],
    satellite_peak_data: Dict[str, Any],
    fft_shape: Tuple[int, int],
    lx_nm: float,
    ly_nm: float
) -> Optional[Dict[str, float]]:
    """
    Calculates various domain wall parameters based on a main and satellite peak.

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
            logger.warning("Domain wall parameter calculation failed: Incomplete peak data.")
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
        logger.error(f"Error in calculate_domain_wall_parameters: {e}")
        return None
    
def create_ase_supercell_from_2d_vectors(
    a1_vec_nm: np.ndarray, 
    a2_vec_nm: np.ndarray, 
    atom_symbol: str = 'Au', 
    size: Tuple[int, int] = (21, 21)
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
        # Rozszerz wektory 2D do 3D, dodając 0 jako współrzędną z
        a1_3d = np.append(a1_vec_nm, 0)
        a2_3d = np.append(a2_vec_nm, 0)

        # Zdefiniuj trzeci wektor prostopadły, tworząc próżnię (2 nm = 20 Å)
        a3_3d = np.array([0, 0, 2.0])

        # Złóż macierz komórki elementarnej
        cell_3d = np.array([a1_3d, a2_3d, a3_3d])

        # Stwórz prymitywną komórkę z jednym atomem
        # Pozycja atomu jest w środku warstwy próżni (z=0.5)
        primitive_cell = Atoms(
            symbols=[atom_symbol],
            scaled_positions=[(0, 0, 0.5)],
            cell=cell_3d,
            pbc=[True, True, False]  # Okresowość tylko w płaszczyźnie XY
        )

        # Zbuduj superkomórkę, powielając komórkę prymitywną
        # Macierz transformacji P dla superkomórki
        P = np.array([[size[0], 0, 0],
                      [0, size[1], 0],
                      [0, 0, 1]])
        supercell = make_supercell(primitive_cell, P)
        return supercell
    except Exception as e:
        logger.error(f"Failed to create ASE structure: {e}")
        return None
