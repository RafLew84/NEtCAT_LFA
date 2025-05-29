# lfa/analysis/lattice.py
"""
Functions and data related to crystal lattices and reciprocal space.
"""
import logging
import numpy as np
from typing import Dict, Tuple, List, Optional, Union, Any

LATTICE_TYPE_HEXAGONAL = "hexagonal"
LATTICE_TYPE_SQUARE = "square"

LATTICE_TYPE_HEXAGONAL = "hexagonal"
LATTICE_TYPE_SQUARE = "square"
LATTICE_TYPE_UNKNOWN = "Unknown" # Dodajemy typ "Unknown"

ADSORBATE_LATTICE_TYPE_UNKNOWN = "Unknown"
ADSORBATE_LATTICE_TYPE_HEXAGONAL = "Hexagonal"
ADSORBATE_LATTICE_TYPE_SQUARE = "Square"


logger = logging.getLogger(__name__)

# --- Lattice Definitions ---
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
    # Add more lattices here
}

# --- Reciprocal Lattice Calculation ---

def get_reciprocal_vectors(lattice_info: Dict) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    """Calculates the primary reciprocal lattice vectors b1*, b2* (without 2pi)."""
    l_type = lattice_info.get("type")
    a = lattice_info.get("a_surf") # Use surface lattice constant

    if not l_type or not a:
        logger.error("Lattice info missing 'type' or 'a_surf'.")
        return None

    if l_type == "hexagonal":
        # b vectors have magnitude 1/d, where d is spacing between (10) lines = a*sqrt(3)/2
        # |b*| = 1/d = 2 / (a * sqrt(3)) - Correct magnitude WITHOUT 2pi factor
        # Or using formulas for reciprocal lattice vectors:
        # a1 = a * [1, 0], a2 = a * [np.cos(60), np.sin(60)]
        # area = |a1 x a2| = a^2 * sin(60) = a^2 * sqrt(3)/2
        # b1* = (1/area) * rot90(a2) = (2/(a^2*sqrt(3))) * a * [-sin(60), cos(60)]
        # b2* = (1/area) * rot90(-a1) = (2/(a^2*sqrt(3))) * rot90([-a, 0]) = (2/(a^2*sqrt(3))) * [0, -a]
        # |b1*| = (2/(a*sqrt(3))) * sqrt(sin^2+cos^2) = 2/(a*sqrt(3)) - Consistent
        # b1_star = (2 / (a * np.sqrt(3))) * np.array([-np.sin(np.pi/3), np.cos(np.pi/3)])
        # b2_star = (2 / (a * np.sqrt(3))) * np.array([np.sin(np.pi/3), np.cos(np.pi/3)]) # Check this
        # Let's use the magnitudes and angles approach, simpler for 2D
        b_mag = (1.0 / a) * (2.0 / np.sqrt(3))
        # Define b1* along x-axis
        b1_star = np.array([b_mag, 0.0])
        # b2* is rotated by 60 degrees
        b2_star = np.array([b_mag * np.cos(np.pi / 3), b_mag * np.sin(np.pi / 3)])
        return b1_star, b2_star
    elif l_type == "square":
        # a1 = a*[1,0], a2 = a*[0,1], area = a^2
        # b1* = rot90(a2)/area = [0, a]/a^2 = [0, 1/a]
        # b2* = rot90(-a1)/area = rot90([-a,0])/a^2 = [0, -a]/a^2 ?? Incorrect manual calc
        # Use standard definition |b*|=1/a, angle=90 deg
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
        display_name = lattice_info.get("name", "Custom") # Użyj nazwy z dict lub "Custom"
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
    num_points_hex: int = 6, # Domyślnie 6 dla heksagonalnej
    num_points_square: int = 4 # Domyślnie 4 dla kwadratowej
) -> Optional[List[Tuple[float, float]]]:
    """
    Generates a specific number of nearest non-zero reciprocal lattice points
    (G* = h*b1* + k*b2*) around the center (0,0).

    Args:
        lattice_name_or_info: Name of a known lattice or a lattice info dictionary.
        num_points_hex (int): Number of nearest points to return for a hexagonal lattice.
        num_points_square (int): Number of nearest points to return for a square lattice.

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

    # Wygeneruj wystarczająco dużą pulę punktów, aby mieć pewność, że znajdziemy najbliższe.
    # max_hk=1 da 8 punktów, max_hk=2 da 24 punkty. Dla 6 najbliższych, max_hk=1 wystarczy.
    # Dla bezpieczeństwa można użyć max_hk=2, jeśli struktura byłaby bardziej złożona.
    # Na razie załóżmy, że max_hk=1 jest wystarczające dla typowych sieci.
    # Jeśli chcemy być pewni, że mamy np. drugich najbliższych sąsiadów, trzeba by to dostosować.
    # Dla heksagonalnej, 6 najbliższych to (1,0) i jego symetryczne odpowiedniki.
    # Dla kwadratowej, 4 najbliższe to (1,0), (0,1) i ich symetryczne.
    # max_hk=1 powinno być wystarczające.
    candidate_points = get_reciprocal_points(lattice_info_dict, max_hk=1)

    if not candidate_points:
        logger.error(f"get_nearest_reciprocal_points: Could not generate candidate points for '{display_name}'.")
        return None

    # Posortuj punkty według odległości od (0,0) - czyli ich magnitudy
    # Magnitude kwadratowa jest wystarczająca do sortowania: Gx^2 + Gy^2
    sorted_points = sorted(candidate_points, key=lambda p: p[0]**2 + p[1]**2)

    num_points_to_return = 0
    if lattice_type == LATTICE_TYPE_HEXAGONAL:
        num_points_to_return = num_points_hex
    elif lattice_type == LATTICE_TYPE_SQUARE:
        num_points_to_return = num_points_square
    else: # pragma: no cover
        logger.warning(f"get_nearest_reciprocal_points: Unsupported lattice type '{lattice_type}' for specific point count. Returning all from max_hk=1.")
        # Można zwrócić błąd lub np. pierwszych N najbliższych, jeśli typ nieznany.
        # Na razie zwróćmy tyle, ile jest w sorted_points, ale nie więcej niż np. 6.
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
            # Można próbować wybrać z mniejszej liczby, ale będzie to mniej pewne.
            # Na razie, jeśli nie ma 6, zwróć None lub zaimplementuj logikę awaryjną.
            if len(g_vectors_relative_px) >= 2: # Prosty fallback, może być niepoprawny geometrycznie
                 logger.warning("Fallback: Taking first two available spots for hexagonal basis, may be incorrect.")
                 return g_vectors_relative_px[0], g_vectors_relative_px[1]
            return None

        # Sortuj wektory po kącie, aby uzyskać spójną kolejność
        # Kąt liczony od dodatniej osi X (kx) przeciwnie do ruchu wskazówek zegara
        sorted_g_vecs = sorted(g_vectors_relative_px, key=lambda v: np.arctan2(v[1], v[0]))
        
        # Uśrednij długości pierwszych 6 wektorów (powinny być podobne)
        lengths = [np.sqrt(v[0]**2 + v[1]**2) for v in sorted_g_vecs[:6]]
        if not lengths or any(l < 1e-6 for l in lengths): # Sprawdzenie czy długości są sensowne
            logger.warning("Hexagonal g-vectors have zero or near-zero length(s). Cannot determine basis.")
            return None
        avg_len_px = np.mean(lengths)

        # Wybierz pierwszy wektor z posortowanej listy jako referencję dla g1
        # Jego kierunek jest już ustalony przez sortowanie (najmniejszy kąt)
        g1_ref_dir = sorted_g_vecs[0]
        g1_ref_len = lengths[0] # Długość tego konkretnego wektora
        
        # Skaluj g1_ref_dir do średniej długości
        g1_px = (g1_ref_dir[0] * avg_len_px / g1_ref_len, 
                 g1_ref_dir[1] * avg_len_px / g1_ref_len)

        # Skonstruuj g2 obrócony o 60 stopni (pi/3 radiana) względem g1_px
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

    # Determinant D = g1x*g2y - g1y*g2x
    # This is equivalent to |g1* x g2*|_z / (2pi)^2 if 2pi was included,
    # or Area_reciprocal_cell / (2pi)^2.
    # Without 2pi, D is Area_reciprocal_cell.
    determinant = g1x * g2y - g1y * g2x

    if abs(determinant) < 1e-9: # Check for collinearity (determinant is zero)
        logger.error("Reciprocal vectors are collinear; cannot calculate real space vectors.")
        return None

    # Real space basis vectors (without 2pi factor in formulas):
    # a1 = (1/D) * [g2y, -g2x]
    # a2 = (1/D) * [-g1y, g1x]
    a1_nm = ( (1.0 / determinant) * g2y,
              (1.0 / determinant) * (-g2x) )
    
    a2_nm = ( (1.0 / determinant) * (-g1y),
              (1.0 / determinant) * g1x )
              
    return a1_nm, a2_nm

def get_real_space_lattice_parameters(
    # Lista 6 (hex) lub 4 (square) idealnych/skorygowanych pików substratu,
    # których współrzędne są WEKTORAMI od centrum FFT, w pikselach.
    # Te piki powinny już być wybrane i stanowić bazę do obliczeń.
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
    if not (Lx_nm > 0 and Ly_nm > 0 and fft_shape_cols_kx > 0 and fft_shape_rows_ky > 0): # pragma: no cover
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

    if g1_s_nm_inv is None or g2_s_nm_inv is None: # pragma: no cover
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

    if a1_s_mag_nm < 1e-9 or a2_s_mag_nm < 1e-9: # pragma: no cover
        logger.error("Calculated real space vectors have zero or near-zero magnitude.")
        return None
        
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
        "g1_vec_px": g1_s_px,     # Tuple (gkx, gky) w pikselach
        "g2_vec_px": g2_s_px,
        "g1_vec_nm_inv": g1_s_nm_inv, # Tuple (gkx, gky) w nm^-1
        "g2_vec_nm_inv": g2_s_nm_inv
    }
    logger.info(f"Calculated real space params: a1={a1_s_mag_nm:.4f}nm, a2={a2_s_mag_nm:.4f}nm, alpha={alpha_s_deg:.2f}deg")
    return results

def select_adsorbate_reciprocal_basis_vectors_px(
    corrected_g_vectors_relative_px: List[Tuple[float, float]], # Wektory g* adsorbatu od centrum idealnego systemu
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

    # --- Scenariusz 1: Użytkownik wybrał dokładnie 2 spoty (traktowane jako g1* i g2*) ---
    if num_spots_selected == 2:
        g1_px = corrected_g_vectors_relative_px[0]
        g2_px = corrected_g_vectors_relative_px[1]
        
        # Sprawdź, czy nie są zerowe lub współliniowe
        norm_g1 = np.linalg.norm(g1_px)
        norm_g2 = np.linalg.norm(g2_px)
        if norm_g1 < 1e-6 or norm_g2 < 1e-6:
            logger.warning("Adsorbate (2 spots): One or both selected g-vectors are zero length.")
            return None
        
        # Iloczyn wektorowy (składowa z) do sprawdzenia współliniowości
        cross_product_z = g1_px[0] * g2_px[1] - g1_px[1] * g2_px[0]
        if abs(cross_product_z) / (norm_g1 * norm_g2) < 1e-3: # Sprawdzenie sinusa kąta
            logger.warning("Adsorbate (2 spots): Selected g-vectors are collinear.")
            return None
            
        logger.info(f"Adsorbate (2 spots defined): Using g1*={g1_px}, g2*={g2_px}")
        return g1_px, g2_px

    # --- Scenariusz 2: Spodziewany typ "Hexagonal" ---
    elif expected_lattice_type == LATTICE_TYPE_HEXAGONAL.capitalize(): # Porównaj z wielką literą, jak w ComboBoxie
        if num_spots_selected < 2 : # Potrzebujemy co najmniej 2 do zdefiniowania, ideał to 6
             logger.warning(f"Hexagonal Adsorbate: Not enough spots (need >= 2, got {num_spots_selected}).")
             return None
        if num_spots_selected < 6:
            logger.warning(f"Hexagonal Adsorbate: {num_spots_selected} spots provided, less than ideal 6. "
                           "Attempting to find basis. Result may be less accurate.")
        
        # Sortuj wektory po długości (od najkrótszego) - zakładamy, że piki pierwszego rzędu są najsilniejsze/najbliższe
        # A następnie po kącie, aby uzyskać spójność
        sorted_g_vecs_by_len = sorted(corrected_g_vectors_relative_px, key=lambda v: np.linalg.norm(v))
        
        # Weź 'num_to_consider' najkrótszych wektorów (np. 6, jeśli dostępne, lub wszystkie)
        num_to_consider = min(num_spots_selected, 6)
        candidate_vecs = sorted_g_vecs_by_len[:num_to_consider]
        
        # Jeśli mamy 6 spotów, zastosuj metodę uśredniania dla anizotropii
        if num_spots_selected >= 6: # Używamy >= 6, aby obsłużyć też przypadek, gdy użytkownik kliknął więcej
            # Sortuj 6 najkrótszych po kącie
            sorted_angularly = sorted(candidate_vecs, key=lambda v: np.arctan2(v[1], v[0]))

            # Wektory osi (uśredniające przeciwległe pary)
            # Zakładamy, że sorted_angularly[0] i sorted_angularly[3] to para, itd.
            d1 = (np.array(sorted_angularly[0]) - np.array(sorted_angularly[3])) / 2.0
            d2 = (np.array(sorted_angularly[1]) - np.array(sorted_angularly[4])) / 2.0
            d3 = (np.array(sorted_angularly[2]) - np.array(sorted_angularly[5])) / 2.0
            
            # Lista wektorów osi, które nie są zerowe
            axis_vectors = [v for v in [d1, d2, d3] if np.linalg.norm(v) > 1e-6]
            if len(axis_vectors) < 2:
                logger.warning("Hexagonal Adsorbate (6 spots): Could not determine at least two distinct axes.")
                return None
            
            # Wybierz d1 jako g1_A* (pierwszy niezerowy wektor osi)
            g1_px = tuple(axis_vectors[0])
            
            # Wybierz jako g2_A* ten z pozostałych wektorów osi, który tworzy kąt najbliższy 60 stopni z g1_px
            best_g2_candidate = None
            min_angle_diff_to_60 = float('inf')
            target_angle_rad = np.pi / 3 # 60 stopni
            norm_g1 = np.linalg.norm(g1_px)

            for v_cand_np in axis_vectors[1:]: # Sprawdź pozostałe wektory osi
                norm_v_cand = np.linalg.norm(v_cand_np)
                if norm_v_cand < 1e-6: continue

                dot_product = np.dot(g1_px, v_cand_np)
                cos_theta = np.clip(dot_product / (norm_g1 * norm_v_cand), -1.0, 1.0)
                angle_rad = np.arccos(cos_theta)
                
                angle_diff = abs(angle_rad - target_angle_rad) # Różnica od 60 stopni
                # Można też sprawdzić różnicę od 120, jeśli to ma sens dla wyboru bazy
                # angle_diff_120 = abs(angle_rad - 2*target_angle_rad) 
                # current_min_diff = min(angle_diff, angle_diff_120)

                if angle_diff < min_angle_diff_to_60:
                    min_angle_diff_to_60 = angle_diff
                    best_g2_candidate = v_cand_np
            
            if best_g2_candidate is not None:
                g2_px = tuple(best_g2_candidate)
                logger.info(f"Adsorbate Hexagonal (6 spots aniso): Basis g1*={g1_px}, g2*={g2_px}")
                return g1_px, g2_px
            else: # pragma: no cover (powinno znaleźć, jeśli są co najmniej 2 osie)
                logger.warning("Adsorbate Hexagonal (6 spots aniso): Could not determine a suitable second basis vector from axes.")
                return None

        elif num_spots_selected >= 2: # Mniej niż 6, ale co najmniej 2 dla Hexagonal
            logger.warning("Hexagonal Adsorbate: Less than 6 spots.")
            # Użyj logiki dla "Unknown" - znajdź dwa najkrótsze, liniowo niezależne
            return None


    # --- Scenariusz 3: Spodziewany typ "Square" ---
    elif expected_lattice_type == LATTICE_TYPE_SQUARE.capitalize():
        if num_spots_selected < 2: # Potrzebujemy co najmniej 2, ideał to 4
            logger.warning(f"Square Adsorbate: Not enough spots (need >= 2, got {num_spots_selected}).")
            return None
        if num_spots_selected < 4:
            logger.warning(f"Square Adsorbate: {num_spots_selected} spots provided, less than ideal 4. "
                           "Attempting to find basis. Result may be less accurate.")

        sorted_g_vecs_by_len = sorted(corrected_g_vectors_relative_px, key=lambda v: np.linalg.norm(v))
        num_to_consider = min(num_spots_selected, 4)
        candidate_vecs = sorted_g_vecs_by_len[:num_to_consider]

        if num_spots_selected >= 4: # Używamy >= 4
            # Sortuj 4 najkrótsze po kącie
            sorted_angularly = sorted(candidate_vecs, key=lambda v: np.arctan2(v[1], v[0]))
            # Wektory osi
            d1 = (np.array(sorted_angularly[0]) - np.array(sorted_angularly[2])) / 2.0
            d2 = (np.array(sorted_angularly[1]) - np.array(sorted_angularly[3])) / 2.0

            norm_d1 = np.linalg.norm(d1); norm_d2 = np.linalg.norm(d2)
            if norm_d1 < 1e-6 or norm_d2 < 1e-6:
                logger.warning("Square Adsorbate (4 spots): Axis vectors have zero length.")
                return None

            # Sprawdź prostopadłość d1 i d2
            cos_angle_d1_d2 = np.dot(d1, d2) / (norm_d1 * norm_d2)
            if abs(cos_angle_d1_d2) > 0.2: # Kąt znacząco różny od 90 stopni (cos(90)=0, cos(78)~0.2)
                logger.warning(f"Square Adsorbate (4 spots): Axis vectors d1, d2 are not orthogonal (cos_angle={cos_angle_d1_d2:.2f}). "
                               "Consider re-selecting spots or using 'Unknown' type.")
                # Można spróbować wybrać inną parę lub zwrócić błąd, lub przejść do logiki "Unknown"
                # Na razie, jeśli nie są prostopadłe, użyjemy ich tak czy inaczej, aby opisać równoległobok
            
            g1_px = tuple(d1)
            g2_px = tuple(d2)
            logger.info(f"Adsorbate Square/Rect (4 spots): Basis g1*={g1_px}, g2*={g2_px}")
            return g1_px, g2_px

    # --- Scenariusz 4: Spodziewany typ "Unknown" (lub fallback z innych typów) ---
    elif expected_lattice_type == ADSORBATE_LATTICE_TYPE_UNKNOWN: # num_spots_selected >=2 jest już sprawdzane na początku
        # Znajdź dwa najkrótsze, liniowo niezależne wektory
        if len(corrected_g_vectors_relative_px) < 2: # pragma: no cover (już sprawdzane)
             return None 
        
        # Sortuj wszystkie dostępne wektory po długości
        sorted_g_vecs_by_len = sorted(corrected_g_vectors_relative_px, key=lambda v: np.linalg.norm(v))
        
        g1_px_cand = np.array(sorted_g_vecs_by_len[0])
        if np.linalg.norm(g1_px_cand) < 1e-6:
            logger.warning("Adsorbate (Unknown): Shortest g-vector is zero length.")
            return None

        g2_px_cand = None
        for i in range(1, len(sorted_g_vecs_by_len)):
            current_g2_cand = np.array(sorted_g_vecs_by_len[i])
            if np.linalg.norm(current_g2_cand) < 1e-6: continue # Pomiń wektory zerowe

            # Sprawdź współliniowość z g1_px_cand
            cross_product_z = g1_px_cand[0] * current_g2_cand[1] - g1_px_cand[1] * current_g2_cand[0]
            # Sprawdź sinus kąta, aby uniknąć problemów z normalizacją dla bardzo krótkich wektorów
            if abs(cross_product_z) / (np.linalg.norm(g1_px_cand) * np.linalg.norm(current_g2_cand) + 1e-9) > 1e-3: # Nie są współliniowe
                g2_px_cand = current_g2_cand
                break
        
        if g2_px_cand is None:
            logger.warning("Adsorbate (Unknown): Could not find two non-collinear g-vectors.")
            return None
            
        g1_px = tuple(g1_px_cand)
        g2_px = tuple(g2_px_cand)
        logger.info(f"Adsorbate (Unknown type, {num_spots_selected} spots): Selected basis g1*={g1_px}, g2*={g2_px}")
        return g1_px, g2_px

    else: # pragma: no cover (nie powinno się zdarzyć, jeśli typy są ograniczone w UI)
        logger.error(f"Unsupported expected_lattice_type '{expected_lattice_type}' or insufficient spots for selection.")
        return None
