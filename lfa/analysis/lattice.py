# lfa/analysis/lattice.py
"""
Functions and data related to crystal lattices and reciprocal space.
"""
import logging
import numpy as np
from typing import Dict, Tuple, List, Optional, Union

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
