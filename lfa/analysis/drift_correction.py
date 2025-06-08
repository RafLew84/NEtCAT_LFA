# lfa/analysis/drift_correction.py
"""
Functions for calculating and analyzing affine transformations
to correct for drift and rotation in STM/FFT data.
Includes a function to match measured points to ideal points
using the Hungarian algorithm before fitting the transformation.
"""
import logging
import numpy as np
from scipy.linalg import polar
from scipy.optimize import linear_sum_assignment
from typing import List, Tuple, Dict, Optional, Any

logger = logging.getLogger(__name__)

# --- Podstawowe Funkcje Transformacji Afinicznej (dostarczone wcześniej) ---

def fit_affine_measured_to_ideal(
    measured_pts: np.ndarray,
    ideal_pts: np.ndarray
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    """
    Solve for transformation (F, t) such that: ideal_pts approx = measured_pts @ F.T + t.
    This means F transforms vectors from the 'measured' coordinate system
    to the 'ideal' coordinate system.

    Args:
        measured_pts (np.ndarray): Nx2 array of measured points (x, y).
        ideal_pts (np.ndarray): Nx2 array of corresponding ideal points (x, y).

    Returns:
        Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
            F (np.ndarray): 2x2 transformation matrix, or None on error.
            t (np.ndarray): 2x1 translation vector, or None on error.
    """
    measured_pts = np.asarray(measured_pts, dtype=float)
    ideal_pts = np.asarray(ideal_pts, dtype=float)

    if measured_pts.shape[0] != ideal_pts.shape[0]:
        logger.error("Mismatch in the number of measured and ideal points.")
        return None, None
    if measured_pts.shape[1] != 2 or ideal_pts.shape[1] != 2:
        logger.error("Points must be 2D (Nx2 array).")
        return None, None
    if measured_pts.shape[0] < 3:
        logger.error("At least 3 point pairs are required for a unique affine fit.")
        # lstsq might still return something for <3 points, but it's not a full affine fit
        return None, None

    # Augment measured_pts with a column of ones for the translation part
    A = np.hstack([measured_pts, np.ones((len(ideal_pts), 1))])

    try:
        M, _, rank, _ = np.linalg.lstsq(A, ideal_pts, rcond=None)

        if rank < A.shape[1]: # Check if the system was underdetermined
             logger.warning(f"Rank deficiency in lstsq fit (rank {rank} < {A.shape[1]}). "
                            "Points might be collinear or too few unique points.")

        F_matrix = M[:2, :].T 
        t_vector = M[2, :]   

        logger.info(f"Affine fit successful. F = \n{F_matrix}\n t = {t_vector}")
        return F_matrix, t_vector

    except np.linalg.LinAlgError as e:
        logger.error(f"Linear algebra error during affine fit: {e}")
        return None, None
    except Exception as e:
        logger.exception(f"Unexpected error in fit_affine_measured_to_ideal: {e}")
        return None, None

def apply_affine_transform(
    points_to_transform: np.ndarray,
    F: np.ndarray,
    t: np.ndarray
) -> Optional[np.ndarray]:
    """
    Applies the affine transformation (F, t) to a set of points.
    Formula: transformed_pts = points_to_transform @ F.T + t

    Args:
        points_to_transform (np.ndarray): Nx2 array of points (x, y) to transform.
        F (np.ndarray): 2x2 transformation matrix.
        t (np.ndarray): 2x1 (or 1x2) translation vector.

    Returns:
        Optional[np.ndarray]: Nx2 array of transformed points, or None on error.
    """
    points_to_transform = np.asarray(points_to_transform, dtype=float)
    F = np.asarray(F, dtype=float)
    t = np.asarray(t, dtype=float).reshape(1, 2) # Ensure t is a row vector for broadcasting

    if points_to_transform.ndim != 2 or points_to_transform.shape[1] != 2:
        logger.error("Points to transform must be an Nx2 array.")
        return None
    if F.shape != (2, 2):
        logger.error("Transformation matrix F must be 2x2.")
        return None
    if t.shape != (1, 2):
        logger.error("Translation vector t must be 2x1 or 1x2 (will be reshaped).")
        return None

    try:
        transformed_points = points_to_transform @ F.T + t
        return transformed_points
    except Exception as e:
        logger.exception(f"Error applying affine transform: {e}")
        return None

def analyze_affine_transform(F: np.ndarray) -> Optional[Dict[str, Any]]:
    """
    Analyze a 2x2 affine transformation matrix F.
    Performs polar decomposition F = R @ U, where R is rotation and U is stretch.

    Args:
        F (np.ndarray): 2x2 transformation matrix.

    Returns:
        Optional[Dict[str, Any]]: A dictionary containing:
            'rotation_matrix' (np.ndarray): R
            'stretch_matrix' (np.ndarray): U (symmetric positive-definite)
            'linearized_strain' (np.ndarray): 0.5 * (F + F.T) - I for small deformation approximation
            'rotation_angle_deg' (float): Rotation angle in degrees
            'eigenvalues_stretch' (np.ndarray): Eigenvalues of U (principal stretches)
            'eigenvectors_stretch' (np.ndarray): Eigenvectors of U (principal stretch directions)
            or None if analysis fails.
    """
    F = np.asarray(F, dtype=float)
    if F.shape != (2, 2):
        logger.error("Transformation matrix F must be 2x2 for analysis.")
        return None

    try:
        # Polar decomposition: F = R @ U
        # R is unitary (orthogonal for real matrices), U is positive semi-definite Hermitian (symmetric for real F)
        R, U = polar(F) # R: rotation, U: right stretch matrix (symmetric positive-definite)

        # Linearized strain tensor (small deformation theory approximation)
        # strain_tensor_linearized = 0.5 * (F + F.T) - np.eye(2)
        # The user's original code might have a slight variation or convention, let's re-check
        # Original: strain_lin = 0.5*(F + F.T) - I. This is a common definition.
        # F.dot(F.T) gives left Cauchy-Green tensor C. U.dot(U) also gives C if F=RU.
        # U is the right stretch tensor sqrt(F.T @ F).
        # For small strains, U approx I + epsilon, where epsilon is small strain tensor.
        # F = R @ U.
        # If using Green-Lagrange strain E = 0.5 * (F.T @ F - I) = 0.5 * (U @ U - I)
        # The user's code uses: linearized_strain = 0.5*(F + F.T) - I. This is different from above.
        # Let's stick to the user's definition for now.
        linearized_strain = 0.5 * (F + F.T) - np.eye(2)


        # Rotation angle from rotation matrix R
        # R = [[cos(O), -sin(O)], [sin(O), cos(O)]]
        # So, R[0,0] = cos(O), R[1,0] = sin(O)
        cos_O = R[0, 0]
        sin_O = R[1, 0]
        rotation_angle_rad = np.arctan2(sin_O, cos_O)
        rotation_angle_deg = np.degrees(rotation_angle_rad)

        # Principal stretches and directions (from eigenvalues/vectors of U)
        eigenvalues_U, eigenvectors_U = np.linalg.eig(U)

        logger.info(f"Transform analysis: Angle={rotation_angle_deg:.2f} deg, Stretch evals={eigenvalues_U}")

        return {
            'rotation_matrix': R,
            'stretch_matrix': U,
            'linearized_strain': linearized_strain, # As per user's original code
            'rotation_angle_deg': rotation_angle_deg,
            'principal_stretches': eigenvalues_U, # Eigenvalues of U
            'principal_stretch_directions': eigenvectors_U # Columns are eigenvectors
        }
    except Exception as e:
        logger.exception(f"Error analyzing affine transform: {e}")
        return None

# --- Funkcja z Algorytmem Węgierskim ---

def match_and_fit_transform(
    measured_pts_px: np.ndarray,
    ideal_pts_pool_px: np.ndarray,
    num_expected_matches: int
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], Optional[Dict[str, Any]], Optional[List[Tuple[int, int]]]]:
    """
    Matches measured points to a pool of ideal points using the Hungarian algorithm,
    then fits an affine transformation.

    Args:
        measured_pts_px (np.ndarray): Nx2 array of measured points (pixels).
        ideal_pts_pool_px (np.ndarray): Mx2 array of ideal points (pixels), M >= N.
        num_expected_matches (int): The number of points (N) to match.

    Returns:
        Tuple containing:
            - F (Optional[np.ndarray]): 2x2 transformation matrix or None.
            - t (Optional[np.ndarray]): 2x1 translation vector or None.
            - analysis_results (Optional[Dict[str, Any]]): Dictionary from analyze_affine_transform or None.
            - point_pairs (Optional[List[Tuple[int, int]]]): List of (measured_idx, ideal_idx) pairs or None.
    """
    measured_pts_px = np.asarray(measured_pts_px, dtype=float)
    ideal_pts_pool_px = np.asarray(ideal_pts_pool_px, dtype=float)

    if measured_pts_px.shape[0] != num_expected_matches:
        logger.error(f"Number of measured points ({measured_pts_px.shape[0]}) "
                     f"does not match num_expected_matches ({num_expected_matches}).")
        return None, None, None, None

    if ideal_pts_pool_px.shape[0] < num_expected_matches:
        logger.error(f"Not enough ideal points in pool ({ideal_pts_pool_px.shape[0]}) "
                     f"to find {num_expected_matches} matches.")
        return None, None, None, None

    # Select target ideal points from the pool
    if ideal_pts_pool_px.shape[0] == num_expected_matches:
        target_ideal_pts_px = ideal_pts_pool_px
    else:
        # If more ideal points than measured, select N closest to the centroid of measured points
        if ideal_pts_pool_px.shape[0] > num_expected_matches:
             logger.warning(f"Ideal points pool ({ideal_pts_pool_px.shape[0]}) is larger than "
                            f"expected matches ({num_expected_matches}). Taking the first {num_expected_matches}.")
             target_ideal_pts_px = ideal_pts_pool_px[:num_expected_matches]
        else: # Should have been caught by earlier check
             target_ideal_pts_px = ideal_pts_pool_px

    # Build cost matrix (squared Euclidean distance for efficiency with linear_sum_assignment)
    cost_matrix = np.sum(
        (measured_pts_px[:, np.newaxis, :] - target_ideal_pts_px[np.newaxis, :, :]) ** 2,
        axis=2
    )

    # Apply Hungarian algorithm for optimal assignment
    try:
        measured_indices, ideal_indices = linear_sum_assignment(cost_matrix)
        min_cost = cost_matrix[measured_indices, ideal_indices].sum()
        logger.info(f"Hungarian assignment successful. Min cost: {min_cost:.2f}")
    except Exception as e:
        logger.exception(f"Error during Hungarian assignment: {e}")
        return None, None, None, None

    # Prepare ordered points for affine fit
    final_measured_pts = measured_pts_px[measured_indices]
    final_ideal_pts = target_ideal_pts_px[ideal_indices]
    
    point_pairs_indices = list(zip(measured_indices, ideal_indices))
    
    # Print matched point pairs with coordinates for debugging
    print("\nMatched point pairs (measured -> ideal):")
    for i, (m_idx, i_idx) in enumerate(point_pairs_indices):
        print(f"Pair {i+1}: ({final_measured_pts[m_idx][0]:.2f}, {final_measured_pts[m_idx][1]:.2f}) -> "
              f"({final_ideal_pts[i_idx][0]:.2f}, {final_ideal_pts[i_idx][1]:.2f})")

    # Fit affine transformation
    F, t = fit_affine_measured_to_ideal(final_measured_pts, final_ideal_pts)
    if F is None or t is None:
        logger.error("Affine fitting failed after Hungarian assignment.")
        return None, None, None, point_pairs_indices

    # Analyze transformation
    analysis_results = analyze_affine_transform(F)
    if analysis_results is None:
        logger.error("Analysis of the fitted transform failed.")
        return F, t, None, point_pairs_indices

    # Calculate RMSE for fit quality assessment
    try:
        predicted_ideal_from_measured = apply_affine_transform(final_measured_pts, F, t)
        if predicted_ideal_from_measured is not None:
            residuals = final_ideal_pts - predicted_ideal_from_measured
            rmse = np.sqrt(np.mean(np.sum(residuals**2, axis=1)))
            analysis_results['rmse'] = rmse
            logger.info(f"RMSE of fit: {rmse:.4f} pixels")
        else:
            analysis_results['rmse'] = None
    except Exception as e:
        logger.exception(f"Error calculating RMSE: {e}")
        analysis_results['rmse'] = None

    return F, t, analysis_results, point_pairs_indices