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
from typing import List, Tuple, Dict, Optional, Any, Sequence

from .uncertainty import propagate_linear

logger = logging.getLogger(__name__)

# --- Core Affine Transformation Utilities ---

def fit_affine_measured_to_ideal(
    measured_pts: np.ndarray,
    ideal_pts: np.ndarray
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], Optional[Dict[str, Any]]]:
    """
    Solve for transformation (F, t) such that: ideal_pts approx measured_pts @ F.T + t.

    Returns the fitted matrix, translation vector, and diagnostic information that
    includes residuals and an estimated covariance for the affine parameters.
    """
    measured_pts = np.asarray(measured_pts, dtype=float)
    ideal_pts = np.asarray(ideal_pts, dtype=float)

    if measured_pts.shape[0] != ideal_pts.shape[0]:
        logger.error("Mismatch in the number of measured and ideal points.")
        return None, None, None
    if measured_pts.shape[1] != 2 or ideal_pts.shape[1] != 2:
        logger.error("Points must be 2D (Nx2 array).")
        return None, None, None
    if measured_pts.shape[0] < 3:
        logger.error("At least 3 point pairs are required for a unique affine fit.")
        return None, None, None

    num_points = measured_pts.shape[0]
    # Build design matrix for linear least squares.
    # For each point we create two equations (x' and y').
    design = np.zeros((num_points * 2, 6), dtype=float)
    target = np.zeros(num_points * 2, dtype=float)
    for idx, ((mx, my), (ix, iy)) in enumerate(zip(measured_pts, ideal_pts)):
        row_x = 2 * idx
        row_y = row_x + 1
        # x' = F11 * mx + F12 * my + t1
        design[row_x, 0] = mx
        design[row_x, 1] = my
        design[row_x, 4] = 1.0
        target[row_x] = ix

        # y' = F21 * mx + F22 * my + t2
        design[row_y, 2] = mx
        design[row_y, 3] = my
        design[row_y, 5] = 1.0
        target[row_y] = iy

    try:
        params, residuals, rank, _ = np.linalg.lstsq(design, target, rcond=None)
    except np.linalg.LinAlgError as error:
        logger.error("Linear algebra error during affine fit: %s", error)
        return None, None, None
    except Exception as exc:  # pragma: no cover - defensive
        logger.exception("Unexpected error in fit_affine_measured_to_ideal: %s", exc)
        return None, None, None

    if rank < design.shape[1]:
        logger.warning(
            "Rank deficiency in affine fit (rank %s < %s). Points might be collinear or degenerate.",
            rank,
            design.shape[1],
        )

    F_matrix = np.array(
        [[params[0], params[1]], [params[2], params[3]]],
        dtype=float,
    )
    t_vector = np.array([params[4], params[5]], dtype=float)

    # Compute residual information and parameter covariance if possible.
    fitted = design @ params
    residual_vector = target - fitted
    rss = residuals[0] if residuals.size else float(np.dot(residual_vector, residual_vector))
    dof = (num_points * 2) - design.shape[1]
    sigma2 = rss / dof if dof > 0 else None

    param_covariance = None
    try:
        xtx_inv = np.linalg.inv(design.T @ design)
        if sigma2 is not None:
            param_covariance = sigma2 * xtx_inv
    except np.linalg.LinAlgError:
        logger.warning("Could not invert (X^T X) for parameter covariance estimation.")

    diagnostics: Dict[str, Any] = {
        "residuals_xy": residual_vector.reshape(num_points, 2),
        "rss": rss,
        "degrees_of_freedom": dof,
        "sigma2": sigma2,
        "rank": rank,
        "parameter_covariance": param_covariance,
    }

    logger.info("Affine fit successful. F = \n%s\n t = %s", F_matrix, t_vector)
    return F_matrix, t_vector, diagnostics

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

# --- Hungarian Matching Utility ---

def match_and_fit_transform(
    measured_pts_px: np.ndarray,
    ideal_pts_pool_px: np.ndarray,
    num_expected_matches: int,
    measured_covariances_px: Optional[Sequence[Optional[np.ndarray]]] = None,
) -> Tuple[
    Optional[np.ndarray],
    Optional[np.ndarray],
    Optional[Dict[str, Any]],
    Optional[List[Tuple[int, int]]],
    Optional[List[Optional[np.ndarray]]],
]:
    """
    Matches measured points to a pool of ideal points using the Hungarian algorithm,
    then fits an affine transformation and propagates measurement uncertainties.
    """
    measured_pts_px = np.asarray(measured_pts_px, dtype=float)
    ideal_pts_pool_px = np.asarray(ideal_pts_pool_px, dtype=float)

    if measured_pts_px.shape[0] != num_expected_matches:
        logger.error(
            "Number of measured points (%s) does not match expected matches (%s).",
            measured_pts_px.shape[0],
            num_expected_matches,
        )
        return None, None, None, None, None

    if ideal_pts_pool_px.shape[0] < num_expected_matches:
        logger.error(
            "Not enough ideal points (%s) to find %s matches.",
            ideal_pts_pool_px.shape[0],
            num_expected_matches,
        )
        return None, None, None, None, None

    # Normalise covariance inputs (if provided) before any reordering.
    ordered_input_covariances: Optional[List[Optional[np.ndarray]]] = None
    if measured_covariances_px is not None:
        ordered_input_covariances = []
        for cov in measured_covariances_px:
            if cov is None:
                ordered_input_covariances.append(None)
                continue
            cov_arr = np.asarray(cov, dtype=float)
            if cov_arr.shape != (2, 2):
                logger.warning("Measured covariance has invalid shape %s; ignoring entry.", cov_arr.shape)
                ordered_input_covariances.append(None)
            else:
                ordered_input_covariances.append(cov_arr)

    # Select target ideal points from the pool
    if ideal_pts_pool_px.shape[0] == num_expected_matches:
        target_ideal_pts_px = ideal_pts_pool_px
    elif ideal_pts_pool_px.shape[0] > num_expected_matches:
        logger.warning(
            "Ideal points pool (%s) is larger than expected matches (%s). Using first %s entries.",
            ideal_pts_pool_px.shape[0],
            num_expected_matches,
            num_expected_matches,
        )
        target_ideal_pts_px = ideal_pts_pool_px[:num_expected_matches]
    else:  # pragma: no cover - defensive
        target_ideal_pts_px = ideal_pts_pool_px

    cost_matrix = np.sum(
        (measured_pts_px[:, np.newaxis, :] - target_ideal_pts_px[np.newaxis, :, :]) ** 2,
        axis=2,
    )

    try:
        measured_indices, ideal_indices = linear_sum_assignment(cost_matrix)
        min_cost = cost_matrix[measured_indices, ideal_indices].sum()
        logger.info("Hungarian assignment successful. Min cost: %.2f", min_cost)
    except Exception as exc:  # pragma: no cover - defensive
        logger.exception("Error during Hungarian assignment: %s", exc)
        return None, None, None, None, None

    final_measured_pts = measured_pts_px[measured_indices]
    final_ideal_pts = target_ideal_pts_px[ideal_indices]
    point_pairs_indices = list(zip(measured_indices, ideal_indices))

    matched_covariances: Optional[List[Optional[np.ndarray]]] = None
    if ordered_input_covariances is not None:
        matched_covariances = []
        for idx in measured_indices:
            matched_covariances.append(
                ordered_input_covariances[idx] if idx < len(ordered_input_covariances) else None
            )

    F, t, fit_diagnostics = fit_affine_measured_to_ideal(final_measured_pts, final_ideal_pts)
    if F is None or t is None:
        logger.error("Affine fitting failed after Hungarian assignment.")
        return None, None, None, point_pairs_indices, matched_covariances

    analysis_results = analyze_affine_transform(F)
    if analysis_results is None:
        logger.error("Analysis of the fitted transform failed.")
        return F, t, None, point_pairs_indices, matched_covariances

    if fit_diagnostics is not None:
        analysis_results["fit_diagnostics"] = fit_diagnostics

        param_covariance = fit_diagnostics.get("parameter_covariance")
        if param_covariance is not None:
            try:
                param_covariance = np.asarray(param_covariance, dtype=float)
                if param_covariance.shape[0] >= 4 and param_covariance.shape[1] >= 4:
                    F_covariance = param_covariance[:4, :4]
                    F_vector = np.array([F[0, 0], F[0, 1], F[1, 0], F[1, 1]], dtype=float)

                    def _vec_to_matrix(vec: np.ndarray) -> np.ndarray:
                        return np.array([[vec[0], vec[1]], [vec[2], vec[3]]], dtype=float)

                    def _rotation_from_vec(vec: np.ndarray) -> np.ndarray:
                        F_mat = _vec_to_matrix(vec)
                        R, _ = polar(F_mat)
                        angle_deg = np.degrees(np.arctan2(R[1, 0], R[0, 0]))
                        return np.array([angle_deg], dtype=float)

                    rotation_propagation = propagate_linear(_rotation_from_vec, F_vector, F_covariance)
                    rotation_variance = float(rotation_propagation.covariance[0, 0])
                    if rotation_variance >= 0.0:
                        analysis_results["rotation_angle_deg_sigma"] = float(np.sqrt(rotation_variance))
                        analysis_results["rotation_angle_deg_covariance"] = rotation_propagation.covariance

                    def _stretches_from_vec(vec: np.ndarray) -> np.ndarray:
                        F_mat = _vec_to_matrix(vec)
                        _, U = polar(F_mat)
                        eigenvalues = np.linalg.eigvals(U)
                        eigenvalues = np.real_if_close(eigenvalues)
                        return np.array(eigenvalues, dtype=float)

                    stretches_propagation = propagate_linear(_stretches_from_vec, F_vector, F_covariance)
                    stretches_covariance = np.asarray(stretches_propagation.covariance, dtype=float)
                    if stretches_covariance.shape == (2, 2):
                        diag_entries = np.clip(np.diag(stretches_covariance), 0.0, None)
                        analysis_results["principal_stretches_sigma"] = (
                            float(np.sqrt(diag_entries[0])),
                            float(np.sqrt(diag_entries[1])),
                        )
                        analysis_results["principal_stretches_covariance"] = stretches_covariance
            except Exception as exc:  # pragma: no cover - defensive
                logger.debug("Failed to propagate affine transform uncertainties: %s", exc)

    try:
        predicted_ideal_from_measured = apply_affine_transform(final_measured_pts, F, t)
        if predicted_ideal_from_measured is not None:
            residuals = final_ideal_pts - predicted_ideal_from_measured
            rmse = np.sqrt(np.mean(np.sum(residuals ** 2, axis=1)))
            analysis_results["rmse"] = rmse
            logger.info("RMSE of fit: %.4f pixels", rmse)
        else:
            analysis_results["rmse"] = None
    except Exception as exc:  # pragma: no cover - defensive
        logger.exception("Error calculating RMSE: %s", exc)
        analysis_results["rmse"] = None

    transformed_covariances: Optional[List[Optional[np.ndarray]]] = None
    if matched_covariances is not None:
        transformed_covariances = []
        for cov in matched_covariances:
            if cov is None:
                transformed_covariances.append(None)
            else:
                try:
                    transformed_covariances.append(F @ cov @ F.T)
                except Exception as exc:  # pragma: no cover - defensive
                    logger.warning("Failed to propagate covariance through affine transform: %s", exc)
                    transformed_covariances.append(None)
        analysis_results["matched_measured_covariances_px"] = matched_covariances
        analysis_results["fitted_spot_covariances_px"] = transformed_covariances

    return F, t, analysis_results, point_pairs_indices, transformed_covariances
