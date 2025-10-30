from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from ....analysis.lattice import get_nearest_reciprocal_points
from ....analysis.drift_correction import (
    match_and_fit_transform,
    apply_affine_transform,
)
from ....core.constants import (
    LATTICE_TYPE_CUSTOM,
    LATTICE_TYPE_HEXAGONAL,
    LATTICE_TYPE_SQUARE,
)
from ....logic.history_manager import HistoryManager

logger = logging.getLogger(__name__)


@dataclass
class SubstrateSpotState:
    selected_spots: List[Tuple[float, float]]
    selected_spot_covariances: List[Optional[np.ndarray]]
    lattice_type: str
    selected_definition: Optional[str]
    custom_definition: Optional[Dict[str, Any]]
    custom_a_surf: Optional[float]
    transform_matrix_F: Optional[np.ndarray]
    transform_translation_t: Optional[np.ndarray]
    transform_analysis: Optional[Dict[str, Any]]
    fitted_spots_px: List[Tuple[float, float]]
    fitted_spot_covariances: List[Optional[np.ndarray]] = field(default_factory=list)
    ideal_spots_px_for_reference: List[Tuple[float, float]] = field(default_factory=list)


@dataclass
class TransformComputation:
    matrix_F: np.ndarray
    translation_t: np.ndarray
    analysis: Dict[str, Any]
    fitted_spots_px: List[Tuple[float, float]]
    measured_spots_px: np.ndarray
    ideal_spots_px: np.ndarray
    matched_pairs: Sequence[Tuple[int, int]]


class TransformComputationError(Exception):
    def __init__(self, user_message: str, status_message: str, severity: str = "warning") -> None:
        super().__init__(user_message)
        self.user_message = user_message
        self.status_message = status_message
        self.severity = severity  # "warning" | "critical"


class SubstrateSpotPresenter:
    """
    Encapsulates non-UI logic for the substrate spot dialog.

    Public API
    ----------
    state : SubstrateSpotState
        Mutable container holding dialog data (spots, lattice, transforms).
    build_lattice_info_dict(preferred_point_count) -> dict | None
        Prepare lattice info for calculations.
    calculate_transform(preferred_point_count) -> TransformComputation
        Run affine transform fitting; updates internal state.
    build_results_dict() -> dict
        Create payload for controller consumption.
    """

    def __init__(
        self,
        *,
        history_manager: HistoryManager,
        fft_node_id: str,
        fft_data: Optional[np.ndarray],
        state: SubstrateSpotState,
    ) -> None:
        self._history = history_manager
        self._fft_node_id = fft_node_id
        self._fft_data = fft_data
        self.state = state

    # ------------------------------------------------------------------ Lattice helpers
    def build_lattice_info_dict(self, *, preferred_point_count: Optional[int] = None) -> Optional[Dict[str, Any]]:
        state = self.state
        if state.lattice_type == LATTICE_TYPE_CUSTOM:
            if not state.custom_definition:
                return None
            info = dict(state.custom_definition)
            if preferred_point_count and preferred_point_count > 0:
                info["preferred_point_count"] = preferred_point_count
            return info

        if (
            state.lattice_type in (LATTICE_TYPE_HEXAGONAL, LATTICE_TYPE_SQUARE)
            and state.custom_a_surf
            and state.custom_a_surf > 0
        ):
            info: Dict[str, Any] = {"type": state.lattice_type, "a_surf": state.custom_a_surf}
            if preferred_point_count and preferred_point_count > 0:
                info["preferred_point_count"] = preferred_point_count
            return info

        return None

    # ------------------------------------------------------------------ Transform computation
    def calculate_transform(self, *, preferred_point_count: int) -> TransformComputation:
        spots = self.state.selected_spots
        if len(spots) < 3:
            raise TransformComputationError(
                "Please select at least 3 substrate spots for transformation.",
                "Select >= 3 spots.",
                severity="warning",
            )

        lattice_info = self.build_lattice_info_dict(preferred_point_count=preferred_point_count)
        if lattice_info is None:
            raise TransformComputationError(
                "Please define lattice parameters first.",
                "Define lattice first.",
                severity="warning",
            )

        if self._fft_data is None:
            raise TransformComputationError(
                "FFT data is not available.",
                "Error: No FFT data.",
                severity="critical",
            )

        root_node = self._history.get_root_node_for_node(self._fft_node_id)
        if not (root_node and root_node.operation_name == "Original" and root_node.parameters):
            raise TransformComputationError(
                "Could not retrieve original image calibration data.",
                "Error: No calibration.",
                severity="critical",
            )

        Lx_nm = root_node.parameters.get("size_nm_x")
        Ly_nm = root_node.parameters.get("size_nm_y")
        if not (Lx_nm and Ly_nm and Lx_nm > 0 and Ly_nm > 0):
            raise TransformComputationError(
                "Missing or invalid calibration data (Lx/Ly).",
                "Error: Invalid Lx/Ly.",
                severity="critical",
            )

        measured_spots_px = np.array(spots, dtype=float)
        fft_rows_ky, fft_cols_kx = self._fft_data.shape

        ideal_points_g_nm_inv = get_nearest_reciprocal_points(lattice_info)
        if not ideal_points_g_nm_inv:
            raise TransformComputationError(
                "Could not generate ideal reciprocal points.",
                "Error: No ideal pts.",
                severity="critical",
            )

        center_kx_px = fft_cols_kx / 2.0
        center_ky_px = fft_rows_ky / 2.0
        ideal_spots_pool_px = np.array(
            [
                (center_ky_px + (Gy * Ly_nm), center_kx_px + (Gx * Lx_nm))
                for Gx, Gy in ideal_points_g_nm_inv
            ],
            dtype=float,
        )

        num_expected = len(measured_spots_px)
        if ideal_spots_pool_px.shape[0] < num_expected:
            raise TransformComputationError(
                f"Not enough ideal points generated ({ideal_spots_pool_px.shape[0]}) to match {num_expected} measured spots.",
                "Error: Ideal pts pool too small.",
                severity="warning",
            )

        try:
            F, t, analysis, point_pairs = match_and_fit_transform(
                measured_pts_px=measured_spots_px,
                ideal_pts_pool_px=ideal_spots_pool_px,
                num_expected_matches=num_expected,
            )
        except ImportError as exc:
            raise TransformComputationError(
                "Drift correction module not available.",
                "Error: Module missing.",
                severity="critical",
            ) from exc
        except np.linalg.LinAlgError as exc:
            raise TransformComputationError(
                "Linear algebra error while computing transform.",
                "LinAlg Error.",
                severity="warning",
            ) from exc

        if F is None or t is None or analysis is None or point_pairs is None:
            raise TransformComputationError(
                "Failed to calculate affine transformation. Points might be degenerate or too few unique matches.",
                "Transform failed.",
                severity="warning",
            )

        matched_measured = measured_spots_px[[idx for idx, _ in point_pairs]]
        matched_ideal = ideal_spots_pool_px[[idx for _, idx in point_pairs]]
        transformed_measured = apply_affine_transform(matched_measured, F, t)
        fitted_spots = [tuple(pt) for pt in transformed_measured] if transformed_measured is not None else []

        # Update internal state
        self.state.transform_matrix_F = F
        self.state.transform_translation_t = t
        self.state.transform_analysis = analysis
        self.state.fitted_spots_px = fitted_spots
        self.state.ideal_spots_px_for_reference = [tuple(pt) for pt in matched_ideal]

        logger.debug("SubstrateSpotPresenter: Transform computed with RMSE=%s", analysis.get("rmse"))

        return TransformComputation(
            matrix_F=F,
            translation_t=t,
            analysis=analysis,
            fitted_spots_px=fitted_spots,
            measured_spots_px=matched_measured,
            ideal_spots_px=matched_ideal,
            matched_pairs=point_pairs,
        )

    # ------------------------------------------------------------------ Result helpers
    def build_results_dict(self) -> Dict[str, Any]:
        state = self.state
        selected_spots = list(state.selected_spots)
        selected_covariances = list(state.selected_spot_covariances)
        if len(selected_covariances) < len(selected_spots):
            selected_covariances.extend([None] * (len(selected_spots) - len(selected_covariances)))
        elif len(selected_covariances) > len(selected_spots):
            selected_covariances = selected_covariances[: len(selected_spots)]

        fitted_spots = list(state.fitted_spots_px)
        fitted_covariances = list(state.fitted_spot_covariances)
        if len(fitted_covariances) < len(fitted_spots):
            fitted_covariances.extend([None] * (len(fitted_spots) - len(fitted_covariances)))
        elif len(fitted_covariances) > len(fitted_spots):
            fitted_covariances = fitted_covariances[: len(fitted_spots)]

        return {
            "spots": selected_spots,
            "lattice_type": state.lattice_type,
            "a_surf": state.custom_a_surf,
            "substrate_definition": state.selected_definition,
            "custom_definition": dict(state.custom_definition) if state.custom_definition else None,
            "spot_covariances": [
                np.array(cov, dtype=float) if cov is not None else None
                for cov in selected_covariances
            ],
            "transformation_F_m2i": state.transform_matrix_F,
            "translation_t_m2i": state.transform_translation_t,
            "transform_analysis_m2i": state.transform_analysis,
            "displayable_fitted_spots": fitted_spots,
            "fitted_spot_covariances": [
                np.array(cov, dtype=float) if cov is not None else None
                for cov in fitted_covariances
            ],
            "ideal_substrate_spots_px_for_reference": list(state.ideal_spots_px_for_reference),
        }
