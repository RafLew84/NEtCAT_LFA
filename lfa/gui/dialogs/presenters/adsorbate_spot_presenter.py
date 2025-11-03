from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from ....analysis.drift_correction import apply_affine_transform
from ....core.constants import (
    ADSORBATE_LATTICE_TYPE_HEXAGONAL,
    ADSORBATE_LATTICE_TYPE_SQUARE,
    ADSORBATE_LATTICE_TYPE_UNKNOWN,
)

logger = logging.getLogger(__name__)


def _coerce_pair(point: Any) -> Tuple[float, float]:
    """
    Accept tuples, numpy arrays, QPointF, etc. and normalise to float pairs.
    Keeps dialog/presenter boundary clean.
    """
    if isinstance(point, np.ndarray):
        if point.ndim == 1 and point.shape[0] >= 2:
            return float(point[0]), float(point[1])
        raise TypeError("Expected 1D numpy array with at least two entries.")

    try:
        x, y = point  # tuple/list-like unpack
    except TypeError:
        if hasattr(point, "x") and hasattr(point, "y"):
            return float(point.x()), float(point.y())
        raise

    return float(x), float(y)


@dataclass
class AdsorbateSpotState:
    """
    Mutable bag-of-data owned by the presenter; the dialog reads from here.

    Attributes
    ----------
    set_index:
        Adsorbate set index that the dialog edits.
    raw_spots:
        Spots selected on FFT (measurement frame).
    corrected_spots:
        Spots transformed into the ideal lattice frame.
    expected_type:
        User-selected lattice type hint for downstream analysis.
    substrate_matrix_F:
        Affine transform matrix (M -> I) provided by substrate fit.
    substrate_translation_t:
        Affine translation vector (M -> I).
    substrate_analysis:
        Metadata dictionary describing transform quality (RMSE, rotation, ...).
    ideal_reference_spots_px / fitted_reference_spots_px:
        Optional overlays coming from substrate dialog.
    """

    set_index: int
    raw_spots: List[Tuple[float, float]] = field(default_factory=list)
    raw_spot_covariances: List[Optional[np.ndarray]] = field(default_factory=list)
    corrected_spots: List[Tuple[float, float]] = field(default_factory=list)
    corrected_spot_covariances: List[Optional[np.ndarray]] = field(default_factory=list)
    expected_type: str = ADSORBATE_LATTICE_TYPE_UNKNOWN
    substrate_matrix_F: Optional[np.ndarray] = None
    substrate_translation_t: Optional[np.ndarray] = None
    substrate_analysis: Optional[Dict[str, Any]] = None
    ideal_reference_spots_px: List[Tuple[float, float]] = field(default_factory=list)
    fitted_reference_spots_px: List[Tuple[float, float]] = field(default_factory=list)


class AdsorbateSpotPresenterError(RuntimeError):
    """Base class for presenter-originated errors intended for the dialog."""


class MissingTransformError(AdsorbateSpotPresenterError):
    """Raised when affine transform data is not available for corrections."""


class AdsorbateSpotPresenter:
    """
    Business logic for the adsorbate spot dialog.

    Keeps mutation away from Qt widget code, easing unit testing.
    """

    VALID_EXPECTED_TYPES: Sequence[str] = (
        ADSORBATE_LATTICE_TYPE_UNKNOWN,
        ADSORBATE_LATTICE_TYPE_HEXAGONAL,
        ADSORBATE_LATTICE_TYPE_SQUARE,
    )

    def __init__(self, *, state: AdsorbateSpotState) -> None:
        self.state = state

    # ------------------------------------------------------------------ State mutation helpers
    def add_raw_spot(
        self,
        spot: Tuple[float, float],
        covariance: Optional[np.ndarray] = None,
    ) -> bool:
        """Append a raw spot if not already present."""
        spot = _coerce_pair(spot)
        if spot in self.state.raw_spots:
            index = self.state.raw_spots.index(spot)
            if covariance is not None and 0 <= index < len(self.state.raw_spot_covariances):
                self.state.raw_spot_covariances[index] = np.array(covariance, dtype=float)
            return False
        self.state.raw_spots.append(spot)
        cov_value = np.array(covariance, dtype=float) if covariance is not None else None
        self.state.raw_spot_covariances.append(cov_value)
        self.state.corrected_spot_covariances.append(None)
        logger.debug("AdsorbatePresenter: added raw spot %s", spot)
        return True

    def remove_raw_spot(self, index: int) -> bool:
        if 0 <= index < len(self.state.raw_spots):
            removed = self.state.raw_spots.pop(index)
            logger.debug("AdsorbatePresenter: removed raw spot %s at index %s", removed, index)
            if 0 <= index < len(self.state.raw_spot_covariances):
                self.state.raw_spot_covariances.pop(index)
            if self.state.corrected_spots:
                try:
                    self.state.corrected_spots.pop(index)
                    if 0 <= index < len(self.state.corrected_spot_covariances):
                        self.state.corrected_spot_covariances.pop(index)
                except IndexError:
                    pass
            return True
        return False

    def clear_raw_spots(self) -> None:
        logger.debug("AdsorbatePresenter: clearing raw & corrected spots.")
        self.state.raw_spots.clear()
        self.state.corrected_spots.clear()
        self.state.raw_spot_covariances.clear()
        self.state.corrected_spot_covariances.clear()

    def set_expected_type(self, lattice_type: str) -> None:
        if lattice_type not in self.VALID_EXPECTED_TYPES:
            raise ValueError(f"Invalid expected lattice type '{lattice_type}'.")
        self.state.expected_type = lattice_type
        logger.debug("AdsorbatePresenter: expected lattice type set to %s", lattice_type)

    def update_reference_spots(
        self,
        *,
        ideal_spots_px: Optional[Sequence[Tuple[float, float]]] = None,
        fitted_spots_px: Optional[Sequence[Tuple[float, float]]] = None,
    ) -> None:
        if ideal_spots_px is not None:
            self.state.ideal_reference_spots_px = [tuple(map(float, pt)) for pt in ideal_spots_px]
        if fitted_spots_px is not None:
            self.state.fitted_reference_spots_px = [tuple(map(float, pt)) for pt in fitted_spots_px]

    def update_substrate_transform(
        self,
        *,
        matrix_F: Optional[np.ndarray],
        translation_t: Optional[np.ndarray],
        analysis: Optional[Dict[str, Any]],
    ) -> None:
        self.state.substrate_matrix_F = matrix_F
        self.state.substrate_translation_t = translation_t
        self.state.substrate_analysis = analysis

    # ------------------------------------------------------------------ Affine correction
    def can_apply_correction(self) -> bool:
        return bool(
            self.state.raw_spots
            and self.state.substrate_matrix_F is not None
            and self.state.substrate_translation_t is not None
        )

    def apply_substrate_correction(self) -> List[Tuple[float, float]]:
        if not self.can_apply_correction():
            raise MissingTransformError(
                "Substrate transformation is not available or no adsorbate spots selected."
            )

        raw_np = np.array(self.state.raw_spots, dtype=float)
        corrected_np = apply_affine_transform(
            raw_np,
            self.state.substrate_matrix_F,
            self.state.substrate_translation_t,
        )

        if corrected_np is None:
            raise AdsorbateSpotPresenterError("Failed to transform adsorbate spots.")

        corrected = [tuple(map(float, pt)) for pt in corrected_np]
        self.state.corrected_spots = corrected

        corrected_covariances: List[Optional[np.ndarray]] = []
        raw_covariances = list(self.state.raw_spot_covariances)
        if raw_covariances and self.state.substrate_matrix_F is not None:
            F = np.asarray(self.state.substrate_matrix_F, dtype=float)
            for cov in raw_covariances:
                if cov is None:
                    corrected_covariances.append(None)
                    continue
                try:
                    cov_arr = np.asarray(cov, dtype=float)
                except (TypeError, ValueError):
                    corrected_covariances.append(None)
                    continue
                if cov_arr.shape != (2, 2):
                    corrected_covariances.append(None)
                    continue
                try:
                    corrected_covariances.append(F @ cov_arr @ F.T)
                except Exception as exc:  # pragma: no cover
                    logger.warning("Failed to propagate adsorbate covariance: %s", exc)
                    corrected_covariances.append(None)
        else:
            corrected_covariances = [None] * len(raw_covariances)

        if len(corrected_covariances) < len(corrected):
            corrected_covariances.extend([None] * (len(corrected) - len(corrected_covariances)))
        elif len(corrected_covariances) > len(corrected):
            corrected_covariances = corrected_covariances[: len(corrected)]

        self.state.corrected_spot_covariances = corrected_covariances
        logger.info("AdsorbatePresenter: corrected %s spots.", len(corrected))
        return corrected

    # ------------------------------------------------------------------ Result helpers
    def build_results_dict(self) -> Dict[str, Any]:
        raw_spots = [tuple(map(float, pt)) for pt in self.state.raw_spots]
        raw_covariances = list(self.state.raw_spot_covariances)
        if len(raw_covariances) < len(raw_spots):
            raw_covariances.extend([None] * (len(raw_spots) - len(raw_covariances)))
        elif len(raw_covariances) > len(raw_spots):
            raw_covariances = raw_covariances[: len(raw_spots)]

        corrected_spots = [tuple(map(float, pt)) for pt in self.state.corrected_spots]
        corrected_covariances = list(self.state.corrected_spot_covariances)
        if len(corrected_covariances) < len(corrected_spots):
            corrected_covariances.extend([None] * (len(corrected_spots) - len(corrected_covariances)))
        elif len(corrected_covariances) > len(corrected_spots):
            corrected_covariances = corrected_covariances[: len(corrected_spots)]

        return {
            "adsorbate_set_index": self.state.set_index,
            "raw_adsorbate_spots": raw_spots,
            "corrected_adsorbate_spots_in_ideal_system": corrected_spots,
            "expected_type": self.state.expected_type,
            "raw_adsorbate_spot_covariances": [
                np.array(cov, dtype=float) if cov is not None else None
                for cov in raw_covariances
            ],
            "corrected_adsorbate_spot_covariances": [
                np.array(cov, dtype=float) if cov is not None else None
                for cov in corrected_covariances
            ],
        }
