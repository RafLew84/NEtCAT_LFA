"""Common request/result contracts for local peak-fit models in AtomMapper."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING, Any, Optional, Tuple

import numpy as np

if TYPE_CHECKING:
    from .fit_settings import FitSettingsState


class LocalFitModelType(str, Enum):
    """Supported local model families for ROI-based peak fitting."""

    GAUSSIAN = "gaussian"
    LORENTZIAN = "lorentzian"
    VOIGT = "voigt"


@dataclass
class LocalFitRequest:
    """Normalized request passed to a local fit-model backend."""

    model: LocalFitModelType
    roi_patch: np.ndarray = field(repr=False)
    roi_origin_yx: Tuple[int, int] = (0, 0)
    compute_uncertainty: bool = True
    fit_mask: Optional[np.ndarray] = field(default=None, repr=False)
    fit_settings_state: Optional["FitSettingsState"] = field(default=None, repr=False)

    def normalized_patch(self) -> np.ndarray:
        """Return the ROI patch as a float array without mutating the caller input."""

        return np.asarray(self.roi_patch, dtype=float)

    def normalized_mask(self) -> Optional[np.ndarray]:
        """Return the fit mask as a boolean array if provided."""

        if self.fit_mask is None:
            return None
        return np.asarray(self.fit_mask, dtype=bool)


@dataclass
class LocalPeakFitResult:
    """Model-agnostic fit result used by preview widgets, point saving and errors."""

    model: LocalFitModelType
    center_patch_yx: Optional[Tuple[float, float]]
    center_image_yx: Optional[Tuple[float, float]]
    center_std_yx: Optional[Tuple[float, float]]
    amplitude: Optional[float]
    width_y: Optional[float]
    width_x: Optional[float]
    theta_rad: Optional[float]
    offset: Optional[float]
    method: str
    success: bool
    error_message: Optional[str]
    roi_patch: np.ndarray = field(repr=False)
    raw_result: Optional[Any] = field(default=None, repr=False)
    model_patch: Optional[np.ndarray] = field(default=None, repr=False)
    fit_mask: Optional[np.ndarray] = field(default=None, repr=False)
    shape_parameters: dict[str, float] = field(default_factory=dict, repr=False)

    @property
    def sigma_y(self) -> Optional[float]:
        """Backward-compatible alias for the primary Y width parameter."""

        return self.width_y

    @property
    def sigma_x(self) -> Optional[float]:
        """Backward-compatible alias for the primary X width parameter."""

        return self.width_x
