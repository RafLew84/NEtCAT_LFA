"""Shared preprocessing state contracts for AtomMapper dialogs and preview code."""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from enum import Enum
from math import isfinite
from typing import Any, Optional

import numpy as np


class PreprocessingMethod(str, Enum):
    """Supported preprocessing methods exposed by the dialog."""

    BLUR = "blur"
    NLM = "nlm"
    BM3D = "bm3d"

    @classmethod
    def normalize(cls, value: object) -> "PreprocessingMethod":
        """Return a normalized preprocessing method enum."""

        if isinstance(value, cls):
            return value

        normalized = str(value).strip().lower()
        try:
            return cls(normalized)
        except ValueError as exc:
            raise ValueError(f"Unsupported preprocessing method: {value!r}") from exc

    @property
    def label(self) -> str:
        """Return the user-facing label for the method."""

        labels = {
            PreprocessingMethod.BLUR: "Blur",
            PreprocessingMethod.NLM: "Non-local means",
            PreprocessingMethod.BM3D: "BM3D",
        }
        return labels[self]


def _normalize_float(value: object, *, default: float, minimum: float) -> float:
    """Return a finite float clamped to the configured minimum."""

    try:
        number = float(value)
    except (TypeError, ValueError):
        number = default

    if not isfinite(number) or number < minimum:
        return float(max(default, minimum))
    return float(number)


def _normalize_int(value: object, *, default: int, minimum: int) -> int:
    """Return an integer clamped to the configured minimum."""

    try:
        number = int(round(float(value)))
    except (TypeError, ValueError):
        number = default

    if number < minimum:
        return max(default, minimum)
    return int(number)


def _normalize_odd_int(value: object, *, default: int, minimum: int) -> int:
    """Return an odd integer clamped to the configured minimum."""

    number = _normalize_int(value, default=default, minimum=minimum)
    if number % 2 == 0:
        number += 1
    return int(number)


@dataclass(frozen=True)
class BlurParameters:
    """Parameters for Gaussian blur preview and apply operations."""

    sigma_px: float = 1.0
    mode: str = "nearest"

    def normalized(self) -> "BlurParameters":
        """Return normalized blur parameters."""

        mode = str(self.mode).strip().lower() or "nearest"
        return BlurParameters(
            sigma_px=_normalize_float(self.sigma_px, default=1.0, minimum=0.05),
            mode=mode,
        )

    def to_dict(self) -> dict[str, Any]:
        """Return a serialized representation."""

        normalized = self.normalized()
        return {
            "sigma_px": normalized.sigma_px,
            "mode": normalized.mode,
        }

    @classmethod
    def from_dict(cls, data: Optional[dict[str, Any]]) -> "BlurParameters":
        """Build parameters from a serialized representation."""

        payload = data or {}
        return cls(
            sigma_px=payload.get("sigma_px", 1.0),
            mode=payload.get("mode", "nearest"),
        ).normalized()


@dataclass(frozen=True)
class NonLocalMeansParameters:
    """Parameters for non-local means denoising."""

    h: float = 0.1
    patch_size: int = 5
    patch_distance: int = 6
    fast_mode: bool = True

    def normalized(self) -> "NonLocalMeansParameters":
        """Return normalized NLM parameters."""

        return NonLocalMeansParameters(
            h=_normalize_float(self.h, default=0.1, minimum=0.001),
            patch_size=_normalize_odd_int(self.patch_size, default=5, minimum=3),
            patch_distance=_normalize_int(self.patch_distance, default=6, minimum=1),
            fast_mode=bool(self.fast_mode),
        )

    def to_dict(self) -> dict[str, Any]:
        """Return a serialized representation."""

        normalized = self.normalized()
        return {
            "h": normalized.h,
            "patch_size": normalized.patch_size,
            "patch_distance": normalized.patch_distance,
            "fast_mode": normalized.fast_mode,
        }

    @classmethod
    def from_dict(cls, data: Optional[dict[str, Any]]) -> "NonLocalMeansParameters":
        """Build parameters from a serialized representation."""

        payload = data or {}
        return cls(
            h=payload.get("h", 0.1),
            patch_size=payload.get("patch_size", 5),
            patch_distance=payload.get("patch_distance", 6),
            fast_mode=payload.get("fast_mode", True),
        ).normalized()


@dataclass(frozen=True)
class BM3DParameters:
    """Parameters for BM3D denoising."""

    sigma_psd: float = 0.1
    stage: str = "all_stages"

    def normalized(self) -> "BM3DParameters":
        """Return normalized BM3D parameters."""

        stage = str(self.stage).strip().lower() or "all_stages"
        return BM3DParameters(
            sigma_psd=_normalize_float(self.sigma_psd, default=0.1, minimum=0.001),
            stage=stage,
        )

    def to_dict(self) -> dict[str, Any]:
        """Return a serialized representation."""

        normalized = self.normalized()
        return {
            "sigma_psd": normalized.sigma_psd,
            "stage": normalized.stage,
        }

    @classmethod
    def from_dict(cls, data: Optional[dict[str, Any]]) -> "BM3DParameters":
        """Build parameters from a serialized representation."""

        payload = data or {}
        return cls(
            sigma_psd=payload.get("sigma_psd", 0.1),
            stage=payload.get("stage", "all_stages"),
        ).normalized()


PreprocessingParameters = BlurParameters | NonLocalMeansParameters | BM3DParameters


@dataclass(frozen=True)
class PreprocessingState:
    """Single source of truth for dialog method selection and parameters."""

    method: PreprocessingMethod = PreprocessingMethod.BLUR
    blur: BlurParameters = field(default_factory=BlurParameters)
    nlm: NonLocalMeansParameters = field(default_factory=NonLocalMeansParameters)
    bm3d: BM3DParameters = field(default_factory=BM3DParameters)

    def normalized(self) -> "PreprocessingState":
        """Return a normalized preprocessing state."""

        return PreprocessingState(
            method=PreprocessingMethod.normalize(self.method),
            blur=self.blur.normalized(),
            nlm=self.nlm.normalized(),
            bm3d=self.bm3d.normalized(),
        )

    @property
    def active_parameters(self) -> PreprocessingParameters:
        """Return parameters for the active method."""

        normalized = self.normalized()
        if normalized.method is PreprocessingMethod.BLUR:
            return normalized.blur
        if normalized.method is PreprocessingMethod.NLM:
            return normalized.nlm
        return normalized.bm3d

    def with_method(self, method: object) -> "PreprocessingState":
        """Return a copy of the state with the active method changed."""

        normalized = self.normalized()
        return replace(normalized, method=PreprocessingMethod.normalize(method))

    def to_dict(self) -> dict[str, Any]:
        """Return a serialized representation of the preprocessing state."""

        normalized = self.normalized()
        return {
            "method": normalized.method.value,
            "parameters": {
                "blur": normalized.blur.to_dict(),
                "nlm": normalized.nlm.to_dict(),
                "bm3d": normalized.bm3d.to_dict(),
            },
        }

    @classmethod
    def from_dict(cls, data: Optional[dict[str, Any]]) -> "PreprocessingState":
        """Build state from a serialized representation."""

        payload = data or {}
        parameters = payload.get("parameters") or {}
        return cls(
            method=PreprocessingMethod.normalize(payload.get("method", PreprocessingMethod.BLUR.value)),
            blur=BlurParameters.from_dict(parameters.get("blur")),
            nlm=NonLocalMeansParameters.from_dict(parameters.get("nlm")),
            bm3d=BM3DParameters.from_dict(parameters.get("bm3d")),
        ).normalized()


@dataclass(frozen=True)
class PreviewViewport:
    """Optional preview viewport for future synchronized Original/Processed panels."""

    x: int = 0
    y: int = 0
    width: int = 0
    height: int = 0

    def normalized(self) -> "PreviewViewport":
        """Return a viewport with normalized non-negative geometry."""

        return PreviewViewport(
            x=_normalize_int(self.x, default=0, minimum=0),
            y=_normalize_int(self.y, default=0, minimum=0),
            width=_normalize_int(self.width, default=0, minimum=0),
            height=_normalize_int(self.height, default=0, minimum=0),
        )


@dataclass(frozen=True)
class PreprocessingPreviewRequest:
    """Request payload passed from the dialog to preview workers."""

    image_id: str
    source_group_id: str
    state: PreprocessingState
    viewport: Optional[PreviewViewport] = None

    def normalized(self) -> "PreprocessingPreviewRequest":
        """Return a normalized request payload."""

        image_id = str(self.image_id).strip()
        if not image_id:
            raise ValueError("image_id must be a non-empty string.")

        source_group_id = str(self.source_group_id).strip()
        if not source_group_id:
            raise ValueError("source_group_id must be a non-empty string.")

        return PreprocessingPreviewRequest(
            image_id=image_id,
            source_group_id=source_group_id,
            state=self.state.normalized(),
            viewport=self.viewport.normalized() if self.viewport is not None else None,
        )


@dataclass(frozen=True)
class PreprocessingPreviewResult:
    """Structured preview result returned by preprocessing backends."""

    request: PreprocessingPreviewRequest
    success: bool
    processed_image: Optional[np.ndarray] = field(default=None, repr=False)
    status_message: str = ""
    error_message: Optional[str] = None

    @classmethod
    def from_success(
        cls,
        request: PreprocessingPreviewRequest,
        processed_image: np.ndarray,
        *,
        status_message: str = "Preview ready.",
    ) -> "PreprocessingPreviewResult":
        """Build a successful preview result."""

        image_array = np.asarray(processed_image, dtype=float)
        if image_array.ndim != 2:
            raise ValueError(f"Expected 2D preview image data, got shape {image_array.shape!r}.")

        return cls(
            request=request.normalized(),
            success=True,
            processed_image=image_array,
            status_message=status_message,
            error_message=None,
        )

    @classmethod
    def from_failure(
        cls,
        request: PreprocessingPreviewRequest,
        error_message: str,
        *,
        status_message: str = "Preview failed.",
    ) -> "PreprocessingPreviewResult":
        """Build a failed preview result."""

        return cls(
            request=request.normalized(),
            success=False,
            processed_image=None,
            status_message=status_message,
            error_message=str(error_message).strip() or "Unknown preprocessing error.",
        )
