"""Serializable state contracts for AtomMapper local-fit settings."""

from __future__ import annotations

from dataclasses import dataclass, field, fields, replace
from enum import Enum
from math import isfinite
from typing import Any, TypeAlias

from .fit_models import LocalFitModelType


class FitParameterTier(str, Enum):
    """UI tier used to split parameters into basic and advanced groups."""

    BASIC = "basic"
    ADVANCED = "advanced"


@dataclass(frozen=True)
class FitParameterDescriptor:
    """Declarative metadata for a single editable fit parameter field."""

    name: str
    label: str
    tier: FitParameterTier


def _parameter_field(
    default: Any,
    *,
    label: str,
    tier: FitParameterTier,
) -> Any:
    return field(default=default, metadata={"label": label, "tier": tier})


def _normalize_optional_float(value: object) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if not isfinite(number):
        return None
    return float(number)


def _normalize_optional_positive_float(value: object, *, minimum: float = 0.001) -> float | None:
    number = _normalize_optional_float(value)
    if number is None:
        return None
    if number < minimum:
        return minimum
    return number


def _normalize_int(value: object, *, default: int, minimum: int) -> int:
    try:
        number = int(round(float(value)))
    except (TypeError, ValueError):
        number = default
    if number < minimum:
        return minimum
    return number


@dataclass(frozen=True)
class ParameterBounds:
    """Optional lower/upper bounds for a single scalar parameter."""

    lower: float | None = None
    upper: float | None = None

    def normalized(self) -> "ParameterBounds":
        lower = _normalize_optional_float(self.lower)
        upper = _normalize_optional_float(self.upper)
        if lower is not None and upper is not None and lower > upper:
            lower, upper = upper, lower
        return ParameterBounds(lower=lower, upper=upper)

    @property
    def is_active(self) -> bool:
        normalized = self.normalized()
        return normalized.lower is not None or normalized.upper is not None

    def to_dict(self) -> dict[str, Any]:
        normalized = self.normalized()
        return {"lower": normalized.lower, "upper": normalized.upper}

    @classmethod
    def from_dict(cls, payload: dict[str, Any] | None) -> "ParameterBounds":
        data = payload or {}
        return cls(
            lower=data.get("lower"),
            upper=data.get("upper"),
        ).normalized()


@dataclass(frozen=True)
class CommonFitSettings:
    """Model-agnostic options shared by all local fit backends."""

    compute_uncertainty: bool = _parameter_field(
        False,
        label="Compute uncertainty",
        tier=FitParameterTier.BASIC,
    )
    use_custom_initial_guess: bool = _parameter_field(
        False,
        label="Use custom initial guess",
        tier=FitParameterTier.BASIC,
    )
    use_custom_bounds: bool = _parameter_field(
        False,
        label="Use custom bounds",
        tier=FitParameterTier.BASIC,
    )
    max_nfev: int = _parameter_field(
        5000,
        label="Max function evaluations",
        tier=FitParameterTier.ADVANCED,
    )

    def normalized(self) -> "CommonFitSettings":
        return CommonFitSettings(
            compute_uncertainty=bool(self.compute_uncertainty),
            use_custom_initial_guess=bool(self.use_custom_initial_guess),
            use_custom_bounds=bool(self.use_custom_bounds),
            max_nfev=_normalize_int(self.max_nfev, default=5000, minimum=10),
        )

    def to_dict(self) -> dict[str, Any]:
        normalized = self.normalized()
        return {
            "compute_uncertainty": normalized.compute_uncertainty,
            "use_custom_initial_guess": normalized.use_custom_initial_guess,
            "use_custom_bounds": normalized.use_custom_bounds,
            "max_nfev": normalized.max_nfev,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any] | None) -> "CommonFitSettings":
        data = payload or {}
        return cls(
            compute_uncertainty=data.get("compute_uncertainty", False),
            use_custom_initial_guess=data.get("use_custom_initial_guess", False),
            use_custom_bounds=data.get("use_custom_bounds", False),
            max_nfev=data.get("max_nfev", 5000),
        ).normalized()


@dataclass(frozen=True)
class GaussianFitSettings:
    """Model-specific parameters exposed for 2D Gaussian fitting."""

    amplitude_init: float | None = _parameter_field(None, label="Amplitude init", tier=FitParameterTier.BASIC)
    sigma_y_init: float | None = _parameter_field(None, label="Sigma Y init", tier=FitParameterTier.BASIC)
    sigma_x_init: float | None = _parameter_field(None, label="Sigma X init", tier=FitParameterTier.BASIC)
    offset_init: float | None = _parameter_field(None, label="Offset init", tier=FitParameterTier.BASIC)
    center_y_init: float | None = _parameter_field(None, label="Center Y init", tier=FitParameterTier.ADVANCED)
    center_x_init: float | None = _parameter_field(None, label="Center X init", tier=FitParameterTier.ADVANCED)
    theta_init_rad: float | None = _parameter_field(None, label="Theta init [rad]", tier=FitParameterTier.ADVANCED)
    amplitude_bounds: ParameterBounds = field(default_factory=ParameterBounds, metadata={"label": "Amplitude bounds", "tier": FitParameterTier.ADVANCED})
    sigma_y_bounds: ParameterBounds = field(default_factory=ParameterBounds, metadata={"label": "Sigma Y bounds", "tier": FitParameterTier.ADVANCED})
    sigma_x_bounds: ParameterBounds = field(default_factory=ParameterBounds, metadata={"label": "Sigma X bounds", "tier": FitParameterTier.ADVANCED})
    offset_bounds: ParameterBounds = field(default_factory=ParameterBounds, metadata={"label": "Offset bounds", "tier": FitParameterTier.ADVANCED})
    center_y_bounds: ParameterBounds = field(default_factory=ParameterBounds, metadata={"label": "Center Y bounds", "tier": FitParameterTier.ADVANCED})
    center_x_bounds: ParameterBounds = field(default_factory=ParameterBounds, metadata={"label": "Center X bounds", "tier": FitParameterTier.ADVANCED})
    theta_bounds_rad: ParameterBounds = field(default_factory=ParameterBounds, metadata={"label": "Theta bounds [rad]", "tier": FitParameterTier.ADVANCED})

    def normalized(self) -> "GaussianFitSettings":
        return GaussianFitSettings(
            amplitude_init=_normalize_optional_float(self.amplitude_init),
            sigma_y_init=_normalize_optional_positive_float(self.sigma_y_init),
            sigma_x_init=_normalize_optional_positive_float(self.sigma_x_init),
            offset_init=_normalize_optional_float(self.offset_init),
            center_y_init=_normalize_optional_float(self.center_y_init),
            center_x_init=_normalize_optional_float(self.center_x_init),
            theta_init_rad=_normalize_optional_float(self.theta_init_rad),
            amplitude_bounds=self.amplitude_bounds.normalized(),
            sigma_y_bounds=self.sigma_y_bounds.normalized(),
            sigma_x_bounds=self.sigma_x_bounds.normalized(),
            offset_bounds=self.offset_bounds.normalized(),
            center_y_bounds=self.center_y_bounds.normalized(),
            center_x_bounds=self.center_x_bounds.normalized(),
            theta_bounds_rad=self.theta_bounds_rad.normalized(),
        )

    def to_dict(self) -> dict[str, Any]:
        normalized = self.normalized()
        return {
            "amplitude_init": normalized.amplitude_init,
            "sigma_y_init": normalized.sigma_y_init,
            "sigma_x_init": normalized.sigma_x_init,
            "offset_init": normalized.offset_init,
            "center_y_init": normalized.center_y_init,
            "center_x_init": normalized.center_x_init,
            "theta_init_rad": normalized.theta_init_rad,
            "bounds": {
                "amplitude": normalized.amplitude_bounds.to_dict(),
                "sigma_y": normalized.sigma_y_bounds.to_dict(),
                "sigma_x": normalized.sigma_x_bounds.to_dict(),
                "offset": normalized.offset_bounds.to_dict(),
                "center_y": normalized.center_y_bounds.to_dict(),
                "center_x": normalized.center_x_bounds.to_dict(),
                "theta_rad": normalized.theta_bounds_rad.to_dict(),
            },
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any] | None) -> "GaussianFitSettings":
        data = payload or {}
        bounds = data.get("bounds") or {}
        return cls(
            amplitude_init=data.get("amplitude_init"),
            sigma_y_init=data.get("sigma_y_init"),
            sigma_x_init=data.get("sigma_x_init"),
            offset_init=data.get("offset_init"),
            center_y_init=data.get("center_y_init"),
            center_x_init=data.get("center_x_init"),
            theta_init_rad=data.get("theta_init_rad"),
            amplitude_bounds=ParameterBounds.from_dict(bounds.get("amplitude")),
            sigma_y_bounds=ParameterBounds.from_dict(bounds.get("sigma_y")),
            sigma_x_bounds=ParameterBounds.from_dict(bounds.get("sigma_x")),
            offset_bounds=ParameterBounds.from_dict(bounds.get("offset")),
            center_y_bounds=ParameterBounds.from_dict(bounds.get("center_y")),
            center_x_bounds=ParameterBounds.from_dict(bounds.get("center_x")),
            theta_bounds_rad=ParameterBounds.from_dict(bounds.get("theta_rad")),
        ).normalized()


@dataclass(frozen=True)
class LorentzianFitSettings:
    """Model-specific parameters exposed for 2D Lorentzian fitting."""

    amplitude_init: float | None = _parameter_field(None, label="Amplitude init", tier=FitParameterTier.BASIC)
    gamma_y_init: float | None = _parameter_field(None, label="Gamma Y init", tier=FitParameterTier.BASIC)
    gamma_x_init: float | None = _parameter_field(None, label="Gamma X init", tier=FitParameterTier.BASIC)
    offset_init: float | None = _parameter_field(None, label="Offset init", tier=FitParameterTier.BASIC)
    center_y_init: float | None = _parameter_field(None, label="Center Y init", tier=FitParameterTier.ADVANCED)
    center_x_init: float | None = _parameter_field(None, label="Center X init", tier=FitParameterTier.ADVANCED)
    theta_init_rad: float | None = _parameter_field(None, label="Theta init [rad]", tier=FitParameterTier.ADVANCED)
    amplitude_bounds: ParameterBounds = field(default_factory=ParameterBounds, metadata={"label": "Amplitude bounds", "tier": FitParameterTier.ADVANCED})
    gamma_y_bounds: ParameterBounds = field(default_factory=ParameterBounds, metadata={"label": "Gamma Y bounds", "tier": FitParameterTier.ADVANCED})
    gamma_x_bounds: ParameterBounds = field(default_factory=ParameterBounds, metadata={"label": "Gamma X bounds", "tier": FitParameterTier.ADVANCED})
    offset_bounds: ParameterBounds = field(default_factory=ParameterBounds, metadata={"label": "Offset bounds", "tier": FitParameterTier.ADVANCED})
    center_y_bounds: ParameterBounds = field(default_factory=ParameterBounds, metadata={"label": "Center Y bounds", "tier": FitParameterTier.ADVANCED})
    center_x_bounds: ParameterBounds = field(default_factory=ParameterBounds, metadata={"label": "Center X bounds", "tier": FitParameterTier.ADVANCED})
    theta_bounds_rad: ParameterBounds = field(default_factory=ParameterBounds, metadata={"label": "Theta bounds [rad]", "tier": FitParameterTier.ADVANCED})

    def normalized(self) -> "LorentzianFitSettings":
        return LorentzianFitSettings(
            amplitude_init=_normalize_optional_float(self.amplitude_init),
            gamma_y_init=_normalize_optional_positive_float(self.gamma_y_init),
            gamma_x_init=_normalize_optional_positive_float(self.gamma_x_init),
            offset_init=_normalize_optional_float(self.offset_init),
            center_y_init=_normalize_optional_float(self.center_y_init),
            center_x_init=_normalize_optional_float(self.center_x_init),
            theta_init_rad=_normalize_optional_float(self.theta_init_rad),
            amplitude_bounds=self.amplitude_bounds.normalized(),
            gamma_y_bounds=self.gamma_y_bounds.normalized(),
            gamma_x_bounds=self.gamma_x_bounds.normalized(),
            offset_bounds=self.offset_bounds.normalized(),
            center_y_bounds=self.center_y_bounds.normalized(),
            center_x_bounds=self.center_x_bounds.normalized(),
            theta_bounds_rad=self.theta_bounds_rad.normalized(),
        )

    def to_dict(self) -> dict[str, Any]:
        normalized = self.normalized()
        return {
            "amplitude_init": normalized.amplitude_init,
            "gamma_y_init": normalized.gamma_y_init,
            "gamma_x_init": normalized.gamma_x_init,
            "offset_init": normalized.offset_init,
            "center_y_init": normalized.center_y_init,
            "center_x_init": normalized.center_x_init,
            "theta_init_rad": normalized.theta_init_rad,
            "bounds": {
                "amplitude": normalized.amplitude_bounds.to_dict(),
                "gamma_y": normalized.gamma_y_bounds.to_dict(),
                "gamma_x": normalized.gamma_x_bounds.to_dict(),
                "offset": normalized.offset_bounds.to_dict(),
                "center_y": normalized.center_y_bounds.to_dict(),
                "center_x": normalized.center_x_bounds.to_dict(),
                "theta_rad": normalized.theta_bounds_rad.to_dict(),
            },
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any] | None) -> "LorentzianFitSettings":
        data = payload or {}
        bounds = data.get("bounds") or {}
        return cls(
            amplitude_init=data.get("amplitude_init"),
            gamma_y_init=data.get("gamma_y_init"),
            gamma_x_init=data.get("gamma_x_init"),
            offset_init=data.get("offset_init"),
            center_y_init=data.get("center_y_init"),
            center_x_init=data.get("center_x_init"),
            theta_init_rad=data.get("theta_init_rad"),
            amplitude_bounds=ParameterBounds.from_dict(bounds.get("amplitude")),
            gamma_y_bounds=ParameterBounds.from_dict(bounds.get("gamma_y")),
            gamma_x_bounds=ParameterBounds.from_dict(bounds.get("gamma_x")),
            offset_bounds=ParameterBounds.from_dict(bounds.get("offset")),
            center_y_bounds=ParameterBounds.from_dict(bounds.get("center_y")),
            center_x_bounds=ParameterBounds.from_dict(bounds.get("center_x")),
            theta_bounds_rad=ParameterBounds.from_dict(bounds.get("theta_rad")),
        ).normalized()


@dataclass(frozen=True)
class VoigtFitSettings:
    """Model-specific parameters exposed for 2D Voigt fitting."""

    amplitude_init: float | None = _parameter_field(None, label="Amplitude init", tier=FitParameterTier.BASIC)
    sigma_y_init: float | None = _parameter_field(None, label="Sigma Y init", tier=FitParameterTier.BASIC)
    sigma_x_init: float | None = _parameter_field(None, label="Sigma X init", tier=FitParameterTier.BASIC)
    gamma_y_init: float | None = _parameter_field(None, label="Gamma Y init", tier=FitParameterTier.BASIC)
    gamma_x_init: float | None = _parameter_field(None, label="Gamma X init", tier=FitParameterTier.BASIC)
    offset_init: float | None = _parameter_field(None, label="Offset init", tier=FitParameterTier.BASIC)
    center_y_init: float | None = _parameter_field(None, label="Center Y init", tier=FitParameterTier.ADVANCED)
    center_x_init: float | None = _parameter_field(None, label="Center X init", tier=FitParameterTier.ADVANCED)
    theta_init_rad: float | None = _parameter_field(None, label="Theta init [rad]", tier=FitParameterTier.ADVANCED)
    amplitude_bounds: ParameterBounds = field(default_factory=ParameterBounds, metadata={"label": "Amplitude bounds", "tier": FitParameterTier.ADVANCED})
    sigma_y_bounds: ParameterBounds = field(default_factory=ParameterBounds, metadata={"label": "Sigma Y bounds", "tier": FitParameterTier.ADVANCED})
    sigma_x_bounds: ParameterBounds = field(default_factory=ParameterBounds, metadata={"label": "Sigma X bounds", "tier": FitParameterTier.ADVANCED})
    gamma_y_bounds: ParameterBounds = field(default_factory=ParameterBounds, metadata={"label": "Gamma Y bounds", "tier": FitParameterTier.ADVANCED})
    gamma_x_bounds: ParameterBounds = field(default_factory=ParameterBounds, metadata={"label": "Gamma X bounds", "tier": FitParameterTier.ADVANCED})
    offset_bounds: ParameterBounds = field(default_factory=ParameterBounds, metadata={"label": "Offset bounds", "tier": FitParameterTier.ADVANCED})
    center_y_bounds: ParameterBounds = field(default_factory=ParameterBounds, metadata={"label": "Center Y bounds", "tier": FitParameterTier.ADVANCED})
    center_x_bounds: ParameterBounds = field(default_factory=ParameterBounds, metadata={"label": "Center X bounds", "tier": FitParameterTier.ADVANCED})
    theta_bounds_rad: ParameterBounds = field(default_factory=ParameterBounds, metadata={"label": "Theta bounds [rad]", "tier": FitParameterTier.ADVANCED})

    def normalized(self) -> "VoigtFitSettings":
        return VoigtFitSettings(
            amplitude_init=_normalize_optional_float(self.amplitude_init),
            sigma_y_init=_normalize_optional_positive_float(self.sigma_y_init),
            sigma_x_init=_normalize_optional_positive_float(self.sigma_x_init),
            gamma_y_init=_normalize_optional_positive_float(self.gamma_y_init),
            gamma_x_init=_normalize_optional_positive_float(self.gamma_x_init),
            offset_init=_normalize_optional_float(self.offset_init),
            center_y_init=_normalize_optional_float(self.center_y_init),
            center_x_init=_normalize_optional_float(self.center_x_init),
            theta_init_rad=_normalize_optional_float(self.theta_init_rad),
            amplitude_bounds=self.amplitude_bounds.normalized(),
            sigma_y_bounds=self.sigma_y_bounds.normalized(),
            sigma_x_bounds=self.sigma_x_bounds.normalized(),
            gamma_y_bounds=self.gamma_y_bounds.normalized(),
            gamma_x_bounds=self.gamma_x_bounds.normalized(),
            offset_bounds=self.offset_bounds.normalized(),
            center_y_bounds=self.center_y_bounds.normalized(),
            center_x_bounds=self.center_x_bounds.normalized(),
            theta_bounds_rad=self.theta_bounds_rad.normalized(),
        )

    def to_dict(self) -> dict[str, Any]:
        normalized = self.normalized()
        return {
            "amplitude_init": normalized.amplitude_init,
            "sigma_y_init": normalized.sigma_y_init,
            "sigma_x_init": normalized.sigma_x_init,
            "gamma_y_init": normalized.gamma_y_init,
            "gamma_x_init": normalized.gamma_x_init,
            "offset_init": normalized.offset_init,
            "center_y_init": normalized.center_y_init,
            "center_x_init": normalized.center_x_init,
            "theta_init_rad": normalized.theta_init_rad,
            "bounds": {
                "amplitude": normalized.amplitude_bounds.to_dict(),
                "sigma_y": normalized.sigma_y_bounds.to_dict(),
                "sigma_x": normalized.sigma_x_bounds.to_dict(),
                "gamma_y": normalized.gamma_y_bounds.to_dict(),
                "gamma_x": normalized.gamma_x_bounds.to_dict(),
                "offset": normalized.offset_bounds.to_dict(),
                "center_y": normalized.center_y_bounds.to_dict(),
                "center_x": normalized.center_x_bounds.to_dict(),
                "theta_rad": normalized.theta_bounds_rad.to_dict(),
            },
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any] | None) -> "VoigtFitSettings":
        data = payload or {}
        bounds = data.get("bounds") or {}
        return cls(
            amplitude_init=data.get("amplitude_init"),
            sigma_y_init=data.get("sigma_y_init"),
            sigma_x_init=data.get("sigma_x_init"),
            gamma_y_init=data.get("gamma_y_init"),
            gamma_x_init=data.get("gamma_x_init"),
            offset_init=data.get("offset_init"),
            center_y_init=data.get("center_y_init"),
            center_x_init=data.get("center_x_init"),
            theta_init_rad=data.get("theta_init_rad"),
            amplitude_bounds=ParameterBounds.from_dict(bounds.get("amplitude")),
            sigma_y_bounds=ParameterBounds.from_dict(bounds.get("sigma_y")),
            sigma_x_bounds=ParameterBounds.from_dict(bounds.get("sigma_x")),
            gamma_y_bounds=ParameterBounds.from_dict(bounds.get("gamma_y")),
            gamma_x_bounds=ParameterBounds.from_dict(bounds.get("gamma_x")),
            offset_bounds=ParameterBounds.from_dict(bounds.get("offset")),
            center_y_bounds=ParameterBounds.from_dict(bounds.get("center_y")),
            center_x_bounds=ParameterBounds.from_dict(bounds.get("center_x")),
            theta_bounds_rad=ParameterBounds.from_dict(bounds.get("theta_rad")),
        ).normalized()


FitModelParameters: TypeAlias = GaussianFitSettings | LorentzianFitSettings | VoigtFitSettings


@dataclass(frozen=True)
class FitSettingsState:
    """Single source of truth for local-fit model selection and parameters."""

    model: LocalFitModelType = LocalFitModelType.GAUSSIAN
    common: CommonFitSettings = field(default_factory=CommonFitSettings)
    gaussian: GaussianFitSettings = field(default_factory=GaussianFitSettings)
    lorentzian: LorentzianFitSettings = field(default_factory=LorentzianFitSettings)
    voigt: VoigtFitSettings = field(default_factory=VoigtFitSettings)

    def normalized(self) -> "FitSettingsState":
        return FitSettingsState(
            model=LocalFitModelType(self.model),
            common=self.common.normalized(),
            gaussian=self.gaussian.normalized(),
            lorentzian=self.lorentzian.normalized(),
            voigt=self.voigt.normalized(),
        )

    @property
    def active_parameters(self) -> FitModelParameters:
        normalized = self.normalized()
        if normalized.model is LocalFitModelType.GAUSSIAN:
            return normalized.gaussian
        if normalized.model is LocalFitModelType.LORENTZIAN:
            return normalized.lorentzian
        return normalized.voigt

    def parameters_for(self, model: LocalFitModelType | str) -> FitModelParameters:
        normalized_model = LocalFitModelType(model)
        normalized = self.normalized()
        if normalized_model is LocalFitModelType.GAUSSIAN:
            return normalized.gaussian
        if normalized_model is LocalFitModelType.LORENTZIAN:
            return normalized.lorentzian
        return normalized.voigt

    def with_model(self, model: LocalFitModelType | str) -> "FitSettingsState":
        return replace(self.normalized(), model=LocalFitModelType(model))

    def to_dict(self) -> dict[str, Any]:
        normalized = self.normalized()
        return {
            "model": normalized.model.value,
            "common": normalized.common.to_dict(),
            "parameters": {
                "gaussian": normalized.gaussian.to_dict(),
                "lorentzian": normalized.lorentzian.to_dict(),
                "voigt": normalized.voigt.to_dict(),
            },
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any] | None) -> "FitSettingsState":
        data = payload or {}
        parameters = data.get("parameters") or {}
        return cls(
            model=LocalFitModelType(data.get("model", LocalFitModelType.GAUSSIAN.value)),
            common=CommonFitSettings.from_dict(data.get("common")),
            gaussian=GaussianFitSettings.from_dict(parameters.get("gaussian")),
            lorentzian=LorentzianFitSettings.from_dict(parameters.get("lorentzian")),
            voigt=VoigtFitSettings.from_dict(parameters.get("voigt")),
        ).normalized()


def describe_fit_parameters(parameter_state: type[Any] | Any) -> tuple[FitParameterDescriptor, ...]:
    """Return declarative metadata for rendering editable parameter fields in the GUI."""

    cls = parameter_state if isinstance(parameter_state, type) else type(parameter_state)
    descriptors: list[FitParameterDescriptor] = []
    for item in fields(cls):
        tier = item.metadata.get("tier")
        label = item.metadata.get("label")
        if tier is None or label is None:
            continue
        descriptors.append(
            FitParameterDescriptor(
                name=item.name,
                label=str(label),
                tier=FitParameterTier(tier),
            )
        )
    return tuple(descriptors)
