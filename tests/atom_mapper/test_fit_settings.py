"""Tests for AtomMapper local-fit settings contracts."""

from __future__ import annotations

from AtomMapper.app.fit_models import LocalFitModelType
from AtomMapper.app.fit_settings import (
    CommonFitSettings,
    FitParameterTier,
    FitSettingsState,
    GaussianFitSettings,
    ParameterBounds,
    VoigtFitSettings,
    describe_fit_parameters,
)


def test_fit_settings_state_roundtrips_and_selects_active_model_parameters():
    state = FitSettingsState(
        model=LocalFitModelType.VOIGT,
        common=CommonFitSettings(
            compute_uncertainty=False,
            use_custom_initial_guess=True,
            use_custom_bounds=True,
            max_nfev=3200,
        ),
        gaussian=GaussianFitSettings(
            amplitude_init=10.0,
            sigma_y_init=1.2,
            sigma_x_init=1.8,
        ),
        voigt=VoigtFitSettings(
            amplitude_init=12.0,
            sigma_y_init=1.5,
            sigma_x_init=1.7,
            gamma_y_init=0.8,
            gamma_x_init=0.9,
        ),
    )

    restored = FitSettingsState.from_dict(state.to_dict())

    assert restored == state.normalized()
    assert isinstance(restored.active_parameters, VoigtFitSettings)
    assert restored.active_parameters.gamma_y_init == 0.8


def test_fit_settings_normalization_clamps_invalid_values_and_orders_bounds():
    state = FitSettingsState(
        common=CommonFitSettings(max_nfev=0),
        gaussian=GaussianFitSettings(
            sigma_y_init=-4.0,
            sigma_x_init="nan",
            amplitude_bounds=ParameterBounds(lower=5.0, upper=-2.0),
        ),
    ).normalized()

    assert state.common.max_nfev == 10
    assert state.gaussian.sigma_y_init == 0.001
    assert state.gaussian.sigma_x_init is None
    assert state.gaussian.amplitude_bounds == ParameterBounds(lower=-2.0, upper=5.0)


def test_describe_fit_parameters_splits_fields_into_basic_and_advanced_groups():
    descriptors = describe_fit_parameters(GaussianFitSettings)
    tiers = {descriptor.name: descriptor.tier for descriptor in descriptors}

    assert tiers["amplitude_init"] is FitParameterTier.BASIC
    assert tiers["sigma_y_init"] is FitParameterTier.BASIC
    assert tiers["theta_init_rad"] is FitParameterTier.ADVANCED
    assert tiers["amplitude_bounds"] is FitParameterTier.ADVANCED


def test_common_fit_settings_defaults_disable_uncertainty_for_interactive_gui():
    state = FitSettingsState().normalized()

    assert state.common.compute_uncertainty is False
