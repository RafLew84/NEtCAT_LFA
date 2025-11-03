from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from ...utils.display import format_float, format_value_with_sigma


@dataclass(frozen=True)
class ValueDisplay:
    text: str = "- "
    tooltip: str = ""


@dataclass(frozen=True)
class RealSpaceLabelBundle:
    substrate_a1: ValueDisplay
    substrate_a2: ValueDisplay
    substrate_alpha: ValueDisplay
    adsorbate_a1: ValueDisplay
    adsorbate_a2: ValueDisplay
    adsorbate_alpha: ValueDisplay
    calibration_text: str
    angle_button_enabled: bool


@dataclass(frozen=True)
class AdsorbateSetInfo:
    index: int
    label: str
    has_real_space_result: bool


@dataclass(frozen=True)
class AdsorbateSetsSummary:
    sets: List[AdsorbateSetInfo]
    selected_index: Optional[int]


@dataclass(frozen=True)
class AngleCalculationResult:
    display_text: str
    alignment_angle_rad: Optional[float]
    error_message: Optional[str] = None


class RealSpaceVisualizerPresenter:
    """Data/formatting helper for :class:`RealSpaceFFTVisualizerDialog`."""

    def __init__(self, app_controller, logger=None) -> None:
        self._controller = app_controller
        self._logger = logger

    # ------------------------------------------------------------------ Adsorbate set helpers
    def get_adsorbate_sets_summary(self) -> AdsorbateSetsSummary:
        ctrl = self._controller
        if not ctrl:
            return AdsorbateSetsSummary([], None)

        num_sets = len(ctrl.adsorbate_spot_sets or [])
        if num_sets == 0 and ctrl.corrected_adsorbate_spot_sets:
            num_sets = len(ctrl.corrected_adsorbate_spot_sets)

        sets: List[AdsorbateSetInfo] = []
        for idx in range(num_sets):
            label = f"Set {idx + 1}"
            has_real_space = idx in getattr(ctrl, "adsorbate_real_space_results", {})
            sets.append(AdsorbateSetInfo(index=idx, label=label, has_real_space_result=has_real_space))

        selected_index: Optional[int] = None
        if sets:
            candidate = getattr(ctrl, "current_adsorbate_set_index", 0) or 0
            if 0 <= candidate < len(sets):
                selected_index = candidate
            else:
                selected_index = 0

        return AdsorbateSetsSummary(sets=sets, selected_index=selected_index)

    # ------------------------------------------------------------------ Label formatting
    def build_real_space_label_bundle(
        self,
        active_adsorbate_index: Optional[int],
    ) -> RealSpaceLabelBundle:
        ctrl = self._controller
        substrate_display = self._build_substrate_displays(ctrl)
        adsorbate_display = self._build_adsorbate_displays(ctrl, active_adsorbate_index)

        calibration_text = "- nm"
        if substrate_display.pixel_sigma_text:
            calibration_text = substrate_display.pixel_sigma_text
        elif adsorbate_display.pixel_sigma_text:
            calibration_text = adsorbate_display.pixel_sigma_text

        angle_enabled = bool(
            ctrl
            and ctrl.substrate_real_space_results
            and active_adsorbate_index is not None
            and active_adsorbate_index in ctrl.adsorbate_real_space_results
            and ctrl.adsorbate_real_space_results[active_adsorbate_index]
        )

        return RealSpaceLabelBundle(
            substrate_a1=substrate_display.a1,
            substrate_a2=substrate_display.a2,
            substrate_alpha=substrate_display.alpha,
            adsorbate_a1=adsorbate_display.a1,
            adsorbate_a2=adsorbate_display.a2,
            adsorbate_alpha=adsorbate_display.alpha,
            calibration_text=calibration_text,
            angle_button_enabled=angle_enabled,
        )

    # ------------------------------------------------------------------ Angle calculation
    def calculate_sub_ads_angle(self, adsorbate_index: int) -> AngleCalculationResult:
        ctrl = self._controller
        if not ctrl:
            return AngleCalculationResult(display_text="Error", alignment_angle_rad=None, error_message="Controller unavailable.")

        sub_params = ctrl.substrate_real_space_results
        ads_params = ctrl.adsorbate_real_space_results.get(adsorbate_index)

        if not (sub_params and "a1_vec_nm" in sub_params and ads_params and "a1_vec_nm" in ads_params):
            return AngleCalculationResult(display_text="N/A (Params missing)", alignment_angle_rad=None)

        try:
            a1_s_vec = np.array(sub_params["a1_vec_nm"], dtype=float)
            a2_s_vec = np.array(sub_params.get("a2_vec_nm", (0.0, 0.0)), dtype=float)
            a1_a_vec = np.array(ads_params["a1_vec_nm"], dtype=float)
            a2_a_vec = np.array(ads_params.get("a2_vec_nm", (0.0, 0.0)), dtype=float)

            norm_s = np.linalg.norm(a1_s_vec)
            norm_a = np.linalg.norm(a1_a_vec)
            if norm_s <= 1e-9 or norm_a <= 1e-9:
                return AngleCalculationResult(display_text="N/A (Zero vector)", alignment_angle_rad=None)

            dot_product = float(np.dot(a1_s_vec, a1_a_vec))
            cos_theta = float(np.clip(dot_product / (norm_s * norm_a), -1.0, 1.0))
            angle_for_display_deg = float(np.degrees(np.arccos(cos_theta)))
            angle_sigma_deg = self._estimate_sub_ads_angle_sigma_deg(sub_params, ads_params)
            display = format_value_with_sigma(
                angle_for_display_deg,
                angle_sigma_deg,
                "deg",
                value_precision=3,
                sigma_precision=3,
            )

            alignment_angle_rad = self._calculate_alignment_rotation(a1_s_vec, a2_s_vec, a1_a_vec, a2_a_vec)
            if self._logger:
                self._logger.info(
                    "Presenter: Angle between default a1 vectors %.3f deg (sigma=%s)",
                    angle_for_display_deg,
                    format_float(angle_sigma_deg, 3) if angle_sigma_deg is not None else "n/a",
                )

            return AngleCalculationResult(display_text=display, alignment_angle_rad=alignment_angle_rad)
        except Exception as exc:  # pragma: no cover
            if self._logger:
                self._logger.exception("Error calculating substrate-adsorbate angle: {0}", exc)
            return AngleCalculationResult(display_text="Error", alignment_angle_rad=None, error_message=str(exc))

    # ------------------------------------------------------------------ Internal helpers
    @dataclass
    class _MetricDisplays:
        a1: ValueDisplay
        a2: ValueDisplay
        alpha: ValueDisplay
        pixel_sigma_text: Optional[str]

    def _build_substrate_displays(self, ctrl) -> "_MetricDisplays":
        params = ctrl.substrate_real_space_results if ctrl else None
        if not params:
            return self._MetricDisplays(
                a1=ValueDisplay(text="- nm"),
                a2=ValueDisplay(text="- nm"),
                alpha=ValueDisplay(text="- deg"),
                pixel_sigma_text=None,
            )

        a1 = self._format_metric(params.get("a1_nm"), params.get("a1_nm_sigma"), "nm", 3, 3)
        a2 = self._format_metric(params.get("a2_nm"), params.get("a2_nm_sigma"), "nm", 3, 3)
        alpha = self._format_metric(params.get("alpha_deg"), params.get("alpha_deg_sigma"), "deg", 2, 2)
        pixel_sigma = self._resolve_sigma_text(params.get("pixel_calibration_sigma_nm"))
        return self._MetricDisplays(a1=a1, a2=a2, alpha=alpha, pixel_sigma_text=pixel_sigma)

    def _build_adsorbate_displays(
        self,
        ctrl,
        active_index: Optional[int],
    ) -> "_MetricDisplays":
        if (
            not ctrl
            or active_index is None
            or active_index not in ctrl.adsorbate_real_space_results
        ):
            return self._MetricDisplays(
                a1=ValueDisplay(text="- nm"),
                a2=ValueDisplay(text="- nm"),
                alpha=ValueDisplay(text="- deg"),
                pixel_sigma_text=None,
            )

        params = ctrl.adsorbate_real_space_results[active_index]
        a1 = self._format_metric(params.get("a1_nm"), params.get("a1_nm_sigma"), "nm", 3, 3)
        a2 = self._format_metric(params.get("a2_nm"), params.get("a2_nm_sigma"), "nm", 3, 3)
        alpha = self._format_metric(params.get("alpha_deg"), params.get("alpha_deg_sigma"), "deg", 2, 2)
        pixel_sigma = self._resolve_sigma_text(params.get("pixel_calibration_sigma_nm"))
        return self._MetricDisplays(a1=a1, a2=a2, alpha=alpha, pixel_sigma_text=pixel_sigma)

    @staticmethod
    def _format_metric(
        value: Optional[float],
        sigma: Optional[float],
        unit: str,
        value_precision: int,
        sigma_precision: int,
    ) -> ValueDisplay:
        text = format_value_with_sigma(
            value,
            sigma,
            unit,
            value_precision=value_precision,
            sigma_precision=sigma_precision,
        )
        tooltip = text if text and not text.startswith("-") else ""
        return ValueDisplay(text=text, tooltip=tooltip)

    @staticmethod
    def _resolve_sigma_text(pair: Optional[Tuple[float, float]]) -> Optional[str]:
        if not pair:
            return None
        sx = format_float(pair[0], 4)
        sy = format_float(pair[1], 4)
        if sx == "-" or sy == "-":
            return None
        return f"({sx}, {sy}) nm"

    def _estimate_sub_ads_angle_sigma_deg(
        self,
        substrate_params: Dict[str, Any],
        adsorbate_params: Dict[str, Any],
    ) -> Optional[float]:
        a1_sub_sigma = substrate_params.get("a1_nm_sigma")
        a2_sub_sigma = substrate_params.get("a2_nm_sigma")
        alpha_sub_sigma = substrate_params.get("alpha_deg_sigma")
        a1_ads_sigma = adsorbate_params.get("a1_nm_sigma")
        a2_ads_sigma = adsorbate_params.get("a2_nm_sigma")
        alpha_ads_sigma = adsorbate_params.get("alpha_deg_sigma")

        if None in (a1_sub_sigma, a2_sub_sigma, alpha_sub_sigma, a1_ads_sigma, a2_ads_sigma, alpha_ads_sigma):
            return None

        sigma_components = [
            float(a1_sub_sigma) if a1_sub_sigma is not None else 0.0,
            float(a2_sub_sigma) if a2_sub_sigma is not None else 0.0,
            float(alpha_sub_sigma) if alpha_sub_sigma is not None else 0.0,
            float(a1_ads_sigma) if a1_ads_sigma is not None else 0.0,
            float(a2_ads_sigma) if a2_ads_sigma is not None else 0.0,
            float(alpha_ads_sigma) if alpha_ads_sigma is not None else 0.0,
        ]
        variance = sum(component ** 2 for component in sigma_components)
        if variance <= 0.0:
            return None
        return float(np.sqrt(variance))

    @staticmethod
    def _calculate_alignment_rotation(
        a1_sub: np.ndarray,
        a2_sub: np.ndarray,
        a1_ads: np.ndarray,
        a2_ads: np.ndarray,
    ) -> Optional[float]:
        y_axis_vector = np.array([0.0, 1.0], dtype=float)

        def find_most_vertical_vector(a1, a2):
            candidate_vectors = [
                a1,
                a2,
                a1 + a2,
                a1 - a2,
                a2 - a1,
                2 * a1 - a2,
                a1 - 2 * a2,
                2 * a1 + a2,
                a1 + 2 * a2,
            ]
            best_vector, max_dot_product = None, -1.0
            for vec in candidate_vectors:
                norm = np.linalg.norm(vec)
                if norm < 1e-9:
                    continue
                dot_product = abs(np.dot(vec / norm, y_axis_vector))
                if dot_product > max_dot_product:
                    max_dot_product, best_vector = dot_product, vec
            return best_vector

        vertical_sub_vec = find_most_vertical_vector(a1_sub, a2_sub)
        vertical_ads_vec = find_most_vertical_vector(a1_ads, a2_ads)
        if vertical_sub_vec is None or vertical_ads_vec is None:
            return None

        angle_sub_rad = float(np.arctan2(vertical_sub_vec[1], vertical_sub_vec[0]))
        angle_ads_rad = float(np.arctan2(vertical_ads_vec[1], vertical_ads_vec[0]))
        alignment_angle_rad = angle_sub_rad - angle_ads_rad
        while alignment_angle_rad <= -np.pi:
            alignment_angle_rad += 2 * np.pi
        while alignment_angle_rad > np.pi:
            alignment_angle_rad -= 2 * np.pi
        return alignment_angle_rad
