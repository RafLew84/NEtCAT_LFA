from __future__ import annotations

import uuid
import math
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Tuple, TypeVar, TYPE_CHECKING

import numpy as np

from ..core.data_models import OriginalImageRecord

if TYPE_CHECKING:  # pragma: no cover
    from .app_controller import AppController
    from .history_manager import HistoryManager

Point = Optional[Tuple[float, float]]
Pair = Tuple[Point, Point]


def _coerce_point(value: Any) -> Point:
    if value is None:
        return None
    if isinstance(value, LayerOffsetNm):
        return value.to_tuple()
    if isinstance(value, dict):
        dx = value.get("dx") or value.get("x") or value.get("0", 0.0)
        dy = value.get("dy") or value.get("y") or value.get("1", 0.0)
        return _coerce_point((dx, dy))
    if isinstance(value, (list, tuple)) and len(value) == 2:
        try:
            first = float(value[0])
            second = float(value[1])
            if any(obj is None for obj in (value[0], value[1])):
                return None
            if not (LayerOffsetNm._is_finite(first) and LayerOffsetNm._is_finite(second)):
                return None
            return (first, second)
        except (TypeError, ValueError):
            return None
    return None


def _coerce_pairs(sequence: Iterable[Any]) -> List[Pair]:
    result: List[Pair] = []
    for item in sequence:
        if isinstance(item, dict):
            raw = _coerce_point(item.get("raw"))
            transformed = _coerce_point(item.get("transformed"))
        elif isinstance(item, (list, tuple)) and len(item) == 2:
            raw = _coerce_point(item[0])
            transformed = _coerce_point(item[1])
        else:
            raw = transformed = None
        result.append((raw, transformed))
    return result


def _pairs_to_payload(pairs: Iterable[Pair]) -> List[Dict[str, Point]]:
    payload: List[Dict[str, Point]] = []
    for raw, transformed in pairs:
        payload.append({"raw": raw, "transformed": transformed})
    return payload


def _covariance_to_payload_matrix(cov: Any) -> Optional[List[List[float]]]:
    if cov is None:
        return None
    try:
        arr = np.asarray(cov, dtype=float)
    except Exception:  # pragma: no cover - defensive
        return None
    if arr.shape != (2, 2):
        return None
    return [
        [float(arr[0, 0]), float(arr[0, 1])],
        [float(arr[1, 0]), float(arr[1, 1])],
    ]


def _normalise_covariance_payload(value: Any) -> Optional[List[List[float]]]:
    if value is None:
        return None
    try:
        rows = list(value)
    except TypeError:  # pragma: no cover - defensive
        return None
    if len(rows) != 2:
        return None
    matrix: List[List[float]] = []
    for row in rows:
        try:
            cols = list(row)
        except TypeError:  # pragma: no cover - defensive
            return None
        if len(cols) != 2:
            return None
        try:
            matrix.append([float(cols[0]), float(cols[1])])
        except (TypeError, ValueError):  # pragma: no cover - defensive
            return None
    return matrix


def _covariance_from_payload_matrix(matrix: Any) -> Optional[np.ndarray]:
    if matrix is None:
        return None
    try:
        arr = np.array(matrix, dtype=float)
    except Exception:  # pragma: no cover - defensive
        return None
    if arr.shape != (2, 2):
        return None
    return arr


def _covariance_sequence_to_payload(sequence: Optional[Iterable[Any]]) -> List[Optional[List[List[float]]]]:
    result: List[Optional[List[List[float]]]] = []
    if not sequence:
        return result
    for cov in sequence:
        result.append(_covariance_to_payload_matrix(cov))
    return result


def _covariance_sequence_from_payload(sequence: Optional[Iterable[Any]]) -> List[Optional[List[List[float]]]]:
    result: List[Optional[List[List[float]]]] = []
    if not sequence:
        return result
    for cov in sequence:
        result.append(_normalise_covariance_payload(cov))
    return result


def _nested_covariance_sequence_to_payload(
    sequences: Optional[Iterable[Iterable[Any]]]
) -> List[List[Optional[List[List[float]]]]]:
    result: List[List[Optional[List[List[float]]]]] = []
    if not sequences:
        return result
    for sequence in sequences:
        result.append(_covariance_sequence_to_payload(sequence))
    return result


def _nested_covariance_sequence_from_payload(
    sequences: Optional[Iterable[Iterable[Any]]]
) -> List[List[Optional[List[List[float]]]]]:
    result: List[List[Optional[List[List[float]]]]] = []
    if not sequences:
        return result
    for sequence in sequences:
        result.append(_covariance_sequence_from_payload(sequence))
    return result


T = TypeVar("T", bound="LayerOffsetNm")


@dataclass
class LayerOffsetNm:
    dx: float = 0.0
    dy: float = 0.0

    @classmethod
    def _is_finite(cls, value: float) -> bool:
        try:
            return math.isfinite(float(value))
        except (TypeError, ValueError):  # pragma: no cover - defensive
            return False

    @classmethod
    def from_any(cls: type[T], value: Any) -> T:
        point = _coerce_point(value)
        if point is None:
            return cls()
        return cls(dx=float(point[0]), dy=float(point[1]))

    def to_tuple(self) -> Tuple[float, float]:
        return (float(self.dx), float(self.dy))


@dataclass
class OriginalImageState:
    image_id: str
    display_name: str
    source_path: Optional[str] = None
    extra_metadata: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_payload(cls, payload: Dict[str, Any]) -> "OriginalImageState":
        image_id = payload.get("image_id") or str(uuid.uuid4())
        return cls(
            image_id=image_id,
            display_name=payload.get("display_name", "Original Image"),
            source_path=payload.get("source_path"),
            extra_metadata=dict(payload.get("extra_metadata") or {}),
        )

    def to_payload(self) -> Dict[str, Any]:
        return {
            "image_id": self.image_id,
            "display_name": self.display_name,
            "source_path": self.source_path,
            "extra_metadata": self.extra_metadata,
        }

    def to_record(self) -> OriginalImageRecord:
        return OriginalImageRecord(
            image_id=self.image_id,
            display_name=self.display_name,
            stm_image=None,
            source_path=self.source_path,
            extra_metadata=dict(self.extra_metadata),
        )


@dataclass
class ControllerState:
    original_file_path: Optional[str] = None
    spot_selection_mode: str = "Substrate"
    spot_refinement_method: str = "Direct Click"
    refinement_roi_size: int = 5
    user_selected_substrate_spots: List[Tuple[float, float]] = field(default_factory=list)
    user_selected_substrate_covariances: List[Optional[List[List[float]]]] = field(default_factory=list)
    substrate_lattice_type: Optional[str] = None
    substrate_a_surf: Optional[float] = None
    substrate_definition_name: str = ""
    substrate_F_m2i: Optional[Any] = None
    substrate_t_m2i: Optional[Any] = None
    substrate_transform_analysis_m2i: Optional[Dict[str, Any]] = None
    displayable_fitted_substrate_spots_on_fft: List[Tuple[float, float]] = field(default_factory=list)
    fitted_substrate_spot_covariances: List[Optional[List[List[float]]]] = field(default_factory=list)
    substrate_spot_pairs: List[Pair] = field(default_factory=list)
    show_substrate_raw_spots: bool = True
    show_substrate_transformed_spots: bool = True
    show_adsorbate_raw_spots: bool = True
    show_adsorbate_transformed_spots: bool = True
    show_substrate_spots_markers: bool = True
    show_adsorbate_spots_markers: bool = True
    substrate_real_space_results: Optional[Dict[str, Any]] = None
    adsorbate_spot_sets: List[Any] = field(default_factory=lambda: [[]])
    corrected_adsorbate_spot_sets: List[Any] = field(default_factory=lambda: [[]])
    adsorbate_spot_covariance_sets: List[List[Optional[List[List[float]]]]] = field(default_factory=lambda: [[]])
    corrected_adsorbate_covariance_sets: List[List[Optional[List[List[float]]]]] = field(default_factory=lambda: [[]])
    current_adsorbate_set_index: int = 0

    adsorbate_real_space_results: Dict[int, Any] = field(default_factory=dict)
    adsorbate_expected_lattice_types: Dict[int, str] = field(default_factory=lambda: {0: "Unknown"})
    superstructure_periodicity_results: Optional[Dict[str, Any]] = None
    substrate_visual_offset_nm: LayerOffsetNm = field(default_factory=LayerOffsetNm)
    adsorbate_visual_offsets_nm: Dict[int, LayerOffsetNm] = field(default_factory=dict)
    adsorbate_spot_pairs: Dict[int, List[Pair]] = field(default_factory=lambda: {0: []})
    pixel_calibration_sigma_nm: Tuple[float, float] = field(default_factory=lambda: (0.0, 0.0))

    @classmethod
    def from_controller(cls, controller: "AppController") -> "ControllerState":
        substrate_pairs = _coerce_pairs(getattr(controller, "substrate_spot_pairs", []))
        adsorbate_pairs_raw = getattr(controller, "adsorbate_spot_pairs", {}) or {}
        adsorbate_pairs: Dict[int, List[Pair]] = {}
        if isinstance(adsorbate_pairs_raw, dict):
            for key, pairs in adsorbate_pairs_raw.items():
                try:
                    idx = int(key)
                except (TypeError, ValueError):
                    continue
                adsorbate_pairs[idx] = _coerce_pairs(pairs)

        ads_offsets_raw = getattr(controller, "adsorbate_visual_offsets_nm", {}) or {}
        ads_offsets: Dict[int, LayerOffsetNm] = {}
        if isinstance(ads_offsets_raw, dict):
            for key, value in ads_offsets_raw.items():
                try:
                    idx = int(key)
                except (TypeError, ValueError):
                    continue
                ads_offsets[idx] = LayerOffsetNm.from_any(value)

        user_covariances_raw = getattr(controller, "user_selected_substrate_covariances", []) or []
        fitted_covariances_raw = getattr(controller, "fitted_substrate_spot_covariances", []) or []
        adsorbate_cov_sets_raw = getattr(controller, "adsorbate_spot_covariance_sets", [[]]) or [[]]
        corrected_adsorbate_cov_sets_raw = getattr(
            controller,
            "corrected_adsorbate_covariance_sets",
            [[]],
        ) or [[]]
        pixel_sigma_raw = getattr(controller, "pixel_calibration_sigma_nm", (0.0, 0.0))
        pixel_sigma = _coerce_point(pixel_sigma_raw) if pixel_sigma_raw is not None else None
        if pixel_sigma is None:
            pixel_sigma = (0.0, 0.0)

        return cls(
            original_file_path=getattr(controller, "original_file_path", None),
            spot_selection_mode=getattr(controller, "spot_selection_mode", "Substrate"),
            spot_refinement_method=getattr(controller, "spot_refinement_method", "Direct Click"),
            refinement_roi_size=int(getattr(controller, "refinement_roi_size", 5)),
            user_selected_substrate_spots=[
                tuple(spot) for spot in getattr(controller, "user_selected_substrate_spots", [])
            ],
            user_selected_substrate_covariances=_covariance_sequence_to_payload(user_covariances_raw),
            substrate_lattice_type=getattr(controller, "substrate_lattice_type", None),
            substrate_a_surf=getattr(controller, "substrate_a_surf", None),
            substrate_definition_name=getattr(controller, "substrate_definition_name", ""),
            substrate_F_m2i=getattr(controller, "substrate_F_m2i", None),
            substrate_t_m2i=getattr(controller, "substrate_t_m2i", None),
            substrate_transform_analysis_m2i=getattr(controller, "substrate_transform_analysis_m2i", None),
            displayable_fitted_substrate_spots_on_fft=list(
                getattr(controller, "displayable_fitted_substrate_spots_on_fft", [])
            ),
            fitted_substrate_spot_covariances=_covariance_sequence_to_payload(fitted_covariances_raw),
            substrate_spot_pairs=substrate_pairs,
            show_substrate_raw_spots=bool(getattr(controller, "show_substrate_raw_spots", True)),
            show_substrate_transformed_spots=bool(getattr(controller, "show_substrate_transformed_spots", True)),
            show_adsorbate_raw_spots=bool(getattr(controller, "show_adsorbate_raw_spots", True)),
            show_adsorbate_transformed_spots=bool(getattr(controller, "show_adsorbate_transformed_spots", True)),
            show_substrate_spots_markers=bool(getattr(controller, "show_substrate_spots_markers", True)),
            show_adsorbate_spots_markers=bool(getattr(controller, "show_adsorbate_spots_markers", True)),
            substrate_real_space_results=getattr(controller, "substrate_real_space_results", None),
            adsorbate_spot_sets=list(getattr(controller, "adsorbate_spot_sets", [[]])),
            corrected_adsorbate_spot_sets=list(getattr(controller, "corrected_adsorbate_spot_sets", [[]])),
            adsorbate_spot_covariance_sets=_nested_covariance_sequence_to_payload(adsorbate_cov_sets_raw),
            corrected_adsorbate_covariance_sets=_nested_covariance_sequence_to_payload(
                corrected_adsorbate_cov_sets_raw
            ),
            current_adsorbate_set_index=int(getattr(controller, "current_adsorbate_set_index", 0)),
            adsorbate_real_space_results=dict(getattr(controller, "adsorbate_real_space_results", {})),
            adsorbate_expected_lattice_types=dict(
                getattr(controller, "adsorbate_expected_lattice_types", {0: "Unknown"})
            ),
            superstructure_periodicity_results=getattr(controller, "superstructure_periodicity_results", None),
            substrate_visual_offset_nm=LayerOffsetNm.from_any(
                getattr(controller, "substrate_visual_offset_nm", None)
            ),
            adsorbate_visual_offsets_nm=ads_offsets,
            adsorbate_spot_pairs=adsorbate_pairs or {0: []},
            pixel_calibration_sigma_nm=(float(pixel_sigma[0]), float(pixel_sigma[1])),
        )

    def apply_to(self, controller: "AppController") -> None:
        controller.original_file_path = self.original_file_path
        controller.spot_selection_mode = self.spot_selection_mode
        controller.spot_refinement_method = self.spot_refinement_method
        controller.refinement_roi_size = self.refinement_roi_size
        controller.user_selected_substrate_spots = list(self.user_selected_substrate_spots)
        user_covariances = [
            _covariance_from_payload_matrix(cov) for cov in self.user_selected_substrate_covariances
        ]
        expected_user_len = len(controller.user_selected_substrate_spots)
        if len(user_covariances) < expected_user_len:
            user_covariances.extend([None] * (expected_user_len - len(user_covariances)))
        elif len(user_covariances) > expected_user_len:
            user_covariances = user_covariances[:expected_user_len]
        controller.user_selected_substrate_covariances = user_covariances
        controller.substrate_lattice_type = self.substrate_lattice_type
        controller.substrate_a_surf = self.substrate_a_surf
        controller.substrate_definition_name = self.substrate_definition_name
        controller.substrate_F_m2i = self.substrate_F_m2i
        controller.substrate_t_m2i = self.substrate_t_m2i
        controller.substrate_transform_analysis_m2i = self.substrate_transform_analysis_m2i
        controller.displayable_fitted_substrate_spots_on_fft = list(
            self.displayable_fitted_substrate_spots_on_fft
        )
        fitted_covariances = [
            _covariance_from_payload_matrix(cov) for cov in self.fitted_substrate_spot_covariances
        ]
        expected_fitted_len = len(controller.displayable_fitted_substrate_spots_on_fft)
        if len(fitted_covariances) < expected_fitted_len:
            fitted_covariances.extend([None] * (expected_fitted_len - len(fitted_covariances)))
        elif len(fitted_covariances) > expected_fitted_len:
            fitted_covariances = fitted_covariances[:expected_fitted_len]
        controller.fitted_substrate_spot_covariances = fitted_covariances
        controller.substrate_spot_pairs = [
            (tuple(raw) if raw is not None else None, tuple(transformed) if transformed is not None else None)
            for raw, transformed in self.substrate_spot_pairs
        ]
        controller.show_substrate_raw_spots = self.show_substrate_raw_spots
        controller.show_substrate_transformed_spots = self.show_substrate_transformed_spots
        controller.show_adsorbate_raw_spots = self.show_adsorbate_raw_spots
        controller.show_adsorbate_transformed_spots = self.show_adsorbate_transformed_spots
        controller.show_substrate_spots_markers = self.show_substrate_spots_markers
        controller.show_adsorbate_spots_markers = self.show_adsorbate_spots_markers
        controller.substrate_real_space_results = self.substrate_real_space_results
        controller.adsorbate_spot_sets = list(self.adsorbate_spot_sets)
        controller.corrected_adsorbate_spot_sets = list(self.corrected_adsorbate_spot_sets)

        def _convert_cov_sets(
            spot_sets: List[List[Any]],
            payload_cov_sets: List[List[Optional[List[List[float]]]]],
        ) -> List[List[Optional[np.ndarray]]]:
            converted: List[List[Optional[np.ndarray]]] = []
            if not spot_sets and not payload_cov_sets:
                return [[]]
            max_len = len(spot_sets)
            for idx in range(max_len):
                spots = spot_sets[idx]
                payload_covs = payload_cov_sets[idx] if idx < len(payload_cov_sets) else []
                cov_list: List[Optional[np.ndarray]] = []
                for spot_idx in range(len(spots)):
                    payload_cov = payload_covs[spot_idx] if spot_idx < len(payload_covs) else None
                    cov_list.append(_covariance_from_payload_matrix(payload_cov))
                converted.append(cov_list)
            if len(payload_cov_sets) > max_len:
                for idx in range(max_len, len(payload_cov_sets)):
                    extra_payload = payload_cov_sets[idx]
                    converted.append([_covariance_from_payload_matrix(cov) for cov in extra_payload])
            if not converted:
                converted = [[]]
            return converted

        controller.adsorbate_spot_covariance_sets = _convert_cov_sets(
            controller.adsorbate_spot_sets,
            self.adsorbate_spot_covariance_sets,
        )
        controller.corrected_adsorbate_covariance_sets = _convert_cov_sets(
            controller.corrected_adsorbate_spot_sets,
            self.corrected_adsorbate_covariance_sets,
        )
        controller.current_adsorbate_set_index = self.current_adsorbate_set_index
        controller.adsorbate_real_space_results = dict(self.adsorbate_real_space_results)
        controller.adsorbate_expected_lattice_types = dict(self.adsorbate_expected_lattice_types)
        controller.superstructure_periodicity_results = self.superstructure_periodicity_results
        controller.substrate_visual_offset_nm = self.substrate_visual_offset_nm.to_tuple()
        controller.adsorbate_visual_offsets_nm = {
            idx: offset.to_tuple() for idx, offset in self.adsorbate_visual_offsets_nm.items()
        }
        controller.adsorbate_spot_pairs = {
            idx: [
                (tuple(raw) if raw is not None else None, tuple(transformed) if transformed is not None else None)
                for raw, transformed in pairs
            ]
            for idx, pairs in self.adsorbate_spot_pairs.items()
        }
        controller.pixel_calibration_sigma_nm = tuple(self.pixel_calibration_sigma_nm)

        controller.set_substrate_raw_visibility(self.show_substrate_raw_spots)
        controller.set_substrate_transformed_visibility(self.show_substrate_transformed_spots)
        controller.set_adsorbate_raw_visibility(self.show_adsorbate_raw_spots)
        controller.set_adsorbate_transformed_visibility(self.show_adsorbate_transformed_spots)

    @classmethod
    def from_payload(cls, payload: Dict[str, Any]) -> "ControllerState":
        if 'domain_wall_analysis_results' in payload and 'superstructure_periodicity_results' not in payload:
            payload = dict(payload)
            payload['superstructure_periodicity_results'] = payload.get('domain_wall_analysis_results')

        substrate_pairs = payload.get("substrate_spot_pairs") or []
        adsorbate_pairs = payload.get("adsorbate_spot_pairs") or {}
        user_covariances = _covariance_sequence_from_payload(payload.get("user_selected_substrate_covariances"))
        fitted_covariances = _covariance_sequence_from_payload(payload.get("fitted_substrate_spot_covariances"))
        adsorbate_covariance_sets = _nested_covariance_sequence_from_payload(
            payload.get("adsorbate_spot_covariance_sets")
        )
        corrected_adsorbate_covariance_sets = _nested_covariance_sequence_from_payload(
            payload.get("corrected_adsorbate_covariance_sets")
        )
        if not adsorbate_covariance_sets:
            adsorbate_covariance_sets = [[]]
        if not corrected_adsorbate_covariance_sets:
            corrected_adsorbate_covariance_sets = [[]]

        ads_offsets_raw = payload.get("adsorbate_visual_offsets_nm", {}) or {}
        ads_offsets: Dict[int, LayerOffsetNm] = {}
        if isinstance(ads_offsets_raw, dict):
            for key, value in ads_offsets_raw.items():
                try:
                    idx = int(key)
                except (TypeError, ValueError):
                    continue
                ads_offsets[idx] = LayerOffsetNm.from_any(value)
        pixel_sigma_payload = _coerce_point(payload.get("pixel_calibration_sigma_nm"))
        if pixel_sigma_payload is None:
            pixel_sigma_payload = (0.0, 0.0)

        return cls(
            original_file_path=payload.get("original_file_path"),
            spot_selection_mode=payload.get("spot_selection_mode", "Substrate"),
            spot_refinement_method=payload.get("spot_refinement_method", "Direct Click"),
            refinement_roi_size=int(payload.get("refinement_roi_size", 5)),
            user_selected_substrate_spots=[
                tuple(spot) for spot in payload.get("user_selected_substrate_spots", [])
            ],
            user_selected_substrate_covariances=user_covariances,
            substrate_lattice_type=payload.get("substrate_lattice_type"),
            substrate_a_surf=payload.get("substrate_a_surf"),
            substrate_definition_name=payload.get("substrate_definition_name", ""),
            substrate_F_m2i=payload.get("substrate_F_m2i"),
            substrate_t_m2i=payload.get("substrate_t_m2i"),
            substrate_transform_analysis_m2i=payload.get("substrate_transform_analysis_m2i"),
            displayable_fitted_substrate_spots_on_fft=list(
                payload.get("displayable_fitted_substrate_spots_on_fft", [])
            ),
            fitted_substrate_spot_covariances=fitted_covariances,
            substrate_spot_pairs=_coerce_pairs(substrate_pairs),
            show_substrate_raw_spots=bool(payload.get("show_substrate_raw_spots", True)),
            show_substrate_transformed_spots=bool(payload.get("show_substrate_transformed_spots", True)),
            show_adsorbate_raw_spots=bool(payload.get("show_adsorbate_raw_spots", True)),
            show_adsorbate_transformed_spots=bool(payload.get("show_adsorbate_transformed_spots", True)),
            show_substrate_spots_markers=bool(payload.get("show_substrate_spots_markers", True)),
            show_adsorbate_spots_markers=bool(payload.get("show_adsorbate_spots_markers", True)),
            substrate_real_space_results=payload.get("substrate_real_space_results"),
            adsorbate_spot_sets=list(payload.get("adsorbate_spot_sets", [[]])),
            corrected_adsorbate_spot_sets=list(payload.get("corrected_adsorbate_spot_sets", [[]])),
            adsorbate_spot_covariance_sets=adsorbate_covariance_sets,
            corrected_adsorbate_covariance_sets=corrected_adsorbate_covariance_sets,
            current_adsorbate_set_index=int(payload.get("current_adsorbate_set_index", 0)),
            adsorbate_real_space_results=dict(payload.get("adsorbate_real_space_results", {})),
            adsorbate_expected_lattice_types=dict(
                payload.get("adsorbate_expected_lattice_types", {0: "Unknown"})
            ),
            superstructure_periodicity_results=payload.get("superstructure_periodicity_results"),
            substrate_visual_offset_nm=LayerOffsetNm.from_any(payload.get("substrate_visual_offset_nm")),
            adsorbate_visual_offsets_nm=ads_offsets,
            adsorbate_spot_pairs={
                int(key): _coerce_pairs(value) for key, value in (adsorbate_pairs.items() if isinstance(adsorbate_pairs, dict) else [])
            } or {0: []},
            pixel_calibration_sigma_nm=(float(pixel_sigma_payload[0]), float(pixel_sigma_payload[1])),
        )

    def to_payload(self) -> Dict[str, Any]:
        return {
            "original_file_path": self.original_file_path,
            "spot_selection_mode": self.spot_selection_mode,
            "spot_refinement_method": self.spot_refinement_method,
            "refinement_roi_size": self.refinement_roi_size,
            "user_selected_substrate_spots": self.user_selected_substrate_spots,
            "user_selected_substrate_covariances": _covariance_sequence_to_payload(
                self.user_selected_substrate_covariances
            ),
            "substrate_lattice_type": self.substrate_lattice_type,
            "substrate_a_surf": self.substrate_a_surf,
            "substrate_definition_name": self.substrate_definition_name,
            "substrate_F_m2i": self.substrate_F_m2i,
            "substrate_t_m2i": self.substrate_t_m2i,
            "substrate_transform_analysis_m2i": self.substrate_transform_analysis_m2i,
            "displayable_fitted_substrate_spots_on_fft": self.displayable_fitted_substrate_spots_on_fft,
            "fitted_substrate_spot_covariances": _covariance_sequence_to_payload(
                self.fitted_substrate_spot_covariances
            ),
            "substrate_spot_pairs": _pairs_to_payload(self.substrate_spot_pairs),
            "show_substrate_raw_spots": self.show_substrate_raw_spots,
            "show_substrate_transformed_spots": self.show_substrate_transformed_spots,
            "show_adsorbate_raw_spots": self.show_adsorbate_raw_spots,
            "show_adsorbate_transformed_spots": self.show_adsorbate_transformed_spots,
            "show_substrate_spots_markers": self.show_substrate_spots_markers,
            "show_adsorbate_spots_markers": self.show_adsorbate_spots_markers,
            "substrate_real_space_results": self.substrate_real_space_results,
            "adsorbate_spot_sets": self.adsorbate_spot_sets,
            "corrected_adsorbate_spot_sets": self.corrected_adsorbate_spot_sets,
            "adsorbate_spot_covariance_sets": _nested_covariance_sequence_to_payload(
                self.adsorbate_spot_covariance_sets
            ),
            "corrected_adsorbate_covariance_sets": _nested_covariance_sequence_to_payload(
                self.corrected_adsorbate_covariance_sets
            ),
            "current_adsorbate_set_index": self.current_adsorbate_set_index,
            "adsorbate_real_space_results": self.adsorbate_real_space_results,
            "adsorbate_expected_lattice_types": self.adsorbate_expected_lattice_types,
            "superstructure_periodicity_results": self.superstructure_periodicity_results,
            "substrate_visual_offset_nm": self.substrate_visual_offset_nm.to_tuple(),
            "adsorbate_visual_offsets_nm": {
                idx: offset.to_tuple() for idx, offset in self.adsorbate_visual_offsets_nm.items()
            },
            "adsorbate_spot_pairs": {
                idx: _pairs_to_payload(pairs)
                for idx, pairs in self.adsorbate_spot_pairs.items()
            },
            "pixel_calibration_sigma_nm": tuple(float(v) for v in self.pixel_calibration_sigma_nm),
        }


@dataclass
class HistoryState:
    tree: Dict[str, Any] = field(default_factory=dict)
    current_node_id: Optional[str] = None
    original_images: List[OriginalImageState] = field(default_factory=list)
    original_order: List[str] = field(default_factory=list)

    @classmethod
    def from_history_manager(cls, history_manager: "HistoryManager") -> "HistoryState":
        tree = getattr(history_manager, "history", {}) or {}
        current_node_id = getattr(history_manager, "current_node_id", None)

        records: List[OriginalImageState] = []
        order: List[str] = []
        if hasattr(history_manager, "iter_original_image_ids"):
            for image_id in history_manager.iter_original_image_ids():
                record = history_manager.get_original_image_record(image_id)
                if not record:
                    continue
                records.append(
                    OriginalImageState(
                        image_id=image_id,
                        display_name=record.display_name,
                        source_path=record.source_path,
                        extra_metadata=dict(record.extra_metadata),
                    )
                )
                order.append(image_id)

        return cls(
            tree=tree,
            current_node_id=current_node_id,
            original_images=records,
            original_order=order,
        )

    def apply_to(self, history_manager: "HistoryManager") -> None:
        history_manager.history = self.tree or {}

        if hasattr(history_manager, "original_images"):
            history_manager.original_images.clear()
        if hasattr(history_manager, "_original_order"):
            history_manager._original_order = []
        if hasattr(history_manager, "_root_nodes_by_image_id"):
            history_manager._root_nodes_by_image_id.clear()

        for record_state in self.original_images:
            record = record_state.to_record()
            if hasattr(history_manager, "register_original_image"):
                history_manager.register_original_image(record)

        if hasattr(history_manager, "rebuild_indexes"):
            history_manager.rebuild_indexes()

        if hasattr(history_manager, "_original_order") and self.original_order:
            history_manager._original_order = list(self.original_order)

        if hasattr(history_manager, "refresh_widget"):
            history_manager.refresh_widget()

        history_manager.set_current_node_by_id(self.current_node_id, emit_signal=False)

    @classmethod
    def from_payload(cls, payload: Dict[str, Any]) -> "HistoryState":
        records_payload = payload.get("original_images", []) or []
        records = [OriginalImageState.from_payload(entry) for entry in records_payload]
        return cls(
            tree=payload.get("tree", {}) or {},
            current_node_id=payload.get("current_node_id"),
            original_images=records,
            original_order=list(payload.get("original_order", []) or []),
        )

    def to_payload(self) -> Dict[str, Any]:
        return {
            "tree": self.tree,
            "current_node_id": self.current_node_id,
            "original_images": [record.to_payload() for record in self.original_images],
            "original_order": list(self.original_order),
        }


@dataclass
class SessionState:
    format_version: str
    controller: ControllerState
    history: HistoryState

    @classmethod
    def from_runtime(
        cls,
        controller: "AppController",
        history_manager: "HistoryManager",
        format_version: str,
    ) -> "SessionState":
        return cls(
            format_version=format_version,
            controller=ControllerState.from_controller(controller),
            history=HistoryState.from_history_manager(history_manager),
        )

    def apply_to(self, controller: "AppController", history_manager: "HistoryManager") -> None:
        self.controller.apply_to(controller)
        self.history.apply_to(history_manager)

    @classmethod
    def from_payload(cls, payload: Dict[str, Any]) -> "SessionState":
        format_version = payload.get("format_version", "")
        controller_state = ControllerState.from_payload(payload.get("controller_state", {}))
        history_state = HistoryState.from_payload(payload.get("history_data", {}))
        return cls(
            format_version=format_version,
            controller=controller_state,
            history=history_state,
        )

    def to_payload(self) -> Dict[str, Any]:
        return {
            "format_version": self.format_version,
            "controller_state": self.controller.to_payload(),
            "history_data": self.history.to_payload(),
        }
