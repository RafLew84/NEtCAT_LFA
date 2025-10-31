"""Utilities for exporting and reporting lattice real-space analysis results."""

from __future__ import annotations

import math
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np

# Type alias for readability
RealSpaceResult = Dict[str, Any]
AdsorbateResults = Dict[int, RealSpaceResult]

_DEFAULT_FLOAT_PRECISION = 4


def build_real_space_summary(
    substrate: Optional[RealSpaceResult],
    adsorbate: Optional[AdsorbateResults],
) -> str:
    """
    Build a human-readable summary of real-space lattice parameters.

    Returns:
        A multiline string describing substrate/adsorbate results.
    """

    lines: List[str] = []

    if substrate:
        lines.append("Substrate Lattice:")
        lines.extend(_summary_lines_for_result(substrate, indent="  "))
    else:
        lines.append("Substrate Lattice: not calculated.")

    if not adsorbate:
        lines.append("Adsorbate Lattice: no sets calculated.")
    else:
        for index in sorted(adsorbate):
            result = adsorbate[index]
            if not result:
                lines.append(f"Adsorbate Set {index + 1}: not calculated.")
                continue
            lines.append(f"Adsorbate Set {index + 1}:")
            lines.extend(_summary_lines_for_result(result, indent="  "))

    return "\n".join(lines)


def build_real_space_json(
    substrate: Optional[RealSpaceResult],
    adsorbate: Optional[AdsorbateResults],
) -> Dict[str, Any]:
    """
    Build a JSON-serialisable payload describing real-space results.

    Returns:
        Dictionary with ``substrate`` and ``adsorbate`` keys.
    """
    payload: Dict[str, Any] = {
        "substrate": _json_payload_for_result(substrate) if substrate else None,
        "adsorbate": {},
    }

    if adsorbate:
        payload["adsorbate"] = {
            index: _json_payload_for_result(adsorbate[index]) if adsorbate[index] else None
            for index in sorted(adsorbate)
        }

    return payload


def build_real_space_records(
    substrate: Optional[RealSpaceResult],
    adsorbate: Optional[AdsorbateResults],
) -> List[Dict[str, Any]]:
    """
    Convert real-space results into flat records suitable for CSV export.

    Returns:
        List of dictionaries with flattened numeric fields.
    """
    records: List[Dict[str, Any]] = []

    if substrate:
        records.append(_record_for_result(substrate, label="substrate"))

    if adsorbate:
        for index in sorted(adsorbate):
            result = adsorbate[index]
            if not result:
                continue
            records.append(_record_for_result(result, label=f"adsorbate_set_{index}", set_index=index))

    return records


# --------------------------------------------------------------------------- #
# Internal helpers


def _summary_lines_for_result(result: RealSpaceResult, *, indent: str) -> List[str]:
    lines: List[str] = []
    lines.append(
        f"{indent}|a1| = {_format_value_with_sigma(result.get('a1_nm'), result.get('a1_nm_sigma'), 'nm')}"
    )
    lines.append(
        f"{indent}|a2| = {_format_value_with_sigma(result.get('a2_nm'), result.get('a2_nm_sigma'), 'nm')}"
    )
    lines.append(
        f"{indent}alpha = {_format_value_with_sigma(result.get('alpha_deg'), result.get('alpha_deg_sigma'), 'deg')}"
    )

    g1_line = _format_vector_with_covariance(
        result.get("g1_vec_nm_inv"),
        result.get("g1_vec_cov_nm_inv"),
        unit="nm^-1",
    )
    if g1_line:
        lines.append(f"{indent}g1 (nm^-1) = {g1_line}")

    g2_line = _format_vector_with_covariance(
        result.get("g2_vec_nm_inv"),
        result.get("g2_vec_cov_nm_inv"),
        unit="nm^-1",
    )
    if g2_line:
        lines.append(f"{indent}g2 (nm^-1) = {g2_line}")

    return lines


def _json_payload_for_result(result: Optional[RealSpaceResult]) -> Optional[Dict[str, Any]]:
    if not result:
        return None

    payload: Dict[str, Any] = {
        "a1_nm": _safe_float(result.get("a1_nm")),
        "a1_nm_sigma": _safe_float(result.get("a1_nm_sigma")),
        "a2_nm": _safe_float(result.get("a2_nm")),
        "a2_nm_sigma": _safe_float(result.get("a2_nm_sigma")),
        "alpha_deg": _safe_float(result.get("alpha_deg")),
        "alpha_deg_sigma": _safe_float(result.get("alpha_deg_sigma")),
        "a1_vec_nm": _safe_vector(result.get("a1_vec_nm")),
        "a2_vec_nm": _safe_vector(result.get("a2_vec_nm")),
        "g1_vec_nm_inv": _safe_vector(result.get("g1_vec_nm_inv")),
        "g2_vec_nm_inv": _safe_vector(result.get("g2_vec_nm_inv")),
        "g1_vec_cov_nm_inv": _safe_matrix(result.get("g1_vec_cov_nm_inv")),
        "g2_vec_cov_nm_inv": _safe_matrix(result.get("g2_vec_cov_nm_inv")),
        "g1_vec_px": _safe_vector(result.get("g1_vec_px")),
        "g2_vec_px": _safe_vector(result.get("g2_vec_px")),
        "g1_vec_cov_px": _safe_matrix(result.get("g1_vec_cov_px")),
        "g2_vec_cov_px": _safe_matrix(result.get("g2_vec_cov_px")),
        "real_space_metric_covariance": _safe_matrix(result.get("real_space_metric_covariance")),
        "pixel_calibration_sigma_nm": _safe_vector(result.get("pixel_calibration_sigma_nm")),
    }
    return payload


def _record_for_result(
    result: RealSpaceResult,
    *,
    label: str,
    set_index: Optional[int] = None,
) -> Dict[str, Any]:
    record: Dict[str, Any] = {"label": label}
    if set_index is not None:
        record["set_index"] = set_index

    record["a1_nm"] = _safe_float(result.get("a1_nm"))
    record["a1_nm_sigma"] = _safe_float(result.get("a1_nm_sigma"))
    record["a2_nm"] = _safe_float(result.get("a2_nm"))
    record["a2_nm_sigma"] = _safe_float(result.get("a2_nm_sigma"))
    record["alpha_deg"] = _safe_float(result.get("alpha_deg"))
    record["alpha_deg_sigma"] = _safe_float(result.get("alpha_deg_sigma"))

    _add_vector_to_record(record, result.get("a1_vec_nm"), "a1_vec_nm")
    _add_vector_to_record(record, result.get("a2_vec_nm"), "a2_vec_nm")

    _add_vector_with_covariance_to_record(
        record,
        result.get("g1_vec_nm_inv"),
        result.get("g1_vec_cov_nm_inv"),
        base_key="g1_nm_inv",
    )
    _add_vector_with_covariance_to_record(
        record,
        result.get("g2_vec_nm_inv"),
        result.get("g2_vec_cov_nm_inv"),
        base_key="g2_nm_inv",
    )
    _add_vector_with_covariance_to_record(
        record,
        result.get("g1_vec_px"),
        result.get("g1_vec_cov_px"),
        base_key="g1_px",
    )
    _add_vector_with_covariance_to_record(
        record,
        result.get("g2_vec_px"),
        result.get("g2_vec_cov_px"),
        base_key="g2_px",
    )

    _add_matrix_to_record(record, result.get("real_space_metric_covariance"), base_key="rs_metric_cov")
    sigma_pair = _safe_vector(result.get("pixel_calibration_sigma_nm"))
    if sigma_pair:
        record["pixel_sigma_nm_x"] = sigma_pair[0]
        record["pixel_sigma_nm_y"] = sigma_pair[1]

    return record


def _format_value_with_sigma(
    value: Any,
    sigma: Any,
    unit: str,
    *,
    precision: int = _DEFAULT_FLOAT_PRECISION,
) -> str:
    numeric_value = _safe_float(value)
    numeric_sigma = _safe_float(sigma)

    if numeric_value is None:
        return f"- {unit}".strip()

    if numeric_sigma is not None and numeric_sigma >= 0.0:
        return f"{numeric_value:.{precision}f} +/- {numeric_sigma:.{precision}f} {unit}".strip()

    return f"{numeric_value:.{precision}f} {unit}".strip()


def _format_sigma_pair_text(pair: Tuple[float, ...]) -> str:
    if not pair or len(pair) < 2:
        return "- nm"
    sx = _safe_float(pair[0])
    sy = _safe_float(pair[1])
    if sx is None or sy is None:
        return "- nm"
    return f"({sx:.4f}, {sy:.4f}) nm"


def _format_vector_with_covariance(
    vector: Any,
    covariance: Any,
    *,
    unit: str,
    precision: int = _DEFAULT_FLOAT_PRECISION,
) -> Optional[str]:
    components = _safe_vector(vector)
    if components is None:
        return None

    sigmas = _component_sigmas(covariance, len(components))
    formatted_components: List[str] = []
    for idx, value in enumerate(components):
        sigma = sigmas[idx] if idx < len(sigmas) else None
        if sigma is not None:
            formatted_components.append(f"{value:.{precision}f} +/- {sigma:.{precision}f}")
        else:
            formatted_components.append(f"{value:.{precision}f}")

    joined = ", ".join(formatted_components)
    return f"({joined}) {unit}".strip()


def _add_vector_to_record(record: Dict[str, Any], vector: Any, base_key: str) -> None:
    components = _safe_vector(vector)
    if components is None:
        return
    for idx, value in enumerate(components):
        record[f"{base_key}_{idx}"] = value


def _add_vector_with_covariance_to_record(
    record: Dict[str, Any],
    vector: Any,
    covariance: Any,
    *,
    base_key: str,
) -> None:
    components = _safe_vector(vector)
    if components is None:
        return

    variances = _component_sigmas(covariance, len(components))
    for idx, value in enumerate(components):
        record[f"{base_key}_{idx}"] = value
        sigma = variances[idx] if idx < len(variances) else None
        if sigma is not None:
            record[f"{base_key}_sigma_{idx}"] = sigma

    _add_matrix_to_record(record, covariance, base_key=f"{base_key}_cov")


def _add_matrix_to_record(
    record: Dict[str, Any],
    matrix: Any,
    *,
    base_key: str,
) -> None:
    clean_matrix = _safe_matrix(matrix)
    if clean_matrix is None:
        return
    arr = np.asarray(clean_matrix, dtype=float)
    if arr.ndim != 2:
        return
    rows, cols = arr.shape
    for i in range(rows):
        for j in range(cols):
            record[f"{base_key}_{i}{j}"] = float(arr[i, j])


def _component_sigmas(covariance: Any, count: int) -> List[Optional[float]]:
    if covariance is None:
        return [None] * count
    arr = _safe_matrix(covariance)
    if arr is None:
        return [None] * count
    arr_np = np.asarray(arr, dtype=float)
    if arr_np.shape != (count, count):
        return [None] * count

    sigmas: List[Optional[float]] = []
    for idx in range(count):
        variance = float(arr_np[idx, idx])
        if variance < 0:
            sigmas.append(None)
        else:
            sigmas.append(math.sqrt(variance))
    return sigmas


def _safe_float(value: Any) -> Optional[float]:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(numeric):
        return None
    return numeric


def _safe_vector(values: Any) -> Optional[Tuple[float, ...]]:
    if values is None:
        return None
    try:
        arr = np.asarray(values, dtype=float)
    except (TypeError, ValueError):
        return None
    if arr.ndim == 1 and arr.size >= 1:
        return tuple(float(item) for item in arr.tolist())
    return None


def _safe_matrix(matrix: Any) -> Optional[List[List[float]]]:
    if matrix is None:
        return None
    try:
        arr = np.asarray(matrix, dtype=float)
    except (TypeError, ValueError):
        return None
    if arr.ndim != 2:
        return None
    return [[float(value) for value in row] for row in arr.tolist()]
