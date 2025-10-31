import csv
import json
from pathlib import Path
import sys

import numpy as np
import pytest
from PyQt6.QtWidgets import QApplication, QListWidget

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from lfa.logic.app_controller import AppController
from lfa.logic.history_manager import HistoryManager
from lfa.logic.reporting import (
    build_real_space_json,
    build_real_space_records,
    build_real_space_summary,
)


def _sample_real_space_result() -> dict:
    return {
        "a1_nm": 0.25,
        "a1_nm_sigma": 0.005,
        "a2_nm": 0.28,
        "a2_nm_sigma": 0.006,
        "alpha_deg": 60.0,
        "alpha_deg_sigma": 0.2,
        "a1_vec_nm": (0.25, 0.0),
        "a2_vec_nm": (0.14, 0.24),
        "g1_vec_nm_inv": (1.2, 0.3),
        "g1_vec_cov_nm_inv": np.array([[0.01**2, 0.0], [0.0, 0.02**2]], dtype=float),
        "g2_vec_nm_inv": (0.1, 1.1),
        "g2_vec_cov_nm_inv": np.array([[0.03**2, 0.0], [0.0, 0.04**2]], dtype=float),
        "g1_vec_px": (10.0, 5.0),
        "g1_vec_cov_px": np.array([[0.5**2, 0.0], [0.0, 0.6**2]], dtype=float),
        "g2_vec_px": (4.0, 11.0),
        "g2_vec_cov_px": np.array([[0.7**2, 0.0], [0.0, 0.8**2]], dtype=float),
        "real_space_metric_covariance": np.diag([0.005**2, 0.006**2, 0.2**2]),
        "pixel_calibration_sigma_nm": (0.01, 0.02),
    }


def _sample_transform_analysis() -> dict:
    return {
        "rotation_angle_deg": 1.5,
        "rotation_angle_deg_sigma": 0.05,
        "rotation_angle_deg_covariance": np.array([[0.0025]], dtype=float),
        "principal_stretches": (1.0, 1.1),
        "principal_stretches_sigma": (0.02, 0.03),
        "principal_stretches_covariance": np.array(
            [[0.0004, 0.0001], [0.0001, 0.0009]],
            dtype=float,
        ),
        "rmse": 0.01,
    }


def test_build_real_space_summary_includes_uncertainties():
    substrate = _sample_real_space_result()
    adsorbate = {0: _sample_real_space_result()}

    summary = build_real_space_summary(
        substrate,
        adsorbate,
        transform_analysis=_sample_transform_analysis(),
    )

    assert "|a1| = 0.2500 +/- 0.0050 nm" in summary
    assert "Adsorbate Set 1" in summary
    assert "Pixel" in summary
    assert "rotation = 1.5000 +/- 0.0500 deg" in summary
    assert "stretches = (1.0000 +/- 0.0200, 1.1000 +/- 0.0300)" in summary


def test_build_real_space_json_contains_sigma_fields():
    substrate = _sample_real_space_result()
    adsorbate = {2: _sample_real_space_result()}

    transform = _sample_transform_analysis()
    payload = build_real_space_json(substrate, adsorbate, transform_analysis=transform)

    assert pytest.approx(payload["substrate"]["a1_nm_sigma"]) == 0.005
    assert payload["adsorbate"][2]["g1_vec_cov_nm_inv"] == [[0.0001, 0.0], [0.0, 0.0004]]
    assert pytest.approx(payload["substrate"]["pixel_calibration_sigma_nm"][0]) == 0.01
    assert pytest.approx(payload["substrate"]["pixel_calibration_sigma_nm"][1]) == 0.02
    transform_payload = payload["substrate_transform"]
    assert transform_payload is not None
    assert pytest.approx(transform_payload["rotation_angle_deg_sigma"]) == 0.05
    assert transform_payload["principal_stretches"] == [1.0, 1.1]


def test_build_real_space_records_flatten_vectors():
    substrate = _sample_real_space_result()
    adsorbate = {1: _sample_real_space_result()}

    transform = _sample_transform_analysis()
    records = build_real_space_records(substrate, adsorbate, transform_analysis=transform)

    assert records
    first = records[0]
    assert first["label"] == "substrate"
    assert pytest.approx(first["g1_nm_inv_sigma_0"]) == 0.01
    assert "rs_metric_cov_00" in first
    assert pytest.approx(first["pixel_sigma_nm_x"]) == 0.01
    assert pytest.approx(first["pixel_sigma_nm_y"]) == 0.02
    assert pytest.approx(first["transform_rotation_deg"]) == 1.5
    assert pytest.approx(first["transform_principal_stretch_sigma_0"]) == 0.02


def test_app_controller_real_space_exports(qtbot, tmp_path: Path):
    widget = QListWidget()
    qtbot.addWidget(widget)
    history_manager = HistoryManager(widget)
    controller = AppController(history_manager)

    substrate = _sample_real_space_result()
    adsorbate = _sample_real_space_result()

    controller.substrate_real_space_results = substrate
    controller.substrate_transform_analysis_m2i = _sample_transform_analysis()
    controller.adsorbate_real_space_results = {0: adsorbate}

    assert controller.copy_real_space_summary_to_clipboard() is True
    clipboard_text = QApplication.clipboard().text()
    assert "|a1| = 0.2500 +/- 0.0050 nm" in clipboard_text
    assert "Pixel" in clipboard_text
    assert "rotation = 1.5000 +/- 0.0500 deg" in clipboard_text

    json_path = tmp_path / "real_space_report.json"
    controller.export_real_space_report_to_json(str(json_path))
    data = json.loads(json_path.read_text(encoding="utf-8"))
    assert pytest.approx(data["substrate"]["alpha_deg_sigma"]) == 0.2
    assert pytest.approx(data["substrate"]["pixel_calibration_sigma_nm"][0]) == 0.01
    assert pytest.approx(data["substrate"]["pixel_calibration_sigma_nm"][1]) == 0.02
    assert pytest.approx(data["substrate_transform"]["rotation_angle_deg_sigma"]) == 0.05

    csv_path = tmp_path / "real_space_report.csv"
    controller.export_real_space_report_to_csv(str(csv_path))
    with csv_path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert rows
    assert rows[0]["a2_nm_sigma"] == "0.006"
    assert pytest.approx(float(rows[0]["pixel_sigma_nm_x"])) == 0.01
    assert pytest.approx(float(rows[0]["pixel_sigma_nm_y"])) == 0.02
    assert pytest.approx(float(rows[0]["transform_rotation_deg_sigma"])) == 0.05
