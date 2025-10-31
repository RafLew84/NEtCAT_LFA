import csv
import json
from pathlib import Path

import numpy as np
import pytest
from PyQt6.QtWidgets import QApplication, QListWidget

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
    }


def test_build_real_space_summary_includes_uncertainties():
    substrate = _sample_real_space_result()
    adsorbate = {0: _sample_real_space_result()}

    summary = build_real_space_summary(substrate, adsorbate)

    assert "|a1| = 0.2500 +/- 0.0050 nm" in summary
    assert "Adsorbate Set 1" in summary


def test_build_real_space_json_contains_sigma_fields():
    substrate = _sample_real_space_result()
    adsorbate = {2: _sample_real_space_result()}

    payload = build_real_space_json(substrate, adsorbate)

    assert pytest.approx(payload["substrate"]["a1_nm_sigma"]) == 0.005
    assert payload["adsorbate"][2]["g1_vec_cov_nm_inv"] == [[0.0001, 0.0], [0.0, 0.0004]]


def test_build_real_space_records_flatten_vectors():
    substrate = _sample_real_space_result()
    adsorbate = {1: _sample_real_space_result()}

    records = build_real_space_records(substrate, adsorbate)

    assert records
    first = records[0]
    assert first["label"] == "substrate"
    assert pytest.approx(first["g1_nm_inv_sigma_0"]) == 0.01
    assert "rs_metric_cov_00" in first


def test_app_controller_real_space_exports(qtbot, tmp_path: Path):
    widget = QListWidget()
    qtbot.addWidget(widget)
    history_manager = HistoryManager(widget)
    controller = AppController(history_manager)

    substrate = _sample_real_space_result()
    adsorbate = _sample_real_space_result()

    controller.substrate_real_space_results = substrate
    controller.adsorbate_real_space_results = {0: adsorbate}

    assert controller.copy_real_space_summary_to_clipboard() is True
    clipboard_text = QApplication.clipboard().text()
    assert "|a1| = 0.2500 +/- 0.0050 nm" in clipboard_text

    json_path = tmp_path / "real_space_report.json"
    controller.export_real_space_report_to_json(str(json_path))
    data = json.loads(json_path.read_text(encoding="utf-8"))
    assert pytest.approx(data["substrate"]["alpha_deg_sigma"]) == 0.2

    csv_path = tmp_path / "real_space_report.csv"
    controller.export_real_space_report_to_csv(str(csv_path))
    with csv_path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert rows
    assert rows[0]["a2_nm_sigma"] == "0.006"
