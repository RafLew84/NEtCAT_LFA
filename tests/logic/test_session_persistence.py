import pickle
from pathlib import Path

import numpy as np
import pytest
pytest.importorskip("PyQt6", reason="PyQt6 is required for session persistence tests")
pytest.importorskip("pytestqt", reason="pytest-qt is required for qtbot fixture")

from PyQt6.QtWidgets import QListWidget, QMessageBox, QFileDialog

from lfa.logic.history_manager import HistoryManager
from lfa.logic.app_controller import AppController
from lfa.logic.session_migrations import CURRENT_SESSION_VERSION, migrate_payload
from lfa.logic.session_serializer import SessionSerializer
from lfa.core.history import HistoryNode
from lfa.core.data_models import OriginalImageRecord



def _create_controller_with_images(qtbot) -> tuple[AppController, OriginalImageRecord, OriginalImageRecord]:
    list_widget = QListWidget()
    qtbot.addWidget(list_widget)
    history_manager = HistoryManager(list_widget)
    controller = AppController(history_manager)

    controller.original_file_path = "img1.stp"

    rec1 = OriginalImageRecord(display_name="Original Image 1")
    history_manager.register_original_image(rec1)
    root1 = HistoryNode(
        operation_name="Original",
        parameters={"original_label": rec1.display_name, "source_image_label": rec1.display_name},
        image_data=np.zeros((4, 4), dtype=np.float32),
        data_type="STM",
        original_image_id=rec1.image_id,
    )
    history_manager.add_node(root1)
    blur1 = HistoryNode(
        parent_id=root1.node_id,
        operation_name="Gaussian Blur",
        parameters={"sigma": 1.0, "source_image_label": rec1.display_name},
        image_data=np.ones((4, 4), dtype=np.float32),
        data_type="STM",
        original_image_id=rec1.image_id,
    )
    history_manager.add_node(blur1)
    history_manager.set_current_node_by_id(blur1.node_id)

    rec2 = OriginalImageRecord(display_name="Original Image 2")
    history_manager.register_original_image(rec2)
    root2 = HistoryNode(
        operation_name="Original",
        parameters={"original_label": rec2.display_name, "source_image_label": rec2.display_name},
        image_data=np.ones((4, 4), dtype=np.float32),
        data_type="STM",
        original_image_id=rec2.image_id,
    )
    history_manager.add_node(root2)

    return controller, rec1, rec2


def test_save_session_includes_original_images(tmp_path, qtbot, monkeypatch):
    controller, rec1, rec2 = _create_controller_with_images(qtbot)

    session_path = Path(tmp_path) / "multi_image_session.lfa_proj"
    monkeypatch.setattr(QFileDialog, "getSaveFileName", lambda *_, **__: (str(session_path), ""))
    monkeypatch.setattr(QMessageBox, "information", lambda *_, **__: None)

    controller.save_analysis_session()

    with session_path.open("rb") as fh:
        session_data = pickle.load(fh)

    assert session_data["format_version"] == CURRENT_SESSION_VERSION
    history_data = session_data["history_data"]
    assert history_data["current_node_id"] == controller.history_manager.current_node_id
    original_images = {entry["image_id"]: entry for entry in history_data["original_images"]}
    assert rec1.image_id in original_images and rec2.image_id in original_images
    assert original_images[rec1.image_id]["display_name"] == "Original Image 1"
    assert history_data["original_order"] == controller.history_manager.iter_original_image_ids()


def test_load_session_restores_original_images(tmp_path, qtbot, monkeypatch):
    controller, rec1, rec2 = _create_controller_with_images(qtbot)

    session_path = Path(tmp_path) / "multi_image_session.lfa_proj"
    monkeypatch.setattr(QFileDialog, "getSaveFileName", lambda *_, **__: (str(session_path), ""))
    monkeypatch.setattr(QMessageBox, "information", lambda *_, **__: None)
    controller.save_analysis_session()

    list_widget = QListWidget()
    qtbot.addWidget(list_widget)
    history_manager = HistoryManager(list_widget)
    restored_controller = AppController(history_manager)

    messages: list[str] = []
    restored_controller.file_loaded_successfully.connect(messages.append)

    monkeypatch.setattr(QFileDialog, "getOpenFileName", lambda *_, **__: (str(session_path), ""))
    restored_controller.load_analysis_session()

    assert messages and messages[0] == "img1.stp"
    ids = history_manager.iter_original_image_ids()
    assert ids == [rec1.image_id, rec2.image_id]
    assert len(history_manager.original_images) == 2
    assert history_manager.get_current_node() is not None
    for node in history_manager.history.values():
        assert node.original_image_id in history_manager.original_images


def test_load_legacy_v1_session(tmp_path, qtbot, monkeypatch):
    list_widget = QListWidget()
    qtbot.addWidget(list_widget)
    history_manager = HistoryManager(list_widget)

    root = HistoryNode(
        operation_name="Original",
        parameters={"filename": "legacy.stp"},
        image_data=np.zeros((2, 2), dtype=np.float32),
        data_type="STM",
    )
    child = HistoryNode(
        parent_id=root.node_id,
        operation_name="Gaussian Blur",
        parameters={"sigma": 1.2},
        image_data=np.zeros((2, 2), dtype=np.float32),
        data_type="STM",
    )
    history_tree = {root.node_id: root, child.node_id: child}
    legacy_session = {
        "format_version": "1.0",
        "history_data": {"tree": history_tree, "current_node_id": child.node_id},
        "controller_state": {"original_file_path": "legacy.stp"},
    }
    legacy_path = Path(tmp_path) / "legacy_session.lfa_proj"
    with legacy_path.open("wb") as fh:
        pickle.dump(legacy_session, fh)

    controller = AppController(history_manager)
    monkeypatch.setattr(QFileDialog, "getOpenFileName", lambda *_, **__: (str(legacy_path), ""))

    controller.load_analysis_session()

    ids = history_manager.iter_original_image_ids()
    assert len(ids) == 1
    assert history_manager.original_images[ids[0]].display_name.startswith("Original Image")
    for node in history_manager.history.values():
        assert node.original_image_id == ids[0]


def test_session_round_trip_preserves_offsets(qtbot):
    controller, _, _ = _create_controller_with_images(qtbot)

    controller.substrate_visual_offset_nm = (1.25, -0.75)
    controller.adsorbate_spot_sets = [
        [(1.0, 2.0), (3.0, 4.0)],
        [(5.0, 6.0)],
    ]
    controller.corrected_adsorbate_spot_sets = [
        [(1.1, 2.1), (3.1, 4.1)],
        [(5.1, 6.1)],
    ]
    controller.adsorbate_visual_offsets_nm = {0: (0.0, 0.0), 1: (0.5, -0.5)}
    controller.adsorbate_spot_pairs = {
        0: [((1.0, 2.0), (1.1, 2.1))],
        1: [((5.0, 6.0), (5.1, 6.1))],
    }
    controller.current_adsorbate_set_index = 1

    serializer = SessionSerializer(controller.history_manager)
    session_state = serializer.build_session_state(controller)

    list_widget = QListWidget()
    qtbot.addWidget(list_widget)
    restored_history = HistoryManager(list_widget)
    restored_controller = AppController(restored_history)

    SessionSerializer(restored_history).restore_session(restored_controller, session_state)

    assert restored_controller.substrate_visual_offset_nm == pytest.approx((1.25, -0.75))
    assert restored_controller.adsorbate_visual_offsets_nm[1] == pytest.approx((0.5, -0.5))
    assert restored_controller.current_adsorbate_set_index == 1
    assert restored_controller.adsorbate_spot_pairs[1][0][0] == pytest.approx((5.0, 6.0))
    assert restored_controller.adsorbate_spot_pairs[1][0][1] == pytest.approx((5.1, 6.1))


def test_session_round_trip_preserves_covariances(qtbot):
    controller, _, _ = _create_controller_with_images(qtbot)

    user_cov0 = np.array([[0.1, 0.01], [0.01, 0.2]], dtype=float)
    fitted_cov0 = np.array([[0.05, 0.0], [0.0, 0.07]], dtype=float)
    fitted_cov1 = np.array([[0.08, 0.01], [0.01, 0.09]], dtype=float)
    ads_cov00 = np.array([[0.02, 0.0], [0.0, 0.03]], dtype=float)
    ads_cov10 = np.array([[0.04, 0.0], [0.0, 0.05]], dtype=float)
    corr_cov00 = np.array([[0.06, 0.0], [0.0, 0.07]], dtype=float)
    corr_cov01 = np.array([[0.08, 0.01], [0.01, 0.09]], dtype=float)

    controller.user_selected_substrate_spots = [(10.0, 20.0), (30.0, 40.0)]
    controller.user_selected_substrate_covariances = [user_cov0, None]
    controller.displayable_fitted_substrate_spots_on_fft = [(11.0, 21.0), (31.0, 41.0)]
    controller.fitted_substrate_spot_covariances = [fitted_cov0, fitted_cov1]
    controller.adsorbate_spot_sets = [
        [(1.0, 2.0), (3.0, 4.0)],
        [(5.0, 6.0)],
    ]
    controller.corrected_adsorbate_spot_sets = [
        [(1.1, 2.1), (3.1, 4.1)],
        [(5.1, 6.1)],
    ]
    controller.substrate_transform_analysis_m2i = {
        "rotation_angle_deg": 1.5,
        "rotation_angle_deg_sigma": 0.05,
        "rotation_angle_deg_covariance": np.array([[0.0025]], dtype=float),
        "principal_stretches": np.array([1.0, 1.1], dtype=float),
        "principal_stretches_sigma": (0.02, 0.03),
        "principal_stretches_covariance": np.array(
            [[0.0004, 0.0001], [0.0001, 0.0009]],
            dtype=float,
        ),
    }
    controller.adsorbate_spot_covariance_sets = [
        [ads_cov00, None],
        [ads_cov10],
    ]
    controller.corrected_adsorbate_covariance_sets = [
        [corr_cov00, corr_cov01],
        [None],
    ]
    controller.pixel_calibration_sigma_nm = (0.01, 0.02)

    serializer = SessionSerializer(controller.history_manager)
    session_state = serializer.build_session_state(controller)

    list_widget = QListWidget()
    qtbot.addWidget(list_widget)
    restored_history = HistoryManager(list_widget)
    restored_controller = AppController(restored_history)

    SessionSerializer(restored_history).restore_session(restored_controller, session_state)

    assert len(restored_controller.user_selected_substrate_covariances) == 2
    np.testing.assert_allclose(restored_controller.user_selected_substrate_covariances[0], user_cov0)
    assert restored_controller.user_selected_substrate_covariances[1] is None

    assert len(restored_controller.fitted_substrate_spot_covariances) == 2
    np.testing.assert_allclose(restored_controller.fitted_substrate_spot_covariances[0], fitted_cov0)
    np.testing.assert_allclose(restored_controller.fitted_substrate_spot_covariances[1], fitted_cov1)
    restored_analysis = restored_controller.substrate_transform_analysis_m2i
    assert restored_analysis is not None
    assert restored_analysis["rotation_angle_deg"] == pytest.approx(1.5)
    assert restored_analysis["rotation_angle_deg_sigma"] == pytest.approx(0.05)
    np.testing.assert_allclose(
        np.array(restored_analysis["rotation_angle_deg_covariance"], dtype=float),
        np.array([[0.0025]], dtype=float),
    )
    np.testing.assert_allclose(
        np.array(restored_analysis["principal_stretches"], dtype=float),
        np.array([1.0, 1.1], dtype=float),
    )
    assert tuple(restored_analysis["principal_stretches_sigma"]) == pytest.approx((0.02, 0.03))
    np.testing.assert_allclose(
        np.array(restored_analysis["principal_stretches_covariance"], dtype=float),
        np.array([[0.0004, 0.0001], [0.0001, 0.0009]], dtype=float),
    )

    assert len(restored_controller.adsorbate_spot_covariance_sets) == 2
    first_set = restored_controller.adsorbate_spot_covariance_sets[0]
    second_set = restored_controller.adsorbate_spot_covariance_sets[1]
    assert len(first_set) == 2 and len(second_set) == 1
    np.testing.assert_allclose(first_set[0], ads_cov00)
    assert first_set[1] is None
    np.testing.assert_allclose(second_set[0], ads_cov10)

    assert len(restored_controller.corrected_adsorbate_covariance_sets) == 2
    corr_first = restored_controller.corrected_adsorbate_covariance_sets[0]
    corr_second = restored_controller.corrected_adsorbate_covariance_sets[1]
    assert len(corr_first) == 2 and len(corr_second) == 1
    np.testing.assert_allclose(corr_first[0], corr_cov00)
    np.testing.assert_allclose(corr_first[1], corr_cov01)
    assert corr_second[0] is None
    assert restored_controller.pixel_calibration_sigma_nm == pytest.approx((0.01, 0.02))


def test_migrate_legacy_offsets_and_domain_wall_results():
    payload = {
        "format_version": "1.1",
        "controller_state": {
            "domain_wall_analysis_results": {"foo": "bar"},
            "substrate_visual_offset_nm": {"dx": "1.5", "dy": "-2.5"},
            "adsorbate_visual_offsets_nm": {"0": ["3.0", None], "1": {"x": "4.5", "y": "-6.25"}},
            "substrate_spot_pairs": [
                {"raw": ["1.0", "2.0"], "transformed": ["3.0", "4.0"]},
                [["not-a-number", 5.0], [6.0, 7.0]],
            ],
            "adsorbate_spot_pairs": {
                "1": [
                    {"raw": ["8.0", "9.0"], "transformed": ["10.0", "11.0"]},
                    [["bad", "data"], ["12.0", "13.5"]],
                ]
            },
        },
        "history_data": {},
    }

    migrated = migrate_payload(payload)

    controller_state = migrated["controller_state"]
    assert migrated["format_version"] == CURRENT_SESSION_VERSION
    assert controller_state["superstructure_periodicity_results"] == {"foo": "bar"}
    assert controller_state["substrate_visual_offset_nm"] == (1.5, -2.5)
    assert controller_state["adsorbate_visual_offsets_nm"][0] == (0.0, 0.0)
    assert controller_state["adsorbate_visual_offsets_nm"][1] == (4.5, -6.25)

    spot_pairs = controller_state["substrate_spot_pairs"]
    assert spot_pairs[0]["raw"] == (1.0, 2.0)
    assert spot_pairs[0]["transformed"] == (3.0, 4.0)
    assert spot_pairs[1]["raw"] is None
    assert spot_pairs[1]["transformed"] == (6.0, 7.0)

    ads_pairs = controller_state["adsorbate_spot_pairs"][1]
    assert ads_pairs[0]["raw"] == (8.0, 9.0)
    assert ads_pairs[0]["transformed"] == (10.0, 11.0)
    assert ads_pairs[1]["raw"] is None
    assert ads_pairs[1]["transformed"] == (12.0, 13.5)

def test_history_manager_emits_active_node_event(qtbot):
    list_widget = QListWidget()
    qtbot.addWidget(list_widget)
    history_manager = HistoryManager(list_widget)

    captured = []
    legacy = []
    history_manager.active_node_changed.connect(captured.append)
    history_manager.current_node_changed.connect(legacy.append)

    record = OriginalImageRecord(display_name="Event Image")
    history_manager.register_original_image(record)
    node = HistoryNode(
        operation_name="Original",
        parameters={"original_label": record.display_name, "source_image_label": record.display_name},
        image_data=np.zeros((2, 2), dtype=np.float32),
        data_type="STM",
        original_image_id=record.image_id,
    )
    history_manager.add_node(node)
    history_manager.set_current_node_by_id(node.node_id)

    assert captured, "Expected active_node_changed to emit when selecting a node"
    payload = captured[-1]
    assert payload.node_id == node.node_id
    assert payload.node is node
    assert payload.reason == "selection-changed"
    assert legacy and legacy[-1] is node


def test_history_manager_original_image_events(qtbot):
    list_widget = QListWidget()
    qtbot.addWidget(list_widget)
    history_manager = HistoryManager(list_widget)

    added = []
    removed = []
    history_manager.original_image_added.connect(added.append)
    history_manager.original_image_removed.connect(removed.append)

    record = OriginalImageRecord(display_name="Root")
    history_manager.register_original_image(record)
    assert added and added[0].image_id == record.image_id

    history_manager.unregister_original_image(record.image_id)
    assert removed and removed[0].image_id == record.image_id
