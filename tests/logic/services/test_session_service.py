from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

pytest.importorskip("PyQt6", reason="PyQt6 is required for session service tests")

from lfa.core.constants import ADSORBATE_LATTICE_TYPE_UNKNOWN, PREDEFINED_SUBSTRATE_NONE
from lfa.core.data_models import OriginalImageRecord
from lfa.core.history import HistoryNode
from lfa.logic.services import SessionService


class FakeSignal:
    def __init__(self) -> None:
        self.calls = []

    def emit(self, *args, **kwargs) -> None:
        self.calls.append((args, kwargs))


class StubHistory:
    def __init__(self) -> None:
        self.current_node = object()
        self.cleared = False
        self.refreshed = False
        self.registered = None
        self.selected_node = None

    def get_current_node(self):
        return self.current_node

    def clear_history(self):
        self.cleared = True

    def refresh_widget(self):
        self.refreshed = True

    def register_original_image(self, record):
        self.registered = record

    def add_node_and_select(self, node):
        self.selected_node = node


def _build_reset_ready_controller():
    controller = SimpleNamespace()
    controller.session_serializer = SimpleNamespace(
        build_session_state=lambda _: {"state": "ok"},
        restore_session=lambda *_: None,
    )
    controller.original_file_path = "C:/data/sample.stp"
    controller.clear_all_spot_data_called = False
    controller.clear_all_spot_data = lambda: setattr(controller, "clear_all_spot_data_called", True)

    controller.reference_ideal_substrate_spots_px = [(1.0, 1.0)]
    controller.custom_lattice_info = {"foo": "bar"}
    controller.last_selected_substrate = "Pt(111)"
    controller.current_substrate_a_surf = 0.25
    controller.current_substrate_type = "Hex"
    controller.current_substrate_name = "Sample"
    controller.substrate_definition_name = "Pt(111)"
    controller.substrate_lattice_type = "hex"
    controller.substrate_a_surf = 0.25
    controller.substrate_F_m2i = np.eye(2)
    controller.substrate_t_m2i = np.zeros(2)
    controller.substrate_transform_analysis_m2i = {"rotation_deg": 5}
    controller.displayable_fitted_substrate_spots_on_fft = [(0.0, 0.0)]
    controller.show_ideal_lattice = False
    controller.current_fft_data_shape = (128, 128)
    controller.user_selected_substrate_spots = [(1.0, 2.0)]
    controller.substrate_visual_offset_nm = (1.0, -1.0)
    controller.adsorbate_visual_offsets_nm = {0: (0.5, 0.5)}
    controller.pixel_calibration_sigma_nm = (0.1, 0.1)

    controller.substrate_definition_changed = FakeSignal()
    controller.substrate_transform_results_updated = FakeSignal()
    controller.substrate_real_space_params_updated = FakeSignal()
    controller.adsorbate_sets_structure_changed = FakeSignal()
    controller.adsorbate_set_updated = FakeSignal()
    controller.adsorbate_real_space_params_updated = FakeSignal()
    controller.adsorbate_expected_type_updated = FakeSignal()
    controller.superstructure_periodicity_results_updated = FakeSignal()
    controller.spot_lists_updated = FakeSignal()

    controller.session_serializer.restore_session = lambda *_: setattr(controller, "restored", True)

    return controller


def test_save_session_serializes_when_path_selected(monkeypatch, tmp_path):
    history = StubHistory()
    controller = _build_reset_ready_controller()

    captured = {}

    def fake_get_save(*_args, **_kwargs):
        return str(tmp_path / "output.lfa_proj"), "filter"

    def fake_dump(path, payload):
        captured["path"] = path
        captured["payload"] = payload

    info_calls = []
    monkeypatch.setattr("lfa.logic.services.session_service.QFileDialog.getSaveFileName", fake_get_save)
    monkeypatch.setattr("lfa.logic.services.session_service.SessionSerializer.dump_to_file", fake_dump)
    monkeypatch.setattr("lfa.logic.services.session_service.QMessageBox.information", lambda *args: info_calls.append(args))

    service = SessionService(controller, history)
    service.save_session()

    assert Path(captured["path"]).name == "output.lfa_proj"
    assert captured["payload"] == {"state": "ok"}
    assert info_calls


def test_save_session_requires_active_node(monkeypatch):
    history = StubHistory()
    history.current_node = None
    controller = _build_reset_ready_controller()

    messages = []
    monkeypatch.setattr("lfa.logic.services.session_service.QMessageBox.information", lambda *args: messages.append(args))

    service = SessionService(controller, history)
    service.save_session()

    assert not messages or "No active analysis" in messages[0][2]


def test_load_session_restores_controller(monkeypatch):
    history = StubHistory()
    controller = _build_reset_ready_controller()

    open_calls = []
    monkeypatch.setattr(
        "lfa.logic.services.session_service.QFileDialog.getOpenFileName",
        lambda *args: ("session.lfa_proj", ""),
    )
    monkeypatch.setattr(
        "lfa.logic.services.session_service.SessionSerializer.load_from_file",
        lambda path: SimpleNamespace(marker="session"),
    )

    restored = {}

    def fake_restore(ctrl, session_state):
        restored["controller"] = ctrl
        restored["session"] = session_state

    controller.session_serializer.restore_session = fake_restore

    service = SessionService(controller, history)
    service.load_session()

    assert history.cleared is True
    assert controller.clear_all_spot_data_called is True
    assert restored["controller"] is controller
    assert restored["session"].marker == "session"


def test_load_session_handles_version_error(monkeypatch):
    history = StubHistory()
    controller = _build_reset_ready_controller()

    monkeypatch.setattr(
        "lfa.logic.services.session_service.QFileDialog.getOpenFileName",
        lambda *args: ("session.lfa_proj", ""),
    )
    monkeypatch.setattr(
        "lfa.logic.services.session_service.SessionSerializer.load_from_file",
        lambda path: SimpleNamespace(marker="session"),
    )
    controller.session_serializer.restore_session = lambda *_: (_ for _ in ()).throw(ValueError("bad version"))

    warnings = []
    monkeypatch.setattr("lfa.logic.services.session_service.QMessageBox.warning", lambda *args: warnings.append(args))

    service = SessionService(controller, history)
    service.load_session()

    assert warnings and "bad version" in warnings[0][2]


def test_reset_session_resets_fields_and_emits(monkeypatch):
    history = StubHistory()
    controller = _build_reset_ready_controller()

    service = SessionService(controller, history)
    service.reset_session()

    assert controller.original_file_path is None
    assert controller.reference_ideal_substrate_spots_px == []
    assert controller.custom_lattice_info is None
    assert controller.last_selected_substrate == PREDEFINED_SUBSTRATE_NONE
    assert controller.current_substrate_a_surf is None
    assert controller.substrate_visual_offset_nm == (0.0, 0.0)
    assert controller.adsorbate_visual_offsets_nm == {0: (0.0, 0.0)}
    assert controller.pixel_calibration_sigma_nm == (0.0, 0.0)
    assert history.refreshed is True

    required_signals = [
        controller.substrate_definition_changed,
        controller.substrate_transform_results_updated,
        controller.substrate_real_space_params_updated,
        controller.adsorbate_sets_structure_changed,
        controller.adsorbate_set_updated,
        controller.adsorbate_real_space_params_updated,
        controller.adsorbate_expected_type_updated,
        controller.superstructure_periodicity_results_updated,
        controller.spot_lists_updated,
    ]
    assert all(signal.calls for signal in required_signals)
    assert controller.adsorbate_expected_type_updated.calls[-1] == ((0, ADSORBATE_LATTICE_TYPE_UNKNOWN), {})


def test_register_new_original_records_and_selects():
    history = StubHistory()
    controller = _build_reset_ready_controller()
    service = SessionService(controller, history)

    record = OriginalImageRecord(display_name="Sample")
    node = HistoryNode(
        operation_name="Original",
        image_data=np.zeros((2, 2), dtype=np.float32),
        data_type="STM",
        original_image_id=record.image_id,
    )

    service.register_new_original(record, node)

    assert history.registered is record
    assert history.selected_node is node
