import numpy as np
import pytest

pytest.importorskip("PyQt6", reason="PyQt6 is required for session service tests")
pytest.importorskip("pytestqt", reason="pytest-qt provides the qtbot fixture")

from PyQt6.QtWidgets import QListWidget

from lfa.core.data_models import OriginalImageRecord
from lfa.core.history import HistoryNode
from lfa.logic.history_manager import HistoryManager
from lfa.logic.services import HistoryOrchestrator, SessionService


def test_session_service_registers_original(qtbot):
    widget = QListWidget()
    qtbot.addWidget(widget)
    history_manager = HistoryManager(widget)
    history_service = HistoryOrchestrator(history_manager)

    class DummyController:
        session_serializer = None  # not needed for this test
        original_file_path = None
        def clear_all_spot_data(self):  # pragma: no cover - unused here
            pass

    session_service = SessionService(DummyController(), history_service)

    record = OriginalImageRecord(display_name="Sample")
    node = HistoryNode(
        operation_name="Original",
        image_data=np.zeros((2, 2), dtype=np.float32),
        data_type="STM",
        original_image_id=record.image_id,
    )

    session_service.register_new_original(record, node)

    current = history_manager.get_current_node()
    assert current is node
    assert history_manager.get_original_image_record(record.image_id) is not None
