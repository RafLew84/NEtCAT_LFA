import numpy as np
import pytest

pytest.importorskip("PyQt6", reason="PyQt6 is required for history orchestrator tests")
pytest.importorskip("pytestqt", reason="pytest-qt provides the qtbot fixture")

from PyQt6.QtWidgets import QListWidget

from lfa.core.data_models import OriginalImageRecord
from lfa.core.history import HistoryNode
from lfa.logic.history_manager import HistoryManager
from lfa.logic.services import HistoryOrchestrator


def _prepare_history(qtbot):
    widget = QListWidget()
    qtbot.addWidget(widget)
    manager = HistoryManager(widget)
    service = HistoryOrchestrator(manager)
    return manager, service


def test_history_orchestrator_returns_image_copy(qtbot):
    history_manager, service = _prepare_history(qtbot)

    record = OriginalImageRecord(display_name="Sample")
    history_manager.register_original_image(record)

    image = np.arange(4, dtype=np.float32).reshape(2, 2)
    node = HistoryNode(
        operation_name="Original",
        image_data=image,
        data_type="STM",
        original_image_id=record.image_id,
        parameters={"original_label": record.display_name, "source_image_label": record.display_name},
    )
    history_manager.add_node(node)
    history_manager.set_current_node_by_id(node.node_id)

    copy = service.get_current_image_data_copy()
    assert copy is not None
    assert np.array_equal(copy, image)
    assert copy is not image
    assert not np.shares_memory(copy, image)

    info = service.get_current_node_info_for_dialogs()
    assert info is not None
    node_id, data_type, image_copy, original_id, label = info
    assert node_id == node.node_id
    assert data_type == "STM"
    assert np.array_equal(image_copy, image)
    assert image_copy is not image
    assert original_id == record.image_id
    assert label == record.display_name


def test_history_orchestrator_handles_missing_node(qtbot):
    _, service = _prepare_history(qtbot)
    assert service.get_current_node() is None
    assert service.get_current_image_data_copy() is None
    assert service.get_current_node_info_for_dialogs() is None


def test_history_orchestrator_adds_and_selects_node(qtbot):
    history_manager, service = _prepare_history(qtbot)

    record = OriginalImageRecord(display_name="Sample")
    history_manager.register_original_image(record)

    node = HistoryNode(
        operation_name="Derived",
        image_data=np.zeros((2, 2), dtype=np.float32),
        data_type="STM",
        original_image_id=record.image_id,
    )

    service.add_node_and_select(node)
    current = history_manager.get_current_node()
    assert current is node
