import numpy as np
import pytest
from PyQt6.QtWidgets import QListWidget

from lfa.core.data_models import OriginalImageRecord
from lfa.core.history import HistoryNode
from lfa.logic.app_controller import AppController
from lfa.logic.history_manager import HistoryManager


@pytest.fixture
def controller_with_branch(qtbot):
    list_widget = QListWidget()
    qtbot.addWidget(list_widget)
    history_manager = HistoryManager(list_widget)
    controller = AppController(history_manager)

    record = OriginalImageRecord(display_name="Original Image 1")
    history_manager.register_original_image(record)

    root = HistoryNode(
        operation_name="Original",
        parameters={"original_label": record.display_name, "source_image_label": record.display_name},
        image_data=np.zeros((4, 4), dtype=np.float32),
        data_type="STM",
        original_image_id=record.image_id,
    )
    history_manager.add_node(root)

    child = HistoryNode(
        parent_id=root.node_id,
        operation_name="Gaussian Blur",
        parameters={"sigma": 1.0, "source_image_label": record.display_name},
        image_data=np.ones((4, 4), dtype=np.float32),
        data_type="STM",
        original_image_id=record.image_id,
    )
    history_manager.add_node(child)
    history_manager.set_current_node_by_id(child.node_id)

    return controller, history_manager, record, root, child


def test_delete_history_step_keeps_context(controller_with_branch):
    controller, history_manager, record, root, child = controller_with_branch
    controller.substrate_spots = [(0.0, 0.0)]

    success = controller.delete_history_step(child.node_id)

    assert success is True
    assert child.node_id not in history_manager.history
    assert root.node_id in history_manager.history
    assert history_manager.current_node_id == root.node_id
    assert controller.substrate_spots == [(0.0, 0.0)]


def test_delete_original_image_clears_state(controller_with_branch):
    controller, history_manager, record, root, child = controller_with_branch
    controller.substrate_spots = [(0.0, 0.0)]
    controller.adsorbate_spot_sets = [[(1.0, 1.0)]]
    controller.adsorbate_visual_offsets_nm = {0: (0.5, 0.5)}

    success = controller.delete_original_image(record.image_id)

    assert success is True
    assert history_manager.history == {}
    assert history_manager.current_node_id is None
    assert record.image_id not in history_manager.original_images
    assert controller.substrate_spots == []
    assert controller.adsorbate_spot_sets == [[]]
    assert controller.adsorbate_visual_offsets_nm == {0: (0.0, 0.0)}
