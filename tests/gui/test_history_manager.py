# tests/gui/test_history_manager.py

import numpy as np
import pytest
from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import QListWidget

# Import klas do testowania i pomocniczych
# Dostosuj ścieżki importu, jeśli HistoryManager jest w innym miejscu (np. lfa.logic)
try:
    from lfa.core.data_models import OriginalImageRecord
    from lfa.core.history import HistoryNode
    from lfa.logic.history_manager import HistoryManager
except ImportError as e:
    pytest.fail(f"Could not import HistoryManager or HistoryNode: {e}", pytrace=False)

# Fixtury dla pytest-qt (automatycznie dostępne, jeśli pytest-qt jest zainstalowany)
# qtbot: do interakcji z widgetami Qt

# --- Fixtures ---

@pytest.fixture
def list_widget(qtbot) -> QListWidget:
    """Tworzy pusty QListWidget dla testów."""
    widget = QListWidget()
    qtbot.addWidget(widget) # Rejestruje widget do automatycznego czyszczenia przez qtbot
    return widget

@pytest.fixture
def history_manager(list_widget) -> HistoryManager:
    """Tworzy instancję HistoryManager z podłączonym list_widget."""
    manager = HistoryManager(history_list_widget=list_widget)
    return manager

@pytest.fixture
def sample_nodes() -> list[HistoryNode]:
    """Zwraca listę przykładowych węzłów historii."""
    node1 = HistoryNode(node_id="node1", operation_name="Original")
    node2 = HistoryNode(node_id="node2", parent_id="node1", operation_name="Blur", parameters={"sigma": 1.0})
    node3 = HistoryNode(node_id="node3", parent_id="node2", operation_name="FFT", parameters={"window": "hann"})
    return [node1, node2, node3]

# --- Test Functions ---

def test_history_manager_initialization(history_manager: HistoryManager, list_widget: QListWidget):
    """Testuje poprawność inicjalizacji HistoryManager."""
    assert history_manager.history == {}, "Historia powinna być pusta na starcie."
    assert history_manager.current_node_id is None, "current_node_id powinien być None na starcie."
    assert history_manager.history_list_widget is list_widget, "Niepoprawnie przypisany list_widget."
    assert list_widget.count() == 0, "List_widget powinien być pusty na starcie."

def test_add_node(history_manager: HistoryManager, list_widget: QListWidget):
    """Testuje dodawanie pojedynczego węzła."""
    node = HistoryNode(node_id="test_node_01", operation_name="TestOp")
    
    list_item = history_manager.add_node(node)

    assert node.node_id in history_manager.history, "Węzeł nie został dodany do słownika historii."
    assert history_manager.history[node.node_id] is node, "Niepoprawny obiekt węzła w słowniku."
    assert list_widget.count() == 1, "Nie dodano elementu do QListWidget."
    assert list_widget.item(0) is list_item, "Zwrócony QListWidgetItem nie zgadza się z tym w widgecie."
    assert list_widget.item(0).text() == node.get_display_text(), "Tekst elementu w QListWidget jest niepoprawny."
    assert list_widget.item(0).data(Qt.ItemDataRole.UserRole) == node.node_id, "Dane użytkownika (ID węzła) w QListWidgetItem są niepoprawne."

def test_add_multiple_nodes(history_manager: HistoryManager, list_widget: QListWidget, sample_nodes: list[HistoryNode]):
    """Testuje dodawanie wielu węzłów."""
    for node in sample_nodes:
        history_manager.add_node(node)
    
    assert len(history_manager.history) == len(sample_nodes), "Niepoprawna liczba węzłów w słowniku historii."
    assert list_widget.count() == len(sample_nodes), "Niepoprawna liczba elementów w QListWidget."
    for i, node in enumerate(sample_nodes):
        assert list_widget.item(i).data(Qt.ItemDataRole.UserRole) == node.node_id

def test_add_duplicate_node_id(history_manager: HistoryManager, list_widget: QListWidget):
    """Testuje próbę dodania węzła o ID, które już istnieje."""
    node1 = HistoryNode(node_id="duplicate_id", operation_name="Op1")
    node2 = HistoryNode(node_id="duplicate_id", operation_name="Op2_Different") # To samo ID
    
    history_manager.add_node(node1)
    item_returned = history_manager.add_node(node2) # Próba dodania duplikatu

    assert len(history_manager.history) == 1, "Duplikat ID nie powinien zostać dodany do słownika."
    assert list_widget.count() == 1, "Duplikat ID nie powinien stworzyć nowego elementu w QListWidget."
    assert history_manager.history["duplicate_id"].operation_name == "Op1", "Operacja oryginalnego węzła powinna pozostać."
    assert item_returned is list_widget.item(0), "Powinien zwrócić istniejący QListWidgetItem."

def test_add_invalid_node(history_manager: HistoryManager, list_widget: QListWidget):
    """Testuje próbę dodania None lub niepoprawnego węzła."""
    assert history_manager.add_node(None) is None # type: ignore
    assert list_widget.count() == 0
    # Można by dodać test dla węzła bez node_id, jeśli konstruktor HistoryNode by na to pozwalał.

def test_get_node_by_id(history_manager: HistoryManager, sample_nodes: list[HistoryNode]):
    """Testuje pobieranie węzła po ID."""
    for node in sample_nodes:
        history_manager.add_node(node)
    
    assert history_manager.get_node_by_id(sample_nodes[1].node_id) is sample_nodes[1]
    assert history_manager.get_node_by_id("non_existent_id") is None
    assert history_manager.get_node_by_id(None) is None # type: ignore

def test_get_root_node_for_node(history_manager: HistoryManager, sample_nodes: list[HistoryNode]):
    """Testuje znajdowanie węzła głównego."""
    node1, node2, node3 = sample_nodes
    history_manager.add_node(node1)
    history_manager.add_node(node2) # parent_id="node1"
    history_manager.add_node(node3) # parent_id="node2"

    assert history_manager.get_root_node_for_node(node3.node_id) is node1
    assert history_manager.get_root_node_for_node(node2.node_id) is node1
    assert history_manager.get_root_node_for_node(node1.node_id) is node1 # Korzeń jest swoim korzeniem
    assert history_manager.get_root_node_for_node("non_existent") is None
    assert history_manager.get_root_node_for_node(None) is None

def test_get_root_node_with_original_name(history_manager: HistoryManager):
    """Testuje znajdowanie korzenia, gdy korzeń ma nazwę "Original"."""
    root = HistoryNode(node_id="root", operation_name="Original")
    child = HistoryNode(node_id="child", parent_id="root", operation_name="Blur")
    grandchild = HistoryNode(node_id="grandchild", parent_id="child", operation_name="FFT")
    history_manager.add_node(root)
    history_manager.add_node(child)
    history_manager.add_node(grandchild)
    
    assert history_manager.get_root_node_for_node(grandchild.node_id) is root

def test_get_root_node_broken_history(history_manager: HistoryManager):
    """Testuje znajdowanie korzenia w przypadku przerwanego łańcucha rodziców."""
    node_a = HistoryNode(node_id="a")
    node_b = HistoryNode(node_id="b", parent_id="a")
    node_c = HistoryNode(node_id="c", parent_id="broken_parent_id") # Rodzic nie istnieje
    history_manager.add_node(node_a)
    history_manager.add_node(node_b)
    history_manager.add_node(node_c)

    # Powinien zwrócić sam node_c jako najstarszego przodka, którego można prześledzić
    assert history_manager.get_root_node_for_node(node_c.node_id) is node_c

def _add_original_image(history_manager: HistoryManager, label: str) -> tuple[OriginalImageRecord, HistoryNode]:
    record = OriginalImageRecord(display_name=label)
    history_manager.register_original_image(record)
    root_node = HistoryNode(
        operation_name="Original",
        parameters={"original_label": label, "source_image_label": label},
        image_data=np.zeros((4, 4), dtype=np.float32),
        data_type="STM",
        original_image_id=record.image_id,
    )
    history_manager.add_node(root_node)
    return record, root_node


def test_delete_node_branch_removes_descendants(history_manager: HistoryManager, list_widget: QListWidget, qtbot):
    rec, root = _add_original_image(history_manager, "Original Image 1")
    child = HistoryNode(
        parent_id=root.node_id,
        operation_name="Gaussian Blur",
        parameters={"sigma": 1.2, "source_image_label": rec.display_name},
        image_data=np.ones((4, 4), dtype=np.float32),
        data_type="STM",
        original_image_id=rec.image_id,
    )
    grandchild = HistoryNode(
        parent_id=child.node_id,
        operation_name="FFT",
        parameters={"scaling_mode": "log", "source_image_label": rec.display_name},
        image_data=np.ones((4, 4), dtype=np.float32),
        data_type="FFT",
        original_image_id=rec.image_id,
    )
    history_manager.add_node(child)
    history_manager.add_node(grandchild)
    history_manager.set_current_node_by_id(grandchild.node_id)

    result = history_manager.delete_node_branch(child.node_id)
    qtbot.wait(0)

    assert result is not None
    assert child.node_id not in history_manager.history
    assert grandchild.node_id not in history_manager.history
    assert root.node_id in history_manager.history
    assert history_manager.current_node_id == root.node_id
    assert list_widget.count() == 1
    assert child.node_id in result["deleted_node_ids"]
    assert grandchild.node_id in result["deleted_node_ids"]
    assert result["removed_original_image_id"] is None


def test_delete_original_image_branch_removes_all(history_manager: HistoryManager, list_widget: QListWidget, qtbot):
    rec1, root1 = _add_original_image(history_manager, "Original Image 1")
    child1 = HistoryNode(
        parent_id=root1.node_id,
        operation_name="Median Filter",
        parameters={"size": 3, "source_image_label": rec1.display_name},
        image_data=np.ones((4, 4), dtype=np.float32),
        data_type="STM",
        original_image_id=rec1.image_id,
    )
    history_manager.add_node(child1)

    rec2, root2 = _add_original_image(history_manager, "Original Image 2")
    child2 = HistoryNode(
        parent_id=root2.node_id,
        operation_name="FFT",
        parameters={"scaling_mode": "log", "source_image_label": rec2.display_name},
        image_data=np.ones((4, 4), dtype=np.float32),
        data_type="FFT",
        original_image_id=rec2.image_id,
    )
    history_manager.add_node(child2)
    history_manager.set_current_node_by_id(child2.node_id)

    result = history_manager.delete_original_image_branch(rec1.image_id)
    qtbot.wait(0)

    assert result is not None
    assert result["removed_original_image_id"] == rec1.image_id
    assert root1.node_id not in history_manager.history
    assert child1.node_id not in history_manager.history
    assert rec1.image_id not in history_manager.original_images
    assert rec1.image_id not in history_manager.iter_original_image_ids()
    assert rec2.image_id in history_manager.original_images
    assert list_widget.count() == 2
    assert history_manager.current_node_id == child2.node_id


def test_multi_original_images_grouping(history_manager: HistoryManager, list_widget: QListWidget):
    """Węzły powinny być grupowane według obrazu źródłowego i zachowywać porządek."""
    rec1, root1 = _add_original_image(history_manager, "Original Image 1")
    child1 = HistoryNode(
        parent_id=root1.node_id,
        operation_name="Gaussian Blur",
        parameters={"sigma": 1.0, "source_image_label": rec1.display_name},
        image_data=np.ones((4, 4), dtype=np.float32),
        data_type="STM",
        original_image_id=rec1.image_id,
    )
    history_manager.add_node(child1)

    rec2, root2 = _add_original_image(history_manager, "Original Image 2")

    assert list_widget.count() == 3
    assert list_widget.item(0).text() == rec1.display_name
    assert list_widget.item(0).font().bold()
    assert "[Original Image 1]" in list_widget.item(1).text()
    assert list_widget.item(2).text() == rec2.display_name

    ids = history_manager.iter_original_image_ids()
    assert ids == [rec1.image_id, rec2.image_id]
    assert list_widget.item(1).data(Qt.ItemDataRole.UserRole + 2) == rec1.image_id
    assert list_widget.item(2).data(Qt.ItemDataRole.UserRole + 2) == rec2.image_id


def test_refresh_widget_preserves_order(history_manager: HistoryManager, list_widget: QListWidget):
    """Odświeżenie widoku powinno zachować kolejność i zaznaczenie."""
    rec1, root1 = _add_original_image(history_manager, "Original Image 1")
    child1 = HistoryNode(
        parent_id=root1.node_id,
        operation_name="Gaussian Blur",
        parameters={"sigma": 0.5, "source_image_label": rec1.display_name},
        image_data=np.zeros((2, 2), dtype=np.float32),
        data_type="STM",
        original_image_id=rec1.image_id,
    )
    history_manager.add_node(child1)
    rec2, root2 = _add_original_image(history_manager, "Original Image 2")

    history_manager.set_current_node_by_id(child1.node_id)
    history_manager.refresh_widget()

    assert list_widget.count() == 3
    assert list_widget.item(0).text() == rec1.display_name
    assert "[Original Image 1]" in list_widget.item(1).text()
    assert list_widget.item(2).text() == rec2.display_name
    assert history_manager.get_current_node() is history_manager.history[child1.node_id]
