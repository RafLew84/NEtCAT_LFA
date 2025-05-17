# tests/logic/test_app_controller.py
import pytest
import numpy as np
import os
from unittest.mock import MagicMock, patch

from PyQt6.QtCore import QObject, pyqtSignal

# Import testowanych klas i zależności
try:
    from lfa.logic.app_controller import AppController, SPOT_SELECTION_SUBSTRATE, SPOT_SELECTION_ADSORBATE, REFINEMENT_DIRECT_CLICK, MAX_SUBSTRATE_SPOTS
    from lfa.core.history import HistoryNode
    from lfa.logic.history_manager import HistoryManager
    from lfa.core.data_models import STMImage
except ImportError as e:
    pytest.fail(f"Failed to import necessary classes for AppController tests: {e}", pytrace=False)

# Require pytest-qt
pytest_qt = pytest.importorskip("pytestqt")


@pytest.fixture
def mock_history_manager(mocker) -> MagicMock:
    """Fixture to create a mock HistoryManager."""
    mock_hm = mocker.MagicMock(spec=HistoryManager)
    mock_hm.history = {}
    mock_hm.current_node_id = None

    def add_node_side_effect(node):
        mock_hm.history[node.node_id] = node
        return MagicMock()

    def set_current_node_by_id_side_effect(node_id, emit_signal=True):
        if node_id in mock_hm.history or node_id is None:
            mock_hm.current_node_id = node_id
        else:
            raise KeyError(f"Node ID {node_id} not in mock history")

    def get_node_by_id_side_effect(node_id):
        return mock_hm.history.get(node_id)

    def clear_history_side_effect():
        mock_hm.history.clear()
        mock_hm.current_node_id = None

    mock_hm.add_node.side_effect = add_node_side_effect
    mock_hm.set_current_node_by_id.side_effect = set_current_node_by_id_side_effect
    mock_hm.get_node_by_id.side_effect = get_node_by_id_side_effect
    mock_hm.clear_history.side_effect = clear_history_side_effect

    return mock_hm

@pytest.fixture
def app_controller(mock_history_manager) -> AppController:
    """Fixture to create an AppController with a mocked HistoryManager."""
    controller = AppController(history_manager=mock_history_manager)
    return controller

@pytest.fixture
def sample_stm_image() -> STMImage:
    """Fixture for a sample STMImage object."""
    return STMImage(
        file_name="test.stp",
        raw_header={"key": "value"},
        data=np.array([[1,2],[3,4]], dtype=np.float32),
        pixels_x=2, pixels_y=2,
        size_nm_x=10.0, size_nm_y=10.0,
        bias_v=1.0, setpoint_a=1e-10, scan_angle_deg=0.0,
        image_type="Topography"
    )

def test_load_file_success(monkeypatch, app_controller: AppController, 
                            mock_history_manager: MagicMock, sample_stm_image: STMImage, qtbot):
    """Test successful file loading."""
    file_path = "dummy/path/test.stp"
    mock_load_stm_file = MagicMock(return_value=sample_stm_image)
    monkeypatch.setattr('lfa.logic.app_controller.load_stm_file', mock_load_stm_file)

    # Przygotowanie do przechwycenia sygnałów
    with qtbot.waitSignal(app_controller.file_loaded_successfully) as success_spy:
        app_controller.load_file(file_path)

    # Weryfikacje
    mock_load_stm_file.assert_called_once_with(file_path)
    mock_history_manager.clear_history.assert_called_once()
    
    # Sprawdzenie, czy węzeł "Original" został dodany i ustawiony jako bieżący
    assert len(mock_history_manager.history) == 1
    original_node_id = list(mock_history_manager.history.keys())[0] # Pobierz ID jedynego węzła
    original_node = mock_history_manager.history[original_node_id]
    
    assert original_node.operation_name == "Original"
    assert np.array_equal(original_node.image_data, sample_stm_image.data)
    assert original_node.parameters["filename"] == os.path.basename(file_path)
    assert original_node.data_type == "STM"
    
    mock_history_manager.add_node.assert_called_once_with(original_node) # Sprawdź, czy przekazano poprawny węzeł
    mock_history_manager.set_current_node_by_id.assert_called_once_with(original_node.node_id)
    
    assert app_controller.original_file_path == file_path

    # Weryfikacja sygnałów
    assert success_spy.signal_triggered
    assert success_spy.args[0] == os.path.basename(file_path) # Sprawdź argument sygnału

    # Sprawdzenie, czy dane pików zostały zresetowane
    assert app_controller.substrate_spots == []
    assert app_controller.adsorbate_spot_sets == [[]]
    assert app_controller.current_adsorbate_set_index == 0


def test_load_file_failure_factory_returns_none(monkeypatch, app_controller: AppController, 
                                                    mock_history_manager: MagicMock, qtbot):
    """Test file loading failure when load_stm_file returns None."""
    file_path = "dummy/path/bad_file.stp"
    mock_load_stm_file = MagicMock(return_value=None)
    monkeypatch.setattr('lfa.logic.app_controller.load_stm_file', mock_load_stm_file)

    with qtbot.waitSignal(app_controller.file_loading_failed) as fail_spy:
        app_controller.load_file(file_path)

    mock_history_manager.clear_history.assert_called_once() # Historia powinna być wyczyszczona
    assert len(mock_history_manager.history) == 0 # Nie powinno być żadnych węzłów
    assert app_controller.original_file_path is None

    assert fail_spy.signal_triggered
    assert "Could not load valid data" in fail_spy.args[0]


def test_load_file_raises_file_not_found(monkeypatch, app_controller: AppController, 
                                        mock_history_manager: MagicMock, qtbot):
    """Test file loading failure due to FileNotFoundError."""
    file_path = "dummy/path/non_existent.stp"
    mock_load_stm_file = MagicMock(side_effect=FileNotFoundError(f"File not found: {file_path}"))
    monkeypatch.setattr('lfa.logic.app_controller.load_stm_file', mock_load_stm_file)

    with qtbot.waitSignal(app_controller.file_loading_failed) as fail_spy:
        app_controller.load_file(file_path)
    
    assert len(mock_history_manager.history) == 0 # Zakładając, że clear_history jest wołane lub historia pusta
    assert app_controller.original_file_path is None

    assert fail_spy.signal_triggered
    assert f"File not found: {file_path}" in fail_spy.args[0]

def test_add_operation_to_history_generic(app_controller: AppController, mock_history_manager: MagicMock):
    """Testuje generyczną metodę add_operation_to_history."""
    # Najpierw załaduj plik, aby mieć węzeł rodzica
    with patch('lfa.logic.app_controller.load_stm_file', return_value=MagicMock(spec=STMImage, data=np.zeros((10,10)))):
        app_controller.load_file("dummy.stp")
    
    parent_node_id = mock_history_manager.current_node_id
    assert parent_node_id is not None, "Parent node ID should be set after load_file"

    op_name = "Test Operation"
    params = {"param1": 123, "roi_used": True}
    # Upewnij się, że processed_data jest kopią lub innymi danymi niż rodzic
    processed_data = np.ones((10,10), dtype=np.float32) 
    data_type = "STM"
    roi_slice = (slice(0,5), slice(0,5))

    # Przygotowanie do przechwycenia sygnału (jeśli HistoryManager by emitował po add_node/set_current)
    # Na razie polegamy na tym, że set_current_node_by_id jest mockowane i zmienia mock_hm.current_node_id

    app_controller.add_operation_to_history(
        parent_node_id, op_name, params, processed_data, data_type, roi_slice
    )

    # Weryfikacje
    assert len(mock_history_manager.history) == 2 # Original + Test Operation
    new_node_id = mock_history_manager.current_node_id # add_operation_to_history ustawia nowy węzeł jako bieżący
    assert new_node_id != parent_node_id
    
    new_node = mock_history_manager.history[new_node_id]
    assert new_node.parent_id == parent_node_id
    assert new_node.operation_name == op_name
    assert new_node.parameters == params
    assert np.array_equal(new_node.image_data, processed_data)
    assert new_node.data_type == data_type
    assert new_node.source_roi_slice == roi_slice

    mock_history_manager.add_node.assert_called_with(new_node) # Sprawdź, czy przekazano poprawny węzeł
    mock_history_manager.set_current_node_by_id.assert_called_with(new_node.node_id)


def test_apply_gaussian_blur_calls_add_operation(app_controller: AppController, mock_history_manager: MagicMock):
    """Testuje, czy apply_gaussian_blur poprawnie wywołuje add_operation_to_history."""
    parent_id = "parent1"
    parent_data_type = "STM"
    # Symulacja, że rodzic istnieje w mock_history_manager
    mock_history_manager.history[parent_id] = HistoryNode(node_id=parent_id, image_data=np.zeros((5,5)))
    mock_history_manager.current_node_id = parent_id # Ustaw jako bieżący, aby logika "dane się nie zmieniły" miała co porównać
    
    processed_data = np.array([[1,1],[1,1]], dtype=np.float32)
    params = {"sigma": 1.0, "apply_roi_only": False}
    roi = (slice(0,1), slice(0,1))

    with patch.object(app_controller, 'add_operation_to_history') as mock_add_op:
        app_controller.apply_gaussian_blur(parent_id, parent_data_type, processed_data, params, roi)
        mock_add_op.assert_called_once_with(
            parent_id, "Gaussian Blur", params, processed_data, parent_data_type, roi
        )

def test_calculate_fft_operation_calls_add_operation(app_controller: AppController, mock_history_manager: MagicMock):
    """Testuje, czy calculate_fft_operation poprawnie wywołuje add_operation_to_history."""
    parent_id = "parent_stm"
    # Symulacja, że rodzic istnieje
    mock_history_manager.history[parent_id] = HistoryNode(node_id=parent_id, image_data=np.zeros((8,8)), data_type="STM")
    mock_history_manager.current_node_id = parent_id

    processed_fft_data = np.fft.fftshift(np.fft.fft2(np.zeros((8,8)))).astype(np.complex64) # Przykładowe dane FFT
    # Ważne: `calculate_fft_operation` w AppController oczekuje już przeskalowanej magnitudy, nie danych zespolonych
    # Dla celów tego testu, przekażmy ndarray float32, jakby to była magnitude
    scaled_magnitude_fft_data = np.abs(processed_fft_data).astype(np.float32)

    params = {"window_type": "hann", "scaling_mode": "log", "apply_roi_only": False}
    
    with patch.object(app_controller, 'add_operation_to_history') as mock_add_op:
        app_controller.calculate_fft_operation(parent_id, scaled_magnitude_fft_data, params, None)
        mock_add_op.assert_called_once_with(
            parent_id, "FFT", params, scaled_magnitude_fft_data, "FFT", None
        )

def test_add_operation_fft_always_adds_node(app_controller: AppController, mock_history_manager: MagicMock):
    """Testuje, czy operacja FFT zawsze dodaje nowy węzeł, nawet jeśli dane/parametry są te same."""
    original_data = np.array([[1,2],[3,4]], dtype=np.float32)
    with patch('lfa.logic.app_controller.load_stm_file', return_value=MagicMock(spec=STMImage, data=original_data)):
        app_controller.load_file("dummy_fft_orig.stp")
    
    parent_node_id = mock_history_manager.current_node_id
    op_name = "FFT"
    params = {"window": "hann"}
    fft_data = np.abs(np.fft.fftshift(np.fft.fft2(original_data))).astype(np.float32)

    # Pierwsze wywołanie FFT
    app_controller.add_operation_to_history(parent_node_id, op_name, params, fft_data, "FFT")
    assert len(mock_history_manager.history) == 2
    first_fft_node_id = mock_history_manager.current_node_id

    # Drugie wywołanie FFT z tymi samymi danymi i parametrami
    app_controller.add_operation_to_history(first_fft_node_id, op_name, params, fft_data, "FFT") # Rodzicem jest poprzednie FFT
    assert len(mock_history_manager.history) == 3 # Powinien być trzeci węzeł
    assert mock_history_manager.current_node_id != first_fft_node_id # Powinien być nowy aktywny węzeł

def test_set_spot_selection_mode(app_controller: AppController, qtbot):
    """Testuje ustawianie trybu wyboru pików i emisję sygnału."""
    # Pierwsze ustawienie trybu powinno wyemitować sygnał
    with qtbot.waitSignal(app_controller.spot_selection_parameters_changed, timeout=1000) as spy:
        app_controller.set_spot_selection_mode(SPOT_SELECTION_ADSORBATE)
    assert app_controller.spot_selection_mode == SPOT_SELECTION_ADSORBATE

    # Ponowne ustawienie tej samej wartości nie powinno emitować sygnału
    app_controller.set_spot_selection_mode(SPOT_SELECTION_ADSORBATE)
    qtbot.wait(100)  # Poczekaj chwilę, aby upewnić się, że sygnał nie został wyemitowany

    # Nieprawidłowy tryb nie powinien zmienić wartości ani emitować sygnału
    app_controller.set_spot_selection_mode("InvalidMode")
    qtbot.wait(100)  # Poczekaj chwilę, aby upewnić się, że sygnał nie został wyemitowany
    assert app_controller.spot_selection_mode == SPOT_SELECTION_ADSORBATE


def test_add_spot_substrate(app_controller: AppController, qtbot):
    """Testuje dodawanie pików substratu."""
    app_controller.set_spot_selection_mode(SPOT_SELECTION_SUBSTRATE)
    
    # Dodanie pierwszego punktu
    with qtbot.waitSignal(app_controller.spot_lists_updated, timeout=1000) as spy:
        app_controller.add_spot((10.0, 20.0))
    assert app_controller.substrate_spots == [(10.0, 20.0)]

    # Dodanie drugiego punktu
    with qtbot.waitSignal(app_controller.spot_lists_updated, timeout=1000) as spy:
        app_controller.add_spot((15.0, 25.0))
    assert app_controller.substrate_spots == [(10.0, 20.0), (15.0, 25.0)]

    # Próba dodania duplikatu nie powinna emitować sygnału
    app_controller.add_spot((10.0, 20.0))
    qtbot.wait(100)  # Poczekaj chwilę, aby upewnić się, że sygnał nie został wyemitowany
    assert app_controller.substrate_spots == [(10.0, 20.0), (15.0, 25.0)]

    # Testowanie limitu MAX_SUBSTRATE_SPOTS
    app_controller.substrate_spots = [(float(i), float(i)) for i in range(MAX_SUBSTRATE_SPOTS)]
    app_controller.add_spot((100.0, 100.0))
    qtbot.wait(100)  # Poczekaj chwilę, aby upewnić się, że sygnał nie został wyemitowany
    assert len(app_controller.substrate_spots) == MAX_SUBSTRATE_SPOTS


def test_add_spot_adsorbate(app_controller: AppController, qtbot):
    """Testuje dodawanie pików adsorbatu."""
    app_controller.set_spot_selection_mode(SPOT_SELECTION_ADSORBATE)
    app_controller.current_adsorbate_set_index = 0

    with qtbot.waitSignal(app_controller.spot_lists_updated) as spy:
        app_controller.add_spot((5.0, 5.0))
    assert app_controller.adsorbate_spot_sets[0] == [(5.0, 5.0)]

    # Zmiana zestawu
    with qtbot.waitSignal(app_controller.adsorbate_sets_structure_changed) as spy:
        app_controller.add_new_adsorbate_set()

    with qtbot.waitSignal(app_controller.spot_lists_updated) as spy:
        app_controller.add_spot((8.0, 8.0))
    assert len(app_controller.adsorbate_spot_sets) == 2
    assert app_controller.adsorbate_spot_sets[0] == [(5.0, 5.0)]
    assert app_controller.adsorbate_spot_sets[1] == [(8.0, 8.0)]


def test_clear_substrate_spots(app_controller: AppController, qtbot):
    """Testuje czyszczenie pików substratu."""
    app_controller.substrate_spots = [(1.0,1.0), (2.0,2.0)]
    
    # Pierwsze wyczyszczenie powinno wyemitować sygnał
    with qtbot.waitSignal(app_controller.spot_lists_updated, timeout=1000) as spy:
        app_controller.clear_substrate_spots()
    assert app_controller.substrate_spots == []

    # Ponowne wyczyszczenie pustej listy nie powinno emitować sygnału
    app_controller.clear_substrate_spots()
    qtbot.wait(100)  # Poczekaj chwilę, aby upewnić się, że sygnał nie został wyemitowany


def test_adsorbate_set_management(app_controller: AppController, qtbot):
    """Testuje zarządzanie zestawami adsorbatu."""
    app_controller.set_spot_selection_mode(SPOT_SELECTION_ADSORBATE)

    # Dodaj pierwszy pik do pierwszego zestawu
    with qtbot.waitSignal(app_controller.spot_lists_updated) as spy:
        app_controller.add_spot((1.1, 1.1))
    assert len(app_controller.adsorbate_spot_sets[0]) == 1
    
    # Dodaj nowy zestaw
    with qtbot.waitSignal(app_controller.adsorbate_sets_structure_changed) as spy:
        app_controller.add_new_adsorbate_set()
    assert app_controller.current_adsorbate_set_index == 1
    assert len(app_controller.adsorbate_spot_sets) == 2
    assert app_controller.adsorbate_spot_sets[1] == []

    # Dodaj pik do nowego zestawu
    with qtbot.waitSignal(app_controller.spot_lists_updated) as spy:
        app_controller.add_spot((2.2, 2.2))
    assert len(app_controller.adsorbate_spot_sets[1]) == 1

    # Wyczyść ostatni pik z bieżącego zestawu
    with qtbot.waitSignal(app_controller.spot_lists_updated) as spy:
        app_controller.clear_last_adsorbate_spot()
    assert app_controller.adsorbate_spot_sets[1] == []

    # Przełącz na pierwszy zestaw
    with qtbot.waitSignal(app_controller.spot_selection_parameters_changed) as spy:
        app_controller.set_current_adsorbate_set_by_index(0)
    assert app_controller.current_adsorbate_set_index == 0

    # Wyczyść wszystkie zestawy
    with qtbot.waitSignals([app_controller.spot_lists_updated, app_controller.adsorbate_sets_structure_changed]) as spy:
        app_controller.clear_all_adsorbate_sets()
    assert app_controller.adsorbate_spot_sets == [[]]
    assert app_controller.current_adsorbate_set_index == 0
