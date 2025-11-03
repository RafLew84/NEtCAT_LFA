# tests/gui/ui_setup/test_dock_panel_manager.py

import pytest
from PyQt6.QtWidgets import QDockWidget, QListWidget, QMainWindow, QMenu

# Import testowanej klasy
try:
    from lfa.gui.panels.fft_analysis_panel import FFTAnalysisPanel
    from lfa.gui.ui_setup.dock_panel_manager import DockPanelManager

    # Importuj klasy widgetów, które są przekazywane do DockPanelManager
    from lfa.gui.widgets.metadata_widget import MetadataWidget
except ImportError as e:
    pytest.fail(f"Could not import DockPanelManager or its dependencies: {e}", pytrace=False)

# Wymaga pytest-qt
pytest.importorskip("pytestqt.qt_compat", reason="pytest-qt not found, skipping GUI tests")


class TestMainWindow(QMainWindow):
    """Testowa klasa MainWindow z wymaganymi slotami."""
    def __init__(self):
        super().__init__()
        self.view_menu = QMenu("&View", self)
        self.fft_analysis_dock = None


@pytest.fixture
def main_window(qtbot) -> TestMainWindow:
    """Fixture to create a real QMainWindow with required slots."""
    window = TestMainWindow()
    qtbot.addWidget(window)
    return window


@pytest.fixture
def history_list_widget(qtbot) -> QListWidget:
    """Fixture for a real QListWidget (content of history_dock)."""
    widget = QListWidget()
    qtbot.addWidget(widget)
    return widget


@pytest.fixture
def metadata_widget(qtbot) -> MetadataWidget:
    """Fixture for a real MetadataWidget."""
    widget = MetadataWidget()
    qtbot.addWidget(widget)
    return widget


@pytest.fixture
def fft_analysis_panel_widget(qtbot) -> FFTAnalysisPanel:
    """Fixture for a real FFTAnalysisPanel."""
    widget = FFTAnalysisPanel()
    qtbot.addWidget(widget)
    return widget


@pytest.fixture
def dock_manager(main_window, history_list_widget,
                 metadata_widget, fft_analysis_panel_widget) -> DockPanelManager:
    """Fixture to create a DockPanelManager with real widgets."""
    manager = DockPanelManager(
        main_window=main_window,
        history_list_widget=history_list_widget,
        metadata_widget=metadata_widget,
        fft_analysis_panel_widget=fft_analysis_panel_widget
    )
    return manager


def test_dock_manager_initialization_creates_docks(dock_manager: DockPanelManager, main_window: TestMainWindow):
    """Testuje, czy DockPanelManager tworzy wszystkie oczekiwane doki."""
    # Sprawdź, czy wewnętrzne referencje do doków w managerze zostały ustawione
    assert dock_manager.history_dock is not None
    assert dock_manager.metadata_dock is not None
    assert dock_manager.fft_analysis_dock is not None
    
    # Sprawdź, czy fft_analysis_dock został przypisany do main_window
    assert main_window.fft_analysis_dock is dock_manager.fft_analysis_dock

    # Sprawdź, czy akcje toggleViewAction zostały dodane do view_menu
    assert len(main_window.view_menu.actions()) == 3


def test_history_dock_setup(dock_manager: DockPanelManager, 
                            main_window: TestMainWindow, 
                            history_list_widget: QListWidget):
    """Szczegółowy test konfiguracji History Dock."""
    history_dock = dock_manager.history_dock
    assert isinstance(history_dock, QDockWidget)
    assert history_dock.widget() is history_list_widget
    assert history_dock.windowTitle() == "History"
    
    # Sprawdzenie, czy akcja dla tego doku została dodana do view_menu
    action_titles = [action.text() for action in main_window.view_menu.actions()]
    assert "History Panel" in action_titles


def test_metadata_dock_setup(dock_manager: DockPanelManager, 
                                main_window: TestMainWindow, 
                                metadata_widget: MetadataWidget):
    """Szczegółowy test konfiguracji Metadata Dock."""
    metadata_dock = dock_manager.metadata_dock
    assert isinstance(metadata_dock, QDockWidget)
    assert metadata_dock.widget() is metadata_widget
    assert metadata_dock.windowTitle() == "Metadata"
    
    action_titles = [action.text() for action in main_window.view_menu.actions()]
    assert "Metadata Panel" in action_titles


def test_fft_analysis_dock_setup(dock_manager: DockPanelManager, 
                                    main_window: TestMainWindow, 
                                    fft_analysis_panel_widget: FFTAnalysisPanel):
    """Szczegółowy test konfiguracji FFT Analysis Tools Dock."""
    fft_dock = dock_manager.fft_analysis_dock
    assert isinstance(fft_dock, QDockWidget)
    assert fft_dock.widget() is fft_analysis_panel_widget
    assert fft_dock.windowTitle() == "FFT Analysis Tools"
    assert fft_dock.isVisible() is False  # Powinien być domyślnie niewidoczny
    
    action_titles = [action.text() for action in main_window.view_menu.actions()]
    assert "FFT Analysis Tools Panel" in action_titles
    
    # Sprawdzenie, czy DockPanelManager ustawił atrybut w MainWindow
    assert main_window.fft_analysis_dock is fft_dock


def test_dock_creation_when_view_menu_missing(mocker, history_list_widget, metadata_widget, fft_analysis_panel_widget):
    """Testuje zachowanie, gdy view_menu nie jest dostępne w MainWindow."""
    mock_mw_no_view_menu = mocker.MagicMock(spec=QMainWindow)
    # Celowo nie ustawiamy mock_mw_no_view_menu.view_menu
    # lub ustawiamy na None, jeśli atrybut by istniał
    if hasattr(mock_mw_no_view_menu, 'view_menu'):
        del mock_mw_no_view_menu.view_menu  # Upewnij się, że nie ma atrybutu
    
    # Sprawdź, czy DockPanelManager loguje błąd, ale nie rzuca wyjątku
    mock_logger = mocker.MagicMock()
    mocker.patch('lfa.gui.ui_setup.dock_panel_manager.logger', mock_logger)
    
    manager = DockPanelManager(
        main_window=mock_mw_no_view_menu,
        history_list_widget=history_list_widget,
        metadata_widget=metadata_widget,
        fft_analysis_panel_widget=fft_analysis_panel_widget
    )
    
    # Wszystkie doki powinny być None, bo nie udało się ich stworzyć poprawnie
    assert manager.history_dock is None
    assert manager.metadata_dock is None
    assert manager.fft_analysis_dock is None
    
    # Sprawdź, czy log.error został wywołany (przynajmniej raz na próbę stworzenia doku)
    assert mock_logger.error.call_count >= 1
    # Sprawdź treść pierwszego logu
    assert "main_window.view_menu is not available" in mock_logger.error.call_args_list[0][0][0]
