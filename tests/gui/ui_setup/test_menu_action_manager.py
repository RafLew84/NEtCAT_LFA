# tests/gui/ui_setup/test_menu_action_manager.py
import pytest
from unittest.mock import MagicMock, call

from PyQt6.QtWidgets import QMainWindow, QMenuBar, QMenu
from PyQt6.QtGui import QAction

# Import testowanej klasy
try:
    from lfa.gui.ui_setup.menu_action_manager import MenuActionManager
except ImportError as e:
    pytest.fail(f"Could not import MenuActionManager: {e}", pytrace=False)

# Wymaga pytest-qt
pytest.importorskip("pytestqt.qt_compat", reason="pytest-qt not found, skipping GUI tests")


class TestMainWindow(QMainWindow):
    """Testowa klasa MainWindow z wymaganymi slotami."""
    def __init__(self):
        super().__init__()
        self.open_file_dialog = MagicMock()
        self.close = MagicMock()
        self.open_gaussian_blur_dialog = MagicMock()
        self.open_gaussian_sharpening_dialog = MagicMock()
        self.open_plane_leveling_dialog = MagicMock()
        self.open_median_filter_dialog = MagicMock()
        self.open_nlmeans_dialog = MagicMock()
        self.open_bm3d_dialog = MagicMock()
        self.open_fft_dialog = MagicMock()
        self.show_about_dialog = MagicMock()


@pytest.fixture
def main_window(qtbot) -> TestMainWindow:
    """Fixture to create a real QMainWindow with required slots."""
    window = TestMainWindow()
    qtbot.addWidget(window)
    return window


@pytest.fixture
def menu_manager(main_window) -> MenuActionManager:
    """Fixture to create a MenuActionManager with a real MainWindow."""
    manager = MenuActionManager(main_window=main_window)
    return manager


def test_menu_manager_initialization(menu_manager: MenuActionManager, main_window: TestMainWindow):
    """Testuje, czy MenuActionManager jest poprawnie inicjalizowany i tworzy menu."""
    # Sprawdź, czy główne menu zostały dodane do paska menu
    expected_menu_titles = ["&File", "&Preprocessing", "&Analysis", "&View", "&Help"]
    menu_bar = main_window.menuBar()
    actual_menu_titles = [menu.title() for menu in menu_bar.findChildren(QMenu)]
    
    for title in expected_menu_titles:
        assert title in actual_menu_titles, f"Menu '{title}' not created"
    
    # Sprawdź, czy view_menu zostało ustawione w main_window
    assert hasattr(main_window, 'view_menu')
    assert main_window.view_menu is not None
    assert main_window.view_menu.title() == "&View"


def test_file_menu_actions_created_and_connected(menu_manager: MenuActionManager, main_window: TestMainWindow, qtbot):
    """Testuje tworzenie akcji w menu File i ich połączenia."""
    file_menu = next(menu for menu in main_window.menuBar().findChildren(QMenu) if menu.title() == "&File")
    assert file_menu is not None, "File menu not found"

    # Akcja Open
    open_action = menu_manager.get_action("file", "open")
    assert open_action is not None
    assert open_action.text() == "&Open..."
    assert hasattr(main_window, 'open_file_dialog')

    # Akcja Exit
    exit_action = menu_manager.get_action("file", "exit")
    assert exit_action is not None
    assert exit_action.text() == "&Exit"
    assert hasattr(main_window, 'close')


def test_analysis_menu_actions_created(menu_manager: MenuActionManager, main_window: TestMainWindow):
    """Testuje tworzenie akcji w menu Analysis."""
    analysis_menu = next(menu for menu in main_window.menuBar().findChildren(QMenu) if menu.title() == "&Analysis")
    assert analysis_menu is not None
    
    fft_action = menu_manager.get_action("analysis", "fft")
    assert fft_action is not None
    assert fft_action.text() == "Calculate &FFT..."
    assert fft_action.isEnabled() is False
    assert main_window.fft_action is fft_action
    assert hasattr(main_window, 'open_fft_dialog')


def test_help_menu_actions_created(menu_manager: MenuActionManager, main_window: TestMainWindow):
    """Testuje tworzenie akcji w menu Help."""
    help_menu = next(menu for menu in main_window.menuBar().findChildren(QMenu) if menu.title() == "&Help")
    assert help_menu is not None

    about_action = menu_manager.get_action("help", "about")
    assert about_action is not None
    assert about_action.text() == "&About LFA..."
    assert hasattr(main_window, 'show_about_dialog')