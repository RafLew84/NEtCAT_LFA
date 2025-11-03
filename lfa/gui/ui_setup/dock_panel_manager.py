# lfa/gui/ui_setup/dock_panel_manager.py
import logging

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import QDockWidget, QListWidget, QMainWindow, QWidget

try:
    from ..panels.fft_analysis_panel import FFTAnalysisPanel
    from ..widgets.metadata_widget import MetadataWidget
except ImportError as e:
    logging.error(f"DockPanelManager: Could not import panel/widget classes: {e}")
    MetadataWidget = None
    FFTAnalysisPanel = None


logger = logging.getLogger(__name__)

class DockPanelManager:
    def __init__(self,
                 main_window: QMainWindow,
                 history_list_widget: QListWidget,
                 metadata_widget: MetadataWidget | None, 
                 fft_analysis_panel_widget: FFTAnalysisPanel | None
                 ):
        """
        Manages the creation and setup of dockable panels for the main window.

        Args:
            main_window (QMainWindow): The main window instance.
            history_list_widget (QListWidget): The widget for the history dock.
            metadata_widget (MetadataWidget): The widget for the metadata dock.
            fft_analysis_panel_widget (FFTAnalysisPanel): The widget for the FFT analysis dock.
        """
        self.main_window = main_window
        self.history_list_widget = history_list_widget
        self.metadata_widget = metadata_widget
        self.fft_analysis_panel_widget = fft_analysis_panel_widget

        self.history_dock: QDockWidget | None = None
        self.metadata_dock: QDockWidget | None = None
        self.fft_analysis_dock: QDockWidget | None = None

        self._create_all_dock_panels()
        logger.debug("DockPanelManager initialized and dock panels created.")

    def _create_dock_widget(self, title: str, widget_content: QWidget,
                            allowed_areas=Qt.DockWidgetArea.AllDockWidgetAreas,
                            initial_area=Qt.DockWidgetArea.LeftDockWidgetArea,
                            visible_by_default=True) -> QDockWidget | None:
        """Helper method to create and configure a QDockWidget."""
        if not hasattr(self.main_window, 'view_menu') or not self.main_window.view_menu: # pragma: no cover
            logger.error(f"Cannot create dock '{title}': main_window.view_menu is not available.")
            return None

        dock = QDockWidget(title, self.main_window)
        dock.setWidget(widget_content)
        dock.setAllowedAreas(allowed_areas)
        self.main_window.addDockWidget(initial_area, dock)
        dock.setVisible(visible_by_default)

        toggle_action = dock.toggleViewAction()
        toggle_action.setText(f"{title} Panel")
        self.main_window.view_menu.addAction(toggle_action)
        
        return dock

    def _setup_history_dock(self):
        """Creates and configures the history list dock widget."""
        self.history_dock = self._create_dock_widget(
            title="History",
            widget_content=self.history_list_widget,
            initial_area=Qt.DockWidgetArea.LeftDockWidgetArea,
            allowed_areas=(Qt.DockWidgetArea.LeftDockWidgetArea | Qt.DockWidgetArea.RightDockWidgetArea)
        )
        if self.history_dock:
             logger.debug("History dock panel created.")

    def _setup_metadata_dock(self):
        """Creates and configures the metadata dock widget."""
        if not self.metadata_widget:
            logger.warning("MetadataWidget not available, skipping metadata dock creation.")
            return
            
        self.metadata_dock = self._create_dock_widget(
            title="Metadata",
            widget_content=self.metadata_widget,
            initial_area=Qt.DockWidgetArea.RightDockWidgetArea,
            allowed_areas=(Qt.DockWidgetArea.LeftDockWidgetArea | Qt.DockWidgetArea.RightDockWidgetArea)
        )
        if self.metadata_dock:
            logger.debug("Metadata dock panel created.")


    def _setup_fft_analysis_dock(self):
        """Creates and configures the FFT Analysis Tools dock widget."""
        if not self.fft_analysis_panel_widget:
            logger.warning("FFTAnalysisPanel widget not available, skipping FFT analysis dock creation.")
            return

        self.fft_analysis_dock = self._create_dock_widget(
            title="FFT Analysis Tools",
            widget_content=self.fft_analysis_panel_widget,
            initial_area=Qt.DockWidgetArea.RightDockWidgetArea,
            allowed_areas=(Qt.DockWidgetArea.LeftDockWidgetArea | Qt.DockWidgetArea.RightDockWidgetArea),
            visible_by_default=False
        )
        if self.fft_analysis_dock:
            logger.debug("FFT Analysis dock panel created.")
            self.main_window.fft_analysis_dock = self.fft_analysis_dock


    def _create_all_dock_panels(self):
        """Creates all managed dock panels."""
        self._setup_history_dock()
        self._setup_metadata_dock()
        self._setup_fft_analysis_dock()