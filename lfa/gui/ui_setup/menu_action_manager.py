# lfa/gui/ui_setup/menu_action_manager.py
import logging

from PyQt6.QtGui import QAction
from PyQt6.QtWidgets import QMainWindow, QMenuBar

logger = logging.getLogger(__name__)

class MenuActionManager:
    def __init__(self, main_window: QMainWindow): 
        """
        Manages the creation of menus and actions for the main window.

        Args:
            main_window (QMainWindow): The main window instance, used for context
                                       (e.g., parent for QAction) and connecting slots.
        """
        self.main_window = main_window
        
        self.menu_bar: QMenuBar = self.main_window.menuBar()

        self.file_actions: dict[str, QAction] = {}
        self.preprocessing_actions: dict[str, QAction] = {}
        self.analysis_actions: dict[str, QAction] = {}
        self.view_actions: dict[str, QAction] = {}
        self.help_actions: dict[str, QAction] = {}
        
        self._create_all_menus()
        logger.debug("MenuActionManager initialized and menus created.")

    def _create_action(self, text: str, status_tip: str, triggered_slot, 
                       shortcut: str = "", enabled: bool = True, 
                       is_checkable: bool = False, checked: bool = False) -> QAction:
        """Helper method to create a QAction."""
        action = QAction(text, self.main_window)
        action.setStatusTip(status_tip)
        if triggered_slot:
            action.triggered.connect(triggered_slot)
        if shortcut:
            action.setShortcut(shortcut)
        action.setEnabled(enabled)
        if is_checkable:
            action.setCheckable(True)
            action.setChecked(checked)
        return action

    def _create_file_menu(self):
        """Creates the File menu."""
        file_menu = self.menu_bar.addMenu("&File")
        
        self.file_actions["open"] = self._create_action(
            text="&Open...", 
            status_tip="Open an STM data file",
            triggered_slot=self.main_window.open_file_dialog,
            shortcut="Ctrl+O"
        )
        file_menu.addAction(self.file_actions["open"])
        
        file_menu.addSeparator()

        self.file_actions["save_session"] = self._create_action(
            text="Save Analysis...",
            status_tip="Save the current analysis session to a file",
            triggered_slot=self.main_window.save_analysis, 
            shortcut="Ctrl+S"
        )
        file_menu.addAction(self.file_actions["save_session"])

        self.file_actions["load_session"] = self._create_action(
            text="Load Analysis...",
            status_tip="Load an analysis session from a file",
            triggered_slot=self.main_window.load_analysis, 
            shortcut="Ctrl+L"
        )
        file_menu.addAction(self.file_actions["load_session"])

        self.file_actions["load_metadata"] = self._create_action(
            text="Load Metadata for Save...",
            status_tip="Load metadata from an original STP file to enable saving",
            triggered_slot=self.main_window.load_metadata_for_session, # Nowy slot
            enabled=True 
        )
        file_menu.addAction(self.file_actions["load_metadata"])
    
        file_menu.addSeparator()

        self.file_actions["clear_session"] = self._create_action(
            text="Clear Session",
            status_tip="Remove all loaded data and reset the analysis session",
            triggered_slot=self.main_window.clear_analysis_session
        )
        file_menu.addAction(self.file_actions["clear_session"])

        file_menu.addSeparator()
        
        self.file_actions["exit"] = self._create_action(
            text="&Exit", 
            status_tip="Exit the application",
            triggered_slot=self.main_window.close, 
            shortcut="Ctrl+Q"
        )
        file_menu.addAction(self.file_actions["exit"])

    def _create_preprocessing_menu(self):
        """Creates the Preprocessing menu."""
        preprocessing_menu = self.menu_bar.addMenu("&Preprocessing")

        actions_definitions = [
            ("gaussian_blur", "&Gaussian Blur...", "Apply Gaussian blur filter", self.main_window.open_gaussian_blur_dialog, False),
            ("gaussian_sharpen", "Gaussian &Sharpening...", "Apply Gaussian Sharpening (Unsharp Mask)", self.main_window.open_gaussian_sharpening_dialog, False),
            ("plane_level", "&Plane Leveling...", "Level image by subtracting a fitted plane", self.main_window.open_plane_leveling_dialog, False),
            ("median_filter", "&Median Filter...", "Apply median filter for noise reduction", self.main_window.open_median_filter_dialog, False),
            ("nlmeans", "&NL-Means Denoising...", "Apply Non-Local Means denoising (skimage)", self.main_window.open_nlmeans_dialog, False),
            ("bm3d", "&BM3D Denoising...", "Apply BM3D denoising (Computationally intensive)", self.main_window.open_bm3d_dialog, False),
        ]

        for name, text, tip, slot, enabled in actions_definitions:
            action = self._create_action(text, tip, slot, enabled=enabled)
            self.preprocessing_actions[name] = action
            preprocessing_menu.addAction(action)


    def _create_analysis_menu(self):
        """Creates the Analysis menu."""
        analysis_menu = self.menu_bar.addMenu("&Analysis")
        
        self.analysis_actions["fft"] = self._create_action(
            text="Calculate &FFT...", 
            status_tip="Calculate Fast Fourier Transform",
            triggered_slot=self.main_window.open_fft_dialog,
            enabled=False
        )
        analysis_menu.addAction(self.analysis_actions["fft"])

        analysis_menu.addSeparator()

        sel_subs_action = self._create_action(
            text="Select &Substrate Spots...",
            status_tip="Open dialog to select/edit substrate spots",
            triggered_slot=self.main_window.open_substrate_spot_selection_dialog,
            enabled=False
        )
        analysis_menu.addAction(sel_subs_action)
        self.analysis_actions["select_substrate_spots"] = sel_subs_action

        sel_ads_action = self._create_action(
            text="Select &Adsorbate Spots...",
            status_tip="Open dialog to select/edit adsorbate spots for the current set",
            triggered_slot=self.main_window.open_adsorbate_spot_selection_dialog,
            enabled=False
        )
        analysis_menu.addAction(sel_ads_action)
        self.analysis_actions["select_adsorbate_spots"] = sel_ads_action

        analysis_menu.addSeparator()
        superstructure_action = self._create_action(
            text="Analyze Superstructure Periodicity...",
            status_tip="Analyze superstructure periodicity from satellite peak splittings",
            triggered_slot=self.main_window.open_superstructure_periodicity_dialog,
            enabled=False
        )
        analysis_menu.addAction(superstructure_action)
        self.analysis_actions["superstructure_periodicity"] = superstructure_action


        analysis_menu.addSeparator()
        vis_action = self._create_action(
            text="Visualize Real Space...",
            status_tip="Open dialog for real space and FFT visualization",
            triggered_slot=self.main_window.open_real_space_fft_visualizer,
            enabled=False
        )
        analysis_menu.addAction(vis_action)
        self.analysis_actions["visualize_real_space"] = vis_action
        reconstruction_action = self._create_action(
            text="Real Space Reconstruction...",
            status_tip="Reconstruct real space image from masked FFT",
            triggered_slot=self.main_window.open_real_space_reconstruction_dialog,
            enabled=False 
        )
        analysis_menu.addAction(reconstruction_action)
        self.analysis_actions["real_space_reconstruction"] = reconstruction_action

        stm_transform_action = self._create_action(
            text="STM Transform...",
            status_tip="Perform advanced transformations on STM data",
            triggered_slot=self.main_window.open_stm_transform_dialog,
            enabled=False
        )
        analysis_menu.addAction(stm_transform_action)
        self.analysis_actions["stm_transform"] = stm_transform_action


    def _create_view_menu(self):
        """Creates the View menu."""
        self.main_window.view_menu = self.menu_bar.addMenu("&View")

    def _create_help_menu(self):
        """Creates the Help menu."""
        help_menu = self.menu_bar.addMenu("&Help")
        
        self.help_actions["about"] = self._create_action(
            text="&About LFA...", 
            status_tip="Show information about LFA",
            triggered_slot=self.main_window.show_about_dialog
        )
        help_menu.addAction(self.help_actions["about"])

    def _create_all_menus(self):
        """Creates all main menus."""
        self._create_file_menu()
        self._create_preprocessing_menu()
        self._create_analysis_menu()
        self._create_view_menu()
        self._create_help_menu()

    def get_action(self, menu_key: str, action_key: str) -> QAction | None:
        """
        Retrieve a QAction using its menu key and action key.
        Useful when MainWindow._update_action_states needs access
        to actions without storing them directly as MainWindow attributes.
        """
        menu_map = {
            "file": self.file_actions,
            "preprocessing": self.preprocessing_actions,
            "analysis": self.analysis_actions,
            "view": self.view_actions,
            "help": self.help_actions,
        }
        return menu_map.get(menu_key, {}).get(action_key)

    def action_map_for_ui_state(self) -> dict[str, QAction | None]:
        """Return the subset of actions needed by UIStateBinder."""
        return {
            "load_metadata": self.file_actions.get("load_metadata"),
            "gaussian_blur": self.preprocessing_actions.get("gaussian_blur"),
            "gaussian_sharpen": self.preprocessing_actions.get("gaussian_sharpen"),
            "plane_level": self.preprocessing_actions.get("plane_level"),
            "median_filter": self.preprocessing_actions.get("median_filter"),
            "nlmeans": self.preprocessing_actions.get("nlmeans"),
            "bm3d": self.preprocessing_actions.get("bm3d"),
            "fft": self.analysis_actions.get("fft"),
            "select_substrate_spots": self.analysis_actions.get("select_substrate_spots"),
            "select_adsorbate_spots": self.analysis_actions.get("select_adsorbate_spots"),
            "superstructure_periodicity": self.analysis_actions.get("superstructure_periodicity"),
            "stm_transform": self.analysis_actions.get("stm_transform"),
            "visualize_real_space": self.analysis_actions.get("visualize_real_space"),
            "real_space_reconstruction": self.analysis_actions.get("real_space_reconstruction"),
        }
