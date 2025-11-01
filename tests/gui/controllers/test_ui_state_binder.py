import pytest
from PyQt6.QtGui import QAction
from PyQt6.QtWidgets import QWidget

pytest.importorskip("PyQt6", reason="PyQt6 is required for UIStateBinder tests")
pytest.importorskip("pytestqt", reason="pytest-qt provides the qtbot fixture")

from lfa.gui.controllers.ui_state_binder import UIStateBinder
from lfa.logic.app_controller import FFTPanelState


class _StubPanel:
    def __init__(self) -> None:
        self.edit_substrate = None
        self.edit_adsorbate = None
        self.reselect = None
        self.clear_all = None
        self.calc_substrate = None
        self.calc_adsorbate = None
        self.substrate_rs_display = "unset"
        self.adsorbate_rs_display = "unset"
        self.transform_display = "unset"

    def set_edit_substrate_spots_button_enabled(self, value: bool) -> None:
        self.edit_substrate = value

    def set_edit_adsorbate_spots_button_enabled(self, value: bool) -> None:
        self.edit_adsorbate = value

    def set_reselect_adsorbate_set_button_enabled(self, value: bool) -> None:
        self.reselect = value

    def set_clear_all_adsorbate_sets_button_enabled(self, value: bool) -> None:
        self.clear_all = value

    def set_calculate_substrate_rs_button_enabled(self, value: bool) -> None:
        self.calc_substrate = value

    def set_calculate_adsorbate_rs_button_enabled(self, value: bool) -> None:
        self.calc_adsorbate = value

    def update_substrate_real_space_display(self, value) -> None:
        self.substrate_rs_display = value

    def update_adsorbate_real_space_display(self, value) -> None:
        self.adsorbate_rs_display = value

    def update_transform_results_display(self, value) -> None:
        self.transform_display = value


class _StubDock:
    def __init__(self) -> None:
        self.visible = None

    def setVisible(self, flag: bool) -> None:  # noqa: N802 - Qt naming
        self.visible = flag


class _StubHistory:
    def __init__(self, node) -> None:
        self._node = node

    def get_current_node(self):
        return self._node


class _StubController:
    def __init__(self, panel_state: FFTPanelState, allow: bool = True) -> None:
        self._panel_state = panel_state
        self._allow = allow
        self.can_fft_calls = []

    def evaluate_fft_panel_state(self, current_node, lattice_enabled: bool) -> FFTPanelState:
        return self._panel_state

    def can_load_metadata(self, current_node) -> bool:
        return bool(current_node)

    def can_calculate_fft(self, current_node) -> bool:
        self.can_fft_calls.append(current_node)
        return self._allow and bool(current_node)

    def can_select_spots(self, current_node) -> bool:
        return self._allow and bool(current_node)

    def can_analyze_superstructure(self, current_node) -> bool:
        return self._allow and bool(current_node)

    def can_open_stm_transform(self, current_node) -> bool:
        return self._allow and bool(current_node)

    def can_visualize_real_space(self, current_node) -> bool:
        return self._allow and bool(current_node)

    def can_open_real_space_reconstruction(self, current_node) -> bool:
        return self._allow and bool(current_node)


class _Node:
    def __init__(self) -> None:
        self.data_type = "FFT"
        self.node_id = "fft-node"


def _create_actions(widget: QWidget):
    return {
        "load_metadata": QAction("Load metadata", widget),
        "gaussian_blur": QAction("Gaussian blur", widget),
        "gaussian_sharpen": QAction("Sharpen", widget),
        "plane_level": QAction("Plane level", widget),
        "median_filter": QAction("Median", widget),
        "nlmeans": QAction("NLM", widget),
        "bm3d": QAction("BM3D", widget),
        "fft": QAction("FFT", widget),
        "select_substrate_spots": QAction("Select substrate", widget),
        "select_adsorbate_spots": QAction("Select adsorbate", widget),
        "superstructure_periodicity": QAction("Superstructure", widget),
        "stm_transform": QAction("STM transform", widget),
        "visualize_real_space": QAction("Visualize", widget),
        "real_space_reconstruction": QAction("Reconstruct", widget),
    }


def test_ui_state_binder_enables_controls_for_fft_node(qtbot):
    widget = QWidget()
    qtbot.addWidget(widget)
    actions = _create_actions(widget)

    panel = _StubPanel()
    dock = _StubDock()
    node = _Node()
    history = _StubHistory(node)
    panel_state = FFTPanelState(
        fft_active=True,
        edit_substrate_enabled=True,
        edit_adsorbate_enabled=False,
        reselect_adsorbate_enabled=True,
        clear_all_adsorbate_sets_enabled=True,
        can_calculate_substrate_rs=True,
        can_calculate_adsorbate_rs=False,
    )
    controller = _StubController(panel_state, allow=True)

    binder = UIStateBinder(
        controller,
        history,
        panel,
        dock,
        actions,
        {
            "preprocessing_dialogs": True,
            "spot_dialogs": True,
            "superstructure_dialog": True,
            "stm_transform_dialog": True,
            "lattice_analysis": True,
        },
    )

    binder.refresh()

    assert actions["gaussian_blur"].isEnabled()
    assert actions["select_adsorbate_spots"].isEnabled()
    assert actions["superstructure_periodicity"].isEnabled()
    assert actions["stm_transform"].isEnabled()
    assert panel.edit_substrate is True
    assert panel.edit_adsorbate is False
    assert panel.reselect is True
    assert panel.clear_all is True
    assert panel.calc_substrate is True
    assert panel.calc_adsorbate is False
    assert panel.substrate_rs_display == "unset"
    assert dock.visible is True


def test_ui_state_binder_disables_panel_when_fft_inactive(qtbot):
    widget = QWidget()
    qtbot.addWidget(widget)
    actions = _create_actions(widget)

    panel = _StubPanel()
    dock = _StubDock()
    history = _StubHistory(None)
    panel_state = FFTPanelState(fft_active=False)
    controller = _StubController(panel_state, allow=False)

    binder = UIStateBinder(
        controller,
        history,
        panel,
        dock,
        actions,
        {
            "preprocessing_dialogs": True,
            "spot_dialogs": True,
            "superstructure_dialog": True,
            "stm_transform_dialog": True,
            "lattice_analysis": True,
        },
    )

    binder.refresh()

    for action in actions.values():
        assert not action.isEnabled()

    assert panel.edit_substrate is False
    assert panel.edit_adsorbate is False
    assert panel.reselect is False
    assert panel.clear_all is False
    assert panel.calc_substrate is False
    assert panel.calc_adsorbate is False
    assert panel.substrate_rs_display is None
    assert panel.adsorbate_rs_display is None
    assert panel.transform_display is None
    assert dock.visible is False
