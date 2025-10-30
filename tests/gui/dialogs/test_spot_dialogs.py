from pathlib import Path
import numpy as np
import pytest
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

pytest.importorskip("PyQt6", reason="PyQt6 is required for spot-selection dialog tests")
pytest.importorskip("pytestqt", reason="pytest-qt is required for qtbot fixture")

from PyQt6.QtWidgets import QListWidget

from lfa.gui.dialogs.substrate_spot_dialog import (
    SubstrateSpotSelectionDialog,
    REFINEMENT_MAX_PIXEL
)
from lfa.gui.dialogs.adsorbate_spot_dialog import (
    AdsorbateSpotSelectionDialog,
    REFINEMENT_MAX_PIXEL as ADS_REFINEMENT_MAX_PIXEL
)
from lfa.logic.history_manager import HistoryManager


@pytest.fixture
def substrate_dialog(qtbot, monkeypatch):
    fft_data = np.zeros((16, 16), dtype=np.float32)
    list_widget = QListWidget()
    qtbot.addWidget(list_widget)
    history_manager = HistoryManager(list_widget)
    dialog = SubstrateSpotSelectionDialog(
        fft_image_data=fft_data,
        history_manager=history_manager,
        current_fft_node_id="node",
        default_refinement_method=REFINEMENT_MAX_PIXEL,
    )
    qtbot.addWidget(dialog)
    dialog.selection_roi.setVisible(True)
    dialog.selection_roi.setPos((2, 2))
    dialog.selection_roi.setSize((6, 6))
    return dialog


@pytest.fixture
def adsorbate_dialog(qtbot, monkeypatch):
    fft_data = np.zeros((16, 16), dtype=np.float32)
    list_widget = QListWidget()
    qtbot.addWidget(list_widget)
    history_manager = HistoryManager(list_widget)
    dialog = AdsorbateSpotSelectionDialog(
        fft_image_data=fft_data,
        history_manager=history_manager,
        current_fft_node_id="node",
        current_adsorbate_spots=None,
        adsorbate_set_index=0,
        default_refinement_method=ADS_REFINEMENT_MAX_PIXEL,
    )
    qtbot.addWidget(dialog)
    dialog.selection_roi.setVisible(True)
    dialog.selection_roi.setPos((4, 4))
    dialog.selection_roi.setSize((8, 8))
    return dialog


def test_substrate_max_pixel_centers_marker(substrate_dialog, monkeypatch):
    monkeypatch.setattr(
        "lfa.gui.dialogs.substrate_spot_dialog.find_max_pixel_in_roi",
        lambda data, center, radius: (5, 10)
    )
    substrate_dialog.current_refinement_method = REFINEMENT_MAX_PIXEL
    substrate_dialog.selected_spots.clear()

    substrate_dialog._add_current_roi_spot()

    assert substrate_dialog.selected_spots
    x, y = substrate_dialog.selected_spots[-1]
    assert x == pytest.approx(10.5)
    assert y == pytest.approx(5.5)


def test_adsorbate_max_pixel_centers_marker(adsorbate_dialog, monkeypatch):
    monkeypatch.setattr(
        "lfa.gui.dialogs.adsorbate_spot_dialog.find_max_pixel_in_roi",
        lambda data, center, radius: (6, 12)
    )
    adsorbate_dialog.current_refinement_method = ADS_REFINEMENT_MAX_PIXEL
    adsorbate_dialog.state.raw_spots.clear()

    adsorbate_dialog._add_current_adsorbate_spot_from_roi()

    assert adsorbate_dialog.state.raw_spots
    x, y = adsorbate_dialog.state.raw_spots[-1]
    assert x == pytest.approx(12.5)
    assert y == pytest.approx(6.5)
