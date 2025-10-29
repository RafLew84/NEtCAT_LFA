import numpy as np
import pytest

pytest.importorskip("PyQt6", reason="PyQt6 is required for adsorbate dialog tests")
pytest.importorskip("pytestqt", reason="pytest-qt provides the qtbot fixture")
pytest.importorskip("pyqtgraph", reason="pyqtgraph is required for adsorbate dialog tests")

from lfa.gui.dialogs.adsorbate_spot_dialog import AdsorbateSpotSelectionDialog


def test_adsorbate_dialog_caches_results_on_accept(qtbot):
    data = np.zeros((16, 16), dtype=np.float32)
    dialog = AdsorbateSpotSelectionDialog(
        fft_image_data=data,
        history_manager=None,
        current_fft_node_id="fft-node",
        current_adsorbate_spots=[(1.0, 2.0)],
        adsorbate_set_index=0,
        default_refinement_roi_size=7,
        substrate_F_m2i=np.eye(2),
        substrate_t_m2i=np.zeros(2),
    )

    qtbot.addWidget(dialog)
    dialog.selected_adsorbate_spots_raw = [(1.0, 2.0), (5.0, 6.0)]
    dialog.corrected_adsorbate_spots_in_ideal_system = [(10.0, 11.0)]

    dialog.accept()
    results = dialog.get_dialog_results()

    assert results["adsorbate_set_index"] == 0
    assert results["raw_adsorbate_spots"] == [(1.0, 2.0), (5.0, 6.0)]
    assert results["corrected_adsorbate_spots_in_ideal_system"] == [(10.0, 11.0)]

    dialog.close()
