"""Tests for the non-modal AtomMapper fit-settings panel."""

from __future__ import annotations

import numpy as np
import pytest
from PyQt6.QtWidgets import QLineEdit, QSpinBox

pytest.importorskip("PyQt6", reason="PyQt6 is required for AtomMapper GUI tests")
pytest.importorskip("pytestqt", reason="pytest-qt is required for AtomMapper GUI tests")

from AtomMapper.app.fit_models import LocalFitModelType
from AtomMapper.app.fit_settings_panel import FitSettingsPanelWidget
from AtomMapper.app.models import LoadedImage, ROIState


def _make_loaded_image(name: str) -> LoadedImage:
    data = np.arange(48, dtype=float).reshape((6, 8))
    return LoadedImage(
        source_path=f"/tmp/{name}",
        display_name=name,
        file_extension=".stp",
        image_data=data,
        pixels_x=8,
        pixels_y=6,
        size_nm_x=8.0,
        size_nm_y=6.0,
        metadata={"image_type": "Topo"},
        raw_metadata={},
    )


def test_fit_settings_panel_updates_state_and_context(qtbot):
    panel = FitSettingsPanelWidget()
    qtbot.addWidget(panel)
    image = _make_loaded_image("panel.stp")
    panel.set_context(image, ROIState(x=1, y=2, width=5, height=4))

    emitted_states = []
    panel.fit_settings_changed.connect(emitted_states.append)

    model_index = panel.model_combo.findData(LocalFitModelType.LORENTZIAN)
    panel.model_combo.setCurrentIndex(model_index)

    max_nfev_spinbox = panel.findChild(QSpinBox, "atommapper_fit_common_max_nfev_spinbox")
    assert max_nfev_spinbox is not None
    max_nfev_spinbox.setValue(3200)

    gamma_y_edit = panel.findChild(QLineEdit, "atommapper_fit_lorentzian_gamma_y_init_lineedit")
    assert gamma_y_edit is not None
    gamma_y_edit.setText("1.25")
    gamma_y_edit.editingFinished.emit()

    assert emitted_states
    assert emitted_states[-1].model is LocalFitModelType.LORENTZIAN
    assert emitted_states[-1].common.max_nfev == 3200
    assert emitted_states[-1].lorentzian.gamma_y_init == 1.25
    assert "panel.stp" in panel.context_label.text()
    assert "ROI: x=1, y=2, 5x4 px" in panel.context_label.text()
