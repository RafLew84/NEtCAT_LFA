import numpy as np
import pytest
from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import QDialogButtonBox

pytest.importorskip("PyQt6", reason="PyQt6 is required for FFT dialog tests")
pytest.importorskip("pytestqt", reason="pytest-qt is required for qtbot fixture")

from lfa.gui.dialogs.fft_dialog import FFTDialog


def test_fft_dialog_apply_emits_signal(qtbot, monkeypatch):
    data = np.random.rand(8, 8).astype(np.float32)
    dialog = FFTDialog(data)
    qtbot.addWidget(dialog)
    dialog.show()

    def fake_fft(image_data, apply_window=True, window_type="hann", pad_to_shape=None):
        return np.fft.fft2(image_data)

    monkeypatch.setattr("lfa.gui.dialogs.fft_dialog.calculate_fft", fake_fft)

    emitted = []

    def on_applied(params, processed, complex_data, roi_slice):
        emitted.append((params, processed, complex_data, roi_slice))

    dialog.fftApplied.connect(on_applied)

    apply_button = dialog.button_box.button(QDialogButtonBox.StandardButton.Apply)
    qtbot.mouseClick(apply_button, Qt.MouseButton.LeftButton)

    assert emitted, "Apply should emit fftApplied signal."
    params, processed, complex_data, roi_slice = emitted[0]
    assert isinstance(params, dict)
    assert isinstance(processed, np.ndarray)
    assert processed.shape == data.shape
    assert complex_data is not None
    assert roi_slice is None
    assert dialog.isVisible(), "Dialog should remain open after Apply."

    dialog.close()
