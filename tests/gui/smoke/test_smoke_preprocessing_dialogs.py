
import numpy as np
import pytest

pytest.importorskip("PyQt6", reason="PyQt6 is required for GUI smoke tests")
pytest.importorskip("pytestqt", reason="pytest-qt is required for qtbot fixture")

from lfa.gui.dialogs.preprocessing import GaussianBlurDialog, NLMeansDialog


def create_dummy_image(shape=(32, 32)):
    return np.zeros(shape, dtype=np.float32)


@pytest.mark.parametrize(
    "DialogClass, controller_method",
    [
        (GaussianBlurDialog, "apply_gaussian_blur"),
        (NLMeansDialog, "apply_nlmeans_denoising"),
    ],
)
def test_processing_dialog_smoke(qtbot, DialogClass, controller_method, monkeypatch):
    data = create_dummy_image()
    dialog = DialogClass(data)
    qtbot.addWidget(dialog)
    dialog.show()

    monkeypatch.setattr(
        "lfa.gui.actions.processing.ProcessingDialogLauncher._status",
        lambda self, msg, timeout_ms=3000: None,
    )
    monkeypatch.setattr(
        "lfa.gui.actions.processing.ProcessingDialogLauncher._status",
        lambda self, msg, timeout_ms=3000: None,
    )

    qtbot.waitUntil(lambda: dialog.isVisible())
    dialog.accept()
    assert dialog.was_roi_applied_only() in (True, False)

