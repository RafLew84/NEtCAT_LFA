"""Compatibility wrapper for legacy preprocessing dialog imports.

New code should import from :mod:`lfa.gui.dialogs.preprocessing`.
"""

import warnings

from .preprocessing import (
    BasePreprocessingDialog,
    BM3DDialog,
    GaussianBlurDialog,
    GaussianSharpeningDialog,
    MedianFilterDialog,
    NLMeansDialog,
    PlaneLevelingDialog,
)

warnings.warn(
    "lfa.gui.dialogs.preprocessing_dialogs is deprecated; "
    "import from lfa.gui.dialogs.preprocessing instead.",
    DeprecationWarning,
    stacklevel=2,
)

__all__ = [
    "BasePreprocessingDialog",
    "GaussianBlurDialog",
    "GaussianSharpeningDialog",
    "NLMeansDialog",
    "BM3DDialog",
    "PlaneLevelingDialog",
    "MedianFilterDialog",
]
