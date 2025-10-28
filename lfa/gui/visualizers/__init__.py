"""Helper components used by GUI visualization dialogs."""

from __future__ import annotations

__all__ = [
    "RealSpaceVisualizerState",
    "RealSpacePyVistaAdapter",
    "RealSpaceSceneConfig",
    "PYVISTA_AVAILABLE",
]

try:  # pragma: no cover - optional dependency import
    from .real_space_state import RealSpaceVisualizerState
except ImportError:  # pragma: no cover - during partial imports
    # Leave attribute unavailable if state module cannot be loaded.
    pass

try:  # pragma: no cover - optional dependency import
    from .real_space_pyvista_adapter import (
        RealSpacePyVistaAdapter,
        RealSpaceSceneConfig,
        PYVISTA_AVAILABLE,
    )
except ImportError:  # pragma: no cover - during partial imports
    # Adapter is optional; expose sentinel if available.
    pass
