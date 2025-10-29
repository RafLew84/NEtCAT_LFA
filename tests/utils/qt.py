"""Minimal Qt stand-ins to support headless unit tests."""

from __future__ import annotations

from typing import Any, Callable, Iterable, List, Optional


class FakeSignal:
    """A small replacement for PyQt6 signals used in headless tests."""

    def __init__(self) -> None:
        self._slots: List[Callable[..., None]] = []

    def connect(self, slot: Callable[..., None]) -> None:
        if slot not in self._slots:
            self._slots.append(slot)

    def disconnect(self, slot: Callable[..., None]) -> None:
        if slot in self._slots:
            self._slots.remove(slot)

    def emit(self, *args: Any, **kwargs: Any) -> None:
        for slot in list(self._slots):
            slot(*args, **kwargs)

    def clear(self) -> None:
        self._slots.clear()


class FakeAction:
    """A minimal QAction look-alike with an enable flag and triggered signal."""

    def __init__(self, text: str = "") -> None:
        self.text = text
        self.enabled: bool = True
        self.triggered = FakeSignal()

    def setEnabled(self, enabled: bool) -> None:
        self.enabled = bool(enabled)

    def isEnabled(self) -> bool:
        return self.enabled

    def trigger(self) -> None:
        if self.enabled:
            self.triggered.emit()


class FakeWidget:
    """Simple widget base that tracks visibility and enabled state."""

    def __init__(self) -> None:
        self._visible = False
        self._enabled = True

    def show(self) -> None:
        self._visible = True

    def hide(self) -> None:
        self._visible = False

    def isVisible(self) -> bool:
        return self._visible

    def setEnabled(self, enabled: bool) -> None:
        self._enabled = bool(enabled)

    def isEnabled(self) -> bool:
        return self._enabled


def ensure_app() -> Optional[Any]:
    """
    Ensure a QApplication instance exists for tests that require it.

    Returns the QApplication instance when PyQt6 is available; otherwise None.
    """

    try:
        from PyQt6.QtWidgets import QApplication
    except ImportError:  # pragma: no cover - exercised only in environments without Qt
        return None

    app = QApplication.instance()
    if app is None:
        app = QApplication(["lfa-tests"])
    return app


__all__ = [
    "FakeAction",
    "FakeSignal",
    "FakeWidget",
    "ensure_app",
]
