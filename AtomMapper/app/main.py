"""Standalone entry point for AtomMapper."""

from __future__ import annotations

import logging
import sys
from typing import Optional, Sequence

try:
    from PyQt6.QtWidgets import QApplication
except ImportError:
    print("ERROR: PyQt6 is not installed. Please install it using:")
    print("pip install PyQt6")
    raise SystemExit(1)

from .controller import AtomMapperController
from .main_window import AtomMapperMainWindow


def configure_logging() -> None:
    """Initialize a basic console logger for the application."""

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )


def create_application(argv: Optional[Sequence[str]] = None) -> QApplication:
    """Return a QApplication instance, creating one when needed."""

    app = QApplication.instance()
    if app is not None:
        return app
    return QApplication(list(argv) if argv is not None else sys.argv)


def create_main_window() -> AtomMapperMainWindow:
    """Build the main window used by the AtomMapper bootstrap."""

    controller = AtomMapperController()
    return AtomMapperMainWindow(controller=controller)


def main(argv: Optional[Sequence[str]] = None) -> int:
    """Run the standalone AtomMapper application."""

    configure_logging()
    logger = logging.getLogger(__name__)
    logger.info("Starting AtomMapper...")

    app = create_application(argv)
    window = create_main_window()
    window.show()

    logger.info("AtomMapper main window displayed.")
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
