# lfa/main.py
"""
Main entry point for the Lattice Fourier Analyzer (LFA) application.

Initializes the QApplication, creates the main window, and starts the event loop.
Also configures basic logging for the application.
"""

import logging
import sys

try:
    from PyQt6.QtWidgets import QApplication
except ImportError:
    print("ERROR: PyQt6 is not installed. Please install it using:")
    print("pip install PyQt6")
    sys.exit(1)

# Import the MainWindow from the gui module
try:
    from .gui.main_window import MainWindow
except ImportError as e:
     # Handle cases where the script is run from the wrong directory
     # or structure is incorrect
    print(f"ERROR: Could not import MainWindow: {e}")
    print("Ensure you are running this from the project root directory (lfa-project)")
    print("and that the package structure (lfa/, lfa/gui/, etc.) is correct.")
    # Attempt import relative to current dir if previous failed (less robust)
    try:
        from gui.main_window import MainWindow
    except ImportError:
        print("Could not resolve import path. Exiting.")
        sys.exit(1)


def main():
    """Main application function."""
    log_format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_format) # Set level to INFO or DEBUG
    logger = logging.getLogger(__name__) # Get logger for this module
    logger.info("Starting LFA Application...")

    app = QApplication(sys.argv)

    try:
        main_window = MainWindow()
        main_window.show() # Make the window visible
        logger.info("Main window displayed.")
    except Exception as e:
        logger.exception(f"Failed to create or show the main window: {e}")
        sys.exit(1)

    logger.info("Starting Qt event loop...")
    exit_code = app.exec()
    logger.info(f"Application finished with exit code {exit_code}.")
    sys.exit(exit_code)


if __name__ == "__main__":
    main()