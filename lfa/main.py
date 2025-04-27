# lfa/main.py
"""
Main entry point for the Lattice Fourier Analyzer (LFA) application.

Initializes the QApplication, creates the main window, and starts the event loop.
Also configures basic logging for the application.
"""

import sys
import logging

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
    # --- Configure Logging ---
    # Set up basic logging to console. More advanced configuration (e.g., file logging)
    # could be added here or managed by a dedicated logging utility module.
    log_format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_format) # Set level to INFO or DEBUG
    logger = logging.getLogger(__name__) # Get logger for this module
    logger.info("Starting LFA Application...")

    # --- Create Qt Application ---
    # Pass sys.argv to allow command-line arguments for Qt if needed
    app = QApplication(sys.argv)

    # --- Create and Show Main Window ---
    try:
        main_window = MainWindow()
        main_window.show() # Make the window visible
        logger.info("Main window displayed.")
    except Exception as e:
        logger.exception(f"Failed to create or show the main window: {e}")
        # Optionally show a critical error message to the user here
        sys.exit(1) # Exit if main window creation fails

    # --- Start Qt Event Loop ---
    # This starts the Qt event processing. The application will run until exit() is called
    # or the main window is closed.
    logger.info("Starting Qt event loop...")
    exit_code = app.exec()
    logger.info(f"Application finished with exit code {exit_code}.")
    sys.exit(exit_code)


# Standard Python entry point guard
if __name__ == "__main__":
    main()