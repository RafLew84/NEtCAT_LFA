# scripts/test_io_factory.py
"""
Test script specifically for the LFA file IO factory.

This script uses the `load_stm_file` factory function from lfa.io.factory
to load specified .stp, .s94, and potentially other files.
It verifies format detection, reader dispatching, metadata extraction,
and displays the loaded images using matplotlib.
"""

import numpy as np
import matplotlib.pyplot as plt
import logging
import os
import sys

# --- Path Setup ---
# Ensure the main 'lfa' package is importable by adding the project root to sys.path
# Assumes this script is in 'scripts/' directory, one level below project root.
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# --- Imports from LFA package ---
try:
    # Import the factory function and the data model
    from lfa.io.factory import load_stm_file
    from lfa.core.data_models import STMImage
except ImportError as e:
    print(f"Import Error: {e}")
    print("Please ensure the 'lfa' package is correctly structured (directories, __init__.py files) and accessible.")
    print("Try running this script from the main project directory ('lfa-project').")
    sys.exit(1)

# --- Basic Logging Configuration ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- Helper Functions ---

def scale_to_grayscale(data: np.ndarray) -> np.ndarray | None:
    """
    Scales NumPy array data to the range [0, 1] for grayscale display.

    Args:
        data (np.ndarray): The 2D numpy array containing image data.

    Returns:
        np.ndarray | None: A new NumPy array with values scaled to [0, 1],
                           or None if the input data is invalid or empty.
                           Returns a flat gray image (0.5) if input is constant.
    """
    if data is None or data.size == 0:
        logger.error("Cannot scale empty or None data.")
        return None
    min_val = np.min(data)
    max_val = np.max(data)
    if max_val == min_val:
        logger.warning("Image data is constant (flat). Displaying as uniform gray.")
        return np.full(data.shape, 0.5, dtype=np.float32)
    else:
        scaled_data = (data - min_val) / (max_val - min_val)
        return scaled_data.astype(np.float32)

def display_image(image_data: np.ndarray, title: str):
    """
    Displays a 2D NumPy array as an image using matplotlib.

    Args:
        image_data (np.ndarray): The (scaled) 2D image data to display.
        title (str): The title for the image plot window.
    """
    if image_data is None:
        logger.error(f"No data provided for display: {title}")
        return

    # Flip the image vertically to correct the mirroring
    image_data = np.flipud(image_data)

    plt.figure(figsize=(8, 8))
    im = plt.imshow(image_data, cmap='gray', origin='lower') # origin='lower' -> (0,0) at bottom-left
    plt.colorbar(im, label='Scaled Intensity [0, 1]')
    plt.title(title)
    plt.xlabel("X Pixels")
    plt.ylabel("Y Pixels")
    plt.axis('on')
    plt.show(block=False) # Show plot without blocking

# --- Main Execution Logic ---

def main():
    """
    Main function to test the IO factory with various file types.
    """
    # Define paths to test files (relative to project root)
    # Ensure these files exist in your 'data/' directory
    stp_file_path = os.path.join("data", "85291r.STP")
    s94_file_path = os.path.join("data", "85291r.s94")
    # Add paths to other test files if available
    # e.g., another_stp = os.path.join("data", "other_image.stp")
    unsupported_file_path = os.path.join("data", "test.txt") # Example unsupported file
    non_existent_file = "non_existent_file.stp"

    # List of files to test with the factory
    file_paths_to_test = [
        stp_file_path,
        s94_file_path,
        # Add other valid test file paths here
        unsupported_file_path, # Test handling of unsupported extensions
        non_existent_file      # Test handling of files that don't exist
    ]

    print("--- Starting IO Factory Test ---")

    # Loop through the files and attempt to load them using the factory
    for file_path in file_paths_to_test:
        logging.info(f"\n--- Attempting to load via factory: {file_path} ---")
        try:
            # *** Call the IO factory function ***
            stm_image: STMImage | None = load_stm_file(file_path)

            # Check the result
            if stm_image and stm_image.data is not None:
                # File loaded successfully
                logging.info(f"Successfully loaded: {os.path.basename(file_path)}")
                print(f"  Image Type: {stm_image.image_type}")
                print(f"  Dimensions (px): {stm_image.pixels_x} x {stm_image.pixels_y}")
                print(f"  Size (nm): {stm_image.size_nm_x:.3f} x {stm_image.size_nm_y:.3f}")
                print(f"  Bias Voltage (V): {stm_image.bias_v:.3f}")

                # Scale and display the loaded image
                scaled_data = scale_to_grayscale(stm_image.data)
                display_title = f"Factory Loaded: {os.path.basename(file_path)}\nType: {stm_image.image_type}"
                display_image(scaled_data, display_title)
            else:
                # Loading failed (File not found, unsupported format, read error)
                # The factory function logs the specific reason (warning/error)
                logging.warning(f"Factory returned None for: {file_path}. Check previous logs for details.")

        except Exception as e:
            # Catch any unexpected errors that might occur outside the factory's try-except blocks
            logging.exception(f"An unexpected error occurred while processing {file_path} in the test script: {e}")

    # --- Keep plots open ---
    print("\n--- IO Factory Test Complete ---")
    logging.info("Close plot windows to exit script.")
    plt.show() # Keep matplotlib windows open until manually closed

# Standard Python entry point guard
if __name__ == "__main__":
    main()