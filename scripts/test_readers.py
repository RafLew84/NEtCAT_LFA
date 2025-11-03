# scripts/test_readers.py
"""
Test script for reading STP and S94 files using the LFA file readers.

This script attempts to load specified .stp and .s94 files,
extract metadata, scale the image data to grayscale, and display
the images using matplotlib. It helps verify that the file readers
in lfa.io are functioning correctly and that the basic data
structure (STMImage) is populated as expected.
"""

import logging
import os
import sys

import matplotlib.pyplot as plt
import numpy as np

# --- Path Setup ---
# Ensure the main 'lfa' package is importable
# This adjusts the Python path assuming the script is in the 'scripts/' directory
# one level below the project root ('lfa-project/').
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# --- Imports from LFA package ---
try:
    from lfa.core.data_models import STMImage
    from lfa.io.s94_reader import read_s94_file
    from lfa.io.stp_reader import read_stp_file
except ImportError as e:
    print(f"Import Error: {e}")
    print("Please ensure the 'lfa' package is correctly structured and accessible.")
    print("Try running this script from the main project directory ('lfa-project').")
    sys.exit(1)

# --- Basic Logging Configuration ---
# Configure logging to show informational messages during execution.
# The logger level and format can be adjusted as needed.
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

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
        # Handle flat images (all values are the same)
        logger.warning("Image data is constant (flat). Displaying as uniform gray.")
        # Return an array filled with 0.5 for neutral gray
        return np.full(data.shape, 0.5, dtype=np.float32)
    else:
        # Perform linear scaling to the [0, 1] range
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
    Main function to execute the file reading and display tests.
    """
    # Define paths to the test files relative to the project root
    # Assumes the files are located in a 'data/' subdirectory
    stp_file_path = os.path.join("data", "85291r.STP")
    s94_file_path = os.path.join("data", "85291r.s94")

    # --- Test STP File Reader ---
    logging.info(f"--- Testing STP file: {stp_file_path} ---")
    try:
        # Check if the file exists before attempting to read
        if not os.path.exists(stp_file_path):
             logging.error(f"STP file not found at: {stp_file_path}")
             # Skip this test if file is missing
        else:
            # Call the reader function from the lfa.io package
            stm_image_stp: STMImage | None = read_stp_file(stp_file_path)

            # Process the result if reading was successful
            if stm_image_stp and stm_image_stp.data is not None:
                logging.info("STP file loaded successfully.")
                # Print some key metadata extracted from the file
                print("\n--- STP Metadata ---")
                print(f"  Image Type: {stm_image_stp.image_type}")
                print(f"  Dimensions (px): {stm_image_stp.pixels_x} x {stm_image_stp.pixels_y}")
                print(f"  Size (nm): {stm_image_stp.size_nm_x:.3f} x {stm_image_stp.size_nm_y:.3f}")
                print(f"  Bias Voltage (V): {stm_image_stp.bias_v:.3f}")
                print(f"  Setpoint Current (A): {stm_image_stp.setpoint_a}") # May be None
                print(f"  Scan Angle (deg): {stm_image_stp.scan_angle_deg}")

                # Scale the data for display
                scaled_data_stp = scale_to_grayscale(stm_image_stp.data)
                # Display the scaled image
                display_title = f"STP Image: {os.path.basename(stp_file_path)}\nType: {stm_image_stp.image_type}"
                display_image(scaled_data_stp, display_title)
            else:
                # Log if reading failed or returned no data
                logging.error("Failed to load data from STP file or data array is empty.")

    except FileNotFoundError as fnf_err:
        # Specific handling for FileNotFoundError is redundant due to the os.path.exists check,
        # but kept here for robustness in case the check is removed.
        logging.error(fnf_err)
    except Exception as e:
        # Catch any other exceptions during STP file processing
        logging.exception(f"An error occurred while processing the STP file: {e}")


    # --- Test S94 File Reader ---
    logging.info(f"\n--- Testing S94 file: {s94_file_path} ---")
    try:
        # Check if the file exists
        if not os.path.exists(s94_file_path):
             logging.error(f"S94 file not found at: {s94_file_path}")
             # Skip this test if file is missing
        else:
            # Call the S94 reader function
            stm_image_s94: STMImage | None = read_s94_file(s94_file_path)

            # Process the result if reading was successful
            if stm_image_s94 and stm_image_s94.data is not None:
                logging.info("S94 file loaded successfully.")
                # Print key metadata
                print("\n--- S94 Metadata ---")
                print(f"  Image Type: {stm_image_s94.image_type}")
                print(f"  Dimensions (px): {stm_image_s94.pixels_x} x {stm_image_s94.pixels_y}")
                print(f"  Size (nm): {stm_image_s94.size_nm_x:.3f} x {stm_image_s94.size_nm_y:.3f}")
                print(f"  Bias Voltage (V): {stm_image_s94.bias_v:.3f}")
                print(f"  Scan Speed (nm/s): {stm_image_s94.scan_speed_nm_s}") # May be None
                print(f"  Scan Angle (deg): {stm_image_s94.scan_angle_deg}")
                print(f"  Z Conversion (nm/raw): {stm_image_s94.z_nm_per_raw}") # May be None

                # Scale the data for display
                scaled_data_s94 = scale_to_grayscale(stm_image_s94.data)

                # Note: S94 reader handles transpose based on 'Swapped' flag internally.
                # Display the scaled image
                display_title = f"S94 Image: {os.path.basename(s94_file_path)}\nType: {stm_image_s94.image_type}"
                display_image(scaled_data_s94, display_title)
            else:
                logging.error("Failed to load data from S94 file or data array is empty.")

    except FileNotFoundError as fnf_err:
        logging.error(fnf_err)
    except Exception as e:
        # Catch any other exceptions during S94 file processing
        logging.exception(f"An error occurred while processing the S94 file: {e}")

    # --- Keep plots open ---
    # Call plt.show() at the end to display all figures and prevent
    # the script from exiting immediately. The windows will remain open
    # until manually closed by the user.
    logging.info("\nImage display finished. Close plot windows to exit script.")
    plt.show()

# Standard Python entry point guard
if __name__ == "__main__":
    main()
