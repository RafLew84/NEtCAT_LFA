# lfa/io/factory.py
"""
Factory module for selecting and loading STM file readers.

Provides a function to load STM data from various file formats
by detecting the format based on the file extension and dispatching
to the appropriate reader implementation.
"""

import os
import logging
from typing import Callable, Dict, Optional

# Use relative imports within the 'lfa.io' package
from .stp_reader import read_stp_file
from .s94_reader import read_s94_file
from ..core.data_models import STMImage # Import the data model

logger = logging.getLogger(__name__)

# Define a type hint for reader functions
ReaderFunction = Callable[[str], Optional[STMImage]]

# Dictionary mapping lowercase file extensions to reader functions
# This makes it easy to add support for more formats in the future.
SUPPORTED_FORMATS: Dict[str, ReaderFunction] = {
    ".stp": read_stp_file,
    ".s94": read_s94_file,
}

def load_stm_file(file_path: str) -> Optional[STMImage]:
    """
    Loads STM data from a supported file format.

    Detects the file format based on the extension and calls the
    corresponding reader function.

    Args:
        file_path (str): The full path to the STM data file.

    Returns:
        Optional[STMImage]: An STMImage object containing the loaded data
                            and metadata if successful and the format is
                            supported, otherwise None. Returns None if the
                            file is not found or cannot be read.
    """
    if not isinstance(file_path, str):
        logger.error(f"Invalid file path provided (not a string): {file_path}")
        return None

    if not os.path.exists(file_path):
        logger.error(f"File not found: {file_path}")
        # Let the specific reader raise FileNotFoundError if needed,
        # but good to log here too.
        # Alternatively, uncomment the line below to return None immediately:
        # return None
        pass # Allow the reader to handle FileNotFoundError explicitly

    # Extract the file extension (including the dot) and convert to lowercase
    try:
        _, file_extension = os.path.splitext(file_path)
        file_extension = file_extension.lower()
    except Exception as e:
        logger.error(f"Could not determine file extension for: {file_path} - {e}")
        return None

    # Find the appropriate reader function based on the extension
    reader_function = SUPPORTED_FORMATS.get(file_extension)

    if reader_function:
        logger.info(f"Found reader for extension '{file_extension}'. Attempting to read file: {file_path}")
        try:
            # Call the selected reader function
            stm_image = reader_function(file_path)
            if stm_image:
                logger.info(f"Successfully loaded file using {reader_function.__name__}.")
            else:
                # This case might occur if the reader itself returns None on failure
                logger.warning(f"Reader {reader_function.__name__} returned None for file: {file_path}")
            return stm_image
        except FileNotFoundError:
            # Error already logged by the reader or the check above
            return None
        except ValueError as ve:
            # Header/data format errors, already logged by the reader
            logger.error(f"Format error reading {file_path} with {reader_function.__name__}: {ve}")
            return None
        except Exception as e:
            # Catch any other unexpected errors during reading
            logger.exception(f"An unexpected error occurred while calling reader {reader_function.__name__} for file {file_path}: {e}")
            return None
    else:
        # Handle unsupported file formats
        logger.warning(f"Unsupported file extension: '{file_extension}'. No reader available for file: {file_path}")
        # Consider raising a custom exception like UnsupportedFormatError if needed later.
        return None

# Example usage (can be removed or placed in a test script)
if __name__ == '__main__':
    # Configure basic logging for standalone testing
    logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

    # Create dummy files or paths for testing relative to this script's location if needed
    # Assumes project root is parent of 'lfa', and 'data' is sibling of 'lfa'
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root_dir = os.path.dirname(os.path.dirname(script_dir))
    test_stp = os.path.join(project_root_dir, "data", "85291r.STP") # Adjust path as needed
    test_s94 = os.path.join(project_root_dir, "data", "85291r.s94") # Adjust path as needed
    test_unsupported = os.path.join(project_root_dir, "data", "test.txt")

    print("\n--- Testing Factory ---")

    print(f"\nTesting STP: {test_stp}")
    img_stp = load_stm_file(test_stp)
    if img_stp:
        print(f"  Loaded STP: Type='{img_stp.image_type}', Size='{img_stp.pixels_x}x{img_stp.pixels_y}'")
    else:
        print("  Failed to load STP.")

    print(f"\nTesting S94: {test_s94}")
    img_s94 = load_stm_file(test_s94)
    if img_s94:
        print(f"  Loaded S94: Type='{img_s94.image_type}', Size='{img_s94.pixels_x}x{img_s94.pixels_y}'")
    else:
        print("  Failed to load S94.")

    print(f"\nTesting Unsupported: {test_unsupported}")
    img_txt = load_stm_file(test_unsupported)
    if not img_txt:
        print("  Correctly handled unsupported file.")
    else:
        print("  Error: Should not have loaded unsupported file.")

    print(f"\nTesting Non-existent file:")
    img_none = load_stm_file("non_existent_file.stp")
    if not img_none:
        print("  Correctly handled non-existent file.")
    else:
        print("  Error: Should not have loaded non-existent file.")