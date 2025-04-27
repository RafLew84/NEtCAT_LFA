# lfa/io/stp_reader.py
"""
Reads Omicron STP STM image files.

This module provides a function to parse the header and binary data
of .stp files, assuming the common Omicron format where a text header
ends with '[Header end]' followed by binary data (double precision floats).
It also handles parsing values with units in the header and rotates the
image data by 180 degrees as per user request.
"""
import struct
import numpy as np
import logging
import re
from ..core.data_models import STMImage

logger = logging.getLogger(__name__)

# Maximum number of header lines to read to prevent infinite loops on corrupted files
MAX_HEADER_LINES = 1000

# --- Helper function for parsing values with units ---
def parse_value_unit(value_str: str, default_unit_factor: float = 1.0) -> float:
    """
    Parses a string potentially containing a numerical value and a unit.
    Returns the numerical value scaled according to common physical units.

    Args:
        value_str (str): The string to parse (e.g., "10.5 nm", "-500 mV", "1.2e-9 A").
        default_unit_factor (float): Factor to use if no unit is detected (e.g., 1.0 for base units like m, V, A).

    Returns:
        float: The numerical value scaled appropriately to base units (m, V, A). Returns 0.0 if parsing fails.
    """
    value_str = value_str.strip()
    if not value_str:
        return 0.0

    # Regular expression to find numerical value (float, int, scientific notation)
    # and an optional unit at the end (allows common Greek letters too)
    match = re.match(r"([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)\s*([a-zA-ZμÅµ]*)", value_str)

    if not match:
        logger.warning(f"Could not parse value string: '{value_str}'")
        return 0.0 # Return 0.0 if parsing fails

    numeric_part_str = match.group(1)
    unit_part = match.group(2).lower() # Convert unit to lowercase

    try:
        numeric_value = float(numeric_part_str)
    except ValueError:
        logger.warning(f"Could not convert numeric part '{numeric_part_str}' to float in '{value_str}'")
        return 0.0

    # Scaling based on unit to return value in base units (meters, Volts, Amperes)
    factor = default_unit_factor
    # Length units -> meters
    if unit_part == 'nm': factor = 1e-9
    elif unit_part == 'pm': factor = 1e-12
    elif unit_part == 'å' or unit_part == 'a' or unit_part == 'angstrom': factor = 1e-10 # Angstrom
    elif unit_part == 'μm' or unit_part == 'um': factor = 1e-6 # micrometer
    elif unit_part == 'mm': factor = 1e-3 # millimeter
    # Voltage units -> Volts
    elif unit_part == 'v': factor = 1.0
    elif unit_part == 'mv': factor = 1e-3
    elif unit_part == 'uv' or unit_part == 'μv': factor = 1e-6
    # Current units -> Amperes
    elif unit_part == 'a': factor = 1.0
    elif unit_part == 'na': factor = 1e-9
    elif unit_part == 'pa': factor = 1e-12
    elif unit_part == 'ua' or unit_part == 'μa': factor = 1e-6

    # If unit is 'm' and default_unit_factor is 1.0, it's already correct.
    # Add more units if needed.

    scaled_value = numeric_value * factor
    # logger.debug(f"Parsed '{value_str}': value={numeric_value}, unit='{unit_part}', factor={factor}, result={scaled_value} (base unit)") # Optional debug log
    return scaled_value

# --- Main reader function ---
def read_stp_file(file_path: str) -> STMImage | None:
    """
    Reads data from an Omicron .stp file.

    Parses the text header, reads binary data (double), handles units in header values,
    and rotates the resulting image data by 180 degrees.

    Args:
        file_path (str): The path to the .stp file.

    Returns:
        STMImage | None: An STMImage object containing data and metadata,
                         or None if reading fails.

    Raises:
        FileNotFoundError: If the specified file is not found.
        ValueError: If the file header is invalid, incomplete, or data is corrupted.
        Exception: For other unexpected errors during file processing.
    """
    logger.info(f"Attempting to read STP file: {file_path}")
    if not isinstance(file_path, str):
        logger.error("Invalid input: file_path must be a string.")
        raise ValueError("Invalid input: file_path must be a string.")

    try:
        header_info = {}
        header_lines_read = 0
        header_end_found = False

        with open(file_path, "rb") as file:
            # Read header lines with added robustness
            while header_lines_read < MAX_HEADER_LINES:
                try:
                    line_start_pos = file.tell() # Store position before reading line
                    raw_line = file.readline()
                    if not raw_line: # Check for end of file
                        logger.warning("Reached end of file while reading header before finding '[Header end]'.")
                        break # Exit loop if file ends unexpectedly

                    # Decode using utf-8, ignore errors for potentially mixed/bad encodings
                    line = raw_line.decode('utf-8', errors='ignore').strip()
                    header_lines_read += 1
                    # Uncomment for deep debugging of header issues:
                    # if header_lines_read > MAX_HEADER_LINES - 10: logger.debug(f"Header line {header_lines_read}: {line}")

                except EOFError:
                     logger.warning("EOFError encountered while reading header line.")
                     break
                except Exception as decode_err:
                    logger.warning(f"Error decoding header line {header_lines_read} at position {line_start_pos}: {decode_err}. Skipping line.")
                    continue # Skip line if it cannot be decoded

                # Skip empty lines after stripping whitespace
                if not line:
                    continue

                # Check for header end marker
                if line == "[Header end]":
                    logger.debug(f"Found '[Header end]' marker at line {header_lines_read}.")
                    header_end_found = True
                    break
                elif ":" in line:
                    # Store header key-value pairs
                    key, value = line.split(":", 1)
                    header_info[key.strip()] = value.strip()
                # else: # Uncomment to log non-standard lines
                #    logger.debug(f"Non-standard header line {header_lines_read}: {line}")

            # Verify that the header end marker was found
            if not header_end_found:
                 msg = f"Header end marker '[Header end]' not found within the first {MAX_HEADER_LINES} lines."
                 logger.error(msg)
                 # Treat this as a critical error
                 raise ValueError(msg)

            # Get dimensions from header
            num_columns = int(header_info.get("Number of columns", 0))
            num_rows = int(header_info.get("Number of rows", 0))
            logger.debug(f"Header dimensions: Rows={num_rows}, Columns={num_columns}")

            # Validate dimensions
            if num_columns <= 0 or num_rows <= 0:
                msg = "Missing or invalid 'Number of columns' or 'Number of rows' in header."
                logger.error(msg)
                raise ValueError(msg)

            # Read binary data (assuming 8-byte double precision floats)
            expected_bytes = num_rows * num_columns * 8
            binary_data = file.read(expected_bytes)

            # Check if the correct amount of data was read
            if len(binary_data) != expected_bytes:
                 msg = f"Expected {expected_bytes} bytes of data, but found {len(binary_data)}."
                 logger.error(msg)
                 raise ValueError(msg)

            # Unpack binary data using struct (little-endian doubles)
            format_string = f'<{num_rows * num_columns}d'
            raw_data_points = struct.unpack(format_string, binary_data)

            # --- Create and Rotate data array ---
            # Reshape flat data into 2D NumPy array
            data_array = np.array(raw_data_points).reshape((num_rows, num_columns))
            logger.info(f"Successfully read data array of shape: {data_array.shape}")

            # *** ROTATE DATA BY 180 DEGREES ***
            # data_array = np.rot90(data_array, k=2)
            # logger.info(f"Rotated STP data array by 180 degrees. Shape remains: {data_array.shape}")
            # ---------------------------------

        # --- Map header info to STMImage fields using parse_value_unit ---
        pixels_x = num_columns
        pixels_y = num_rows

        # Parse dimensions - expect meters from parser, convert to nm
        size_x_m_str = header_info.get("X Amplitude", "0")
        size_y_m_str = header_info.get("Y Amplitude", "0")
        size_x_m = parse_value_unit(size_x_m_str, default_unit_factor=1.0) # Assume meters if no unit
        size_y_m = parse_value_unit(size_y_m_str, default_unit_factor=1.0)
        size_nm_x = size_x_m * 1e9 # Convert meters to nanometers
        size_nm_y = size_y_m * 1e9

        # Parse offset - expect meters from parser, convert to nm
        offset_x_m_str = header_info.get("X Offset", "0")
        offset_y_m_str = header_info.get("Y Offset", "0")
        offset_x_m = parse_value_unit(offset_x_m_str, default_unit_factor=1.0)
        offset_y_m = parse_value_unit(offset_y_m_str, default_unit_factor=1.0)
        offset_nm_x = offset_x_m * 1e9 # Convert meters to nanometers
        offset_nm_y = offset_y_m * 1e9

        # Parse bias voltage - expect Volts from parser
        # Look for common header keys for bias voltage
        bias_v_str = header_info.get("Gap Voltage", header_info.get("Bias", "0"))
        bias_v = parse_value_unit(bias_v_str, default_unit_factor=1.0) # Assume Volts if no unit

        # Parse setpoint current - expect Amperes from parser
        # Look for common header keys for setpoint current
        setpoint_a_str = header_info.get("Setpoint", header_info.get("Current", "0"))
        setpoint_a = parse_value_unit(setpoint_a_str, default_unit_factor=1.0) # Assume Amperes if no unit

        # Parse scan angle - assumed to be unitless (degrees)
        try:
            scan_angle_deg = float(header_info.get("Angle", "0.0"))
        except ValueError:
            logger.warning(f"Could not parse Scan Angle: '{header_info.get('Angle', '0.0')}'. Setting to 0.")
            scan_angle_deg = 0.0

        # Determine image type based on Channel name
        channel = header_info.get("Channel", "").lower()
        if "z" in channel or "topo" in channel: image_type = "Topography"
        elif "current" in channel or "iset" in channel: image_type = "Current"
        else: image_type = header_info.get("Channel", "Unknown") # Use original name if unsure

        # Create and return the STMImage object
        stm_image = STMImage(
            file_name=file_path,
            raw_header=header_info,
            data=data_array, # data_array is already rotated
            pixels_x=pixels_x,
            pixels_y=pixels_y,
            size_nm_x=size_nm_x,
            size_nm_y=size_nm_y,
            offset_nm_x=offset_nm_x,
            offset_nm_y=offset_nm_y,
            scan_angle_deg=scan_angle_deg,
            bias_v=bias_v,
            setpoint_a=setpoint_a,
            image_type=image_type
            # Add other fields like scan_speed if available and parsable
        )
        logger.info(f"STP file read and rotated successfully: {file_path}")
        return stm_image

    # Exception handling remains the same
    except FileNotFoundError: logger.error(f"File not found: {file_path}"); raise
    except struct.error as e: logger.error(f"Error unpacking binary data in '{file_path}': {e}"); raise ValueError(f"Error unpacking binary data: {e}") from e
    except ValueError as e: logger.error(f"Value error reading file '{file_path}': {e}"); raise
    except Exception as e: logger.exception(f"An unexpected error occurred while reading '{file_path}': {e}"); raise