# lfa/io/s94_reader.py

r""" .s94 metadata
/*
Converter was created thanks to the courtesy of Gerald Gmachmeir and Krzysztof Wasielewski
s94MetaDataDisplay: display metadata of .s94 STM image files.
(c) 2013 Gerald Gmachmeir    V0.74

s94 file:                   	c:\Test\Gerald\19708.S94
Image size [pixel]:         	256*256	Pixel
x/y scan direction swapped: 	0	(0: no swap;  1: swap of x/y scan direction)
ImageMode:                  	0	(0: topological image [nm];  1: current image [nA])
ImageNumber original:       	19708	(number of the originally recorded image)
Image size [nm]:            	  81.817 *  81.817	nm^2
OffsetX, OffsetY:           	   0.000 ,   0.000	nm^2  (x/y offset of image center)
ScanSpeed:                  	2496.865	nm/s (fast scanning direction)
Digital Bias Voltage:       	 200.000	mV
zGain:                      	3	(Gain of z-feedback circuit: 1/2/3) [5.5/22/88 nm]
Section:                    	2	(Gain of x/y scan ramp amplifiers: 1/2/3)
Scan Angle:                 	80	deg (Rotation with respect to physical x/y scan direction: 0-359 deg)

Image data:
Possible z-range:  -44.000 ... 43.979 nm  (-32768 ...32752 RAW)    (88 nm)
Actual image:       -3.459 ...  7.025 nm  ( -2576 ... 5232 RAW)
   zMax - zMin:                10.484 nm            ( 7808 RAW)    (11.9% F.S.)
   zMean:                       0.777 nm  (  2.26*256 =   578.9 RAW)
   zStdDev                      1.220 nm  (  3.55*256 =   908.7 RAW)
12 bit image:
 1 LSB (12 bit) = delta z Min = 0.0215 nm
   z-range used contains: 489 levels     8.9 bits effective

*/
"""

"""
Based on conv_s94-STP_W10_v2_Z-amp_MD+current.c converter created thanks to the courtesy of Gerald Gmachmeir and Krzysztof Wasielewski.
Reads SPECS S94 STM image files.

This module provides a function to parse the fixed-size binary header
and subsequent binary image data (int16) of .s94 files. It converts
the raw data to physical units (nm for topography, nA for current)
based on information found in the header (like z_gain).
"""
import logging
import struct

import numpy as np

from ..core.data_models import STMImage

logger = logging.getLogger(__name__)

# Define the format string used to read the fixed-size binary header
# < : little-endian
# h : short (2 bytes) * 4
# i : int (4 bytes) * 1
# f : float (4 bytes) * 6
# h : short (2 bytes) * 2
# f : float (4 bytes) * 4
# h : short (2 bytes) * 2
# Total: 8h + 1i + 10f = 19 fields
S94_HEADER_FORMAT = "<hhhhiffffffhhffffhh"
S94_HEADER_SIZE = struct.calcsize(S94_HEADER_FORMAT)
logger.debug(f"Calculated S94 header size: {S94_HEADER_SIZE} bytes for format '{S94_HEADER_FORMAT}'")


# Known z-gain values mapping to full scale range in nm
# Based on original code comments: "zGain: 1/2/3 [5.5/22/88 nm]"
Z_GAIN_RANGES = {1: 5.5, 2: 22.0, 3: 88.0}

def read_s94_file(file_path: str) -> STMImage | None:
    """
    Reads data from a SPECS .s94 file.

    Parses the binary header, reads the int16 image data, converts data
    to physical units (nm or nA), and handles potential x/y swapping.

    Args:
        file_path (str): The path to the .s94 file.

    Returns:
        STMImage | None: An STMImage object containing data and metadata,
                         or None if reading fails.

    Raises:
        FileNotFoundError: If the specified file is not found.
        ValueError: If the file header is invalid, incomplete, or data is corrupted.
        Exception: For other unexpected errors during file processing.
    """
    logger.info(f"Attempting to read S94 file: {file_path}")
    if not isinstance(file_path, str):
        logger.error("Invalid input: file_path must be a string.")
        raise ValueError("Invalid input: file_path must be a string.")

    try:
        with open(file_path, 'rb') as file:
            header_data = file.read(S94_HEADER_SIZE)
            if len(header_data) != S94_HEADER_SIZE:
                msg = f"Incomplete header read: expected {S94_HEADER_SIZE} bytes, got {len(header_data)}."
                logger.error(msg); raise ValueError(msg)

            try:
                (x_points, y_points, swapped, image_mode_raw, image_number,
                 x_size_nm, y_size_nm, x_offset_nm, y_offset_nm, scan_speed_nm_s,
                 bias_mv, z_gain_raw, section, kp, tn, tv, it, scan_angle_deg, z_flag # 19 variables
                ) = struct.unpack(S94_HEADER_FORMAT, header_data)
            except struct.error as unpack_err:
                 # Provide more context in case of struct errors
                 logger.error(f"Struct unpack error: {unpack_err}. Header size {len(header_data)} bytes may be wrong for format '{S94_HEADER_FORMAT}'.")
                 raise ValueError(f"Struct unpack error: {unpack_err}") from unpack_err

            logger.debug(f"Read header: x_points={x_points}, y_points={y_points}, mode={image_mode_raw}, x_size={x_size_nm}, y_size={y_size_nm}, bias={bias_mv}mV, z_gain={z_gain_raw}, angle={scan_angle_deg}")

            if x_points <= 0 or y_points <= 0:
                 msg = f"Invalid dimensions in header: x_points={x_points}, y_points={y_points}"
                 logger.error(msg); raise ValueError(msg)

            expected_data_points = x_points * y_points
            expected_bytes = expected_data_points * 2 # 2 bytes per int16
            image_raw_bytes = file.read(expected_bytes)

            if len(image_raw_bytes) != expected_bytes:
                msg = f"Incomplete image data: expected {expected_bytes} bytes, got {len(image_raw_bytes)}."
                logger.error(msg); raise ValueError(msg)

            image_data_raw = np.frombuffer(image_raw_bytes, dtype=np.int16).reshape((y_points, x_points))
            logger.info(f"Successfully read raw data array of shape: {image_data_raw.shape}")

            z_nm_per_raw = None # Initialize conversion factor
            if image_mode_raw == 0: # S94_TOPOGRAPHY
                image_type = "Topography"
                full_range_nm = Z_GAIN_RANGES.get(z_gain_raw, None)
                if full_range_nm is None:
                    logger.warning(f"Unknown z_gain value: {z_gain_raw}. Cannot calculate z_nm_per_raw. Keeping data in raw units (scaled to float32).")
                    data_array = image_data_raw.astype(np.float32) # Keep raw if gain unknown
                else:
                    z_nm_per_raw = full_range_nm / 65536.0
                    data_array = image_data_raw.astype(np.float32) * z_nm_per_raw
                    logger.debug(f"Calculated z_nm_per_raw = {z_nm_per_raw} for z_gain = {z_gain_raw}")

            elif image_mode_raw == 1: # S94_CURRENT
                image_type = "Current"
                current_nA_per_raw = 20.0 / 65536.0
                data_array = image_data_raw.astype(np.float32) * current_nA_per_raw

            else:
                logger.warning(f"Unknown image mode: {image_mode_raw}. Treating data as raw int16 (scaled to float32).")
                image_type = f"Unknown ({image_mode_raw})"
                data_array = image_data_raw.astype(np.float32)

            if swapped == 1:
                logger.info("x/y scan direction swapped flag is set. Transposing data array and dimensions.")
                data_array = data_array.T
                pixels_x_final = y_points
                pixels_y_final = x_points
                size_nm_x_final = y_size_nm
                size_nm_y_final = x_size_nm
            else:
                pixels_x_final = x_points
                pixels_y_final = y_points
                size_nm_x_final = x_size_nm
                size_nm_y_final = y_size_nm

            raw_header = {
                "x_points": x_points, "y_points": y_points, "Swapped": swapped,
                "image_mode": image_mode_raw, "Image_Number": image_number,
                "x_size": x_size_nm, "y_size": y_size_nm, "x_offset": x_offset_nm,
                "y_offset": y_offset_nm, "Scan_Speed": scan_speed_nm_s,
                "Bias_Voltage_mV": bias_mv, "z_gain": z_gain_raw, "Section": section,
                "Kp": kp, "Tn": tn, "Tv": tv, "It": it, "Scan_Angle": scan_angle_deg,
                "z_Flag": z_flag
            }

            stm_image = STMImage(
                file_name=file_path,
                raw_header=raw_header,
                data=data_array,
                pixels_x=pixels_x_final,
                pixels_y=pixels_y_final,
                size_nm_x=size_nm_x_final,
                size_nm_y=size_nm_y_final,
                offset_nm_x=x_offset_nm,  # Assume offset doesn't swap
                offset_nm_y=y_offset_nm,
                scan_angle_deg=scan_angle_deg,
                bias_v=bias_mv / 1000.0, # Convert mV to V
                scan_speed_nm_s=scan_speed_nm_s,
                z_nm_per_raw=z_nm_per_raw, # Store conversion factor if calculated
                image_type=image_type
            )
            logger.info(f"S94 file read successfully: {file_path}")
            return stm_image

    except FileNotFoundError: logger.error(f"File not found: {file_path}"); raise
    except struct.error as e: logger.error(f"Error unpacking binary data in '{file_path}': {e}"); raise ValueError(f"Error unpacking binary data: {e}") from e
    except ValueError as e: logger.error(f"Value error reading file '{file_path}': {e}"); raise
    except Exception as e: logger.exception(f"An unexpected error occurred while reading '{file_path}': {e}"); raise
