# lfa/io/stp_reader.py
import struct
import numpy as np
import logging
import re
from collections import OrderedDict
from ..core.data_models import STMImage

logger = logging.getLogger(__name__)

def parse_value_unit(value_str: str) -> float:
    """Parse a value+unit string and return the value in meters."""
    value_str = value_str.strip()
    match = re.match(r"([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)\s*([a-zA-ZμÅµ]*)", value_str)
    if not match: return 0.0
    
    numeric_part_str, unit_part = match.group(1), match.group(2).lower()
    try: numeric_value = float(numeric_part_str)
    except ValueError: return 0.0
    
    factor = 1.0 
    if unit_part == 'nm': factor = 1e-9
    elif unit_part == 'pm': factor = 1e-12
    elif unit_part in ('å', 'a', 'angstrom'): factor = 1e-10
    return numeric_value * factor

def read_stp_file(file_path: str) -> STMImage | None:
    """
    Read a WSxM .stp file, parsing the header and binary payload.
    """
    logger.info(f"Reading WSxM STP file: {file_path}")
    try:
        header_info = OrderedDict()
        current_section = None
        
        with open(file_path, "rb") as file:
            while True:
                line_bytes = file.readline()
                if not line_bytes:
                    raise ValueError("Unexpected end of file before '[Header end]'.")
                
                if line_bytes.strip() == b"[Header end]":
                    break
                
                line = line_bytes.decode('utf-8', errors='ignore').strip()
                if not line: continue

                match = re.match(r"\[(.*)\]", line)
                if match:
                    current_section = match.group(1)
                    header_info[current_section] = OrderedDict()
                elif ":" in line:
                    key, value = line.split(":", 1)
                    key, value = key.strip(), value.strip()
                    if current_section is None:
                        if "Header Root" not in header_info:
                            header_info["Header Root"] = OrderedDict()
                        header_info["Header Root"][key] = value
                    else:
                        header_info[current_section][key] = value
                        
            general_info = header_info.get("General Info", {})
            control_info = header_info.get("Control", {})
            num_columns = int(general_info.get("Number of columns", 0))
            num_rows = int(general_info.get("Number of rows", 0))

            if num_columns <= 0 or num_rows <= 0:
                raise ValueError("Could not find valid dimensions in header.")

            expected_bytes = num_rows * num_columns * 8 # 8 bajtów dla 'double'
            binary_data = file.read(expected_bytes)
            if len(binary_data) != expected_bytes:
                raise ValueError(f"Incomplete binary data. Expected {expected_bytes}, got {len(binary_data)}.")

            data_array = np.frombuffer(binary_data, dtype='<f8').reshape((num_rows, num_columns))

        size_nm_x = parse_value_unit(control_info.get("X Amplitude", "0 nm")) * 1e9
        size_nm_y = parse_value_unit(control_info.get("Y Amplitude", "0 nm")) * 1e9
        
        stm_image = STMImage(
            file_name=file_path, raw_header=header_info, data=data_array,
            pixels_x=num_columns, pixels_y=num_rows, size_nm_x=size_nm_x,
            size_nm_y=size_nm_y, offset_nm_x=0, offset_nm_y=0,
            scan_angle_deg=0.0, bias_v=0.0, setpoint_a=0.0, image_type="Topo"
        )
        logger.info(f"WSxM STP file read successfully: {file_path}")
        return stm_image

    except Exception as e:
        logger.exception(f"Failed to read STP file '{file_path}': {e}")
        raise