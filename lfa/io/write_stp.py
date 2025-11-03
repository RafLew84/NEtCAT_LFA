# lfa/io/write_stp.py
import logging
import struct
from typing import Any, Dict

import numpy as np

logger = logging.getLogger(__name__)

def write_STP_file(file_path: str, data_array: np.ndarray, header_info: Dict[str, Any]):
    """
    Writes data to a WSxM .stp file, faithfully reproducing the header structure.
    """
    logger.info(f"Writing WSxM STP file: {file_path}")
    try:
        with open(file_path, 'wb') as f:
            f.write(b"WSxM file copyright UAM\n")
            f.write(b"SxM Image file\n")
            for section_name, section_content in header_info.items():
                if section_name == "Header Root": # Special handling for the initial header lines
                    for key, value in section_content.items():
                        f.write(f"{key}: {value}\n".encode('utf-8'))
                    f.write(b"\n")
                elif isinstance(section_content, dict):
                    f.write(f"[{section_name}]\n\n".encode('utf-8'))
                    for key, value in section_content.items():
                        f.write(f"    {key}: {value}\n".encode('utf-8'))
                    f.write(b"\n")
            
            f.write(b"[Header end]\n")

            flat_data = data_array.flatten()
            binary_data = struct.pack(f'<{len(flat_data)}d', *flat_data)
            f.write(binary_data)
        
        logger.info(f"Successfully wrote STP file.")

    except Exception as e:
        logger.exception(f"Failed to write STP file '{file_path}': {e}")
        raise