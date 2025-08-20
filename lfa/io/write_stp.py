# lfa/io/write_stp.py
import struct
import numpy as np
import logging
from typing import Dict, Any
from collections import OrderedDict

logger = logging.getLogger(__name__)

def write_STP_file(file_path: str, data_array: np.ndarray, header_info: Dict[str, Any]):
    """
    Zapisuje dane do pliku WSxM .stp, wiernie odtwarzając strukturę nagłówka.
    """
    logger.info(f"Writing WSxM STP file: {file_path}")
    try:
        with open(file_path, 'wb') as f:
            # Iterujemy przez zachowaną strukturę OrderedDict
            f.write(b"WSxM file copyright UAM\n")
            f.write(b"SxM Image file\n")
            for section_name, section_content in header_info.items():
                if section_name == "Header Root": # Specjalna obsługa dla linii na początku
                    for key, value in section_content.items():
                        f.write(f"{key}: {value}\n".encode('utf-8'))
                    f.write(b"\n")
                elif isinstance(section_content, dict):
                    f.write(f"[{section_name}]\n\n".encode('utf-8'))
                    for key, value in section_content.items():
                        f.write(f"    {key}: {value}\n".encode('utf-8'))
                    f.write(b"\n")
            
            f.write(b"[Header end]\n")

            # Zapis danych binarnych
            flat_data = data_array.flatten()
            binary_data = struct.pack(f'<{len(flat_data)}d', *flat_data)
            f.write(binary_data)
        
        logger.info(f"Successfully wrote STP file.")

    except Exception as e:
        logger.exception(f"Failed to write STP file '{file_path}': {e}")
        raise