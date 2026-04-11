"""IO helpers for loading STM files into AtomMapper models."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict
from uuid import NAMESPACE_URL, uuid4, uuid5

import numpy as np

from lfa.core.data_models import STMImage
from lfa.io.factory import load_stm_file

from .models import LoadedImage

logger = logging.getLogger(__name__)

SUPPORTED_STM_EXTENSIONS = {".stp", ".s94"}


def _build_loaded_image(stm_image: STMImage, source_path: Path) -> LoadedImage:
    """Convert ``STMImage`` into the simplified AtomMapper data model."""

    image_data = np.asarray(stm_image.data, dtype=float)
    if image_data.ndim != 2:
        raise ValueError(f"Expected 2D STM image data, got shape {image_data.shape!r}.")

    metadata: Dict[str, Any] = {
        "image_type": stm_image.image_type,
        "scan_angle_deg": stm_image.scan_angle_deg,
        "bias_v": stm_image.bias_v,
        "setpoint_a": stm_image.setpoint_a,
        "scan_speed_nm_s": stm_image.scan_speed_nm_s,
        "offset_nm_x": stm_image.offset_nm_x,
        "offset_nm_y": stm_image.offset_nm_y,
        "z_nm_per_raw": stm_image.z_nm_per_raw,
    }

    raw_metadata = dict(stm_image.raw_header) if stm_image.raw_header else {}
    resolved_path = source_path.expanduser().resolve(strict=False)
    source_group_id = uuid5(NAMESPACE_URL, resolved_path.as_posix()).hex

    return LoadedImage(
        source_path=str(source_path),
        display_name=source_path.name,
        file_extension=source_path.suffix.lower(),
        image_data=image_data,
        pixels_x=int(stm_image.pixels_x),
        pixels_y=int(stm_image.pixels_y),
        size_nm_x=float(stm_image.size_nm_x),
        size_nm_y=float(stm_image.size_nm_y),
        image_id=uuid4().hex,
        source_group_id=source_group_id,
        parent_image_id=None,
        variant_name="original",
        metadata=metadata,
        raw_metadata=raw_metadata,
    )


def load_loaded_image(file_path: str | Path) -> LoadedImage:
    """Load a supported STM file and adapt it to the AtomMapper model."""

    source_path = Path(file_path).expanduser()
    suffix = source_path.suffix.lower()

    if suffix not in SUPPORTED_STM_EXTENSIONS:
        raise ValueError(
            f"Unsupported file extension '{suffix or '<none>'}'. "
            f"Supported extensions: {', '.join(sorted(SUPPORTED_STM_EXTENSIONS))}."
        )

    stm_image = load_stm_file(str(source_path))
    if stm_image is None:
        raise ValueError(f"Could not load STM image from '{source_path}'.")

    loaded_image = _build_loaded_image(stm_image, source_path)
    logger.info(
        "Loaded AtomMapper image '%s' (%dx%d px, %.3f x %.3f nm).",
        loaded_image.display_name,
        loaded_image.pixels_x,
        loaded_image.pixels_y,
        loaded_image.size_nm_x,
        loaded_image.size_nm_y,
    )
    return loaded_image
