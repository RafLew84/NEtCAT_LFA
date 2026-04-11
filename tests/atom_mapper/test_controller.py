"""Tests for AtomMapper controller state management."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from AtomMapper.app.controller import AtomMapperController
from AtomMapper.app.models import LoadedImage


def _make_loaded_image(name: str, width: int = 8, height: int = 6) -> LoadedImage:
    image_data = np.arange(width * height, dtype=float).reshape((height, width))
    return LoadedImage(
        source_path=str(Path("/tmp") / name),
        display_name=name,
        file_extension=Path(name).suffix.lower(),
        image_data=image_data,
        pixels_x=width,
        pixels_y=height,
        size_nm_x=float(width),
        size_nm_y=float(height),
        metadata={"image_type": "Topo"},
        raw_metadata={},
    )


def test_controller_tracks_loaded_images_and_selection():
    controller = AtomMapperController()
    first = _make_loaded_image("first.stp")
    second = _make_loaded_image("second.s94")

    controller.set_loaded_images([first, second])

    assert controller.active_image_index == 0
    assert controller.active_image == first
    assert controller.loaded_images == (first, second)

    selected = controller.select_image(1)
    assert selected == second
    assert controller.active_image_index == 1
    assert controller.active_image == second


def test_controller_rejects_invalid_selection():
    controller = AtomMapperController()
    controller.set_loaded_images([_make_loaded_image("only.stp")])

    with pytest.raises(IndexError, match="out of range"):
        controller.select_image(5)
