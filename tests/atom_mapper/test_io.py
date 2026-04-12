"""Tests for the AtomMapper STM loading adapter."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from AtomMapper.app.io import SUPPORTED_STM_EXTENSIONS, load_loaded_image
from lfa.core.data_models import STMImage


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def test_supported_extensions_cover_stp_and_s94():
    assert SUPPORTED_STM_EXTENSIONS == {".s94", ".stp"}


def test_load_loaded_image_from_sample_stp():
    sample_path = PROJECT_ROOT / "data" / "8343.stp"

    loaded = load_loaded_image(sample_path)
    calibration = loaded.physical_calibration

    assert loaded.display_name == "8343.stp"
    assert loaded.file_extension == ".stp"
    assert loaded.image_data.ndim == 2
    assert loaded.image_data.shape == (loaded.pixels_y, loaded.pixels_x)
    assert loaded.pixels_x > 0
    assert loaded.pixels_y > 0
    assert loaded.size_nm_x > 0.0
    assert loaded.size_nm_y > 0.0
    assert calibration is not None
    assert calibration.size_nm_x == pytest.approx(loaded.size_nm_x)
    assert calibration.size_nm_y == pytest.approx(loaded.size_nm_y)
    assert calibration.pixel_size_nm_x == pytest.approx(loaded.size_nm_x / loaded.pixels_x)
    assert calibration.pixel_size_nm_y == pytest.approx(loaded.size_nm_y / loaded.pixels_y)
    assert loaded.pixel_size_nm_x is not None
    assert loaded.pixel_size_nm_y is not None
    assert loaded.metadata["image_type"] == "Topo"
    assert loaded.image_id
    assert loaded.source_group_id
    assert loaded.parent_image_id is None
    assert loaded.variant_name == "original"
    assert loaded.is_original is True


def test_load_loaded_image_from_sample_s94():
    sample_path = PROJECT_ROOT / "data" / "85291r.s94"

    loaded = load_loaded_image(sample_path)
    calibration = loaded.physical_calibration

    assert loaded.display_name == "85291r.s94"
    assert loaded.file_extension == ".s94"
    assert loaded.image_data.ndim == 2
    assert loaded.image_data.shape == (loaded.pixels_y, loaded.pixels_x)
    assert loaded.pixels_x > 0
    assert loaded.pixels_y > 0
    assert loaded.size_nm_x > 0.0
    assert loaded.size_nm_y > 0.0
    assert calibration is not None
    assert calibration.size_nm_x == pytest.approx(loaded.size_nm_x)
    assert calibration.size_nm_y == pytest.approx(loaded.size_nm_y)
    assert calibration.pixel_size_nm_x == pytest.approx(loaded.size_nm_x / loaded.pixels_x)
    assert calibration.pixel_size_nm_y == pytest.approx(loaded.size_nm_y / loaded.pixels_y)
    assert loaded.metadata["image_type"] in {"Topography", "Current", "Unknown (0)", "Unknown (1)"}
    assert "Bias_Voltage_mV" in loaded.raw_metadata
    assert loaded.image_id
    assert loaded.source_group_id
    assert loaded.parent_image_id is None
    assert loaded.variant_name == "original"


@pytest.mark.parametrize(
    ("sample_name", "expected_extension"),
    [
        ("8343.stp", ".stp"),
        ("85291r.s94", ".s94"),
    ],
)
def test_real_sample_files_provide_usable_nm_point_conversion(
    sample_name: str,
    expected_extension: str,
):
    sample_path = PROJECT_ROOT / "data" / sample_name

    loaded = load_loaded_image(sample_path)
    calibration = loaded.physical_calibration

    assert loaded.file_extension == expected_extension
    assert calibration is not None
    x_nm, y_nm = calibration.point_px_to_nm(10.5, 12.25)
    assert x_nm == pytest.approx(10.5 * calibration.pixel_size_nm_x)
    assert y_nm == pytest.approx(12.25 * calibration.pixel_size_nm_y)
    assert x_nm > 0.0
    assert y_nm > 0.0
    assert loaded.calibration_summary is not None
    assert "nm/px" in loaded.calibration_summary


def test_load_loaded_image_rejects_unsupported_extension(tmp_path: Path):
    bad_path = tmp_path / "not_supported.txt"
    bad_path.write_text("dummy", encoding="utf-8")

    with pytest.raises(ValueError, match="Unsupported file extension"):
        load_loaded_image(bad_path)


def test_load_loaded_image_keeps_original_row_and_col_orientation(monkeypatch, tmp_path: Path):
    fake_path = tmp_path / "fake.stp"
    fake_path.write_text("dummy", encoding="utf-8")

    stm_image = STMImage(
        file_name=str(fake_path),
        raw_header={},
        data=np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=float),
        pixels_x=3,
        pixels_y=2,
        size_nm_x=30.0,
        size_nm_y=20.0,
        offset_nm_x=7.0,
        offset_nm_y=9.0,
        image_type="Topo",
    )

    monkeypatch.setattr("AtomMapper.app.io.load_stm_file", lambda _: stm_image)

    loaded = load_loaded_image(fake_path)

    assert loaded.image_data.shape == (2, 3)
    assert loaded.image_data.tolist() == [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]
    assert loaded.pixels_x == 3
    assert loaded.pixels_y == 2
    assert loaded.size_nm_x == 30.0
    assert loaded.size_nm_y == 20.0
    assert loaded.has_physical_calibration is True
    assert loaded.pixel_size_nm_x == pytest.approx(10.0)
    assert loaded.pixel_size_nm_y == pytest.approx(10.0)
    assert loaded.metadata["offset_nm_x"] == 7.0
    assert loaded.metadata["offset_nm_y"] == 9.0
    assert "transposed_for_display" not in loaded.metadata
    assert loaded.variant_name == "original"
    assert loaded.parent_image_id is None
    assert loaded.source_group_id
