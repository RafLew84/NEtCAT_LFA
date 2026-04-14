"""Tests for the versioned AtomMapper session model."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from AtomMapper.app.fit_models import LocalFitModelType
from AtomMapper.app.fit_settings import FitSettingsState, GaussianFitSettings
from AtomMapper.app.models import AtomPoint, AtomRow, LoadedImage, ROIState
from AtomMapper.app.polygon_mask import PolygonMaskState
from AtomMapper.app.plots import PlotUnit, RowPlotMode
from AtomMapper.app.session_model import (
    ATOMMAPPER_SESSION_VERSION,
    AtomMapperSession,
    SessionViewState,
)


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
        raw_metadata={"source": "test"},
    )


def test_session_view_state_roundtrips_metric_and_unit_preferences():
    state = SessionViewState(
        show_gaussian_fit=False,
        row_plot_mode=RowPlotMode.DISTANCE_NM,
        row_plot_unit=PlotUnit.NM,
        row_metrics_unit=PlotUnit.NM,
        global_scatter_unit=PlotUnit.NM,
        active_polygon_mask=PolygonMaskState(
            vertices_xy=((1.0, 2.0), (4.0, 2.0), (3.0, 5.0))
        ),
    )

    restored = SessionViewState.from_dict(state.to_dict())

    assert restored == state


def test_atommapper_session_roundtrips_loaded_images_rows_rois_and_view_state():
    original = _make_loaded_image("sample.stp", width=12, height=10)
    variant = original.derive_variant(
        variant_name="blur",
        image_data=original.image_data + 2.0,
        metadata_updates={"preprocess": "blur"},
    )
    row = AtomRow(
        row_id="row-1",
        source_group_id=original.source_group_id,
        display_name="Row 1",
        points=(
            AtomPoint(
                row_id="row-1",
                image_id=original.image_id,
                source_group_id=original.source_group_id,
                point_index=0,
                x_px=2.0,
                y_px=3.0,
                x_nm=2.0,
                y_nm=3.0,
                point_id="point-1",
            ),
            AtomPoint(
                row_id="row-1",
                image_id=variant.image_id,
                source_group_id=original.source_group_id,
                point_index=1,
                x_px=4.5,
                y_px=6.5,
                x_nm=4.5,
                y_nm=6.5,
                point_id="point-2",
                manual_override=True,
                manual_override_source="drag",
            ),
        ),
    )
    session = AtomMapperSession(
        loaded_images=(original, variant),
        active_image_id=variant.image_id,
        roi_states_by_image_id={
            original.image_id: ROIState(x=1, y=2, width=6, height=6),
            variant.image_id: ROIState(x=3, y=4, width=5, height=5),
        },
        rows=(row,),
        active_row_id_by_source_group={original.source_group_id: row.row_id},
        active_point_id_by_source_group={original.source_group_id: "point-2"},
        fit_settings=FitSettingsState(
            model=LocalFitModelType.GAUSSIAN,
            gaussian=GaussianFitSettings(
                amplitude_init=11.0,
                sigma_y_init=1.4,
                sigma_x_init=1.8,
            ),
        ),
        view_state=SessionViewState(
            show_gaussian_fit=False,
            row_plot_mode=RowPlotMode.DISTANCE_PX,
            row_plot_unit=PlotUnit.PX,
            row_metrics_unit=PlotUnit.NM,
            global_scatter_unit=PlotUnit.NM,
            active_polygon_mask=PolygonMaskState(
                vertices_xy=((3.0, 4.0), (7.0, 4.0), (7.0, 8.0), (3.0, 8.0))
            ),
        ),
    )

    restored = AtomMapperSession.from_dict(session.to_dict())

    assert restored.version == ATOMMAPPER_SESSION_VERSION
    assert restored.active_image_id == variant.image_id
    assert tuple(image.image_id for image in restored.loaded_images) == (
        original.image_id,
        variant.image_id,
    )
    assert np.array_equal(restored.loaded_images[0].image_data, original.image_data)
    assert np.array_equal(restored.loaded_images[1].image_data, variant.image_data)
    assert restored.loaded_images[1].metadata["preprocess"] == "blur"
    assert restored.roi_states_by_image_id[variant.image_id] == ROIState(
        x=3,
        y=4,
        width=5,
        height=5,
    )
    assert len(restored.rows) == 1
    assert restored.rows[0].to_dict() == row.to_dict()
    assert restored.active_row_id_by_source_group == {original.source_group_id: row.row_id}
    assert restored.active_point_id_by_source_group == {original.source_group_id: "point-2"}
    assert restored.fit_settings == session.fit_settings
    assert restored.view_state == session.view_state


def test_atommapper_session_rejects_unsupported_version():
    with pytest.raises(ValueError, match="Unsupported AtomMapper session version"):
        AtomMapperSession.from_dict({"version": 99})


def test_atommapper_session_rejects_broken_active_point_reference():
    image = _make_loaded_image("sample.stp")
    row = AtomRow(
        row_id="row-1",
        source_group_id=image.source_group_id,
        display_name="Row 1",
        points=(),
    )

    with pytest.raises(ValueError, match="existing point_id"):
        AtomMapperSession(
            loaded_images=(image,),
            rows=(row,),
            active_point_id_by_source_group={image.source_group_id: "missing-point"},
        )
