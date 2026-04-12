"""Tests for AtomMapper session runtime snapshot and file I/O."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from AtomMapper.app.controller import AtomMapperController
from AtomMapper.app.models import AtomPoint, LoadedImage, ROIState
from AtomMapper.app.plots import PlotUnit, RowPlotMode
from AtomMapper.app.session_io import (
    build_session_from_runtime,
    load_session_from_file,
    save_session_to_file,
)
from AtomMapper.app.session_model import SessionViewState


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


def test_build_session_from_runtime_captures_controller_and_view_state():
    controller = AtomMapperController()
    original = _make_loaded_image("sample.stp", width=12, height=10)
    controller.set_loaded_images([original])
    controller.update_active_roi_state(ROIState(x=2, y=3, width=6, height=6))
    row = controller.create_row_for_active_source_group(display_name="Row 1")
    controller.add_point_to_row(
        AtomPoint(
            row_id=row.row_id,
            image_id=original.image_id,
            source_group_id=original.source_group_id,
            point_index=0,
            x_px=4.0,
            y_px=5.0,
            point_id="point-1",
        )
    )

    session = build_session_from_runtime(
        controller,
        active_point_id_by_source_group={original.source_group_id: "point-1"},
        view_state=SessionViewState(
            show_gaussian_fit=False,
            row_plot_mode=RowPlotMode.DISTANCE_PX,
            row_plot_unit=PlotUnit.PX,
            row_metrics_unit=PlotUnit.NM,
            global_scatter_unit=PlotUnit.NM,
        ),
    )

    assert session.active_image_id == original.image_id
    assert session.roi_states_by_image_id[original.image_id] == ROIState(x=2, y=3, width=6, height=6)
    assert session.active_row_id_by_source_group[original.source_group_id] == row.row_id
    assert session.active_point_id_by_source_group[original.source_group_id] == "point-1"
    assert session.view_state.show_gaussian_fit is False
    assert session.view_state.row_plot_mode is RowPlotMode.DISTANCE_PX
    assert session.view_state.row_metrics_unit is PlotUnit.NM
    assert session.view_state.global_scatter_unit is PlotUnit.NM


def test_save_session_to_file_writes_json_project(tmp_path: Path):
    controller = AtomMapperController()
    image = _make_loaded_image("sample.stp")
    controller.set_loaded_images([image])

    session = build_session_from_runtime(controller)
    project_path = tmp_path / "sample.atommapper_proj"
    saved_path = save_session_to_file(project_path, session)

    assert saved_path == project_path
    with project_path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)

    assert payload["version"] == 1
    assert len(payload["loaded_images"]) == 1
    assert payload["loaded_images"][0]["display_name"] == "sample.stp"


def test_load_session_from_file_round_trips_saved_project(tmp_path: Path):
    controller = AtomMapperController()
    image = _make_loaded_image("sample.stp", width=10, height=9)
    controller.set_loaded_images([image])
    controller.update_active_roi_state(ROIState(x=1, y=2, width=4, height=5))
    row = controller.create_row_for_active_source_group(display_name="Row 1")
    controller.add_point_to_row(
        AtomPoint(
            row_id=row.row_id,
            image_id=image.image_id,
            source_group_id=image.source_group_id,
            point_index=0,
            x_px=3.0,
            y_px=4.0,
            point_id="point-1",
        )
    )

    session = build_session_from_runtime(
        controller,
        active_point_id_by_source_group={image.source_group_id: "point-1"},
        view_state=SessionViewState(
            show_gaussian_fit=False,
            row_plot_mode=RowPlotMode.DISTANCE_PX,
            row_plot_unit=PlotUnit.PX,
            row_metrics_unit=PlotUnit.NM,
            global_scatter_unit=PlotUnit.NM,
        ),
    )
    project_path = tmp_path / "roundtrip.atommapper_proj"
    save_session_to_file(project_path, session)

    restored = load_session_from_file(project_path)

    assert restored.to_dict() == session.to_dict()
