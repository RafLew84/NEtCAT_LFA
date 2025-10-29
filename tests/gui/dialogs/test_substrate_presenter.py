import numpy as np
import pytest

pytest.importorskip("PyQt6", reason="PyQt6 is required for substrate presenter tests")
pytest.importorskip("pytestqt", reason="pytest-qt provides the qtbot fixture")

from PyQt6.QtWidgets import QListWidget

from lfa.core.constants import (
    LATTICE_TYPE_HEXAGONAL,
    LATTICE_TYPE_CUSTOM,
)
from lfa.core.data_models import OriginalImageRecord
from lfa.core.history import HistoryNode
from lfa.logic.history_manager import HistoryManager
from lfa.gui.dialogs.presenters.substrate_spot_presenter import (
    SubstrateSpotPresenter,
    SubstrateSpotState,
    TransformComputationError,
)


def _make_presenter(qtbot, *, lattice_type=LATTICE_TYPE_HEXAGONAL, custom_a=None):
    widget = QListWidget()
    qtbot.addWidget(widget)
    history_manager = HistoryManager(widget)
    state = SubstrateSpotState(
        selected_spots=[],
        lattice_type=lattice_type,
        selected_definition=None,
        custom_definition=None,
        custom_a_surf=custom_a,
        transform_matrix_F=None,
        transform_translation_t=None,
        transform_analysis=None,
        fitted_spots_px=[],
    )
    presenter = SubstrateSpotPresenter(
        history_manager=history_manager,
        fft_node_id="fft-node",
        fft_data=None,
        state=state,
    )
    return presenter


def test_presenter_builds_lattice_info_for_hexagonal(qtbot):
    presenter = _make_presenter(qtbot, custom_a=0.288)
    info = presenter.build_lattice_info_dict(preferred_point_count=6)
    assert info == {"type": LATTICE_TYPE_HEXAGONAL, "a_surf": 0.288, "preferred_point_count": 6}


def test_presenter_returns_none_without_definition(qtbot):
    presenter = _make_presenter(qtbot, lattice_type=LATTICE_TYPE_CUSTOM)
    info = presenter.build_lattice_info_dict()
    assert info is None


def test_presenter_transform_requires_three_spots(qtbot):
    presenter = _make_presenter(qtbot, custom_a=0.288)
    presenter.state.selected_spots.extend([(0.0, 0.0), (1.0, 1.0)])
    with pytest.raises(TransformComputationError) as excinfo:
        presenter.calculate_transform(preferred_point_count=6)
    assert "at least 3 substrate spots" in excinfo.value.user_message


def test_presenter_calculate_transform_success(qtbot, monkeypatch):
    widget = QListWidget()
    qtbot.addWidget(widget)
    history_manager = HistoryManager(widget)

    record = OriginalImageRecord(display_name="Root")
    history_manager.register_original_image(record)

    root_params = {
        "size_nm_x": 10.0,
        "size_nm_y": 10.0,
    }
    root_node = HistoryNode(
        operation_name="Original",
        image_data=np.zeros((8, 8), dtype=np.float32),
        parameters=root_params,
        data_type="STM",
        original_image_id=record.image_id,
    )
    history_manager.add_node(root_node)

    state = SubstrateSpotState(
        selected_spots=[(10.0, 12.0), (20.0, 22.0), (30.0, 32.0)],
        lattice_type=LATTICE_TYPE_HEXAGONAL,
        selected_definition="Au(111)",
        custom_definition=None,
        custom_a_surf=0.288,
        transform_matrix_F=None,
        transform_translation_t=None,
        transform_analysis=None,
        fitted_spots_px=[],
    )
    presenter = SubstrateSpotPresenter(
        history_manager=history_manager,
        fft_node_id=root_node.node_id,
        fft_data=np.zeros((64, 64), dtype=np.float32),
        state=state,
    )

    def fake_get_nearest_reciprocal_points(_info):
        return [(0.1, 0.2), (0.0, -0.2), (-0.1, 0.0)]

    def fake_match_and_fit_transform(*, measured_pts_px, ideal_pts_pool_px, num_expected_matches):
        assert num_expected_matches == 3
        assert measured_pts_px.shape[0] == 3
        assert ideal_pts_pool_px.shape[0] == 3
        F = np.eye(2)
        t = np.array([0.5, -0.25])
        analysis = {"rotation_angle_deg": 1.23, "principal_stretches": (1.0, 1.0), "rmse": 0.01}
        point_pairs = [(0, 0), (1, 1), (2, 2)]
        return F, t, analysis, point_pairs

    def fake_apply_affine_transform(points, F, t):
        return points @ F.T + t

    monkeypatch.setattr(
        "lfa.gui.dialogs.presenters.substrate_spot_presenter.get_nearest_reciprocal_points",
        fake_get_nearest_reciprocal_points,
    )
    monkeypatch.setattr(
        "lfa.gui.dialogs.presenters.substrate_spot_presenter.match_and_fit_transform",
        fake_match_and_fit_transform,
    )
    monkeypatch.setattr(
        "lfa.gui.dialogs.presenters.substrate_spot_presenter.apply_affine_transform",
        fake_apply_affine_transform,
    )

    result = presenter.calculate_transform(preferred_point_count=3)

    assert np.allclose(result.matrix_F, np.eye(2))
    assert np.allclose(result.translation_t, np.array([0.5, -0.25]))
    assert result.analysis["rmse"] == 0.01
    assert len(result.fitted_spots_px) == 3
    assert presenter.state.transform_matrix_F is not None
    assert presenter.state.transform_translation_t is not None
    assert presenter.state.fitted_spots_px == result.fitted_spots_px
