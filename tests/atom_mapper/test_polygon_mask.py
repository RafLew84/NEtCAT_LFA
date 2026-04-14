"""Tests for polygon-mask helpers used by ROI-restricted local fitting."""

from __future__ import annotations

import numpy as np

from AtomMapper.app.models import ROIState
from AtomMapper.app.polygon_mask import PolygonMaskState, build_polygon_mask_for_roi


def test_build_polygon_mask_for_roi_rasterizes_polygon_in_roi_local_coordinates():
    roi = ROIState(x=10, y=20, width=6, height=6)
    polygon = PolygonMaskState(
        vertices_xy=(
            (11.0, 21.0),
            (15.0, 21.0),
            (15.0, 25.0),
            (11.0, 25.0),
        )
    )

    mask = build_polygon_mask_for_roi(roi, polygon)

    assert mask is not None
    assert mask.dtype == bool
    assert mask.shape == (6, 6)
    assert int(mask.sum()) == 16
    assert np.all(mask[1:5, 1:5])
    assert not np.any(mask[0, :])
    assert not np.any(mask[:, 0])


def test_build_polygon_mask_for_roi_returns_none_for_invalid_polygon():
    roi = ROIState(x=0, y=0, width=8, height=8)
    polygon = PolygonMaskState(vertices_xy=((1.0, 1.0), (2.0, 2.0)))

    mask = build_polygon_mask_for_roi(roi, polygon)

    assert mask is None
