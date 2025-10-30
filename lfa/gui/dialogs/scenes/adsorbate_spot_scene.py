from __future__ import annotations

import logging
from typing import Iterable, Optional, Tuple

try:  # pragma: no cover - guarded import
    import pyqtgraph as pg
except ImportError:  # pragma: no cover
    pg = None  # type: ignore

from .substrate_spot_scene import MarkerSpec

logger = logging.getLogger(__name__)


class AdsorbateSpotScene:
    """
    Wrapper around pyqtgraph artefacts for the adsorbate dialog.

    Responsible for ROI placement and rendering primary overlays.
    """

    def __init__(self, *, initial_roi_size: int = 5) -> None:
        if pg is None:  # pragma: no cover
            raise RuntimeError("pyqtgraph is required for AdsorbateSpotScene.")

        self.plot_widget = pg.GraphicsLayoutWidget()
        self.view_box = self.plot_widget.addViewBox(row=0, col=0, lockAspect=True, invertY=True)
        self.view_box.setMenuEnabled(True)
        self.view_box.setMouseMode(pg.ViewBox.PanMode)
        self.view_box.setMouseEnabled(x=True, y=True)

        self.image_item = pg.ImageItem()
        self.view_box.addItem(self.image_item)

        self.histogram = pg.HistogramLUTItem()
        self.histogram.setImageItem(self.image_item)
        self.plot_widget.addItem(self.histogram, row=0, col=1)

        self.selection_roi = pg.RectROI(
            [0, 0],
            [initial_roi_size, initial_roi_size],
            pen={"color": (255, 0, 255), "width": 2},
            movable=True,
            resizable=True,
            rotatable=False,
            translateSnap=True,
            scaleSnap=True,
        )
        self.selection_roi.setVisible(False)
        self.view_box.addItem(self.selection_roi)

        # Overlay items
        self.raw_markers = pg.ScatterPlotItem()
        self.view_box.addItem(self.raw_markers)

        self.corrected_markers = pg.ScatterPlotItem()
        self.view_box.addItem(self.corrected_markers)

        self.ideal_reference_markers = pg.ScatterPlotItem()
        self.view_box.addItem(self.ideal_reference_markers)

        self.fitted_reference_markers = pg.ScatterPlotItem()
        self.view_box.addItem(self.fitted_reference_markers)

    # ------------------------------------------------------------------ Accessors
    def widget(self) -> pg.GraphicsLayoutWidget:
        return self.plot_widget

    def roi(self) -> pg.RectROI:
        return self.selection_roi

    # ------------------------------------------------------------------ Image control
    def set_image(self, data) -> None:
        if data is None:
            self.image_item.clear()
        else:
            self.image_item.setImage(data.T)

    # ------------------------------------------------------------------ Overlay helpers
    def show_raw_spots(self, spots: Iterable[Tuple[float, float]]) -> None:
        specs = [
            {
                "pos": tuple(map(float, pt)),
                "symbol": "o",
                "size": 10,
                "pen": pg.mkPen("y", width=1.5),
                "brush": pg.mkBrush(0, 0, 255, 120),
            }
            for pt in spots
        ]
        if specs:
            self.raw_markers.setData(spots=specs)
        else:
            self.raw_markers.clear()

    def show_corrected_spots(self, spots: Iterable[Tuple[float, float]]) -> None:
        specs = [
            {
                "pos": tuple(map(float, pt)),
                "symbol": "s",
                "size": 10,
                "pen": pg.mkPen("r", width=1.5),
                "brush": pg.mkBrush(255, 0, 0, 120),
            }
            for pt in spots
        ]
        if specs:
            self.corrected_markers.setData(spots=specs)
        else:
            self.corrected_markers.clear()

    def show_reference_overlay(
        self,
        *,
        ideal_specs: Iterable[MarkerSpec] = (),
        fitted_specs: Iterable[MarkerSpec] = (),
    ) -> None:
        ideal_specs = list(ideal_specs)
        if ideal_specs:
            self.ideal_reference_markers.setData(
                spots=[
                    {
                        "pos": spec.pos,
                        "symbol": spec.symbol,
                        "size": spec.size,
                        "pen": pg.mkPen(spec.pen, width=1.5),
                        "brush": pg.mkBrush(*(spec.brush or (0, 0, 0, 0))),
                    }
                    for spec in ideal_specs
                ]
            )
        else:
            self.ideal_reference_markers.clear()

        fitted_specs = list(fitted_specs)
        if fitted_specs:
            self.fitted_reference_markers.setData(
                spots=[
                    {
                        "pos": spec.pos,
                        "symbol": spec.symbol,
                        "size": spec.size,
                        "pen": pg.mkPen(spec.pen, width=2.0),
                        "brush": pg.mkBrush(*(spec.brush or (0, 0, 0, 0))),
                    }
                    for spec in fitted_specs
                ]
            )
        else:
            self.fitted_reference_markers.clear()

    def clear_all(self) -> None:
        self.show_raw_spots([])
        self.show_corrected_spots([])
        self.show_reference_overlay()

