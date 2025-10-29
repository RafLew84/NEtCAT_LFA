from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Iterable, Optional, Sequence, Tuple

from PyQt6.QtCore import QPointF

try:  # pragma: no cover - guarded import
    import pyqtgraph as pg
except ImportError:  # pragma: no cover
    pg = None  # type: ignore

logger = logging.getLogger(__name__)


@dataclass
class MarkerSpec:
    pos: Tuple[float, float]
    symbol: str
    size: int
    pen: Tuple[int, int, int]
    brush: Optional[Tuple[int, int, int, int]] = None


class SubstrateSpotScene:
    """
    Owns pyqtgraph artefacts for the substrate spot dialog.

    Responsibilities:
    - Provide access to the underlying view box, image item, histogram, ROI.
    - Render selected/fitted spots and ideal lattice overlays.
    - Handle floating guide lines when displaying matched pairs.
    """

    def __init__(self) -> None:
        if pg is None:  # pragma: no cover
            raise RuntimeError("pyqtgraph is required for SubstrateSpotScene.")

        self.plot_widget = pg.GraphicsLayoutWidget()
        self.view_box = self.plot_widget.addViewBox(row=0, col=0, lockAspect=True, invertY=True)
        self.image_item = pg.ImageItem()
        self.view_box.addItem(self.image_item)

        self.selected_markers = pg.ScatterPlotItem()
        self.view_box.addItem(self.selected_markers)
        self.fitted_markers = pg.ScatterPlotItem()
        self.view_box.addItem(self.fitted_markers)

        self.histogram = pg.HistogramLUTItem()
        self.histogram.setImageItem(self.image_item)
        self.plot_widget.addItem(self.histogram, row=0, col=1)

        self.view_box.setMenuEnabled(True)
        self.view_box.setMouseMode(pg.ViewBox.PanMode)
        self.view_box.setMouseEnabled(x=True, y=True)

        self.selection_roi = pg.RectROI(
            [10, 10],
            [50, 50],
            pen={"color": (255, 255, 0), "width": 2},
            movable=True,
            resizable=True,
            rotatable=False,
            translateSnap=True,
            scaleSnap=True,
        )
        self.view_box.addItem(self.selection_roi)
        self.selection_roi.setVisible(False)

        self.ideal_overlay_item: Optional[pg.ScatterPlotItem] = None
        self.pair_lines: list[pg.PlotDataItem] = []

    # ------------------------------------------------------------------ Basic accessors
    def widget(self) -> pg.GraphicsLayoutWidget:
        return self.plot_widget

    def roi(self) -> pg.RectROI:
        return self.selection_roi

    def set_image(self, data) -> None:
        self.image_item.setImage(data.T)

    # ------------------------------------------------------------------ Marker rendering
    def show_selected_spots(self, specs: Iterable[MarkerSpec]) -> None:
        specs = list(specs)
        if specs:
            spots = [
                {
                    "pos": spec.pos,
                    "symbol": spec.symbol,
                    "size": spec.size,
                    "pen": pg.mkPen(spec.pen, width=1.5),
                    "brush": pg.mkBrush(*(spec.brush or (0, 0, 0, 0))),
                }
                for spec in specs
            ]
            self.selected_markers.setData(spots=spots)
        else:
            self.selected_markers.clear()

    def show_fitted_spots(self, specs: Iterable[MarkerSpec]) -> None:
        specs = list(specs)
        if specs:
            spots = [
                {
                    "pos": spec.pos,
                    "symbol": spec.symbol,
                    "size": spec.size,
                    "pen": pg.mkPen(spec.pen, width=2.0),
                    "brush": pg.mkBrush(*(spec.brush or (0, 0, 0, 0))),
                }
                for spec in specs
            ]
            self.fitted_markers.setData(spots=spots)
        else:
            self.fitted_markers.clear()
            self.view_box.scene().update()

    def show_ideal_overlay(self, specs: Iterable[MarkerSpec]) -> None:
        if self.ideal_overlay_item:
            try:
                self.view_box.removeItem(self.ideal_overlay_item)
            except RuntimeError:  # pragma: no cover
                pass
            self.ideal_overlay_item = None

        specs = list(specs)
        if not specs:
            return

        self.ideal_overlay_item = pg.ScatterPlotItem()
        self.ideal_overlay_item.setData(
            spots=[
                {
                    "pos": spec.pos,
                    "symbol": spec.symbol,
                    "size": spec.size,
                    "pen": pg.mkPen(spec.pen, width=1.5),
                    "brush": pg.mkBrush(*(spec.brush or (0, 0, 0, 0))),
                }
                for spec in specs
            ]
        )
        self.view_box.addItem(self.ideal_overlay_item)

    def show_pair_lines(self, pairs: Sequence[Tuple[Tuple[float, float], Tuple[float, float]]]) -> None:
        for line in self.pair_lines:
            try:
                self.view_box.removeItem(line)
            except RuntimeError:  # pragma: no cover
                pass
        self.pair_lines.clear()

        for measured, ideal in pairs:
            line = pg.PlotDataItem(
                x=[measured[0], ideal[0]],
                y=[measured[1], ideal[1]],
                pen=pg.mkPen("y", width=1, style=pg.QtCore.Qt.PenStyle.DashLine),
            )
            self.view_box.addItem(line)
            self.pair_lines.append(line)

    # ------------------------------------------------------------------ Utility
    def map_scene_to_data(self, scene_pos: QPointF) -> Optional[QPointF]:
        pos_viewbox = self.view_box.mapSceneToView(scene_pos)
        return self.image_item.mapToData(pos_viewbox)

    def clear_all(self) -> None:
        self.show_selected_spots([])
        self.show_fitted_spots([])
        self.show_ideal_overlay([])
        self.show_pair_lines([])
