"""Application package for AtomMapper."""

from .controller import AtomMapperController
from .gaussian_fit import GaussianPatchFitResult, fit_gaussian_to_roi_patch
from .gaussian_preview import GaussianFitPreviewWidget
from .global_scatter_plot_widget import GlobalScatterPlotWidget
from .io import SUPPORTED_STM_EXTENSIONS, load_loaded_image
from .main_window import AtomMapperMainWindow
from .models import AtomPoint, AtomRow, LoadedImage, ROIState
from .pyqtgraph_image_view import PyQtGraphSTMViewport
from .pyqtgraph_preview_bridge import PyQtGraphPreviewBridge
from .preprocessing import (
    apply_bm3d,
    apply_blur,
    apply_non_local_means,
    build_bm3d_metadata,
    build_blur_metadata,
    build_nlm_metadata,
    is_bm3d_available,
)
from .preprocessing_dialog import PreprocessingDialog
from .preprocessing_preview import PreprocessingImagePreview
from .preprocessing_state import (
    BM3DParameters,
    BlurParameters,
    NonLocalMeansParameters,
    PreviewViewport,
    PreprocessingMethod,
    PreprocessingPreviewRequest,
    PreprocessingPreviewResult,
    PreprocessingState,
)
from .plots import (
    GlobalScatterSample,
    GlobalScatterSeries,
    RowDistanceMetrics,
    RowMetricSeries,
    RowPlotMode,
    RowSeriesSample,
    build_global_scatter_series,
    build_row_distance_metrics,
    build_row_metric_series,
    sorted_row_points,
)
from .roi_preview import ROIPreviewWidget
from .row_metrics_widget import RowMetricsWidget
from .row_plot_widget import RowPlotWidget

__all__ = [
    "AtomMapperController",
    "AtomMapperMainWindow",
    "AtomPoint",
    "AtomRow",
    "BM3DParameters",
    "BlurParameters",
    "GaussianPatchFitResult",
    "GaussianFitPreviewWidget",
    "GlobalScatterPlotWidget",
    "LoadedImage",
    "NonLocalMeansParameters",
    "PreprocessingDialog",
    "PreprocessingImagePreview",
    "PreprocessingMethod",
    "PreprocessingPreviewRequest",
    "PreprocessingPreviewResult",
    "PreprocessingState",
    "PyQtGraphSTMViewport",
    "PyQtGraphPreviewBridge",
    "PreviewViewport",
    "ROIState",
    "ROIPreviewWidget",
    "RowMetricsWidget",
    "RowPlotWidget",
    "GlobalScatterSample",
    "GlobalScatterSeries",
    "RowDistanceMetrics",
    "RowMetricSeries",
    "RowPlotMode",
    "RowSeriesSample",
    "SUPPORTED_STM_EXTENSIONS",
    "apply_bm3d",
    "apply_blur",
    "apply_non_local_means",
    "build_bm3d_metadata",
    "build_blur_metadata",
    "build_global_scatter_series",
    "build_row_distance_metrics",
    "build_nlm_metadata",
    "build_row_metric_series",
    "fit_gaussian_to_roi_patch",
    "is_bm3d_available",
    "load_loaded_image",
    "sorted_row_points",
]
