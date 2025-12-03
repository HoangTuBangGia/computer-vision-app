# CV Master UI Module
from .main_window import MainWindow
from .image_viewer import ImageViewer
from .zoomable_viewer import ZoomableGraphicsView, SyncedImageViewer, SyncedImageViewerPanel
from .control_panel import ControlPanel
from .basic_tab import BasicTab
from .filters_tab import FiltersTab
from .morphology_tab import MorphologyTab
from .frequency_tab import FrequencyTab
from .segmentation_tab import SegmentationTab
from .pca_tab import PCATab
from .compression_tab import CompressionTab
from .geometry_tab import GeometryTab

__all__ = [
    'MainWindow', 'ImageViewer', 'ControlPanel', 
    'ZoomableGraphicsView', 'SyncedImageViewer', 'SyncedImageViewerPanel',
    'BasicTab', 'FiltersTab', 'MorphologyTab', 'FrequencyTab', 
    'SegmentationTab', 'PCATab', 'CompressionTab', 'GeometryTab'
]
