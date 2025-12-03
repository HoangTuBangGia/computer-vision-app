"""
Control Panel Widget - Contains all control buttons and tabs
"""
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QTabWidget,
    QFrame, QLabel, QSpacerItem, QSizePolicy
)
from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QFont
import qtawesome as qta

from .basic_tab import BasicTab
from .filters_tab import FiltersTab
from .morphology_tab import MorphologyTab
from .frequency_tab import FrequencyTab
from .segmentation_tab import SegmentationTab
from .pca_tab import PCATab
from .compression_tab import CompressionTab
from .geometry_tab import GeometryTab
from .restoration_tab import RestorationTab


class StyledButton(QPushButton):
    """Custom styled button with icon"""
    
    def __init__(self, text: str, icon_name: str = None, parent=None):
        super().__init__(text, parent)
        
        if icon_name:
            icon = qta.icon(icon_name, color='white')
            self.setIcon(icon)
            
        self.setMinimumHeight(45)
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.setStyleSheet("""
            QPushButton {
                background-color: #0078d4;
                color: white;
                border: none;
                border-radius: 8px;
                padding: 10px 20px;
                font-size: 14px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #1084d8;
            }
            QPushButton:pressed {
                background-color: #006cbd;
            }
            QPushButton:disabled {
                background-color: #555555;
                color: #888888;
            }
        """)


class SecondaryButton(QPushButton):
    """Secondary styled button"""
    
    def __init__(self, text: str, icon_name: str = None, parent=None):
        super().__init__(text, parent)
        
        if icon_name:
            icon = qta.icon(icon_name, color='#cccccc')
            self.setIcon(icon)
            
        self.setMinimumHeight(45)
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.setStyleSheet("""
            QPushButton {
                background-color: #404040;
                color: #cccccc;
                border: 1px solid #555555;
                border-radius: 8px;
                padding: 10px 20px;
                font-size: 14px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #4a4a4a;
                border-color: #666666;
            }
            QPushButton:pressed {
                background-color: #353535;
            }
            QPushButton:disabled {
                background-color: #333333;
                color: #555555;
                border-color: #404040;
            }
        """)


class ControlPanel(QWidget):
    """
    Control Panel Widget
    Contains Load/Save buttons and processing tabs
    """
    
    # Signals
    loadImageClicked = Signal()
    saveImageClicked = Signal()
    basicProcessRequested = Signal(str, dict)  # operation, params
    addNoiseRequested = Signal(str, dict)  # noise_type, params
    applyFilterRequested = Signal(str, dict)  # filter_type, params
    morphologyRequested = Signal(str, dict)  # operation, params
    showSpectrumRequested = Signal()  # show FFT spectrum
    frequencyFilterRequested = Signal(str, dict)  # filter_type, params
    showFilteredSpectrumRequested = Signal()  # show filtered spectrum
    otsuRequested = Signal()  # Otsu thresholding
    manualThresholdRequested = Signal(int)  # manual threshold
    kmeansRequested = Signal(dict)  # K-means clustering
    loadFaceDatasetRequested = Signal()  # Load face dataset for PCA
    reconstructFaceRequested = Signal(int, int)  # face_index, n_components
    compressFullImageRequested = Signal(int)  # quality
    compressionQualityChanged = Signal(int)  # quality
    rotateRequested = Signal(float, bool)  # angle, keep_size
    scaleRequested = Signal(float, float)  # scale_x, scale_y
    resizeRequested = Signal(int, int)  # width, height
    flipRequested = Signal(str)  # 'horizontal', 'vertical', 'both'
    geometryResetRequested = Signal()
    restorationRequested = Signal(str, dict)  # operation, params
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("controlPanel")
        self.setMinimumWidth(200)
        self.setMaximumWidth(500)
        self._setupUI()
        
    def _setupUI(self):
        """Setup the control panel UI"""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(15, 15, 15, 15)
        layout.setSpacing(15)
        
        # Header
        header = self._createHeader()
        layout.addWidget(header)
        
        # Load/Save buttons
        buttons_widget = self._createButtonsSection()
        layout.addWidget(buttons_widget)
        
        # Separator
        separator = QFrame()
        separator.setFrameShape(QFrame.Shape.HLine)
        separator.setStyleSheet("background-color: #404040;")
        separator.setMaximumHeight(1)
        layout.addWidget(separator)
        
        # Tabs for different processing options
        self.tab_widget = self._createTabs()
        layout.addWidget(self.tab_widget, 1)
        
        # Set panel style
        self.setStyleSheet("""
            QWidget#controlPanel {
                background-color: #252525;
                border-right: 1px solid #3d3d3d;
            }
        """)
        
    def _createHeader(self) -> QWidget:
        """Create the header section"""
        header = QFrame()
        header.setStyleSheet("""
            QFrame {
                background-color: #1a1a1a;
                border-radius: 10px;
                border: 1px solid #333333;
            }
        """)
        
        layout = QVBoxLayout(header)
        layout.setContentsMargins(24, 32, 24, 32)
        layout.setSpacing(16)
        
        # Chinese original
        chinese = QLabel("「那年冬天，以为同淋雪便可共白头」")
        chinese.setAlignment(Qt.AlignmentFlag.AlignCenter)
        chinese.setStyleSheet("""
            QLabel {
                color: #e0e0e0;
                font-size: 16px;
                font-weight: 500;
            }
        """)
        layout.addWidget(chinese)
        
        # Vietnamese translation
        vietnamese = QLabel("\"Nếu cùng nhau đi dưới tuyết,\nliệu chúng ta có cùng đi đến bạc đầu...\"")
        vietnamese.setAlignment(Qt.AlignmentFlag.AlignCenter)
        vietnamese.setWordWrap(True)
        vietnamese.setStyleSheet("""
            QLabel {
                color: #888888;
                font-size: 13px;
                font-style: italic;
            }
        """)
        layout.addWidget(vietnamese)
        
        return header
        
    def _createButtonsSection(self) -> QWidget:
        """Create the Load/Save buttons section"""
        widget = QWidget()
        layout = QVBoxLayout(widget)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(10)
        
        # Load Image button
        self.load_btn = StyledButton("Load Image", "fa5s.folder-open")
        self.load_btn.clicked.connect(self.loadImageClicked.emit)
        layout.addWidget(self.load_btn)
        
        # Save Image button
        self.save_btn = SecondaryButton("Save Image", "fa5s.save")
        self.save_btn.setEnabled(False)  # Disabled until image is processed
        self.save_btn.clicked.connect(self.saveImageClicked.emit)
        layout.addWidget(self.save_btn)
        
        return widget
        
    def _createTabs(self) -> QTabWidget:
        """Create the processing tabs"""
        tabs = QTabWidget()
        tabs.setDocumentMode(True)
        tabs.setStyleSheet("""
            QTabWidget::pane {
                background-color: #2b2b2b;
                border: 1px solid #3d3d3d;
                border-radius: 8px;
                padding: 5px;
            }
            QTabBar::tab {
                background-color: #353535;
                color: #aaaaaa;
                padding: 8px 12px;
                margin-right: 2px;
                border-top-left-radius: 6px;
                border-top-right-radius: 6px;
                min-width: 50px;
            }
            QTabBar::tab:selected {
                background-color: #2b2b2b;
                color: #ffffff;
            }
            QTabBar::tab:hover:!selected {
                background-color: #404040;
            }
        """)
        
        # Basic tab (fully implemented)
        self.basic_tab = BasicTab()
        self.basic_tab.processRequested.connect(self.basicProcessRequested.emit)
        icon_basic = qta.icon('fa5s.sliders-h', color='#888888')
        tabs.addTab(self.basic_tab, icon_basic, "Basic")
        
        # Filters tab (fully implemented)
        self.filters_tab = FiltersTab()
        self.filters_tab.addNoiseRequested.connect(self.addNoiseRequested.emit)
        self.filters_tab.applyFilterRequested.connect(self.applyFilterRequested.emit)
        icon_filters = qta.icon('fa5s.magic', color='#888888')
        tabs.addTab(self.filters_tab, icon_filters, "Filters")
        
        # Morphology tab (fully implemented)
        self.morphology_tab = MorphologyTab()
        self.morphology_tab.morphologyRequested.connect(self.morphologyRequested.emit)
        icon_morph = qta.icon('fa5s.shapes', color='#888888')
        tabs.addTab(self.morphology_tab, icon_morph, "Morph")
        
        # Frequency tab (fully implemented)
        self.frequency_tab = FrequencyTab()
        self.frequency_tab.showSpectrumRequested.connect(self.showSpectrumRequested.emit)
        self.frequency_tab.applyFilterRequested.connect(self.frequencyFilterRequested.emit)
        self.frequency_tab.showFilteredRequested.connect(self.showFilteredSpectrumRequested.emit)
        icon_freq = qta.icon('fa5s.wave-square', color='#888888')
        tabs.addTab(self.frequency_tab, icon_freq, "Freq")
        
        # Segmentation tab (fully implemented)
        self.segmentation_tab = SegmentationTab()
        self.segmentation_tab.otsuRequested.connect(self.otsuRequested.emit)
        self.segmentation_tab.manualThresholdRequested.connect(self.manualThresholdRequested.emit)
        self.segmentation_tab.kmeansRequested.connect(self.kmeansRequested.emit)
        icon_seg = qta.icon('fa5s.object-group', color='#888888')
        tabs.addTab(self.segmentation_tab, icon_seg, "Seg")
        
        # PCA Face Recognition tab (fully implemented)
        self.pca_tab = PCATab()
        self.pca_tab.loadDatasetRequested.connect(self.loadFaceDatasetRequested.emit)
        self.pca_tab.reconstructRequested.connect(self.reconstructFaceRequested.emit)
        icon_pca = qta.icon('fa5s.user', color='#888888')
        tabs.addTab(self.pca_tab, icon_pca, "Face")
        
        # Compression tab (fully implemented)
        self.compression_tab = CompressionTab()
        self.compression_tab.compressFullImage.connect(
            lambda: self.compressFullImageRequested.emit(self.compression_tab.get_quality())
        )
        self.compression_tab.qualityChanged.connect(self.compressionQualityChanged.emit)
        icon_compress = qta.icon('fa5s.compress', color='#888888')
        tabs.addTab(self.compression_tab, icon_compress, "Compress")
        
        # Geometry tab (fully implemented)
        self.geometry_tab = GeometryTab()
        self.geometry_tab.rotateRequested.connect(self.rotateRequested.emit)
        self.geometry_tab.scaleRequested.connect(self.scaleRequested.emit)
        self.geometry_tab.resizeRequested.connect(self.resizeRequested.emit)
        self.geometry_tab.flipRequested.connect(self.flipRequested.emit)
        self.geometry_tab.resetRequested.connect(self.geometryResetRequested.emit)
        icon_geometry = qta.icon('fa5s.expand-arrows-alt', color='#888888')
        tabs.addTab(self.geometry_tab, icon_geometry, "Geometry")
        
        # Restoration tab (fully implemented)
        self.restoration_tab = RestorationTab()
        self.restoration_tab.operationRequested.connect(self.restorationRequested.emit)
        icon_restore = qta.icon('fa5s.magic', color='#888888')
        tabs.addTab(self.restoration_tab, icon_restore, "Restore")
            
        return tabs
        
    def _createTabContent(self, description: str) -> QWidget:
        """Create placeholder content for a tab"""
        widget = QWidget()
        layout = QVBoxLayout(widget)
        layout.setContentsMargins(10, 10, 10, 10)
        
        # Placeholder label
        label = QLabel(description)
        label.setAlignment(Qt.AlignmentFlag.AlignTop | Qt.AlignmentFlag.AlignLeft)
        label.setWordWrap(True)
        label.setStyleSheet("""
            QLabel {
                color: #888888;
                font-size: 12px;
                padding: 10px;
            }
        """)
        layout.addWidget(label)
        
        # Spacer
        layout.addStretch()
        
        # Coming soon label
        coming_soon = QLabel("🚧 Coming Soon")
        coming_soon.setAlignment(Qt.AlignmentFlag.AlignCenter)
        coming_soon.setStyleSheet("""
            QLabel {
                color: #555555;
                font-size: 11px;
                font-style: italic;
            }
        """)
        layout.addWidget(coming_soon)
        
        return widget
        
    def enableSaveButton(self, enabled: bool = True):
        """Enable or disable the save button"""
        self.save_btn.setEnabled(enabled)
        
    def updateHistogram(self, histogram_data, is_color: bool = True):
        """Update histogram in Basic tab"""
        self.basic_tab.updateHistogram(histogram_data, is_color)
        
    def clearHistogram(self):
        """Clear histogram in Basic tab"""
        self.basic_tab.clearHistogram()
        
    def getBasicTab(self) -> BasicTab:
        """Get reference to Basic tab"""
        return self.basic_tab
        
    def getFiltersTab(self) -> FiltersTab:
        """Get reference to Filters tab"""
        return self.filters_tab

    def getMorphologyTab(self) -> MorphologyTab:
        """Get reference to Morphology tab"""
        return self.morphology_tab

    def getFrequencyTab(self) -> FrequencyTab:
        """Get reference to Frequency tab"""
        return self.frequency_tab

    def getSegmentationTab(self) -> SegmentationTab:
        """Get reference to Segmentation tab"""
        return self.segmentation_tab

    def getPCATab(self) -> PCATab:
        """Get reference to PCA tab"""
        return self.pca_tab

    def getCompressionTab(self) -> CompressionTab:
        """Get reference to Compression tab"""
        return self.compression_tab

    def getGeometryTab(self) -> GeometryTab:
        """Get reference to Geometry tab"""
        return self.geometry_tab

    def getRestorationTab(self) -> RestorationTab:
        """Get reference to Restoration tab"""
        return self.restoration_tab
