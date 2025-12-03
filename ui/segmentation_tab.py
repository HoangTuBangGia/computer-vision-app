"""
Segmentation Tab - Image Segmentation Operations
Otsu Thresholding, K-Means Clustering
"""
import numpy as np
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QSpinBox,
    QPushButton, QFrame, QGroupBox, QSlider,
    QScrollArea, QCheckBox
)
from PySide6.QtCore import Qt, Signal
import qtawesome as qta


class StyledSlider(QSlider):
    """Custom styled horizontal slider"""
    
    def __init__(self, orientation=Qt.Orientation.Horizontal, parent=None):
        super().__init__(orientation, parent)
        self.setStyleSheet("""
            QSlider::groove:horizontal {
                border: 1px solid #555555;
                height: 6px;
                background: #353535;
                border-radius: 3px;
            }
            QSlider::handle:horizontal {
                background: #0078d4;
                border: none;
                width: 16px;
                height: 16px;
                margin: -5px 0;
                border-radius: 8px;
            }
            QSlider::handle:horizontal:hover {
                background: #1084d8;
            }
            QSlider::sub-page:horizontal {
                background: #0078d4;
                border-radius: 3px;
            }
        """)


class ActionButton(QPushButton):
    """Styled action button"""
    
    def __init__(self, text: str, icon_name: str = None, color: str = "#0078d4", parent=None):
        super().__init__(text, parent)
        
        if icon_name:
            icon = qta.icon(icon_name, color='white')
            self.setIcon(icon)
            
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.setMinimumHeight(40)
        self.setStyleSheet(f"""
            QPushButton {{
                background-color: {color};
                color: white;
                border: none;
                border-radius: 8px;
                padding: 10px 15px;
                font-size: 12px;
                font-weight: bold;
            }}
            QPushButton:hover {{
                background-color: {self._lighten_color(color)};
            }}
            QPushButton:pressed {{
                background-color: {self._darken_color(color)};
            }}
        """)
        
    def _lighten_color(self, hex_color: str) -> str:
        hex_color = hex_color.lstrip('#')
        r, g, b = int(hex_color[0:2], 16), int(hex_color[2:4], 16), int(hex_color[4:6], 16)
        r = min(255, r + 20)
        g = min(255, g + 20)
        b = min(255, b + 20)
        return f"#{r:02x}{g:02x}{b:02x}"
        
    def _darken_color(self, hex_color: str) -> str:
        hex_color = hex_color.lstrip('#')
        r, g, b = int(hex_color[0:2], 16), int(hex_color[2:4], 16), int(hex_color[4:6], 16)
        r = max(0, r - 30)
        g = max(0, g - 30)
        b = max(0, b - 30)
        return f"#{r:02x}{g:02x}{b:02x}"


class StyledSpinBox(QSpinBox):
    """Custom styled SpinBox"""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setButtonSymbols(QSpinBox.ButtonSymbols.UpDownArrows)
        self.setStyleSheet("""
            QSpinBox {
                background-color: #353535;
                color: #cccccc;
                border: 1px solid #555555;
                border-radius: 4px;
                padding: 4px 8px;
                font-size: 12px;
                min-height: 28px;
                min-width: 70px;
            }
            QSpinBox:hover {
                border-color: #0078d4;
            }
            QSpinBox:focus {
                border-color: #0078d4;
            }
        """)


class SegmentationTab(QWidget):
    """
    Segmentation Tab - Otsu Thresholding and K-Means Clustering
    """
    
    # Signals
    otsuRequested = Signal()
    adaptiveThresholdRequested = Signal(dict)
    manualThresholdRequested = Signal(int)
    kmeansRequested = Signal(dict)
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self._setupUI()
        self._connectSignals()
        
    def _setupUI(self):
        """Setup the tab UI"""
        # Use scroll area
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.Shape.NoFrame)
        scroll.setStyleSheet("""
            QScrollArea {
                background-color: transparent;
                border: none;
            }
            QScrollBar:vertical {
                background-color: #2b2b2b;
                width: 8px;
                border-radius: 4px;
            }
            QScrollBar::handle:vertical {
                background-color: #555555;
                border-radius: 4px;
                min-height: 20px;
            }
        """)
        
        # Content widget
        content = QWidget()
        layout = QVBoxLayout(content)
        layout.setContentsMargins(5, 5, 5, 5)
        layout.setSpacing(10)
        
        # Thresholding Section
        threshold_group = self._createThresholdGroup()
        layout.addWidget(threshold_group)
        
        # Clustering Section
        cluster_group = self._createClusterGroup()
        layout.addWidget(cluster_group)
        
        # Result Info Section
        info_group = self._createInfoGroup()
        layout.addWidget(info_group)
        
        layout.addStretch()
        
        scroll.setWidget(content)
        
        # Main layout
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.addWidget(scroll)
        
    def _createThresholdGroup(self) -> QGroupBox:
        """Create thresholding group"""
        group = QGroupBox("🎯 Thresholding")
        group.setStyleSheet("""
            QGroupBox {
                color: #ffffff;
                font-size: 12px;
                font-weight: bold;
                border: 1px solid #404040;
                border-radius: 6px;
                margin-top: 10px;
                padding-top: 10px;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 5px;
            }
        """)
        
        layout = QVBoxLayout(group)
        layout.setSpacing(10)
        
        # Otsu section
        otsu_frame = QFrame()
        otsu_frame.setStyleSheet("""
            QFrame {
                background-color: #353535;
                border-radius: 6px;
                padding: 5px;
            }
        """)
        otsu_layout = QVBoxLayout(otsu_frame)
        
        otsu_header = QLabel("Otsu's Method (Automatic)")
        otsu_header.setStyleSheet("color: #ffffff; font-size: 11px; font-weight: bold;")
        otsu_layout.addWidget(otsu_header)
        
        otsu_desc = QLabel("Automatically finds optimal threshold")
        otsu_desc.setStyleSheet("color: #888888; font-size: 10px;")
        otsu_layout.addWidget(otsu_desc)
        
        # Otsu threshold result label
        self.otsu_result_label = QLabel("Threshold: --")
        self.otsu_result_label.setStyleSheet("""
            color: #2ecc71;
            font-size: 12px;
            font-weight: bold;
            padding: 5px;
            background-color: #2b2b2b;
            border-radius: 4px;
        """)
        self.otsu_result_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        otsu_layout.addWidget(self.otsu_result_label)
        
        self.otsu_btn = ActionButton("Run Otsu", "fa5s.magic", "#9b59b6")
        otsu_layout.addWidget(self.otsu_btn)
        
        layout.addWidget(otsu_frame)
        
        # Manual threshold section
        manual_frame = QFrame()
        manual_frame.setStyleSheet("""
            QFrame {
                background-color: #353535;
                border-radius: 6px;
                padding: 5px;
            }
        """)
        manual_layout = QVBoxLayout(manual_frame)
        
        manual_header = QLabel("Manual Threshold")
        manual_header.setStyleSheet("color: #ffffff; font-size: 11px; font-weight: bold;")
        manual_layout.addWidget(manual_header)
        
        # Threshold slider
        thresh_header = QHBoxLayout()
        thresh_label = QLabel("Threshold:")
        thresh_label.setStyleSheet("color: #cccccc; font-size: 10px;")
        thresh_header.addWidget(thresh_label)
        
        self.thresh_value_label = QLabel("128")
        self.thresh_value_label.setStyleSheet("color: #0078d4; font-size: 11px; font-weight: bold;")
        self.thresh_value_label.setAlignment(Qt.AlignmentFlag.AlignRight)
        thresh_header.addWidget(self.thresh_value_label)
        manual_layout.addLayout(thresh_header)
        
        self.thresh_slider = StyledSlider()
        self.thresh_slider.setRange(0, 255)
        self.thresh_slider.setValue(128)
        self.thresh_slider.valueChanged.connect(
            lambda v: self.thresh_value_label.setText(str(v))
        )
        manual_layout.addWidget(self.thresh_slider)
        
        # Live preview checkbox
        self.live_thresh_preview = QCheckBox("Live Preview")
        self.live_thresh_preview.setStyleSheet("""
            QCheckBox {
                color: #cccccc;
                font-size: 10px;
                spacing: 6px;
            }
            QCheckBox::indicator {
                width: 16px;
                height: 16px;
                border-radius: 3px;
            }
            QCheckBox::indicator:unchecked {
                border: 2px solid #555555;
                background-color: #2b2b2b;
            }
            QCheckBox::indicator:checked {
                border: 2px solid #0078d4;
                background-color: #0078d4;
                image: url(resources/checkmark.svg);
            }
            QCheckBox::indicator:hover {
                border-color: #0078d4;
            }
        """)
        manual_layout.addWidget(self.live_thresh_preview)
        
        self.manual_thresh_btn = ActionButton("Apply Threshold", "fa5s.adjust", "#3498db")
        manual_layout.addWidget(self.manual_thresh_btn)
        
        layout.addWidget(manual_frame)
        
        return group
        
    def _createClusterGroup(self) -> QGroupBox:
        """Create clustering group"""
        group = QGroupBox("🎨 K-Means Clustering")
        group.setStyleSheet("""
            QGroupBox {
                color: #ffffff;
                font-size: 12px;
                font-weight: bold;
                border: 1px solid #404040;
                border-radius: 6px;
                margin-top: 10px;
                padding-top: 10px;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 5px;
            }
        """)
        
        layout = QVBoxLayout(group)
        layout.setSpacing(10)
        
        # Info label
        info = QLabel(
            "Segments image by grouping similar colors.\n"
            "Each cluster gets a representative color."
        )
        info.setWordWrap(True)
        info.setStyleSheet("""
            QLabel {
                color: #888888;
                font-size: 10px;
                padding: 5px;
            }
        """)
        layout.addWidget(info)
        
        # Number of clusters (K)
        k_layout = QHBoxLayout()
        k_label = QLabel("Clusters (K):")
        k_label.setStyleSheet("color: #cccccc; font-size: 11px;")
        k_label.setFixedWidth(80)
        k_layout.addWidget(k_label)
        
        self.k_spin = StyledSpinBox()
        self.k_spin.setRange(2, 10)
        self.k_spin.setValue(4)
        k_layout.addWidget(self.k_spin, 1)
        layout.addLayout(k_layout)
        
        # Use vibrant colors checkbox
        self.vibrant_colors = QCheckBox("Use Vibrant Colors")
        self.vibrant_colors.setChecked(True)
        self.vibrant_colors.setStyleSheet("""
            QCheckBox {
                color: #cccccc;
                font-size: 11px;
                padding: 5px;
                spacing: 8px;
            }
            QCheckBox::indicator {
                width: 18px;
                height: 18px;
                border-radius: 4px;
            }
            QCheckBox::indicator:unchecked {
                border: 2px solid #555555;
                background-color: #2b2b2b;
            }
            QCheckBox::indicator:checked {
                border: 2px solid #0078d4;
                background-color: #0078d4;
                image: url(resources/checkmark.svg);
            }
            QCheckBox::indicator:hover {
                border-color: #0078d4;
            }
        """)
        layout.addWidget(self.vibrant_colors)
        
        # K-Means button
        self.kmeans_btn = ActionButton("Run K-Means", "fa5s.object-group", "#e74c3c")
        layout.addWidget(self.kmeans_btn)
        
        # Result label
        self.kmeans_result_label = QLabel("Clusters: --  |  Compactness: --")
        self.kmeans_result_label.setStyleSheet("""
            QLabel {
                color: #888888;
                font-size: 10px;
                padding: 8px;
                background-color: #353535;
                border-radius: 4px;
            }
        """)
        self.kmeans_result_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(self.kmeans_result_label)
        
        return group
        
    def _createInfoGroup(self) -> QGroupBox:
        """Create info group"""
        group = QGroupBox("ℹ️ Information")
        group.setStyleSheet("""
            QGroupBox {
                color: #ffffff;
                font-size: 12px;
                font-weight: bold;
                border: 1px solid #404040;
                border-radius: 6px;
                margin-top: 10px;
                padding-top: 10px;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 5px;
            }
        """)
        
        layout = QVBoxLayout(group)
        
        self.info_label = QLabel(
            "Segmentation Methods:\n\n"
            "• Otsu: Best for bimodal histograms\n"
            "  (clear foreground/background)\n\n"
            "• K-Means: Groups pixels by color\n"
            "  similarity. Good for color-based\n"
            "  region separation."
        )
        self.info_label.setWordWrap(True)
        self.info_label.setStyleSheet("""
            QLabel {
                color: #888888;
                font-size: 10px;
                padding: 8px;
                background-color: #353535;
                border-radius: 4px;
            }
        """)
        layout.addWidget(self.info_label)
        
        return group
        
    def _connectSignals(self):
        """Connect internal signals"""
        self.otsu_btn.clicked.connect(self.otsuRequested.emit)
        self.manual_thresh_btn.clicked.connect(self._onManualThreshold)
        self.kmeans_btn.clicked.connect(self._onKmeans)
        self.thresh_slider.valueChanged.connect(self._onThresholdChanged)
        
    def _onManualThreshold(self):
        """Handle manual threshold button"""
        self.manualThresholdRequested.emit(self.thresh_slider.value())
        
    def _onThresholdChanged(self, value: int):
        """Handle threshold slider change"""
        if self.live_thresh_preview.isChecked():
            self.manualThresholdRequested.emit(value)
            
    def _onKmeans(self):
        """Handle K-Means button"""
        params = {
            "k": self.k_spin.value(),
            "use_vibrant": self.vibrant_colors.isChecked(),
        }
        self.kmeansRequested.emit(params)
        
    def updateOtsuResult(self, threshold: float):
        """Update Otsu threshold result"""
        self.otsu_result_label.setText(f"Threshold: {threshold:.0f}")
        
    def updateKmeansResult(self, k: int, compactness: float):
        """Update K-Means result"""
        self.kmeans_result_label.setText(f"Clusters: {k}  |  Compactness: {compactness:.0f}")
        
    def getCurrentParameters(self) -> dict:
        """Get current parameters"""
        return {
            "manual_threshold": self.thresh_slider.value(),
            "k": self.k_spin.value(),
            "use_vibrant": self.vibrant_colors.isChecked(),
        }
