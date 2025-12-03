"""
Frequency Tab - Frequency Domain Operations
FFT, Lowpass/Highpass Filtering
"""
import numpy as np
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QSpinBox,
    QComboBox, QPushButton, QFrame, QGroupBox, QGridLayout,
    QScrollArea, QSlider, QCheckBox
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


class StyledComboBox(QComboBox):
    """Custom styled ComboBox"""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setStyleSheet("""
            QComboBox {
                background-color: #353535;
                color: #cccccc;
                border: 1px solid #555555;
                border-radius: 6px;
                padding: 8px 12px;
                font-size: 12px;
                min-height: 20px;
            }
            QComboBox:hover {
                border-color: #0078d4;
            }
            QComboBox::drop-down {
                border: none;
                width: 30px;
            }
            QComboBox::down-arrow {
                image: none;
                border-left: 5px solid transparent;
                border-right: 5px solid transparent;
                border-top: 6px solid #888888;
                margin-right: 10px;
            }
            QComboBox QAbstractItemView {
                background-color: #2b2b2b;
                color: #cccccc;
                selection-background-color: #0078d4;
                selection-color: white;
                border: 1px solid #555555;
                border-radius: 4px;
                padding: 4px;
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
        """Lighten a hex color"""
        hex_color = hex_color.lstrip('#')
        r, g, b = int(hex_color[0:2], 16), int(hex_color[2:4], 16), int(hex_color[4:6], 16)
        r = min(255, r + 20)
        g = min(255, g + 20)
        b = min(255, b + 20)
        return f"#{r:02x}{g:02x}{b:02x}"
        
    def _darken_color(self, hex_color: str) -> str:
        """Darken a hex color"""
        hex_color = hex_color.lstrip('#')
        r, g, b = int(hex_color[0:2], 16), int(hex_color[2:4], 16), int(hex_color[4:6], 16)
        r = max(0, r - 30)
        g = max(0, g - 30)
        b = max(0, b - 30)
        return f"#{r:02x}{g:02x}{b:02x}"


class FrequencyTab(QWidget):
    """
    Frequency Tab - FFT and Frequency Filtering
    """
    
    # Signals
    showSpectrumRequested = Signal()  # Request to show magnitude spectrum
    applyFilterRequested = Signal(str, dict)  # filter_type, params
    showFilteredRequested = Signal()  # Show filtered result
    
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
        
        # FFT Section
        fft_group = self._createFFTGroup()
        layout.addWidget(fft_group)
        
        # Filter Section
        filter_group = self._createFilterGroup()
        layout.addWidget(filter_group)
        
        # Action Section
        action_group = self._createActionGroup()
        layout.addWidget(action_group)
        
        # Info Section
        info_group = self._createInfoGroup()
        layout.addWidget(info_group)
        
        layout.addStretch()
        
        scroll.setWidget(content)
        
        # Main layout
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.addWidget(scroll)
        
    def _createFFTGroup(self) -> QGroupBox:
        """Create FFT visualization group"""
        group = QGroupBox("📊 Fourier Transform")
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
            "The Fourier Transform converts an image from\n"
            "spatial domain to frequency domain.\n\n"
            "• Low frequencies → smooth regions\n"
            "• High frequencies → edges, details"
        )
        info.setWordWrap(True)
        info.setStyleSheet("""
            QLabel {
                color: #888888;
                font-size: 10px;
                padding: 8px;
                background-color: #353535;
                border-radius: 4px;
            }
        """)
        layout.addWidget(info)
        
        # Show Spectrum button
        self.show_spectrum_btn = ActionButton(
            "Show Magnitude Spectrum", 
            "fa5s.chart-area",
            "#9b59b6"
        )
        layout.addWidget(self.show_spectrum_btn)
        
        return group
        
    def _createFilterGroup(self) -> QGroupBox:
        """Create filter settings group"""
        group = QGroupBox("🎛️ Frequency Filter")
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
        
        # Filter type selection
        type_layout = QHBoxLayout()
        type_label = QLabel("Type:")
        type_label.setStyleSheet("color: #cccccc; font-size: 11px;")
        type_label.setFixedWidth(60)
        type_layout.addWidget(type_label)
        
        self.filter_type_combo = StyledComboBox()
        self.filter_type_combo.addItems([
            "Ideal Lowpass",
            "Ideal Highpass",
            "Gaussian Lowpass",
            "Gaussian Highpass",
            "Butterworth Lowpass",
            "Butterworth Highpass",
        ])
        type_layout.addWidget(self.filter_type_combo, 1)
        layout.addLayout(type_layout)
        
        # Cutoff frequency (D0) slider
        d0_layout = QVBoxLayout()
        
        d0_header = QHBoxLayout()
        d0_label = QLabel("Cutoff Radius (D₀):")
        d0_label.setStyleSheet("color: #cccccc; font-size: 11px;")
        d0_header.addWidget(d0_label)
        
        self.d0_value_label = QLabel("30")
        self.d0_value_label.setStyleSheet("""
            color: #0078d4;
            font-size: 12px;
            font-weight: bold;
        """)
        self.d0_value_label.setAlignment(Qt.AlignmentFlag.AlignRight)
        d0_header.addWidget(self.d0_value_label)
        d0_layout.addLayout(d0_header)
        
        self.d0_slider = StyledSlider()
        self.d0_slider.setRange(1, 200)
        self.d0_slider.setValue(30)
        self.d0_slider.valueChanged.connect(
            lambda v: self.d0_value_label.setText(str(v))
        )
        d0_layout.addWidget(self.d0_slider)
        layout.addLayout(d0_layout)
        
        # Butterworth order (only for Butterworth filters)
        order_layout = QHBoxLayout()
        order_label = QLabel("Order (n):")
        order_label.setStyleSheet("color: #cccccc; font-size: 11px;")
        order_label.setFixedWidth(60)
        order_layout.addWidget(order_label)
        
        self.order_spin = QSpinBox()
        self.order_spin.setRange(1, 10)
        self.order_spin.setValue(2)
        self.order_spin.setStyleSheet("""
            QSpinBox {
                background-color: #353535;
                color: #cccccc;
                border: 1px solid #555555;
                border-radius: 6px;
                padding: 6px 10px;
                font-size: 11px;
            }
            QSpinBox:hover {
                border-color: #0078d4;
            }
        """)
        order_layout.addWidget(self.order_spin, 1)
        layout.addLayout(order_layout)
        
        # Live preview checkbox
        self.live_preview = QCheckBox("Live Preview Filter Mask")
        self.live_preview.setChecked(False)
        self.live_preview.setStyleSheet("""
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
        layout.addWidget(self.live_preview)
        
        return group
        
    def _createActionGroup(self) -> QGroupBox:
        """Create action buttons group"""
        group = QGroupBox("⚡ Actions")
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
        layout.setSpacing(8)
        
        # Apply Filter & Show Result button
        self.apply_filter_btn = ActionButton(
            "Apply Filter & IFFT",
            "fa5s.filter",
            "#2ecc71"
        )
        layout.addWidget(self.apply_filter_btn)
        
        # Show Filtered Spectrum button
        self.show_filtered_btn = ActionButton(
            "Show Filtered Spectrum",
            "fa5s.eye",
            "#3498db"
        )
        layout.addWidget(self.show_filtered_btn)
        
        return group
        
    def _createInfoGroup(self) -> QGroupBox:
        """Create info group"""
        group = QGroupBox("ℹ️ Filter Info")
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
            "Filter Types:\n\n"
            "• Ideal: Sharp cutoff (may cause ringing)\n"
            "• Gaussian: Smooth transition, no ringing\n"
            "• Butterworth: Adjustable sharpness\n\n"
            "Lowpass: Blur/smooth (remove high freq)\n"
            "Highpass: Edge detection (remove low freq)"
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
        self.show_spectrum_btn.clicked.connect(self.showSpectrumRequested.emit)
        self.apply_filter_btn.clicked.connect(self._onApplyFilter)
        self.show_filtered_btn.clicked.connect(self.showFilteredRequested.emit)
        self.filter_type_combo.currentIndexChanged.connect(self._updateFilterInfo)
        self.d0_slider.valueChanged.connect(self._onParameterChanged)
        self.order_spin.valueChanged.connect(self._onParameterChanged)
        self.live_preview.toggled.connect(self._onParameterChanged)
        
    def _onApplyFilter(self):
        """Handle apply filter button"""
        params = self.getCurrentParameters()
        filter_type = self._getFilterTypeKey()
        self.applyFilterRequested.emit(filter_type, params)
        
    def _onParameterChanged(self):
        """Handle parameter change for live preview"""
        if self.live_preview.isChecked():
            self._onApplyFilter()
            
    def _getFilterTypeKey(self) -> str:
        """Get filter type key from combo box"""
        filter_map = {
            0: "ideal_lowpass",
            1: "ideal_highpass",
            2: "gaussian_lowpass",
            3: "gaussian_highpass",
            4: "butterworth_lowpass",
            5: "butterworth_highpass",
        }
        return filter_map.get(self.filter_type_combo.currentIndex(), "ideal_lowpass")
        
    def _updateFilterInfo(self, index: int):
        """Update info based on selected filter"""
        info_texts = {
            0: "Ideal Lowpass:\nPasses frequencies < D₀\nBlocks frequencies > D₀\nMay cause ringing artifacts",
            1: "Ideal Highpass:\nBlocks frequencies < D₀\nPasses frequencies > D₀\nEnhances edges, may ring",
            2: "Gaussian Lowpass:\nSmooth frequency transition\nNo ringing artifacts\nGood for general smoothing",
            3: "Gaussian Highpass:\nSmooth high-pass filtering\nNo ringing artifacts\nGood edge enhancement",
            4: "Butterworth Lowpass:\nAdjustable transition sharpness\nOrder n controls rolloff\nGood compromise",
            5: "Butterworth Highpass:\nAdjustable transition\nHigher n = sharper cutoff\nFlexible edge detection",
        }
        self.info_label.setText(info_texts.get(index, ""))
        
    def getCurrentParameters(self) -> dict:
        """Get current filter parameters"""
        return {
            "d0": self.d0_slider.value(),
            "order": self.order_spin.value(),
        }
        
    def isLivePreviewEnabled(self) -> bool:
        """Check if live preview is enabled"""
        return self.live_preview.isChecked()
