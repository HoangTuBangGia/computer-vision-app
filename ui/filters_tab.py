"""
Filters Tab - Spatial filtering operations with noise addition
"""
import numpy as np
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QSlider,
    QComboBox, QCheckBox, QPushButton, QFrame, QGroupBox,
    QScrollArea, QSizePolicy, QButtonGroup, QRadioButton
)
from PySide6.QtCore import Qt, Signal
import qtawesome as qta


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


class StyledSlider(QWidget):
    """Custom styled slider with label and value display"""
    
    valueChanged = Signal(int)
    
    def __init__(self, label: str, min_val: int, max_val: int, 
                 default: int, step: int = 1, odd_only: bool = False, parent=None):
        super().__init__(parent)
        
        self.min_val = min_val
        self.max_val = max_val
        self.step = step
        self.odd_only = odd_only
        
        self._setupUI(label, min_val, max_val, default)
        
    def _setupUI(self, label: str, min_val: int, max_val: int, default: int):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 5, 0, 5)
        layout.setSpacing(3)
        
        # Header row with label and value
        header = QHBoxLayout()
        
        self.label = QLabel(label)
        self.label.setStyleSheet("color: #cccccc; font-size: 11px;")
        header.addWidget(self.label)
        
        header.addStretch()
        
        self.value_label = QLabel(str(default))
        self.value_label.setStyleSheet("""
            color: #0078d4; 
            font-size: 11px; 
            font-weight: bold;
            background-color: #353535;
            padding: 2px 8px;
            border-radius: 3px;
        """)
        header.addWidget(self.value_label)
        
        layout.addLayout(header)
        
        # Slider
        self.slider = QSlider(Qt.Orientation.Horizontal)
        self.slider.setMinimum(min_val)
        self.slider.setMaximum(max_val)
        self.slider.setValue(default)
        self.slider.setSingleStep(self.step)
        self.slider.setStyleSheet("""
            QSlider::groove:horizontal {
                border: 1px solid #404040;
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
        self.slider.valueChanged.connect(self._onValueChanged)
        layout.addWidget(self.slider)
        
    def _onValueChanged(self, value: int):
        """Handle slider value change"""
        if self.odd_only and value % 2 == 0:
            value = value + 1 if value < self.max_val else value - 1
            self.slider.blockSignals(True)
            self.slider.setValue(value)
            self.slider.blockSignals(False)
        self.value_label.setText(str(value))
        self.valueChanged.emit(value)
        
    def value(self) -> int:
        """Get current value"""
        val = self.slider.value()
        if self.odd_only and val % 2 == 0:
            val = val + 1 if val < self.max_val else val - 1
        return val
        
    def setValue(self, value: int):
        """Set slider value"""
        self.slider.setValue(value)
        
    def setEnabled(self, enabled: bool):
        """Enable/disable slider"""
        self.slider.setEnabled(enabled)
        opacity = "1.0" if enabled else "0.5"
        self.label.setStyleSheet(f"color: #cccccc; font-size: 11px; opacity: {opacity};")


class NoiseButton(QPushButton):
    """Styled button for adding noise"""
    
    def __init__(self, text: str, icon_name: str = None, parent=None):
        super().__init__(text, parent)
        
        if icon_name:
            icon = qta.icon(icon_name, color='#ff8c00')
            self.setIcon(icon)
            
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.setStyleSheet("""
            QPushButton {
                background-color: #4a3500;
                color: #ffb347;
                border: 1px solid #ff8c00;
                border-radius: 6px;
                padding: 8px 15px;
                font-size: 11px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #5a4000;
                border-color: #ffa500;
            }
            QPushButton:pressed {
                background-color: #3a2800;
            }
        """)


class ApplyButton(QPushButton):
    """Styled apply button"""
    
    def __init__(self, text: str, icon_name: str = None, parent=None):
        super().__init__(text, parent)
        
        if icon_name:
            icon = qta.icon(icon_name, color='white')
            self.setIcon(icon)
            
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.setStyleSheet("""
            QPushButton {
                background-color: #0078d4;
                color: white;
                border: none;
                border-radius: 6px;
                padding: 8px 15px;
                font-size: 11px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #1084d8;
            }
            QPushButton:pressed {
                background-color: #006cbd;
            }
        """)


class FiltersTab(QWidget):
    """
    Filters Tab - Spatial filtering operations
    Contains noise addition and various filter options
    """
    
    # Signals
    addNoiseRequested = Signal(str, dict)  # noise_type, params
    applyFilterRequested = Signal(str, dict)  # filter_type, params
    livePreviewChanged = Signal(bool)  # live preview state
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self._setupUI()
        self._connectSignals()
        
    def _setupUI(self):
        """Setup the tab UI"""
        # Use scroll area for content
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
            QScrollBar::handle:vertical:hover {
                background-color: #666666;
            }
        """)
        
        # Content widget
        content = QWidget()
        layout = QVBoxLayout(content)
        layout.setContentsMargins(5, 5, 5, 5)
        layout.setSpacing(10)
        
        # Noise section
        noise_group = self._createNoiseGroup()
        layout.addWidget(noise_group)
        
        # Filter section
        filter_group = self._createFilterGroup()
        layout.addWidget(filter_group)
        
        # Parameters section
        params_group = self._createParametersGroup()
        layout.addWidget(params_group)
        
        # Spacer
        layout.addStretch()
        
        scroll.setWidget(content)
        
        # Main layout
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.addWidget(scroll)
        
    def _createNoiseGroup(self) -> QGroupBox:
        """Create noise addition group"""
        group = QGroupBox("🐛 Add Noise (for testing)")
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
        
        # Noise type selection
        type_layout = QHBoxLayout()
        type_label = QLabel("Type:")
        type_label.setStyleSheet("color: #cccccc; font-size: 11px;")
        type_layout.addWidget(type_label)
        
        self.noise_combo = StyledComboBox()
        self.noise_combo.addItems(["Gaussian", "Salt & Pepper"])
        type_layout.addWidget(self.noise_combo, 1)
        layout.addLayout(type_layout)
        
        # Noise intensity slider
        self.noise_intensity = StyledSlider("Intensity", 1, 100, 25)
        layout.addWidget(self.noise_intensity)
        
        # Add noise button
        self.add_noise_btn = NoiseButton("Add Noise", "fa5s.bug")
        self.add_noise_btn.clicked.connect(self._onAddNoiseClicked)
        layout.addWidget(self.add_noise_btn)
        
        return group
        
    def _createFilterGroup(self) -> QGroupBox:
        """Create filter selection group"""
        group = QGroupBox("🔧 Filter Type")
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
        
        # Filter type combo
        self.filter_combo = StyledComboBox()
        self.filter_combo.addItems([
            "None (Original)",
            "Mean (Average)",
            "Gaussian Blur",
            "Median",
            "Laplacian (Sharpen)",
            "Unsharp Mask",
            "Bilateral"
        ])
        self.filter_combo.currentIndexChanged.connect(self._onFilterChanged)
        layout.addWidget(self.filter_combo)
        
        # Filter info label
        self.filter_info = QLabel("Select a filter to see description")
        self.filter_info.setWordWrap(True)
        self.filter_info.setStyleSheet("""
            QLabel {
                color: #888888;
                font-size: 10px;
                font-style: italic;
                padding: 5px;
                background-color: #353535;
                border-radius: 4px;
            }
        """)
        layout.addWidget(self.filter_info)
        
        return group
        
    def _createParametersGroup(self) -> QGroupBox:
        """Create parameters group"""
        group = QGroupBox("⚙️ Parameters")
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
        
        # Kernel size slider (odd numbers only)
        self.kernel_slider = StyledSlider("Kernel Size", 1, 31, 3, step=2, odd_only=True)
        layout.addWidget(self.kernel_slider)
        
        # Strength slider (for Laplacian/Unsharp)
        self.strength_slider = StyledSlider("Strength", 1, 30, 10)
        self.strength_slider.setEnabled(False)
        layout.addWidget(self.strength_slider)
        
        # Live preview checkbox
        self.live_preview = QCheckBox("Live Preview")
        self.live_preview.setChecked(True)
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
        self.live_preview.stateChanged.connect(self._onLivePreviewChanged)
        layout.addWidget(self.live_preview)
        
        # Apply button
        btn_layout = QHBoxLayout()
        self.apply_btn = ApplyButton("Apply Filter", "fa5s.check")
        self.apply_btn.clicked.connect(self._onApplyClicked)
        btn_layout.addWidget(self.apply_btn)
        layout.addLayout(btn_layout)
        
        return group
        
    def _connectSignals(self):
        """Connect internal signals"""
        self.kernel_slider.valueChanged.connect(self._onParameterChanged)
        self.strength_slider.valueChanged.connect(self._onParameterChanged)
        
    def _onAddNoiseClicked(self):
        """Handle add noise button click"""
        noise_types = ["gaussian", "salt_pepper"]
        noise_type = noise_types[self.noise_combo.currentIndex()]
        intensity = self.noise_intensity.value()
        
        params = {}
        if noise_type == "gaussian":
            params["sigma"] = intensity
        else:
            prob = intensity / 500.0  # Convert to probability (max ~20%)
            params["salt_prob"] = prob
            params["pepper_prob"] = prob
            
        self.addNoiseRequested.emit(noise_type, params)
        
    def _onFilterChanged(self, index: int):
        """Handle filter selection change"""
        # Update filter info
        filter_infos = [
            "No filter applied - shows original/noisy image",
            "Mean filter: Replaces each pixel with neighborhood average. Good for Gaussian noise.",
            "Gaussian blur: Weighted average using Gaussian kernel. Better edge preservation than mean.",
            "Median filter: Replaces with median value. Best for salt & pepper noise!",
            "Laplacian sharpen: Enhances edges using Laplacian operator.",
            "Unsharp mask: Advanced sharpening by subtracting blurred version.",
            "Bilateral filter: Smooths while preserving edges. Slower but better quality.",
        ]
        self.filter_info.setText(filter_infos[index])
        
        # Enable/disable strength slider
        needs_strength = index in [4, 5]  # Laplacian, Unsharp
        self.strength_slider.setEnabled(needs_strength)
        
        # Enable/disable kernel slider
        needs_kernel = index in [1, 2, 3, 5, 6]  # All except None and Laplacian
        self.kernel_slider.setEnabled(needs_kernel)
        
        # Trigger preview if live preview is enabled
        if self.live_preview.isChecked():
            self._emitFilterRequest()
            
    def _onParameterChanged(self, value):
        """Handle parameter change"""
        if self.live_preview.isChecked():
            self._emitFilterRequest()
            
    def _onLivePreviewChanged(self, state):
        """Handle live preview checkbox change"""
        self.livePreviewChanged.emit(state == Qt.CheckState.Checked.value)
        if state == Qt.CheckState.Checked.value:
            self._emitFilterRequest()
            
    def _onApplyClicked(self):
        """Handle apply button click"""
        self._emitFilterRequest()
        
    def _emitFilterRequest(self):
        """Emit filter request signal with current settings"""
        filter_types = ["none", "mean", "gaussian", "median", "laplacian", "unsharp", "bilateral"]
        filter_type = filter_types[self.filter_combo.currentIndex()]
        
        params = {
            "kernel_size": self.kernel_slider.value(),
            "strength": self.strength_slider.value() / 10.0,  # Convert to 0.1-3.0 range
        }
        
        self.applyFilterRequested.emit(filter_type, params)
        
    def getCurrentFilter(self) -> tuple:
        """Get current filter and parameters"""
        filter_types = ["none", "mean", "gaussian", "median", "laplacian", "unsharp", "bilateral"]
        filter_type = filter_types[self.filter_combo.currentIndex()]
        
        params = {
            "kernel_size": self.kernel_slider.value(),
            "strength": self.strength_slider.value() / 10.0,
        }
        
        return filter_type, params
        
    def isLivePreviewEnabled(self) -> bool:
        """Check if live preview is enabled"""
        return self.live_preview.isChecked()
