"""
Restoration Tab - Image restoration operations
Includes: Noise models, Mean filters, Order-statistics filters,
Adaptive filters, Motion blur, Inverse filtering
"""
import numpy as np
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QSlider,
    QComboBox, QCheckBox, QPushButton, QFrame, QGroupBox,
    QScrollArea, QSizePolicy, QButtonGroup, QRadioButton,
    QDoubleSpinBox, QSpinBox
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
                padding: 10px 20px;
                font-size: 12px;
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


class RestorationTab(QWidget):
    """
    Restoration Tab Widget
    Contains controls for image restoration operations
    """
    
    # Signal emitted when restoration operation is requested
    operationRequested = Signal(str, dict)  # operation, params
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self._setupUI()
        
    def _setupUI(self):
        """Setup the restoration tab UI"""
        # Main scroll area
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        scroll.setStyleSheet("""
            QScrollArea {
                border: none;
                background-color: transparent;
            }
            QScrollBar:vertical {
                background-color: #2b2b2b;
                width: 10px;
                border-radius: 5px;
            }
            QScrollBar::handle:vertical {
                background-color: #555555;
                border-radius: 5px;
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
        layout.setSpacing(15)
        
        # Add Noise Section
        noise_group = self._createNoiseSection()
        layout.addWidget(noise_group)
        
        # Mean Filters Section
        mean_group = self._createMeanFiltersSection()
        layout.addWidget(mean_group)
        
        # Order-Statistics Filters Section
        order_group = self._createOrderStatsSection()
        layout.addWidget(order_group)
        
        # Adaptive Filters Section
        adaptive_group = self._createAdaptiveFiltersSection()
        layout.addWidget(adaptive_group)
        
        # Degradation & Restoration Section
        degrade_group = self._createDegradationSection()
        layout.addWidget(degrade_group)
        
        # Spacer
        layout.addStretch()
        
        scroll.setWidget(content)
        
        # Set main layout
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.addWidget(scroll)
        
    def _createGroupBox(self, title: str) -> QGroupBox:
        """Create a styled group box"""
        group = QGroupBox(title)
        group.setStyleSheet("""
            QGroupBox {
                font-weight: bold;
                font-size: 12px;
                color: #ffffff;
                border: 1px solid #404040;
                border-radius: 8px;
                margin-top: 12px;
                padding-top: 10px;
                background-color: #2d2d2d;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 15px;
                padding: 0 8px;
                background-color: #2d2d2d;
            }
        """)
        return group
        
    def _createNoiseSection(self) -> QGroupBox:
        """Create noise addition section"""
        group = self._createGroupBox("Add Noise")
        layout = QVBoxLayout(group)
        layout.setSpacing(10)
        
        # Noise type selection
        type_layout = QHBoxLayout()
        type_label = QLabel("Type:")
        type_label.setStyleSheet("color: #cccccc; font-size: 11px;")
        type_layout.addWidget(type_label)
        
        self.noise_combo = StyledComboBox()
        self.noise_combo.addItems([
            "Uniform",
            "Rayleigh", 
            "Exponential"
        ])
        type_layout.addWidget(self.noise_combo, 1)
        layout.addLayout(type_layout)
        
        # Noise parameters
        self.noise_param1 = StyledSlider("Param A/Scale", -100, 100, -50)
        layout.addWidget(self.noise_param1)
        
        self.noise_param2 = StyledSlider("Param B", 1, 100, 50)
        layout.addWidget(self.noise_param2)
        
        # Add noise button
        add_noise_btn = NoiseButton("Add Noise", "fa5s.random")
        add_noise_btn.clicked.connect(self._onAddNoise)
        layout.addWidget(add_noise_btn)
        
        return group
        
    def _createMeanFiltersSection(self) -> QGroupBox:
        """Create mean filters section"""
        group = self._createGroupBox("Mean Filters")
        layout = QVBoxLayout(group)
        layout.setSpacing(10)
        
        # Filter type selection
        type_layout = QHBoxLayout()
        type_label = QLabel("Filter:")
        type_label.setStyleSheet("color: #cccccc; font-size: 11px;")
        type_layout.addWidget(type_label)
        
        self.mean_filter_combo = StyledComboBox()
        self.mean_filter_combo.addItems([
            "Arithmetic Mean",
            "Geometric Mean",
            "Harmonic Mean",
            "Contraharmonic Mean"
        ])
        self.mean_filter_combo.currentIndexChanged.connect(self._onMeanFilterChanged)
        type_layout.addWidget(self.mean_filter_combo, 1)
        layout.addLayout(type_layout)
        
        # Kernel size
        self.mean_kernel = StyledSlider("Kernel Size", 3, 15, 3, odd_only=True)
        layout.addWidget(self.mean_kernel)
        
        # Q parameter (for contraharmonic)
        self.q_param = StyledSlider("Q (Order)", -50, 50, 15)
        self.q_param.setEnabled(False)
        layout.addWidget(self.q_param)
        
        # Apply button
        apply_btn = ApplyButton("Apply Mean Filter", "fa5s.check")
        apply_btn.clicked.connect(self._onApplyMeanFilter)
        layout.addWidget(apply_btn)
        
        return group
        
    def _createOrderStatsSection(self) -> QGroupBox:
        """Create order-statistics filters section"""
        group = self._createGroupBox("Order-Statistics Filters")
        layout = QVBoxLayout(group)
        layout.setSpacing(10)
        
        # Filter type selection
        type_layout = QHBoxLayout()
        type_label = QLabel("Filter:")
        type_label.setStyleSheet("color: #cccccc; font-size: 11px;")
        type_layout.addWidget(type_label)
        
        self.order_filter_combo = StyledComboBox()
        self.order_filter_combo.addItems([
            "Median",
            "Max (Pepper noise)",
            "Min (Salt noise)",
            "Midpoint",
            "Alpha-trimmed Mean"
        ])
        self.order_filter_combo.currentIndexChanged.connect(self._onOrderFilterChanged)
        type_layout.addWidget(self.order_filter_combo, 1)
        layout.addLayout(type_layout)
        
        # Kernel size
        self.order_kernel = StyledSlider("Kernel Size", 3, 15, 3, odd_only=True)
        layout.addWidget(self.order_kernel)
        
        # D parameter (for alpha-trimmed)
        self.d_param = StyledSlider("d (Trim count)", 2, 20, 2)
        self.d_param.setEnabled(False)
        layout.addWidget(self.d_param)
        
        # Apply button
        apply_btn = ApplyButton("Apply Order Filter", "fa5s.check")
        apply_btn.clicked.connect(self._onApplyOrderFilter)
        layout.addWidget(apply_btn)
        
        return group
        
    def _createAdaptiveFiltersSection(self) -> QGroupBox:
        """Create adaptive filters section"""
        group = self._createGroupBox("Adaptive Filters")
        layout = QVBoxLayout(group)
        layout.setSpacing(10)
        
        # Filter type selection
        type_layout = QHBoxLayout()
        type_label = QLabel("Filter:")
        type_label.setStyleSheet("color: #cccccc; font-size: 11px;")
        type_layout.addWidget(type_label)
        
        self.adaptive_filter_combo = StyledComboBox()
        self.adaptive_filter_combo.addItems([
            "Adaptive Local Noise Reduction",
            "Adaptive Median"
        ])
        type_layout.addWidget(self.adaptive_filter_combo, 1)
        layout.addLayout(type_layout)
        
        # Kernel size
        self.adaptive_kernel = StyledSlider("Kernel Size", 3, 15, 7, odd_only=True)
        layout.addWidget(self.adaptive_kernel)
        
        # Apply button
        apply_btn = ApplyButton("Apply Adaptive Filter", "fa5s.check")
        apply_btn.clicked.connect(self._onApplyAdaptiveFilter)
        layout.addWidget(apply_btn)
        
        return group
        
    def _createDegradationSection(self) -> QGroupBox:
        """Create degradation and restoration section"""
        group = self._createGroupBox("Degradation & Restoration")
        layout = QVBoxLayout(group)
        layout.setSpacing(10)
        
        # Degradation type
        degrade_layout = QHBoxLayout()
        degrade_label = QLabel("Degradation:")
        degrade_label.setStyleSheet("color: #cccccc; font-size: 11px;")
        degrade_layout.addWidget(degrade_label)
        
        self.degrade_combo = StyledComboBox()
        self.degrade_combo.addItems([
            "Motion Blur",
            "Atmospheric Turbulence"
        ])
        self.degrade_combo.currentIndexChanged.connect(self._onDegradeTypeChanged)
        degrade_layout.addWidget(self.degrade_combo, 1)
        layout.addLayout(degrade_layout)
        
        # Motion blur parameters
        self.motion_length = StyledSlider("Motion Length", 5, 50, 15)
        layout.addWidget(self.motion_length)
        
        self.motion_angle = StyledSlider("Motion Angle (°)", 0, 180, 0)
        layout.addWidget(self.motion_angle)
        
        # Atmospheric turbulence parameter
        self.turbulence_k = StyledSlider("Turbulence K (x0.0001)", 1, 100, 10)
        self.turbulence_k.setEnabled(False)
        layout.addWidget(self.turbulence_k)
        
        # Apply degradation button
        degrade_btn = NoiseButton("Apply Degradation", "fa5s.cloud")
        degrade_btn.clicked.connect(self._onApplyDegradation)
        layout.addWidget(degrade_btn)
        
        # Separator
        sep = QFrame()
        sep.setFrameShape(QFrame.Shape.HLine)
        sep.setStyleSheet("background-color: #404040;")
        layout.addWidget(sep)
        
        # Restoration method
        restore_layout = QHBoxLayout()
        restore_label = QLabel("Restore Method:")
        restore_label.setStyleSheet("color: #cccccc; font-size: 11px;")
        restore_layout.addWidget(restore_label)
        
        self.restore_combo = StyledComboBox()
        self.restore_combo.addItems([
            "Wiener Filter",
            "Inverse Filter"
        ])
        restore_layout.addWidget(self.restore_combo, 1)
        layout.addLayout(restore_layout)
        
        # Wiener K parameter
        self.wiener_k = StyledSlider("K (x0.001)", 1, 100, 10)
        layout.addWidget(self.wiener_k)
        
        # Cutoff ratio (for inverse filter)
        self.cutoff_ratio = StyledSlider("Cutoff Ratio (%)", 10, 100, 70)
        layout.addWidget(self.cutoff_ratio)
        
        # Restore button
        restore_btn = ApplyButton("Restore Image", "fa5s.magic")
        restore_btn.clicked.connect(self._onRestoreImage)
        layout.addWidget(restore_btn)
        
        return group
        
    def _onMeanFilterChanged(self, index: int):
        """Handle mean filter type change"""
        # Enable Q parameter only for Contraharmonic
        self.q_param.setEnabled(index == 3)
        
    def _onOrderFilterChanged(self, index: int):
        """Handle order filter type change"""
        # Enable d parameter only for Alpha-trimmed Mean
        self.d_param.setEnabled(index == 4)
        
    def _onDegradeTypeChanged(self, index: int):
        """Handle degradation type change"""
        is_motion = (index == 0)
        self.motion_length.setEnabled(is_motion)
        self.motion_angle.setEnabled(is_motion)
        self.turbulence_k.setEnabled(not is_motion)
        
    def _onAddNoise(self):
        """Handle add noise button click"""
        noise_type = self.noise_combo.currentText()
        params = {
            "param_a": self.noise_param1.value(),
            "param_b": self.noise_param2.value()
        }
        
        operation_map = {
            "Uniform": "add_uniform_noise",
            "Rayleigh": "add_rayleigh_noise",
            "Exponential": "add_exponential_noise"
        }
        
        operation = operation_map.get(noise_type, "add_uniform_noise")
        self.operationRequested.emit(operation, params)
        
    def _onApplyMeanFilter(self):
        """Handle apply mean filter button click"""
        filter_type = self.mean_filter_combo.currentText()
        params = {
            "kernel_size": self.mean_kernel.value(),
            "q": self.q_param.value() / 10.0  # Scale Q to reasonable range
        }
        
        operation_map = {
            "Arithmetic Mean": "arithmetic_mean_filter",
            "Geometric Mean": "geometric_mean_filter",
            "Harmonic Mean": "harmonic_mean_filter",
            "Contraharmonic Mean": "contraharmonic_mean_filter"
        }
        
        operation = operation_map.get(filter_type, "arithmetic_mean_filter")
        self.operationRequested.emit(operation, params)
        
    def _onApplyOrderFilter(self):
        """Handle apply order filter button click"""
        filter_type = self.order_filter_combo.currentText()
        params = {
            "kernel_size": self.order_kernel.value(),
            "d": self.d_param.value()
        }
        
        operation_map = {
            "Median": "median_filter",
            "Max (Pepper noise)": "max_filter",
            "Min (Salt noise)": "min_filter",
            "Midpoint": "midpoint_filter",
            "Alpha-trimmed Mean": "alpha_trimmed_mean_filter"
        }
        
        operation = operation_map.get(filter_type, "median_filter")
        self.operationRequested.emit(operation, params)
        
    def _onApplyAdaptiveFilter(self):
        """Handle apply adaptive filter button click"""
        filter_type = self.adaptive_filter_combo.currentText()
        params = {
            "kernel_size": self.adaptive_kernel.value()
        }
        
        operation_map = {
            "Adaptive Local Noise Reduction": "adaptive_local_noise_reduction",
            "Adaptive Median": "adaptive_median_filter"
        }
        
        operation = operation_map.get(filter_type, "adaptive_local_noise_reduction")
        self.operationRequested.emit(operation, params)
        
    def _onApplyDegradation(self):
        """Handle apply degradation button click"""
        degrade_type = self.degrade_combo.currentText()
        
        if degrade_type == "Motion Blur":
            params = {
                "length": self.motion_length.value(),
                "angle": self.motion_angle.value()
            }
            self.operationRequested.emit("apply_motion_blur", params)
        else:
            params = {
                "k": self.turbulence_k.value() * 0.0001
            }
            self.operationRequested.emit("apply_atmospheric_blur", params)
            
    def _onRestoreImage(self):
        """Handle restore image button click"""
        degrade_type = self.degrade_combo.currentText()
        restore_method = self.restore_combo.currentText()
        
        method = "wiener" if restore_method == "Wiener Filter" else "inverse"
        
        if degrade_type == "Motion Blur":
            params = {
                "length": self.motion_length.value(),
                "angle": self.motion_angle.value(),
                "method": method,
                "K": self.wiener_k.value() * 0.001,
                "cutoff_ratio": self.cutoff_ratio.value() / 100.0
            }
            self.operationRequested.emit("restore_motion_blur", params)
        else:
            params = {
                "k": self.turbulence_k.value() * 0.0001,
                "method": method,
                "K": self.wiener_k.value() * 0.001,
                "cutoff_ratio": self.cutoff_ratio.value() / 100.0
            }
            self.operationRequested.emit("restore_atmospheric_blur", params)
