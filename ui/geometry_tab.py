"""
Geometry Tab - Rotate, Scale, Flip operations
"""
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QGroupBox,
    QLabel, QPushButton, QSlider, QSpinBox, QDoubleSpinBox,
    QCheckBox, QFrame, QGridLayout
)
from PySide6.QtCore import Qt, Signal
import qtawesome as qta


class GeometryTab(QWidget):
    """
    Tab for geometric transformations:
    - Rotate (slider -180 to 180)
    - Scale (resize)
    - Flip (horizontal/vertical)
    """
    
    # Signals
    rotateRequested = Signal(float, bool)  # angle, keep_size
    scaleRequested = Signal(float, float)  # scale_x, scale_y
    resizeRequested = Signal(int, int)  # width, height
    flipRequested = Signal(str)  # 'horizontal', 'vertical', 'both'
    resetRequested = Signal()
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self._current_angle = 0
        self._setupUI()
        self._connectSignals()
        
    def _setupUI(self):
        """Setup the UI"""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setSpacing(12)
        
        # Rotate Group
        rotate_group = self._createRotateGroup()
        layout.addWidget(rotate_group)
        
        # Scale Group
        scale_group = self._createScaleGroup()
        layout.addWidget(scale_group)
        
        # Flip Group
        flip_group = self._createFlipGroup()
        layout.addWidget(flip_group)
        
        # Reset Button
        self.reset_btn = QPushButton(qta.icon('fa5s.undo', color='white'), "  Reset")
        self.reset_btn.setStyleSheet("""
            QPushButton {
                background-color: #d9534f;
                color: white;
                border: none;
                border-radius: 6px;
                padding: 10px 16px;
                font-size: 12px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #c9302c;
            }
            QPushButton:pressed {
                background-color: #ac2925;
            }
        """)
        layout.addWidget(self.reset_btn)
        
        layout.addStretch()
        
    def _createRotateGroup(self) -> QGroupBox:
        """Create rotation controls"""
        group = QGroupBox("🔄 Rotate")
        group.setStyleSheet("""
            QGroupBox {
                font-weight: bold;
                border: 1px solid #3d3d3d;
                border-radius: 6px;
                margin-top: 12px;
                padding-top: 8px;
                background-color: #252525;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 12px;
                padding: 0 8px;
                color: #0078d4;
            }
        """)
        
        layout = QVBoxLayout(group)
        layout.setContentsMargins(12, 20, 12, 12)
        layout.setSpacing(10)
        
        # Angle slider row
        slider_layout = QHBoxLayout()
        slider_layout.setSpacing(10)
        
        # -180 label
        min_label = QLabel("-180°")
        min_label.setStyleSheet("color: #888888; font-size: 10px;")
        slider_layout.addWidget(min_label)
        
        # Slider
        self.rotate_slider = QSlider(Qt.Orientation.Horizontal)
        self.rotate_slider.setRange(-180, 180)
        self.rotate_slider.setValue(0)
        self.rotate_slider.setTickPosition(QSlider.TickPosition.TicksBelow)
        self.rotate_slider.setTickInterval(45)
        self.rotate_slider.setStyleSheet("""
            QSlider::groove:horizontal {
                background: #3d3d3d;
                height: 6px;
                border-radius: 3px;
            }
            QSlider::handle:horizontal {
                background: #0078d4;
                width: 18px;
                height: 18px;
                margin: -6px 0;
                border-radius: 9px;
            }
            QSlider::handle:horizontal:hover {
                background: #1084d8;
            }
            QSlider::sub-page:horizontal {
                background: #0078d4;
                border-radius: 3px;
            }
        """)
        slider_layout.addWidget(self.rotate_slider, 1)
        
        # +180 label
        max_label = QLabel("+180°")
        max_label.setStyleSheet("color: #888888; font-size: 10px;")
        slider_layout.addWidget(max_label)
        
        layout.addLayout(slider_layout)
        
        # Value display and controls row
        controls_layout = QHBoxLayout()
        controls_layout.setSpacing(10)
        
        # Angle value label
        self.angle_label = QLabel("0°")
        self.angle_label.setFixedWidth(50)
        self.angle_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.angle_label.setStyleSheet("""
            QLabel {
                color: #0078d4;
                font-size: 16px;
                font-weight: bold;
                background-color: #1e1e1e;
                border-radius: 4px;
                padding: 4px;
            }
        """)
        controls_layout.addWidget(self.angle_label)
        
        # Keep size checkbox
        self.keep_size_check = QCheckBox("Keep Size")
        self.keep_size_check.setChecked(True)
        self.keep_size_check.setStyleSheet("""
            QCheckBox {
                color: #cccccc;
                font-size: 11px;
                spacing: 6px;
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
        controls_layout.addWidget(self.keep_size_check)
        
        controls_layout.addStretch()
        
        # Apply button
        self.rotate_btn = QPushButton("Apply")
        self.rotate_btn.setFixedWidth(70)
        self.rotate_btn.setStyleSheet("""
            QPushButton {
                background-color: #0078d4;
                color: white;
                border: none;
                border-radius: 4px;
                padding: 6px 12px;
                font-size: 11px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #1084d8;
            }
        """)
        controls_layout.addWidget(self.rotate_btn)
        
        layout.addLayout(controls_layout)
        
        # Quick rotate buttons
        quick_layout = QHBoxLayout()
        quick_layout.setSpacing(6)
        
        quick_angles = [(-90, "↶ -90°"), (-45, "↶ -45°"), (45, "↷ +45°"), (90, "↷ +90°")]
        self.quick_rotate_btns = []
        
        for angle, text in quick_angles:
            btn = QPushButton(text)
            btn.setFixedHeight(28)
            btn.setProperty("angle", angle)
            btn.setStyleSheet("""
                QPushButton {
                    background-color: #353535;
                    color: #cccccc;
                    border: 1px solid #454545;
                    border-radius: 4px;
                    font-size: 10px;
                    padding: 4px 8px;
                }
                QPushButton:hover {
                    background-color: #404040;
                    border-color: #555555;
                }
            """)
            quick_layout.addWidget(btn)
            self.quick_rotate_btns.append(btn)
            
        layout.addLayout(quick_layout)
        
        return group
        
    def _createScaleGroup(self) -> QGroupBox:
        """Create scale controls"""
        group = QGroupBox("📐 Scale / Resize")
        group.setStyleSheet("""
            QGroupBox {
                font-weight: bold;
                border: 1px solid #3d3d3d;
                border-radius: 6px;
                margin-top: 12px;
                padding-top: 8px;
                background-color: #252525;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 12px;
                padding: 0 8px;
                color: #0078d4;
            }
        """)
        
        layout = QVBoxLayout(group)
        layout.setContentsMargins(12, 20, 12, 12)
        layout.setSpacing(10)
        
        # Scale factor controls
        scale_layout = QGridLayout()
        scale_layout.setSpacing(8)
        
        # Scale X
        scale_x_label = QLabel("Scale X:")
        scale_x_label.setStyleSheet("color: #cccccc; font-size: 11px;")
        scale_layout.addWidget(scale_x_label, 0, 0)
        
        self.scale_x_spin = QDoubleSpinBox()
        self.scale_x_spin.setRange(0.1, 10.0)
        self.scale_x_spin.setValue(1.0)
        self.scale_x_spin.setSingleStep(0.1)
        self.scale_x_spin.setStyleSheet("""
            QDoubleSpinBox {
                background-color: #2b2b2b;
                color: #e0e0e0;
                border: 1px solid #3d3d3d;
                border-radius: 4px;
                padding: 4px 8px;
                font-size: 11px;
            }
        """)
        scale_layout.addWidget(self.scale_x_spin, 0, 1)
        
        # Scale Y
        scale_y_label = QLabel("Scale Y:")
        scale_y_label.setStyleSheet("color: #cccccc; font-size: 11px;")
        scale_layout.addWidget(scale_y_label, 0, 2)
        
        self.scale_y_spin = QDoubleSpinBox()
        self.scale_y_spin.setRange(0.1, 10.0)
        self.scale_y_spin.setValue(1.0)
        self.scale_y_spin.setSingleStep(0.1)
        self.scale_y_spin.setStyleSheet("""
            QDoubleSpinBox {
                background-color: #2b2b2b;
                color: #e0e0e0;
                border: 1px solid #3d3d3d;
                border-radius: 4px;
                padding: 4px 8px;
                font-size: 11px;
            }
        """)
        scale_layout.addWidget(self.scale_y_spin, 0, 3)
        
        layout.addLayout(scale_layout)
        
        # Link checkbox and Scale button
        link_layout = QHBoxLayout()
        link_layout.setSpacing(10)
        
        self.link_scale_check = QCheckBox("🔗 Link X/Y")
        self.link_scale_check.setChecked(True)
        self.link_scale_check.setStyleSheet("""
            QCheckBox {
                color: #cccccc;
                font-size: 11px;
                spacing: 6px;
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
        link_layout.addWidget(self.link_scale_check)
        
        link_layout.addStretch()
        
        self.scale_btn = QPushButton("Scale")
        self.scale_btn.setFixedWidth(70)
        self.scale_btn.setStyleSheet("""
            QPushButton {
                background-color: #0078d4;
                color: white;
                border: none;
                border-radius: 4px;
                padding: 6px 12px;
                font-size: 11px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #1084d8;
            }
        """)
        link_layout.addWidget(self.scale_btn)
        
        layout.addLayout(link_layout)
        
        # Separator
        sep = QFrame()
        sep.setFrameShape(QFrame.Shape.HLine)
        sep.setStyleSheet("background-color: #3d3d3d;")
        layout.addWidget(sep)
        
        # Resize controls
        resize_layout = QGridLayout()
        resize_layout.setSpacing(8)
        
        # Width
        width_label = QLabel("Width:")
        width_label.setStyleSheet("color: #cccccc; font-size: 11px;")
        resize_layout.addWidget(width_label, 0, 0)
        
        self.width_spin = QSpinBox()
        self.width_spin.setRange(1, 10000)
        self.width_spin.setValue(640)
        self.width_spin.setStyleSheet("""
            QSpinBox {
                background-color: #2b2b2b;
                color: #e0e0e0;
                border: 1px solid #3d3d3d;
                border-radius: 4px;
                padding: 4px 8px;
                font-size: 11px;
            }
        """)
        resize_layout.addWidget(self.width_spin, 0, 1)
        
        # Height
        height_label = QLabel("Height:")
        height_label.setStyleSheet("color: #cccccc; font-size: 11px;")
        resize_layout.addWidget(height_label, 0, 2)
        
        self.height_spin = QSpinBox()
        self.height_spin.setRange(1, 10000)
        self.height_spin.setValue(480)
        self.height_spin.setStyleSheet("""
            QSpinBox {
                background-color: #2b2b2b;
                color: #e0e0e0;
                border: 1px solid #3d3d3d;
                border-radius: 4px;
                padding: 4px 8px;
                font-size: 11px;
            }
        """)
        resize_layout.addWidget(self.height_spin, 0, 3)
        
        layout.addLayout(resize_layout)
        
        # Resize button
        resize_btn_layout = QHBoxLayout()
        resize_btn_layout.addStretch()
        
        self.resize_btn = QPushButton("Resize")
        self.resize_btn.setFixedWidth(70)
        self.resize_btn.setStyleSheet("""
            QPushButton {
                background-color: #5cb85c;
                color: white;
                border: none;
                border-radius: 4px;
                padding: 6px 12px;
                font-size: 11px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #449d44;
            }
        """)
        resize_btn_layout.addWidget(self.resize_btn)
        
        layout.addLayout(resize_btn_layout)
        
        return group
        
    def _createFlipGroup(self) -> QGroupBox:
        """Create flip controls"""
        group = QGroupBox("↔️ Flip")
        group.setStyleSheet("""
            QGroupBox {
                font-weight: bold;
                border: 1px solid #3d3d3d;
                border-radius: 6px;
                margin-top: 12px;
                padding-top: 8px;
                background-color: #252525;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 12px;
                padding: 0 8px;
                color: #0078d4;
            }
        """)
        
        layout = QHBoxLayout(group)
        layout.setContentsMargins(12, 20, 12, 12)
        layout.setSpacing(10)
        
        # Horizontal flip
        self.flip_h_btn = QPushButton(qta.icon('fa5s.arrows-alt-h', color='white'), " Horizontal")
        self.flip_h_btn.setStyleSheet("""
            QPushButton {
                background-color: #6c5ce7;
                color: white;
                border: none;
                border-radius: 6px;
                padding: 10px 16px;
                font-size: 11px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #5f4dd0;
            }
        """)
        layout.addWidget(self.flip_h_btn)
        
        # Vertical flip
        self.flip_v_btn = QPushButton(qta.icon('fa5s.arrows-alt-v', color='white'), " Vertical")
        self.flip_v_btn.setStyleSheet("""
            QPushButton {
                background-color: #00b894;
                color: white;
                border: none;
                border-radius: 6px;
                padding: 10px 16px;
                font-size: 11px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #00a383;
            }
        """)
        layout.addWidget(self.flip_v_btn)
        
        # Both flip (180° rotation)
        self.flip_both_btn = QPushButton(qta.icon('fa5s.sync', color='white'), " Both")
        self.flip_both_btn.setStyleSheet("""
            QPushButton {
                background-color: #e17055;
                color: white;
                border: none;
                border-radius: 6px;
                padding: 10px 16px;
                font-size: 11px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #d35844;
            }
        """)
        layout.addWidget(self.flip_both_btn)
        
        return group
        
    def _connectSignals(self):
        """Connect signals"""
        # Rotate slider
        self.rotate_slider.valueChanged.connect(self._onRotateSliderChanged)
        self.rotate_btn.clicked.connect(self._onRotateClicked)
        
        # Quick rotate buttons
        for btn in self.quick_rotate_btns:
            btn.clicked.connect(self._onQuickRotateClicked)
            
        # Scale
        self.scale_x_spin.valueChanged.connect(self._onScaleXChanged)
        self.scale_btn.clicked.connect(self._onScaleClicked)
        
        # Resize
        self.resize_btn.clicked.connect(self._onResizeClicked)
        
        # Flip
        self.flip_h_btn.clicked.connect(lambda: self.flipRequested.emit('horizontal'))
        self.flip_v_btn.clicked.connect(lambda: self.flipRequested.emit('vertical'))
        self.flip_both_btn.clicked.connect(lambda: self.flipRequested.emit('both'))
        
        # Reset
        self.reset_btn.clicked.connect(self.resetRequested.emit)
        
    def _onRotateSliderChanged(self, value: int):
        """Update angle label when slider changes"""
        self._current_angle = value
        self.angle_label.setText(f"{value}°")
        
    def _onRotateClicked(self):
        """Emit rotate signal"""
        keep_size = self.keep_size_check.isChecked()
        self.rotateRequested.emit(float(self._current_angle), keep_size)
        
    def _onQuickRotateClicked(self):
        """Handle quick rotate button click"""
        btn = self.sender()
        angle = btn.property("angle")
        self.rotate_slider.setValue(angle)
        keep_size = self.keep_size_check.isChecked()
        self.rotateRequested.emit(float(angle), keep_size)
        
    def _onScaleXChanged(self, value: float):
        """Sync scale Y when linked"""
        if self.link_scale_check.isChecked():
            self.scale_y_spin.blockSignals(True)
            self.scale_y_spin.setValue(value)
            self.scale_y_spin.blockSignals(False)
            
    def _onScaleClicked(self):
        """Emit scale signal"""
        scale_x = self.scale_x_spin.value()
        scale_y = self.scale_y_spin.value()
        self.scaleRequested.emit(scale_x, scale_y)
        
    def _onResizeClicked(self):
        """Emit resize signal"""
        width = self.width_spin.value()
        height = self.height_spin.value()
        self.resizeRequested.emit(width, height)
        
    def setImageSize(self, width: int, height: int):
        """Set current image size in resize spinboxes"""
        self.width_spin.setValue(width)
        self.height_spin.setValue(height)
        
    def reset(self):
        """Reset all controls"""
        self.rotate_slider.setValue(0)
        self.scale_x_spin.setValue(1.0)
        self.scale_y_spin.setValue(1.0)
