"""
Basic Operations Tab - Point processing operations with histogram
"""
import numpy as np
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QSlider,
    QRadioButton, QButtonGroup, QFrame, QGroupBox, QScrollArea,
    QSizePolicy
)
from PySide6.QtCore import Qt, Signal

# Matplotlib imports with dark theme configuration
import matplotlib
matplotlib.use('QtAgg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

# Configure matplotlib for dark theme
plt.style.use('dark_background')
matplotlib.rcParams.update({
    'figure.facecolor': '#2b2b2b',
    'axes.facecolor': '#2b2b2b',
    'axes.edgecolor': '#555555',
    'axes.labelcolor': '#cccccc',
    'text.color': '#cccccc',
    'xtick.color': '#888888',
    'ytick.color': '#888888',
    'grid.color': '#404040',
    'legend.facecolor': '#2b2b2b',
    'legend.edgecolor': '#555555',
})


class HistogramCanvas(FigureCanvas):
    """Matplotlib canvas for histogram display with dark theme"""
    
    def __init__(self, parent=None, width=3, height=2, dpi=100):
        self.fig = Figure(figsize=(width, height), dpi=dpi, facecolor='#2b2b2b')
        self.axes = self.fig.add_subplot(111)
        self.axes.set_facecolor('#2b2b2b')
        
        super().__init__(self.fig)
        self.setParent(parent)
        
        # Make canvas expand
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        self.setMinimumHeight(150)
        self.setMaximumHeight(200)
        
        # Initial empty plot
        self._setupEmptyPlot()
        
    def _setupEmptyPlot(self):
        """Setup empty plot with styling"""
        self.axes.clear()
        self.axes.set_xlim(0, 255)
        self.axes.set_xlabel('Pixel Value', fontsize=8, color='#888888')
        self.axes.set_ylabel('Frequency', fontsize=8, color='#888888')
        self.axes.tick_params(axis='both', labelsize=7, colors='#888888')
        self.axes.spines['top'].set_visible(False)
        self.axes.spines['right'].set_visible(False)
        self.axes.spines['bottom'].set_color('#555555')
        self.axes.spines['left'].set_color('#555555')
        self.fig.tight_layout(pad=0.5)
        self.draw()
        
    def plotHistogram(self, histogram_data, is_color: bool = True):
        """
        Plot histogram data
        
        Args:
            histogram_data: Either single array (grayscale) or tuple of (B, G, R)
            is_color: Whether the image is color or grayscale
        """
        self.axes.clear()
        
        x = np.arange(256)
        
        if is_color and isinstance(histogram_data, tuple):
            hist_b, hist_g, hist_r = histogram_data
            # Plot RGB histograms with transparency
            self.axes.fill_between(x, hist_r, alpha=0.4, color='#ff6b6b', label='Red')
            self.axes.fill_between(x, hist_g, alpha=0.4, color='#51cf66', label='Green')
            self.axes.fill_between(x, hist_b, alpha=0.4, color='#339af0', label='Blue')
            self.axes.plot(x, hist_r, color='#ff6b6b', linewidth=0.8)
            self.axes.plot(x, hist_g, color='#51cf66', linewidth=0.8)
            self.axes.plot(x, hist_b, color='#339af0', linewidth=0.8)
            self.axes.legend(loc='upper right', fontsize=7, framealpha=0.8)
        else:
            # Grayscale histogram
            hist = histogram_data if isinstance(histogram_data, np.ndarray) else histogram_data[0]
            self.axes.fill_between(x, hist, alpha=0.6, color='#aaaaaa')
            self.axes.plot(x, hist, color='#cccccc', linewidth=0.8)
        
        # Styling
        self.axes.set_xlim(0, 255)
        self.axes.set_xlabel('Pixel Value', fontsize=8, color='#888888')
        self.axes.set_ylabel('Frequency', fontsize=8, color='#888888')
        self.axes.tick_params(axis='both', labelsize=7, colors='#888888')
        self.axes.spines['top'].set_visible(False)
        self.axes.spines['right'].set_visible(False)
        self.axes.spines['bottom'].set_color('#555555')
        self.axes.spines['left'].set_color('#555555')
        
        self.fig.tight_layout(pad=0.5)
        self.draw()
        
    def clear(self):
        """Clear the histogram"""
        self._setupEmptyPlot()


class StyledSlider(QWidget):
    """Custom styled slider with label and value display"""
    
    valueChanged = Signal(float)
    
    def __init__(self, label: str, min_val: float, max_val: float, 
                 default: float, step: float = 0.1, parent=None):
        super().__init__(parent)
        
        self.min_val = min_val
        self.max_val = max_val
        self.step = step
        self.multiplier = int(1 / step) if step < 1 else 1
        
        self._setupUI(label, min_val, max_val, default)
        
    def _setupUI(self, label: str, min_val: float, max_val: float, default: float):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 5, 0, 5)
        layout.setSpacing(3)
        
        # Header row with label and value
        header = QHBoxLayout()
        
        self.label = QLabel(label)
        self.label.setStyleSheet("color: #cccccc; font-size: 11px;")
        header.addWidget(self.label)
        
        header.addStretch()
        
        self.value_label = QLabel(f"{default:.2f}")
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
        self.slider.setMinimum(int(min_val * self.multiplier))
        self.slider.setMaximum(int(max_val * self.multiplier))
        self.slider.setValue(int(default * self.multiplier))
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
        actual_value = value / self.multiplier
        self.value_label.setText(f"{actual_value:.2f}")
        self.valueChanged.emit(actual_value)
        
    def value(self) -> float:
        """Get current value"""
        return self.slider.value() / self.multiplier
        
    def setValue(self, value: float):
        """Set slider value"""
        self.slider.setValue(int(value * self.multiplier))
        
    def setEnabled(self, enabled: bool):
        """Enable/disable slider"""
        self.slider.setEnabled(enabled)
        opacity = "1.0" if enabled else "0.5"
        self.label.setStyleSheet(f"color: #cccccc; font-size: 11px; opacity: {opacity};")


class BasicTab(QWidget):
    """
    Basic Operations Tab
    Contains point processing operations and histogram display
    """
    
    # Signal emitted when processing is requested
    processRequested = Signal(str, dict)  # operation_name, parameters
    
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
        
        # Operation selection group
        op_group = self._createOperationGroup()
        layout.addWidget(op_group)
        
        # Parameters group
        params_group = self._createParametersGroup()
        layout.addWidget(params_group)
        
        # Histogram group
        hist_group = self._createHistogramGroup()
        layout.addWidget(hist_group)
        
        # Spacer
        layout.addStretch()
        
        scroll.setWidget(content)
        
        # Main layout
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.addWidget(scroll)
        
    def _createOperationGroup(self) -> QGroupBox:
        """Create operation selection group"""
        group = QGroupBox("🎨 Point Operations")
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
        layout.setSpacing(5)
        
        # Radio buttons for operations
        self.op_button_group = QButtonGroup(self)
        
        operations = [
            ("original", "Original (No Effect)"),
            ("grayscale", "Grayscale"),
            ("negative", "Negative"),
            ("log", "Log Transform"),
            ("gamma", "Power-law (Gamma)"),
            ("equalize", "Histogram Equalization"),
        ]
        
        for i, (op_id, op_name) in enumerate(operations):
            radio = QRadioButton(op_name)
            radio.setObjectName(op_id)
            radio.setStyleSheet("""
                QRadioButton {
                    color: #cccccc;
                    font-size: 11px;
                    padding: 5px;
                }
                QRadioButton:hover {
                    color: #ffffff;
                }
                QRadioButton::indicator {
                    width: 14px;
                    height: 14px;
                }
                QRadioButton::indicator:unchecked {
                    border: 2px solid #555555;
                    border-radius: 7px;
                    background-color: #2b2b2b;
                }
                QRadioButton::indicator:checked {
                    border: 2px solid #0078d4;
                    border-radius: 7px;
                    background-color: #0078d4;
                }
            """)
            if i == 0:
                radio.setChecked(True)
            self.op_button_group.addButton(radio, i)
            layout.addWidget(radio)
            
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
        
        # Gamma slider
        self.gamma_slider = StyledSlider("Gamma (γ)", 0.1, 5.0, 1.0, 0.1)
        self.gamma_slider.setEnabled(False)
        layout.addWidget(self.gamma_slider)
        
        # Log C constant slider
        self.log_c_slider = StyledSlider("Log Constant (c)", 1.0, 100.0, 46.0, 1.0)
        self.log_c_slider.setEnabled(False)
        layout.addWidget(self.log_c_slider)
        
        # Info label
        self.param_info = QLabel("Select an operation to adjust parameters")
        self.param_info.setWordWrap(True)
        self.param_info.setStyleSheet("""
            QLabel {
                color: #888888;
                font-size: 10px;
                font-style: italic;
                padding: 5px;
                background-color: #353535;
                border-radius: 4px;
            }
        """)
        layout.addWidget(self.param_info)
        
        return group
        
    def _createHistogramGroup(self) -> QGroupBox:
        """Create histogram display group"""
        group = QGroupBox("📊 Histogram")
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
        layout.setContentsMargins(5, 15, 5, 5)
        
        # Histogram canvas
        self.histogram_canvas = HistogramCanvas(self)
        layout.addWidget(self.histogram_canvas)
        
        return group
        
    def _connectSignals(self):
        """Connect signals"""
        self.op_button_group.buttonClicked.connect(self._onOperationChanged)
        self.gamma_slider.valueChanged.connect(self._onParameterChanged)
        self.log_c_slider.valueChanged.connect(self._onParameterChanged)
        
    def _onOperationChanged(self, button: QRadioButton):
        """Handle operation selection change"""
        op_id = button.objectName()
        
        # Update parameter availability
        self.gamma_slider.setEnabled(op_id == "gamma")
        self.log_c_slider.setEnabled(op_id == "log")
        
        # Update info text
        info_texts = {
            "original": "No processing applied - shows original image",
            "grayscale": "Converts image to grayscale using luminosity method",
            "negative": "Inverts all pixel values: s = 255 - r",
            "log": "s = c × log(1 + r) - Expands dark regions",
            "gamma": "s = c × r^γ - γ<1 brightens, γ>1 darkens",
            "equalize": "Redistributes pixel intensities for better contrast",
        }
        self.param_info.setText(info_texts.get(op_id, ""))
        
        # Emit process request
        self._emitProcessRequest()
        
    def _onParameterChanged(self, value: float):
        """Handle parameter value change"""
        self._emitProcessRequest()
        
    def _emitProcessRequest(self):
        """Emit process request signal with current settings"""
        button = self.op_button_group.checkedButton()
        if button:
            op_id = button.objectName()
            params = {
                "gamma": self.gamma_slider.value(),
                "log_c": self.log_c_slider.value(),
            }
            self.processRequested.emit(op_id, params)
            
    def updateHistogram(self, histogram_data, is_color: bool = True):
        """Update histogram display"""
        self.histogram_canvas.plotHistogram(histogram_data, is_color)
        
    def clearHistogram(self):
        """Clear histogram display"""
        self.histogram_canvas.clear()
        
    def getCurrentOperation(self) -> tuple:
        """Get current operation and parameters"""
        button = self.op_button_group.checkedButton()
        if button:
            op_id = button.objectName()
            params = {
                "gamma": self.gamma_slider.value(),
                "log_c": self.log_c_slider.value(),
            }
            return op_id, params
        return "original", {}
