"""
Morphology Tab - Morphological operations for binary images
"""
import numpy as np
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QSpinBox,
    QComboBox, QPushButton, QFrame, QGroupBox, QGridLayout,
    QScrollArea, QSizePolicy, QCheckBox
)
from PySide6.QtCore import Qt, Signal
import qtawesome as qta


class MorphButton(QPushButton):
    """Styled button for morphological operations"""
    
    def __init__(self, text: str, icon_name: str, color: str = "#0078d4", parent=None):
        super().__init__(text, parent)
        
        icon = qta.icon(icon_name, color=color)
        self.setIcon(icon)
        self.setIconSize(self.iconSize() * 1.5)
        
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.setMinimumHeight(50)
        self.setStyleSheet(f"""
            QPushButton {{
                background-color: #353535;
                color: #cccccc;
                border: 1px solid #454545;
                border-radius: 8px;
                padding: 10px;
                font-size: 10px;
                font-weight: bold;
                text-align: center;
            }}
            QPushButton:hover {{
                background-color: #404040;
                border-color: {color};
                color: #ffffff;
            }}
            QPushButton:pressed {{
                background-color: #2a2a2a;
            }}
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


class MorphologyTab(QWidget):
    """
    Morphology Tab - Binary morphological operations
    Contains erosion, dilation, opening, closing, boundary extraction
    """
    
    # Signal emitted when operation is requested
    morphologyRequested = Signal(str, dict)  # operation_name, parameters
    
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
        
        # Parameters section
        params_group = self._createParametersGroup()
        layout.addWidget(params_group)
        
        # Operations section
        ops_group = self._createOperationsGroup()
        layout.addWidget(ops_group)
        
        # Info section
        info_group = self._createInfoGroup()
        layout.addWidget(info_group)
        
        # Spacer
        layout.addStretch()
        
        scroll.setWidget(content)
        
        # Main layout
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.addWidget(scroll)
        
    def _createParametersGroup(self) -> QGroupBox:
        """Create parameters group"""
        group = QGroupBox("⚙️ Structuring Element")
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
        
        # Shape selection
        shape_layout = QHBoxLayout()
        shape_label = QLabel("Shape:")
        shape_label.setStyleSheet("color: #cccccc; font-size: 11px;")
        shape_label.setFixedWidth(50)
        shape_layout.addWidget(shape_label)
        
        self.shape_combo = StyledComboBox()
        self.shape_combo.addItems(["Rectangle", "Cross", "Ellipse"])
        shape_layout.addWidget(self.shape_combo, 1)
        layout.addLayout(shape_layout)
        
        # Kernel size selection
        size_layout = QHBoxLayout()
        size_label = QLabel("Size:")
        size_label.setStyleSheet("color: #cccccc; font-size: 11px;")
        size_label.setFixedWidth(50)
        size_layout.addWidget(size_label)
        
        self.size_spin = StyledSpinBox()
        self.size_spin.setRange(1, 31)
        self.size_spin.setValue(3)
        self.size_spin.setSingleStep(2)  # Odd numbers preferred
        size_layout.addWidget(self.size_spin, 1)
        layout.addLayout(size_layout)
        
        # Iterations (for erosion/dilation)
        iter_layout = QHBoxLayout()
        iter_label = QLabel("Iterations:")
        iter_label.setStyleSheet("color: #cccccc; font-size: 11px;")
        iter_label.setFixedWidth(50)
        iter_layout.addWidget(iter_label)
        
        self.iter_spin = StyledSpinBox()
        self.iter_spin.setRange(1, 10)
        self.iter_spin.setValue(1)
        iter_layout.addWidget(self.iter_spin, 1)
        layout.addLayout(iter_layout)
        
        # Auto binary checkbox
        self.auto_binary = QCheckBox("Auto convert to binary")
        self.auto_binary.setChecked(True)
        self.auto_binary.setStyleSheet("""
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
        layout.addWidget(self.auto_binary)
        
        return group
        
    def _createOperationsGroup(self) -> QGroupBox:
        """Create operations button grid"""
        group = QGroupBox("🔧 Operations")
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
        
        # Create 2x3 grid of operation buttons
        grid = QGridLayout()
        grid.setSpacing(8)
        
        # Define operations with icons
        operations = [
            ("Erosion", "fa5s.compress-arrows-alt", "#e74c3c", "erosion"),
            ("Dilation", "fa5s.expand-arrows-alt", "#2ecc71", "dilation"),
            ("Opening", "fa5s.door-open", "#3498db", "opening"),
            ("Closing", "fa5s.door-closed", "#9b59b6", "closing"),
            ("Boundary", "fa5s.border-style", "#f39c12", "boundary"),
            ("Skeleton", "fa5s.bone", "#e91e63", "skeleton"),
            ("Gradient", "fa5s.wave-square", "#1abc9c", "gradient"),
        ]
        
        self.op_buttons = {}
        for i, (name, icon, color, op_id) in enumerate(operations):
            btn = MorphButton(name, icon, color)
            btn.setProperty("operation", op_id)
            btn.clicked.connect(lambda checked, op=op_id: self._onOperationClicked(op))
            self.op_buttons[op_id] = btn
            grid.addWidget(btn, i // 3, i % 3)
            
        layout.addLayout(grid)
        
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
            "Select an operation:\n\n"
            "• Erosion: Shrinks objects\n"
            "• Dilation: Expands objects\n"
            "• Opening: Removes noise\n"
            "• Closing: Fills holes\n"
            "• Boundary: Edge extraction\n"
            "• Skeleton: Medial axis\n"
            "• Gradient: Edge highlight"
        )
        self.info_label.setWordWrap(True)
        self.info_label.setStyleSheet("""
            QLabel {
                color: #888888;
                font-size: 10px;
                padding: 5px;
                background-color: #353535;
                border-radius: 4px;
            }
        """)
        layout.addWidget(self.info_label)
        
        return group
        
    def _connectSignals(self):
        """Connect internal signals"""
        pass  # Buttons connect in _createOperationsGroup
        
    def _onOperationClicked(self, operation: str):
        """Handle operation button click"""
        # Update info based on operation
        info_texts = {
            "erosion": "Erosion shrinks bright regions.\nUses: Remove noise, separate objects.\nFormula: min(neighbors)",
            "dilation": "Dilation expands bright regions.\nUses: Fill holes, connect objects.\nFormula: max(neighbors)",
            "opening": "Opening = Erosion → Dilation\nUses: Remove small bright spots while preserving shape.",
            "closing": "Closing = Dilation → Erosion\nUses: Fill small holes and gaps while preserving shape.",
            "boundary": "Boundary = Original - Erosion\nExtracts object boundaries/edges.",
            "skeleton": "Skeleton extracts medial axis.\nReduces objects to thin lines while preserving connectivity.",
            "gradient": "Gradient = Dilation - Erosion\nHighlights object edges with thickness.",
        }
        self.info_label.setText(info_texts.get(operation, ""))
        
        # Emit signal
        params = self._getParameters()
        self.morphologyRequested.emit(operation, params)
        
    def _getParameters(self) -> dict:
        """Get current parameters"""
        shape_map = ["rect", "cross", "ellipse"]
        return {
            "shape": shape_map[self.shape_combo.currentIndex()],
            "kernel_size": self.size_spin.value(),
            "iterations": self.iter_spin.value(),
            "auto_binary": self.auto_binary.isChecked(),
        }
        
    def getCurrentParameters(self) -> dict:
        """Get current parameters (public method)"""
        return self._getParameters()
