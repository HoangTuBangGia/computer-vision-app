"""
Compression Tab - JPEG Simulation for educational purposes
Click on image to analyze 8x8 blocks
"""
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QGroupBox,
    QLabel, QPushButton, QSlider, QTableWidget, QTableWidgetItem,
    QTextEdit, QSplitter, QFrame, QGridLayout, QSizePolicy
)
from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QFont, QColor
import qtawesome as qta


class MatrixTableWidget(QTableWidget):
    """Custom table widget for displaying 8x8 matrices"""
    
    def __init__(self, title: str = "", parent=None):
        super().__init__(8, 8, parent)
        self.title = title
        self._setup_ui()
        
    def _setup_ui(self):
        """Setup table appearance"""
        # Set fixed size for cells
        for i in range(8):
            self.setColumnWidth(i, 50)
            self.setRowHeight(i, 24)
            
        # Style
        self.setStyleSheet("""
            QTableWidget {
                background-color: #1e1e1e;
                gridline-color: #3d3d3d;
                border: 1px solid #3d3d3d;
                border-radius: 4px;
                font-family: 'Consolas', 'Monaco', monospace;
                font-size: 10px;
            }
            QTableWidget::item {
                padding: 2px;
                color: #e0e0e0;
            }
            QHeaderView::section {
                background-color: #2d2d2d;
                color: #808080;
                border: 1px solid #3d3d3d;
                font-size: 9px;
            }
        """)
        
        # Hide headers
        self.horizontalHeader().setVisible(False)
        self.verticalHeader().setVisible(False)
        
        # Disable editing
        self.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        
        # Selection
        self.setSelectionMode(QTableWidget.SelectionMode.NoSelection)
        
        # Size policy - allow expansion
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        self.setMinimumWidth(200)
        self.setFixedHeight(200)
        
    def set_matrix(self, matrix, is_float: bool = False, highlight_zeros: bool = False):
        """
        Set matrix values
        
        Args:
            matrix: 8x8 numpy array
            is_float: Whether to display as float
            highlight_zeros: Whether to highlight zero values
        """
        import numpy as np
        
        for i in range(8):
            for j in range(8):
                value = matrix[i, j]
                
                if is_float:
                    text = f"{value:.1f}"
                else:
                    text = f"{int(value)}"
                    
                item = QTableWidgetItem(text)
                item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
                
                # Highlight zeros in green
                if highlight_zeros and abs(value) < 0.5:
                    item.setBackground(QColor("#1a472a"))
                    item.setForeground(QColor("#4ade80"))
                # Highlight large values
                elif abs(value) > 100:
                    item.setBackground(QColor("#3d2929"))
                    item.setForeground(QColor("#f87171"))
                else:
                    item.setForeground(QColor("#e0e0e0"))
                    
                self.setItem(i, j, item)


class ZigzagDisplayWidget(QWidget):
    """Widget to display zig-zag array with trailing zeros highlighted"""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self._setup_ui()
        
    def _setup_ui(self):
        """Setup UI"""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        
        self.text_display = QTextEdit()
        self.text_display.setReadOnly(True)
        self.text_display.setMaximumHeight(100)
        self.text_display.setStyleSheet("""
            QTextEdit {
                background-color: #1e1e1e;
                border: 1px solid #3d3d3d;
                border-radius: 4px;
                font-family: 'Consolas', 'Monaco', monospace;
                font-size: 11px;
                color: #e0e0e0;
                padding: 8px;
            }
        """)
        
        layout.addWidget(self.text_display)
        
    def set_zigzag(self, zigzag_array):
        """
        Display zig-zag array with formatting
        
        Args:
            zigzag_array: 1D array of 64 elements
        """
        import numpy as np
        
        # Find trailing zeros
        last_nonzero = 63
        for i in range(63, -1, -1):
            if abs(zigzag_array[i]) > 0.5:
                last_nonzero = i
                break
        else:
            last_nonzero = -1
            
        # Build HTML display
        html = '<span style="font-family: Consolas, monospace;">'
        
        for i, val in enumerate(zigzag_array):
            int_val = int(val)
            
            if i > 0 and i % 16 == 0:
                html += '<br>'
            elif i > 0:
                html += ' '
                
            # Color coding
            if i > last_nonzero:
                # Trailing zeros - green
                html += f'<span style="color: #4ade80; background-color: #1a472a; padding: 1px 3px;">{int_val:4d}</span>'
            elif int_val == 0:
                # Non-trailing zeros - dim
                html += f'<span style="color: #666666;">{int_val:4d}</span>'
            else:
                # Non-zero values
                html += f'<span style="color: #e0e0e0;">{int_val:4d}</span>'
                
        html += '</span>'
        
        # Add stats
        zero_count = int(np.sum(np.abs(zigzag_array) < 0.5))
        trailing_count = 63 - last_nonzero if last_nonzero >= 0 else 64
        
        stats = f'<br><br><span style="color: #808080; font-size: 10px;">'
        stats += f'Total zeros: <span style="color: #4ade80;">{zero_count}/64</span> | '
        stats += f'Trailing zeros: <span style="color: #4ade80;">{trailing_count}</span>'
        stats += '</span>'
        
        self.text_display.setHtml(html + stats)


class CompressionTab(QWidget):
    """
    Tab 7: Image Compression (JPEG Simulation)
    Educational visualization of DCT, Quantization, Zig-zag
    """
    
    # Signals
    blockClicked = Signal(int, int)  # x, y position
    qualityChanged = Signal(int)
    compressFullImage = Signal()
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self._setup_ui()
        self._connect_signals()
        
    def _setup_ui(self):
        """Setup the UI"""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(12)
        
        # === Header ===
        header = QLabel("JPEG Compression Simulation")
        header.setStyleSheet("""
            QLabel {
                color: #0078d4;
                font-size: 16px;
                font-weight: bold;
                padding: 4px;
            }
        """)
        layout.addWidget(header)
        
        # === Instructions ===
        instructions = QLabel(
            "📍 Click anywhere on the Original image to analyze the 8×8 block at that position. "
            "See how DCT transforms spatial data to frequency domain, quantization removes high-frequency details, "
            "and zig-zag ordering groups zeros together for better compression."
        )
        instructions.setWordWrap(True)
        instructions.setStyleSheet("""
            QLabel {
                color: #a0a0a0;
                font-size: 11px;
                padding: 8px;
                background-color: #252525;
                border-radius: 4px;
            }
        """)
        layout.addWidget(instructions)
        
        # === Quality Control ===
        quality_group = self._create_quality_group()
        layout.addWidget(quality_group)
        
        # === Block Info ===
        self.block_info_label = QLabel("No block selected. Click on the Original image.")
        self.block_info_label.setStyleSheet("""
            QLabel {
                color: #ffa500;
                font-size: 12px;
                font-weight: bold;
                padding: 8px;
                background-color: #2d2d2d;
                border-radius: 4px;
            }
        """)
        layout.addWidget(self.block_info_label)
        
        # === Main Content - 2x2 Grid Layout ===
        matrices_container = QWidget()
        matrices_container.setStyleSheet("background-color: transparent;")
        grid_layout = QGridLayout(matrices_container)
        grid_layout.setContentsMargins(0, 0, 0, 0)
        grid_layout.setSpacing(12)
        
        # Top-left: Original block
        original_group = self._create_matrix_group("📊 Original 8×8 Block")
        self.original_table = MatrixTableWidget()
        original_group.layout().addWidget(self.original_table)
        grid_layout.addWidget(original_group, 0, 0)
        
        # Top-right: After Quantization
        quant_group = self._create_matrix_group("📉 After Quantization")
        self.quantized_table = MatrixTableWidget()
        quant_group.layout().addWidget(self.quantized_table)
        grid_layout.addWidget(quant_group, 0, 1)
        
        # Bottom-left: DCT coefficients
        dct_group = self._create_matrix_group("🔄 DCT Coefficients")
        self.dct_table = MatrixTableWidget()
        dct_group.layout().addWidget(self.dct_table)
        grid_layout.addWidget(dct_group, 1, 0)
        
        # Bottom-right: Zig-zag array
        zigzag_group = self._create_matrix_group("📈 Zig-Zag Sequence")
        self.zigzag_display = ZigzagDisplayWidget()
        zigzag_group.layout().addWidget(self.zigzag_display)
        grid_layout.addWidget(zigzag_group, 1, 1)
        
        # Make columns stretch equally
        grid_layout.setColumnStretch(0, 1)
        grid_layout.setColumnStretch(1, 1)
        
        layout.addWidget(matrices_container, 1)
        
        # === Compression Stats ===
        stats_group = self._create_stats_group()
        layout.addWidget(stats_group)
        
        # === Compress Full Image Button ===
        self.compress_btn = QPushButton(qta.icon('fa5s.compress-arrows-alt', color='white'),
                                         "  Compress Full Image")
        self.compress_btn.setStyleSheet("""
            QPushButton {
                background-color: #0078d4;
                color: white;
                border: none;
                border-radius: 6px;
                padding: 12px 24px;
                font-size: 13px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #1084d8;
            }
            QPushButton:pressed {
                background-color: #006cbd;
            }
        """)
        self.compress_btn.setFixedHeight(44)
        layout.addWidget(self.compress_btn)
        
    def _create_quality_group(self) -> QGroupBox:
        """Create quality control group"""
        group = QGroupBox("Compression Quality")
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
        
        main_layout = QVBoxLayout(group)
        main_layout.setContentsMargins(16, 24, 16, 12)
        main_layout.setSpacing(12)
        
        # Top row: Slider with min/max labels
        slider_layout = QHBoxLayout()
        slider_layout.setSpacing(12)
        
        # Min label
        min_label = QLabel("1")
        min_label.setStyleSheet("color: #f87171; font-weight: bold; font-size: 12px;")
        min_label.setFixedWidth(20)
        min_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        slider_layout.addWidget(min_label)
        
        # Quality slider
        self.quality_slider = QSlider(Qt.Orientation.Horizontal)
        self.quality_slider.setRange(1, 100)
        self.quality_slider.setValue(50)
        self.quality_slider.setFixedHeight(28)
        self.quality_slider.setTracking(True)
        self.quality_slider.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        self.quality_slider.setStyleSheet("""
            QSlider::groove:horizontal {
                background: #3d3d3d;
                height: 8px;
                border-radius: 4px;
            }
            QSlider::handle:horizontal {
                background: #0078d4;
                border: 2px solid #0078d4;
                width: 18px;
                height: 18px;
                margin: -6px 0;
                border-radius: 11px;
            }
            QSlider::handle:horizontal:hover {
                background: #1a8cdb;
                border-color: #1a8cdb;
            }
            QSlider::handle:horizontal:pressed {
                background: #005a9e;
            }
            QSlider::sub-page:horizontal {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                    stop:0 #f87171, stop:0.5 #fbbf24, stop:1 #4ade80);
                height: 8px;
                border-radius: 4px;
            }
        """)
        slider_layout.addWidget(self.quality_slider, 1)
        
        # Max label
        max_label = QLabel("100")
        max_label.setStyleSheet("color: #4ade80; font-weight: bold; font-size: 12px;")
        max_label.setFixedWidth(30)
        max_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        slider_layout.addWidget(max_label)
        
        main_layout.addLayout(slider_layout)
        
        # Bottom row: Current value display centered
        value_layout = QHBoxLayout()
        value_layout.addStretch()
        
        value_container = QHBoxLayout()
        value_container.setSpacing(8)
        
        quality_label = QLabel("Quality:")
        quality_label.setStyleSheet("color: #a0a0a0; font-size: 12px;")
        value_container.addWidget(quality_label)
        
        self.quality_value_label = QLabel("50")
        self.quality_value_label.setFixedWidth(45)
        self.quality_value_label.setFixedHeight(28)
        self.quality_value_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.quality_value_label.setStyleSheet("""
            QLabel {
                color: #ffffff;
                font-weight: bold;
                font-size: 16px;
                background-color: #0078d4;
                border-radius: 6px;
            }
        """)
        value_container.addWidget(self.quality_value_label)
        
        # Quality description
        self.quality_desc_label = QLabel("(Medium)")
        self.quality_desc_label.setStyleSheet("color: #fbbf24; font-size: 11px;")
        value_container.addWidget(self.quality_desc_label)
        
        value_layout.addLayout(value_container)
        value_layout.addStretch()
        
        main_layout.addLayout(value_layout)
        
        return group
        
    def _create_matrix_group(self, title: str) -> QGroupBox:
        """Create a group box for matrix display"""
        group = QGroupBox(title)
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
        
        return group
        
    def _create_stats_group(self) -> QGroupBox:
        """Create compression statistics group"""
        group = QGroupBox("Compression Statistics")
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
        layout.setContentsMargins(16, 20, 16, 16)
        layout.setSpacing(24)
        
        # Block stats
        block_stats = QVBoxLayout()
        block_stats.setSpacing(4)
        
        block_title = QLabel("Block Analysis")
        block_title.setStyleSheet("color: #808080; font-size: 10px; font-weight: normal;")
        block_stats.addWidget(block_title)
        
        self.zeros_label = QLabel("Zeros: -/64")
        self.zeros_label.setStyleSheet("color: #4ade80; font-size: 12px; font-weight: bold;")
        block_stats.addWidget(self.zeros_label)
        
        self.block_error_label = QLabel("Block MSE: -")
        self.block_error_label.setStyleSheet("color: #fbbf24; font-size: 12px; font-weight: normal;")
        block_stats.addWidget(self.block_error_label)
        
        layout.addLayout(block_stats)
        
        # Separator
        sep = QFrame()
        sep.setFrameShape(QFrame.Shape.VLine)
        sep.setStyleSheet("background-color: #3d3d3d;")
        layout.addWidget(sep)
        
        # Full image stats
        image_stats = QVBoxLayout()
        image_stats.setSpacing(4)
        
        image_title = QLabel("Full Image (after compression)")
        image_title.setStyleSheet("color: #808080; font-size: 10px; font-weight: normal;")
        image_stats.addWidget(image_title)
        
        self.psnr_label = QLabel("PSNR: - dB")
        self.psnr_label.setStyleSheet("color: #60a5fa; font-size: 12px; font-weight: bold;")
        image_stats.addWidget(self.psnr_label)
        
        self.ssim_label = QLabel("SSIM: -")
        self.ssim_label.setStyleSheet("color: #c084fc; font-size: 12px; font-weight: normal;")
        image_stats.addWidget(self.ssim_label)
        
        layout.addLayout(image_stats)
        layout.addStretch()
        
        return group
        
    def _connect_signals(self):
        """Connect internal signals"""
        self.quality_slider.valueChanged.connect(self._on_quality_changed)
        self.compress_btn.clicked.connect(self.compressFullImage.emit)
        
    def _on_quality_changed(self, value: int):
        """Handle quality slider change"""
        self.quality_value_label.setText(str(value))
        
        # Update quality description
        if value <= 20:
            desc = "(Very Low)"
            color = "#f87171"
        elif value <= 40:
            desc = "(Low)"
            color = "#fb923c"
        elif value <= 60:
            desc = "(Medium)"
            color = "#fbbf24"
        elif value <= 80:
            desc = "(High)"
            color = "#a3e635"
        else:
            desc = "(Excellent)"
            color = "#4ade80"
            
        self.quality_desc_label.setText(desc)
        self.quality_desc_label.setStyleSheet(f"color: {color}; font-size: 11px;")
        
        self.qualityChanged.emit(value)
        
    def get_quality(self) -> int:
        """Get current quality value"""
        return self.quality_slider.value()
        
    def update_block_info(self, x: int, y: int, block_x: int, block_y: int):
        """Update block position info"""
        self.block_info_label.setText(
            f"🎯 Click at ({x}, {y}) → Block at ({block_x}, {block_y}) to ({block_x+7}, {block_y+7})"
        )
        
    def display_original_block(self, block):
        """Display original pixel values"""
        self.original_table.set_matrix(block, is_float=False)
        
    def display_dct_block(self, dct):
        """Display DCT coefficients"""
        self.dct_table.set_matrix(dct, is_float=True)
        
    def display_quantized_block(self, quantized):
        """Display quantized values with zeros highlighted"""
        self.quantized_table.set_matrix(quantized, is_float=False, highlight_zeros=True)
        
    def display_zigzag(self, zigzag):
        """Display zig-zag array"""
        self.zigzag_display.set_zigzag(zigzag)
        
    def update_block_stats(self, zeros_count: int, mse: float):
        """Update block statistics"""
        self.zeros_label.setText(f"Zeros: {zeros_count}/64")
        self.block_error_label.setText(f"Block MSE: {mse:.2f}")
        
    def update_image_stats(self, psnr: float, ssim: float):
        """Update full image statistics"""
        if psnr == float('inf'):
            self.psnr_label.setText("PSNR: ∞ dB (Perfect)")
        else:
            self.psnr_label.setText(f"PSNR: {psnr:.2f} dB")
        self.ssim_label.setText(f"SSIM: {ssim:.4f}")
        
    def clear_display(self):
        """Clear all displays"""
        import numpy as np
        empty = np.zeros((8, 8))
        
        self.original_table.set_matrix(empty)
        self.dct_table.set_matrix(empty)
        self.quantized_table.set_matrix(empty)
        self.zigzag_display.set_zigzag(np.zeros(64))
        
        self.block_info_label.setText("No block selected. Click on the Original image.")
        self.zeros_label.setText("Zeros: -/64")
        self.block_error_label.setText("Block MSE: -")
        self.psnr_label.setText("PSNR: - dB")
        self.ssim_label.setText("SSIM: -")
