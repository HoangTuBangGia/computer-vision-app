"""
PCA Tab - Face Recognition using Eigenfaces
Load face dataset, compute PCA, reconstruct faces
"""
import numpy as np
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QSpinBox,
    QPushButton, QFrame, QGroupBox, QSlider, QComboBox,
    QScrollArea, QFileDialog, QGridLayout
)
from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QPixmap, QImage
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
            QPushButton:disabled {{
                background-color: #555555;
                color: #888888;
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
                font-size: 11px;
                min-height: 18px;
            }
            QComboBox:hover {
                border-color: #0078d4;
            }
            QComboBox::drop-down {
                border: none;
                width: 25px;
            }
            QComboBox::down-arrow {
                image: none;
                border-left: 4px solid transparent;
                border-right: 4px solid transparent;
                border-top: 5px solid #888888;
                margin-right: 8px;
            }
            QComboBox QAbstractItemView {
                background-color: #2b2b2b;
                color: #cccccc;
                selection-background-color: #0078d4;
                selection-color: white;
                border: 1px solid #555555;
            }
        """)


class ImageDisplayWidget(QLabel):
    """Widget to display small images with border"""
    
    def __init__(self, title: str = "", size: int = 80, parent=None):
        super().__init__(parent)
        self.title = title
        self.display_size = size
        self.setFixedSize(size, size)
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setStyleSheet("""
            QLabel {
                background-color: #2b2b2b;
                border: 2px solid #404040;
                border-radius: 6px;
            }
        """)
        self.setText("No Image")
        self.setStyleSheet(self.styleSheet() + "color: #666666; font-size: 9px;")
        
    def setImage(self, image: np.ndarray):
        """Set image from numpy array (grayscale or color)"""
        if image is None:
            self.setText("No Image")
            return
            
        h, w = image.shape[:2]
        
        if len(image.shape) == 2:
            # Grayscale
            q_image = QImage(image.data, w, h, w, QImage.Format.Format_Grayscale8)
        else:
            # Color (BGR to RGB)
            import cv2
            rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            q_image = QImage(rgb.data, w, h, 3 * w, QImage.Format.Format_RGB888)
            
        pixmap = QPixmap.fromImage(q_image)
        scaled = pixmap.scaled(
            self.display_size - 4, self.display_size - 4,
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation
        )
        self.setPixmap(scaled)


class PCATab(QWidget):
    """
    PCA Tab - Eigenfaces Face Recognition
    """
    
    # Signals
    loadDatasetRequested = Signal()
    reconstructRequested = Signal(int, int)  # face_index, n_components
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self._n_faces = 0
        self._max_components = 1
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
        
        # Dataset Section
        dataset_group = self._createDatasetGroup()
        layout.addWidget(dataset_group)
        
        # Mean Face & Eigenfaces Display
        display_group = self._createDisplayGroup()
        layout.addWidget(display_group)
        
        # Reconstruction Section
        recon_group = self._createReconstructionGroup()
        layout.addWidget(recon_group)
        
        # Info Section
        info_group = self._createInfoGroup()
        layout.addWidget(info_group)
        
        layout.addStretch()
        
        scroll.setWidget(content)
        
        # Main layout
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.addWidget(scroll)
        
    def _createDatasetGroup(self) -> QGroupBox:
        """Create dataset loading group"""
        group = QGroupBox("📁 Face Dataset")
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
        
        # Load button
        self.load_dataset_btn = ActionButton(
            "Load Face Dataset",
            "fa5s.users",
            "#9b59b6"
        )
        layout.addWidget(self.load_dataset_btn)
        
        # Dataset info label
        self.dataset_info_label = QLabel("No dataset loaded")
        self.dataset_info_label.setWordWrap(True)
        self.dataset_info_label.setStyleSheet("""
            QLabel {
                color: #888888;
                font-size: 10px;
                padding: 8px;
                background-color: #353535;
                border-radius: 4px;
            }
        """)
        layout.addWidget(self.dataset_info_label)
        
        return group
        
    def _createDisplayGroup(self) -> QGroupBox:
        """Create mean face and eigenfaces display"""
        group = QGroupBox("👤 Mean Face & Eigenfaces")
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
        
        # Mean face display
        mean_layout = QHBoxLayout()
        
        mean_label = QLabel("Mean Face:")
        mean_label.setStyleSheet("color: #cccccc; font-size: 10px;")
        mean_layout.addWidget(mean_label)
        mean_layout.addStretch()
        
        self.mean_face_display = ImageDisplayWidget("Mean", 70)
        mean_layout.addWidget(self.mean_face_display)
        
        layout.addLayout(mean_layout)
        
        # First 3 eigenfaces
        eigen_label = QLabel("Top Eigenfaces:")
        eigen_label.setStyleSheet("color: #cccccc; font-size: 10px;")
        layout.addWidget(eigen_label)
        
        eigen_layout = QHBoxLayout()
        eigen_layout.setSpacing(5)
        
        self.eigenface_displays = []
        for i in range(3):
            display = ImageDisplayWidget(f"PC{i+1}", 60)
            self.eigenface_displays.append(display)
            eigen_layout.addWidget(display)
            
        eigen_layout.addStretch()
        layout.addLayout(eigen_layout)
        
        return group
        
    def _createReconstructionGroup(self) -> QGroupBox:
        """Create reconstruction controls"""
        group = QGroupBox("🔄 Face Reconstruction")
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
        
        # Face selection
        face_layout = QHBoxLayout()
        face_label = QLabel("Select Face:")
        face_label.setStyleSheet("color: #cccccc; font-size: 11px;")
        face_label.setFixedWidth(70)
        face_layout.addWidget(face_label)
        
        self.face_combo = StyledComboBox()
        self.face_combo.setEnabled(False)
        face_layout.addWidget(self.face_combo, 1)
        layout.addLayout(face_layout)
        
        # Components slider
        comp_header = QHBoxLayout()
        comp_label = QLabel("Components (K):")
        comp_label.setStyleSheet("color: #cccccc; font-size: 11px;")
        comp_header.addWidget(comp_label)
        
        self.comp_value_label = QLabel("1")
        self.comp_value_label.setStyleSheet("""
            color: #0078d4;
            font-size: 12px;
            font-weight: bold;
        """)
        self.comp_value_label.setAlignment(Qt.AlignmentFlag.AlignRight)
        comp_header.addWidget(self.comp_value_label)
        layout.addLayout(comp_header)
        
        self.comp_slider = StyledSlider()
        self.comp_slider.setRange(1, 100)
        self.comp_slider.setValue(1)
        self.comp_slider.setEnabled(False)
        layout.addWidget(self.comp_slider)
        
        # Variance explained
        self.variance_label = QLabel("Variance Explained: --%")
        self.variance_label.setStyleSheet("""
            color: #2ecc71;
            font-size: 11px;
            padding: 5px;
            background-color: #353535;
            border-radius: 4px;
        """)
        self.variance_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(self.variance_label)
        
        # Original vs Reconstructed display
        compare_layout = QHBoxLayout()
        compare_layout.setSpacing(10)
        
        # Original
        orig_container = QVBoxLayout()
        orig_label = QLabel("Original")
        orig_label.setStyleSheet("color: #888888; font-size: 9px;")
        orig_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        orig_container.addWidget(orig_label)
        self.original_display = ImageDisplayWidget("Original", 80)
        orig_container.addWidget(self.original_display)
        compare_layout.addLayout(orig_container)
        
        # Arrow
        arrow_label = QLabel("→")
        arrow_label.setStyleSheet("color: #0078d4; font-size: 20px; font-weight: bold;")
        arrow_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        compare_layout.addWidget(arrow_label)
        
        # Reconstructed
        recon_container = QVBoxLayout()
        recon_label = QLabel("Reconstructed")
        recon_label.setStyleSheet("color: #888888; font-size: 9px;")
        recon_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        recon_container.addWidget(recon_label)
        self.reconstructed_display = ImageDisplayWidget("Reconstructed", 80)
        recon_container.addWidget(self.reconstructed_display)
        compare_layout.addLayout(recon_container)
        
        layout.addLayout(compare_layout)
        
        # MSE label
        self.mse_label = QLabel("MSE: --")
        self.mse_label.setStyleSheet("""
            color: #f39c12;
            font-size: 10px;
            padding: 3px;
        """)
        self.mse_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(self.mse_label)
        
        return group
        
    def _createInfoGroup(self) -> QGroupBox:
        """Create info group"""
        group = QGroupBox("ℹ️ About Eigenfaces")
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
            "PCA Eigenfaces:\n\n"
            "• Computes principal components\n"
            "  from face image dataset\n"
            "• Mean face = average of all faces\n"
            "• Eigenfaces = directions of max\n"
            "  variance in face space\n"
            "• More components = better quality\n"
            "  reconstruction"
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
        self.load_dataset_btn.clicked.connect(self.loadDatasetRequested.emit)
        self.comp_slider.valueChanged.connect(self._onComponentsChanged)
        self.face_combo.currentIndexChanged.connect(self._onFaceSelected)
        
    def _onComponentsChanged(self, value: int):
        """Handle components slider change"""
        self.comp_value_label.setText(str(value))
        if self._n_faces > 0:
            face_index = self.face_combo.currentIndex()
            self.reconstructRequested.emit(face_index, value)
            
    def _onFaceSelected(self, index: int):
        """Handle face selection change"""
        if index >= 0 and self._n_faces > 0:
            n_components = self.comp_slider.value()
            self.reconstructRequested.emit(index, n_components)
            
    def updateDatasetInfo(self, n_faces: int, labels: list, max_components: int):
        """Update UI after dataset is loaded"""
        self._n_faces = n_faces
        self._max_components = max_components
        
        self.dataset_info_label.setText(
            f"✓ Loaded {n_faces} faces\n"
            f"Max components: {max_components}"
        )
        self.dataset_info_label.setStyleSheet("""
            QLabel {
                color: #2ecc71;
                font-size: 10px;
                padding: 8px;
                background-color: #353535;
                border-radius: 4px;
            }
        """)
        
        # Update face combo
        self.face_combo.clear()
        for i, label in enumerate(labels):
            self.face_combo.addItem(f"{i+1}: {label[:15]}")
        self.face_combo.setEnabled(True)
        
        # Update slider
        self.comp_slider.setRange(1, max_components)
        self.comp_slider.setValue(min(10, max_components))
        self.comp_slider.setEnabled(True)
        
    def setMeanFace(self, image: np.ndarray):
        """Set mean face display"""
        self.mean_face_display.setImage(image)
        
    def setEigenfaces(self, eigenfaces: list):
        """Set eigenface displays"""
        for i, display in enumerate(self.eigenface_displays):
            if i < len(eigenfaces):
                display.setImage(eigenfaces[i])
            else:
                display.setText("--")
                
    def setOriginalFace(self, image: np.ndarray):
        """Set original face display"""
        self.original_display.setImage(image)
        
    def setReconstructedFace(self, image: np.ndarray):
        """Set reconstructed face display"""
        self.reconstructed_display.setImage(image)
        
    def updateVariance(self, variance_ratio: float):
        """Update variance explained label"""
        self.variance_label.setText(f"Variance Explained: {variance_ratio*100:.1f}%")
        
    def updateMSE(self, mse: float):
        """Update MSE label"""
        self.mse_label.setText(f"MSE: {mse:.2f}")
        
    def getCurrentParameters(self) -> dict:
        """Get current parameters"""
        return {
            "face_index": self.face_combo.currentIndex(),
            "n_components": self.comp_slider.value(),
        }
