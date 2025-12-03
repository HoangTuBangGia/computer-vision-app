"""
Image Viewer Widget - Displays Original and Processed images side by side
"""
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QFrame, QScrollArea
)
from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QPixmap


class ImageLabel(QLabel):
    """Custom QLabel for displaying images with proper scaling"""
    
    # Signal emitted when image is clicked with image coordinates
    imageClicked = Signal(int, int)  # x, y in image coordinates
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self._pixmap = None
        self._original_size = None  # Store original image size
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setMinimumSize(200, 200)
        self.setStyleSheet("background-color: #2b2b2b; border-radius: 8px;")
        
    def setImage(self, pixmap: QPixmap):
        """Set the image and scale it to fit while keeping aspect ratio"""
        self._pixmap = pixmap
        if pixmap:
            self._original_size = (pixmap.width(), pixmap.height())
        else:
            self._original_size = None
        self._updatePixmap()
        
    def _updatePixmap(self):
        """Update the displayed pixmap with proper scaling"""
        if self._pixmap:
            scaled = self._pixmap.scaled(
                self.size(),
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation
            )
            super().setPixmap(scaled)
            
    def resizeEvent(self, event):
        """Handle resize events to rescale the image"""
        super().resizeEvent(event)
        self._updatePixmap()
        
    def mousePressEvent(self, event):
        """Handle mouse clicks and emit image coordinates"""
        if self._pixmap and self._original_size and self.pixmap():
            # Get the displayed pixmap size
            displayed_pixmap = self.pixmap()
            displayed_w = displayed_pixmap.width()
            displayed_h = displayed_pixmap.height()
            
            # Calculate offset (image is centered)
            label_w = self.width()
            label_h = self.height()
            offset_x = (label_w - displayed_w) // 2
            offset_y = (label_h - displayed_h) // 2
            
            # Get click position relative to image
            click_x = event.pos().x() - offset_x
            click_y = event.pos().y() - offset_y
            
            # Check if click is within image bounds
            if 0 <= click_x < displayed_w and 0 <= click_y < displayed_h:
                # Scale to original image coordinates
                scale_x = self._original_size[0] / displayed_w
                scale_y = self._original_size[1] / displayed_h
                
                img_x = int(click_x * scale_x)
                img_y = int(click_y * scale_y)
                
                # Clamp to valid range
                img_x = max(0, min(img_x, self._original_size[0] - 1))
                img_y = max(0, min(img_y, self._original_size[1] - 1))
                
                self.imageClicked.emit(img_x, img_y)
                
        super().mousePressEvent(event)
        
    def clear(self):
        """Clear the image"""
        self._pixmap = None
        super().clear()


class ImageViewerPanel(QFrame):
    """Single image viewer panel with title"""
    
    # Signal to forward image click
    imageClicked = Signal(int, int)
    
    def __init__(self, title: str, parent=None):
        super().__init__(parent)
        self.setObjectName("imageViewerPanel")
        self.setStyleSheet("""
            QFrame#imageViewerPanel {
                background-color: #2b2b2b;
                border: 1px solid #3d3d3d;
                border-radius: 10px;
            }
        """)
        
        layout = QVBoxLayout(self)
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setSpacing(8)
        
        # Title label
        self.title_label = QLabel(title)
        self.title_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.title_label.setStyleSheet("""
            QLabel {
                color: #ffffff;
                font-size: 14px;
                font-weight: bold;
                padding: 5px;
                background-color: #404040;
                border-radius: 5px;
            }
        """)
        layout.addWidget(self.title_label)
        
        # Image display area
        self.image_label = ImageLabel()
        self.image_label.imageClicked.connect(self.imageClicked.emit)
        layout.addWidget(self.image_label, 1)
        
    def setImage(self, pixmap: QPixmap):
        """Set the image to display"""
        self.image_label.setImage(pixmap)
        
    def clear(self):
        """Clear the image"""
        self.image_label.clear()


class ImageViewer(QWidget):
    """
    Main Image Viewer Widget
    Displays Original and Processed images side by side for comparison
    """
    
    # Signal emitted when original image is clicked
    originalImageClicked = Signal(int, int)  # x, y in image coordinates
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("imageViewer")
        self._setupUI()
        
    def _setupUI(self):
        """Setup the viewer UI"""
        layout = QHBoxLayout(self)
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setSpacing(15)
        
        # Original image panel
        self.original_panel = ImageViewerPanel("📷 Original")
        self.original_panel.imageClicked.connect(self.originalImageClicked.emit)
        layout.addWidget(self.original_panel, 1)
        
        # Processed image panel
        self.processed_panel = ImageViewerPanel("✨ Processed")
        layout.addWidget(self.processed_panel, 1)
        
        # Set dark background
        self.setStyleSheet("""
            QWidget#imageViewer {
                background-color: #1e1e1e;
                border-radius: 10px;
            }
        """)
        
    def setOriginalImage(self, pixmap: QPixmap):
        """Set the original image"""
        self.original_panel.setImage(pixmap)
        
    def setProcessedImage(self, pixmap: QPixmap):
        """Set the processed image"""
        self.processed_panel.setImage(pixmap)
        
    def clearImages(self):
        """Clear both images"""
        self.original_panel.clear()
        self.processed_panel.clear()
