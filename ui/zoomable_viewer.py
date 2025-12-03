"""
Zoomable Graphics View - Custom QGraphicsView with zoom/pan support
Supports mouse wheel zoom at cursor position and left-click drag to pan
"""
from PySide6.QtWidgets import (
    QGraphicsView, QGraphicsScene, QGraphicsPixmapItem,
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QFrame
)
from PySide6.QtCore import Qt, Signal, QRectF, QPointF
from PySide6.QtGui import QPixmap, QWheelEvent, QMouseEvent, QPainter, QTransform


class ZoomableGraphicsView(QGraphicsView):
    """
    Custom QGraphicsView with zoom and pan capabilities
    
    Features:
    - Mouse wheel zoom at cursor position
    - Left-click drag to pan
    - Smooth transformations
    - Emits signals for sync with other views
    """
    
    # Signals for synchronization
    zoomChanged = Signal(float)  # Emitted when zoom level changes
    panChanged = Signal(QPointF)  # Emitted when pan position changes
    viewTransformChanged = Signal(QTransform)  # Full transform for sync
    imageClicked = Signal(int, int)  # x, y in image coordinates
    
    # Zoom constraints
    MIN_ZOOM = 0.1
    MAX_ZOOM = 20.0
    ZOOM_FACTOR = 1.15
    
    def __init__(self, parent=None):
        super().__init__(parent)
        
        # Setup scene
        self._scene = QGraphicsScene(self)
        self.setScene(self._scene)
        
        # Pixmap item for the image
        self._pixmap_item = None
        self._original_pixmap = None
        
        # State tracking
        self._zoom_level = 1.0
        self._is_panning = False
        self._last_pan_point = QPointF()
        self._sync_enabled = True  # For preventing infinite sync loops
        
        # Setup view properties
        self._setupView()
        
    def _setupView(self):
        """Configure view properties"""
        # Render settings
        self.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        self.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform, True)
        
        # Viewport settings
        self.setViewportUpdateMode(QGraphicsView.ViewportUpdateMode.FullViewportUpdate)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        self.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        
        # Transform settings
        self.setTransformationAnchor(QGraphicsView.ViewportAnchor.AnchorUnderMouse)
        self.setResizeAnchor(QGraphicsView.ViewportAnchor.AnchorViewCenter)
        
        # Drag mode
        self.setDragMode(QGraphicsView.DragMode.NoDrag)
        
        # Background
        self.setStyleSheet("""
            QGraphicsView {
                background-color: #2b2b2b;
                border: none;
                border-radius: 6px;
            }
        """)
        
    def setImage(self, pixmap: QPixmap):
        """
        Set the image to display
        
        Args:
            pixmap: QPixmap to display
        """
        self._original_pixmap = pixmap
        
        # Clear existing item
        if self._pixmap_item:
            self._scene.removeItem(self._pixmap_item)
            
        if pixmap and not pixmap.isNull():
            # Create new pixmap item
            self._pixmap_item = QGraphicsPixmapItem(pixmap)
            self._pixmap_item.setTransformationMode(Qt.TransformationMode.SmoothTransformation)
            self._scene.addItem(self._pixmap_item)
            
            # Set scene rect to match pixmap
            self._scene.setSceneRect(QRectF(pixmap.rect()))
            
            # Fit in view initially
            self.fitInView(self._scene.sceneRect(), Qt.AspectRatioMode.KeepAspectRatio)
            self._zoom_level = self.transform().m11()  # Get actual zoom from transform
        else:
            self._pixmap_item = None
            self._scene.clear()
            
    def clear(self):
        """Clear the displayed image"""
        self._original_pixmap = None
        if self._pixmap_item:
            self._scene.removeItem(self._pixmap_item)
            self._pixmap_item = None
        self._scene.clear()
        self._zoom_level = 1.0
        
    def getZoomLevel(self) -> float:
        """Get current zoom level"""
        return self._zoom_level
        
    def setZoomLevel(self, zoom: float, sync: bool = True):
        """
        Set zoom level programmatically
        
        Args:
            zoom: Zoom level (1.0 = 100%)
            sync: Whether to emit signal for sync
        """
        zoom = max(self.MIN_ZOOM, min(self.MAX_ZOOM, zoom))
        
        # Calculate scale factor from current zoom
        scale_factor = zoom / self._zoom_level
        self._zoom_level = zoom
        
        # Apply transformation
        self.scale(scale_factor, scale_factor)
        
        if sync and self._sync_enabled:
            self.zoomChanged.emit(self._zoom_level)
            self.viewTransformChanged.emit(self.transform())
            
    def wheelEvent(self, event: QWheelEvent):
        """Handle mouse wheel for zooming"""
        if self._pixmap_item is None:
            return
            
        # Get the zoom factor
        if event.angleDelta().y() > 0:
            factor = self.ZOOM_FACTOR
        else:
            factor = 1.0 / self.ZOOM_FACTOR
            
        # Calculate new zoom level
        new_zoom = self._zoom_level * factor
        
        # Clamp to limits
        if new_zoom < self.MIN_ZOOM or new_zoom > self.MAX_ZOOM:
            return
            
        self._zoom_level = new_zoom
        
        # Scale the view
        self.scale(factor, factor)
        
        # Emit signal for sync
        if self._sync_enabled:
            self.zoomChanged.emit(self._zoom_level)
            self.viewTransformChanged.emit(self.transform())
            
    def mousePressEvent(self, event: QMouseEvent):
        """Handle mouse press for panning and clicking"""
        if event.button() == Qt.MouseButton.LeftButton:
            # Check if clicking on image
            if self._pixmap_item and self._original_pixmap:
                # Get scene position
                scene_pos = self.mapToScene(event.pos())
                
                # Check if within image bounds
                if self._pixmap_item.contains(scene_pos):
                    img_x = int(scene_pos.x())
                    img_y = int(scene_pos.y())
                    
                    # Clamp to valid range
                    img_x = max(0, min(img_x, self._original_pixmap.width() - 1))
                    img_y = max(0, min(img_y, self._original_pixmap.height() - 1))
                    
                    self.imageClicked.emit(img_x, img_y)
            
            # Start panning
            self._is_panning = True
            self._last_pan_point = event.pos()
            self.setCursor(Qt.CursorShape.ClosedHandCursor)
            
        super().mousePressEvent(event)
        
    def mouseMoveEvent(self, event: QMouseEvent):
        """Handle mouse move for panning"""
        if self._is_panning:
            # Calculate delta
            delta = event.pos() - self._last_pan_point
            self._last_pan_point = event.pos()
            
            # Pan the view
            self.horizontalScrollBar().setValue(
                self.horizontalScrollBar().value() - delta.x()
            )
            self.verticalScrollBar().setValue(
                self.verticalScrollBar().value() - delta.y()
            )
            
            # Emit signal for sync
            if self._sync_enabled:
                center = self.mapToScene(self.viewport().rect().center())
                self.panChanged.emit(center)
                self.viewTransformChanged.emit(self.transform())
                
        super().mouseMoveEvent(event)
        
    def mouseReleaseEvent(self, event: QMouseEvent):
        """Handle mouse release"""
        if event.button() == Qt.MouseButton.LeftButton:
            self._is_panning = False
            self.setCursor(Qt.CursorShape.ArrowCursor)
            
        super().mouseReleaseEvent(event)
        
    def syncTransform(self, transform: QTransform):
        """
        Synchronize with another view's transform
        
        Args:
            transform: Transform to apply
        """
        # Temporarily disable sync to prevent loops
        self._sync_enabled = False
        
        self.setTransform(transform)
        self._zoom_level = transform.m11()
        
        self._sync_enabled = True
        
    def syncScrollBars(self, h_value: int, v_value: int):
        """
        Synchronize scroll bar positions
        
        Args:
            h_value: Horizontal scroll value
            v_value: Vertical scroll value
        """
        self._sync_enabled = False
        
        self.horizontalScrollBar().setValue(h_value)
        self.verticalScrollBar().setValue(v_value)
        
        self._sync_enabled = True
        
    def fitToView(self):
        """Fit image to view"""
        if self._pixmap_item:
            self.fitInView(self._scene.sceneRect(), Qt.AspectRatioMode.KeepAspectRatio)
            self._zoom_level = self.transform().m11()
            
            if self._sync_enabled:
                self.viewTransformChanged.emit(self.transform())
                
    def resetZoom(self):
        """Reset zoom to 100%"""
        self.resetTransform()
        self._zoom_level = 1.0
        
        if self._sync_enabled:
            self.viewTransformChanged.emit(self.transform())
            
    def resizeEvent(self, event):
        """Handle resize events"""
        super().resizeEvent(event)
        # Optionally fit to view on resize
        # self.fitToView()


class SyncedImageViewerPanel(QFrame):
    """
    Single image viewer panel with title and zoomable view
    """
    
    imageClicked = Signal(int, int)
    viewTransformChanged = Signal(QTransform)
    scrollBarsChanged = Signal(int, int)
    
    def __init__(self, title: str, parent=None):
        super().__init__(parent)
        self.setObjectName("imageViewerPanel")
        self._setupUI(title)
        self._connectSignals()
        
    def _setupUI(self, title: str):
        """Setup panel UI"""
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
        
        # Zoomable graphics view
        self.view = ZoomableGraphicsView()
        layout.addWidget(self.view, 1)
        
        # Zoom info label
        self.zoom_label = QLabel("Zoom: 100%")
        self.zoom_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.zoom_label.setStyleSheet("""
            QLabel {
                color: #888888;
                font-size: 11px;
                padding: 3px;
            }
        """)
        layout.addWidget(self.zoom_label)
        
    def _connectSignals(self):
        """Connect internal signals"""
        self.view.imageClicked.connect(self.imageClicked.emit)
        self.view.viewTransformChanged.connect(self.viewTransformChanged.emit)
        self.view.zoomChanged.connect(self._updateZoomLabel)
        
        # Connect scroll bars
        self.view.horizontalScrollBar().valueChanged.connect(self._onScrollChanged)
        self.view.verticalScrollBar().valueChanged.connect(self._onScrollChanged)
        
    def _updateZoomLabel(self, zoom: float):
        """Update zoom percentage label"""
        self.zoom_label.setText(f"Zoom: {zoom * 100:.0f}%")
        
    def _onScrollChanged(self):
        """Handle scroll bar changes"""
        if self.view._sync_enabled:
            h = self.view.horizontalScrollBar().value()
            v = self.view.verticalScrollBar().value()
            self.scrollBarsChanged.emit(h, v)
            
    def setImage(self, pixmap: QPixmap):
        """Set the image to display"""
        self.view.setImage(pixmap)
        self._updateZoomLabel(self.view.getZoomLevel())
        
    def clear(self):
        """Clear the image"""
        self.view.clear()
        self.zoom_label.setText("Zoom: 100%")
        
    def syncTransform(self, transform: QTransform):
        """Sync with transform from another panel"""
        self.view.syncTransform(transform)
        self._updateZoomLabel(self.view.getZoomLevel())
        
    def syncScrollBars(self, h: int, v: int):
        """Sync scroll bar positions"""
        self.view.syncScrollBars(h, v)


class SyncedImageViewer(QWidget):
    """
    Dual synchronized image viewer
    Both panels zoom and pan together
    """
    
    # Signal for click on original image
    originalImageClicked = Signal(int, int)
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("imageViewer")
        self._setupUI()
        self._connectSignals()
        
    def _setupUI(self):
        """Setup the viewer UI"""
        layout = QHBoxLayout(self)
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setSpacing(15)
        
        # Original image panel
        self.original_panel = SyncedImageViewerPanel("📷 Original")
        layout.addWidget(self.original_panel, 1)
        
        # Processed image panel
        self.processed_panel = SyncedImageViewerPanel("✨ Processed")
        layout.addWidget(self.processed_panel, 1)
        
        # Set dark background
        self.setStyleSheet("""
            QWidget#imageViewer {
                background-color: #1e1e1e;
                border-radius: 10px;
            }
        """)
        
    def _connectSignals(self):
        """Setup synchronization between panels"""
        # Original -> Processed sync
        self.original_panel.viewTransformChanged.connect(
            self.processed_panel.syncTransform
        )
        self.original_panel.scrollBarsChanged.connect(
            self.processed_panel.syncScrollBars
        )
        
        # Processed -> Original sync
        self.processed_panel.viewTransformChanged.connect(
            self.original_panel.syncTransform
        )
        self.processed_panel.scrollBarsChanged.connect(
            self.original_panel.syncScrollBars
        )
        
        # Forward click signal from original panel
        self.original_panel.imageClicked.connect(self.originalImageClicked.emit)
        
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
        
    def fitToView(self):
        """Fit both images to their views"""
        self.original_panel.view.fitToView()
        self.processed_panel.view.fitToView()
        
    def resetZoom(self):
        """Reset zoom on both panels"""
        self.original_panel.view.resetZoom()
        self.processed_panel.view.resetZoom()
