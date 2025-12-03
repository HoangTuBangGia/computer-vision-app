"""
Main Window - CV Master Application
"""
import cv2
import numpy as np
from pathlib import Path

from PySide6.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QSplitter, QFileDialog, QMessageBox, QStatusBar, QProgressBar, QLabel
)
from PySide6.QtCore import Qt
from PySide6.QtGui import QPixmap, QImage, QAction

from .zoomable_viewer import SyncedImageViewer
from .control_panel import ControlPanel

# Import core processing functions
from core import point as point_ops
from core import filters as filter_ops
from core import morphology as morph_ops
from core import frequency as freq_ops
from core import segmentation as seg_ops
from core import compression as compress_ops
from core import geometry as geom_ops
from core import restoration as restore_ops
from core.pca import PCAFaceRecognizer
from core.worker import Worker, WorkerManager, create_kmeans_task, create_fft_task, create_compression_task


class MainWindow(QMainWindow):
    """
    Main Application Window for CV Master
    """
    
    SUPPORTED_FORMATS = "Images (*.png *.jpg *.jpeg *.bmp *.tiff *.webp);;All Files (*)"
    
    def __init__(self):
        super().__init__()
        
        # Image data storage
        self.original_image = None  # OpenCV image (BGR)
        self.processed_image = None  # OpenCV image (BGR)
        self.working_image = None  # Image after noise (for filters to work on)
        self.current_file_path = None
        
        # PCA Face Recognizer
        self.pca_recognizer = None
        
        # Worker Manager for threading
        self.worker_manager = WorkerManager()
        
        self._setupWindow()
        self._setupUI()
        self._setupMenuBar()
        self._setupStatusBar()
        self._connectSignals()
        
    def _setupWindow(self):
        """Setup window properties"""
        self.setWindowTitle("CV Master - Computer Vision Application")
        self.setMinimumSize(1000, 700)
        self.resize(1200, 800)
        
        # Set window style
        self.setStyleSheet("""
            QMainWindow {
                background-color: #1e1e1e;
            }
            QSplitter::handle {
                background-color: #3d3d3d;
                width: 3px;
            }
            QSplitter::handle:hover {
                background-color: #0078d4;
            }
        """)
        
    def _setupUI(self):
        """Setup the main UI layout"""
        # Central widget
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        # Main layout
        main_layout = QHBoxLayout(central_widget)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)
        
        # Create splitter for resizable panels
        self.splitter = QSplitter(Qt.Orientation.Horizontal)
        self.splitter.setChildrenCollapsible(False)
        
        # Left panel - Controls
        self.control_panel = ControlPanel()
        self.splitter.addWidget(self.control_panel)
        
        # Right panel - Zoomable Image viewer (synced)
        self.image_viewer = SyncedImageViewer()
        self.splitter.addWidget(self.image_viewer)
        
        # Set initial splitter sizes (300px for control panel, rest for viewer)
        self.splitter.setSizes([300, 900])
        
        main_layout.addWidget(self.splitter)
        
    def _setupMenuBar(self):
        """Setup the menu bar"""
        menubar = self.menuBar()
        menubar.setStyleSheet("""
            QMenuBar {
                background-color: #252525;
                color: #cccccc;
                padding: 5px;
            }
            QMenuBar::item:selected {
                background-color: #404040;
            }
            QMenu {
                background-color: #2b2b2b;
                color: #cccccc;
                border: 1px solid #404040;
            }
            QMenu::item:selected {
                background-color: #0078d4;
            }
        """)
        
        # File menu
        file_menu = menubar.addMenu("&File")
        
        open_action = QAction("&Open Image", self)
        open_action.setShortcut("Ctrl+O")
        open_action.triggered.connect(self._loadImage)
        file_menu.addAction(open_action)
        
        save_action = QAction("&Save Image", self)
        save_action.setShortcut("Ctrl+S")
        save_action.triggered.connect(self._saveImage)
        file_menu.addAction(save_action)
        
        file_menu.addSeparator()
        
        exit_action = QAction("E&xit", self)
        exit_action.setShortcut("Ctrl+Q")
        exit_action.triggered.connect(self.close)
        file_menu.addAction(exit_action)
        
        # Edit menu
        edit_menu = menubar.addMenu("&Edit")
        
        reset_action = QAction("&Reset Image", self)
        reset_action.setShortcut("Ctrl+R")
        reset_action.triggered.connect(self._resetImage)
        edit_menu.addAction(reset_action)
        
        # Help menu
        help_menu = menubar.addMenu("&Help")
        
        about_action = QAction("&About", self)
        about_action.triggered.connect(self._showAbout)
        help_menu.addAction(about_action)
        
    def _setupStatusBar(self):
        """Setup the status bar with progress bar"""
        self.status_bar = QStatusBar()
        self.status_bar.setStyleSheet("""
            QStatusBar {
                background-color: #252525;
                color: #888888;
                border-top: 1px solid #3d3d3d;
                padding: 5px;
            }
        """)
        self.setStatusBar(self.status_bar)
        
        # Progress bar (hidden by default)
        self.progress_bar = QProgressBar()
        self.progress_bar.setFixedWidth(200)
        self.progress_bar.setMaximumHeight(16)
        self.progress_bar.setTextVisible(True)
        self.progress_bar.setStyleSheet("""
            QProgressBar {
                background-color: #3d3d3d;
                border: none;
                border-radius: 8px;
                text-align: center;
                color: #ffffff;
                font-size: 10px;
            }
            QProgressBar::chunk {
                background-color: #0078d4;
                border-radius: 8px;
            }
        """)
        self.progress_bar.hide()
        self.status_bar.addPermanentWidget(self.progress_bar)
        
        # Status message label
        self.status_label = QLabel("Ready - Load an image to get started")
        self.status_label.setStyleSheet("color: #888888;")
        self.status_bar.addWidget(self.status_label, 1)
        
    def _connectSignals(self):
        """Connect signals and slots"""
        self.control_panel.loadImageClicked.connect(self._loadImage)
        self.control_panel.saveImageClicked.connect(self._saveImage)
        self.control_panel.basicProcessRequested.connect(self._processBasicOperation)
        self.control_panel.addNoiseRequested.connect(self._addNoise)
        self.control_panel.applyFilterRequested.connect(self._applyFilter)
        self.control_panel.morphologyRequested.connect(self._applyMorphology)
        self.control_panel.showSpectrumRequested.connect(self._showSpectrum)
        self.control_panel.frequencyFilterRequested.connect(self._applyFrequencyFilter)
        self.control_panel.showFilteredSpectrumRequested.connect(self._showFilteredSpectrum)
        self.control_panel.otsuRequested.connect(self._applyOtsu)
        self.control_panel.manualThresholdRequested.connect(self._applyManualThreshold)
        self.control_panel.kmeansRequested.connect(self._applyKmeans)
        self.control_panel.loadFaceDatasetRequested.connect(self._loadFaceDataset)
        self.control_panel.reconstructFaceRequested.connect(self._reconstructFace)
        self.control_panel.compressFullImageRequested.connect(self._compressFullImage)
        self.control_panel.compressionQualityChanged.connect(self._onCompressionQualityChanged)
        
        # Geometry signals
        self.control_panel.rotateRequested.connect(self._rotateImage)
        self.control_panel.scaleRequested.connect(self._scaleImage)
        self.control_panel.resizeRequested.connect(self._resizeImage)
        self.control_panel.flipRequested.connect(self._flipImage)
        self.control_panel.geometryResetRequested.connect(self._resetImage)
        
        # Restoration signals
        self.control_panel.restorationRequested.connect(self._applyRestoration)
        
        # Image viewer click for compression block analysis
        self.image_viewer.originalImageClicked.connect(self._onOriginalImageClicked)
        
    def _loadImage(self):
        """Load an image from file"""
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Open Image",
            str(Path.home()),
            self.SUPPORTED_FORMATS
        )
        
        if file_path:
            self._openImage(file_path)
            
    def _openImage(self, file_path: str):
        """Open and display an image"""
        try:
            # Read image with OpenCV (returns BGR)
            image = cv2.imread(file_path)
            
            if image is None:
                raise ValueError(f"Cannot read image: {file_path}")
                
            # Store original image
            self.original_image = image.copy()
            self.processed_image = image.copy()
            self.working_image = image.copy()  # For filters
            self.current_file_path = file_path
            
            # Convert BGR to RGB for display
            rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            # Convert to QPixmap
            pixmap = self._cvImageToQPixmap(rgb_image)
            
            # Display in viewer
            self.image_viewer.setOriginalImage(pixmap)
            self.image_viewer.setProcessedImage(pixmap)
            
            # Update histogram
            self._updateHistogram(image)
            
            # Enable save button
            self.control_panel.enableSaveButton(True)
            
            # Update status bar
            h, w = image.shape[:2]
            channels = image.shape[2] if len(image.shape) > 2 else 1
            file_name = Path(file_path).name
            self._showStatus(
                f"Loaded: {file_name} | Size: {w}x{h} | Channels: {channels}"
            )
            
            # Update geometry tab with image size
            self.control_panel.getGeometryTab().setImageSize(w, h)
            
            # Apply current operation if any
            op, params = self.control_panel.getBasicTab().getCurrentOperation()
            if op != "original":
                self._processBasicOperation(op, params)
            
        except Exception as e:
            QMessageBox.critical(
                self,
                "Error",
                f"Failed to load image:\n{str(e)}"
            )
            
    def _cvImageToQPixmap(self, cv_image: np.ndarray) -> QPixmap:
        """Convert OpenCV image (RGB) to QPixmap"""
        height, width = cv_image.shape[:2]
        
        if len(cv_image.shape) == 3:
            # Color image
            bytes_per_line = 3 * width
            q_image = QImage(
                cv_image.data,
                width,
                height,
                bytes_per_line,
                QImage.Format.Format_RGB888
            )
        else:
            # Grayscale image
            bytes_per_line = width
            q_image = QImage(
                cv_image.data,
                width,
                height,
                bytes_per_line,
                QImage.Format.Format_Grayscale8
            )
            
        return QPixmap.fromImage(q_image)
        
    def _saveImage(self):
        """Save the processed image"""
        if self.processed_image is None:
            QMessageBox.warning(
                self,
                "Warning",
                "No image to save. Please load an image first."
            )
            return
            
        # Get save file path
        default_name = "processed_image.png"
        if self.current_file_path:
            original_path = Path(self.current_file_path)
            default_name = f"{original_path.stem}_processed{original_path.suffix}"
            
        file_path, _ = QFileDialog.getSaveFileName(
            self,
            "Save Image",
            str(Path.home() / default_name),
            self.SUPPORTED_FORMATS
        )
        
        if file_path:
            try:
                cv2.imwrite(file_path, self.processed_image)
                self._showStatus(f"Saved: {Path(file_path).name}")
                QMessageBox.information(
                    self,
                    "Success",
                    f"Image saved successfully:\n{file_path}"
                )
            except Exception as e:
                QMessageBox.critical(
                    self,
                    "Error",
                    f"Failed to save image:\n{str(e)}"
                )
                
    def _resetImage(self):
        """Reset processed image to original"""
        if self.original_image is not None:
            self.processed_image = self.original_image.copy()
            rgb_image = cv2.cvtColor(self.processed_image, cv2.COLOR_BGR2RGB)
            pixmap = self._cvImageToQPixmap(rgb_image)
            self.image_viewer.setProcessedImage(pixmap)
            self._updateHistogram(self.processed_image)
            self._showStatus("Image reset to original")
            
            # Reset geometry tab controls
            self.control_panel.getGeometryTab().reset()
            
            # Update geometry tab with image size
            h, w = self.original_image.shape[:2]
            self.control_panel.getGeometryTab().setImageSize(w, h)
            
    def _showStatus(self, message: str):
        """Show message in status bar"""
        self.status_label.setText(message)
        
    def _showProgress(self, current: int, total: int):
        """Show/update progress bar"""
        self.progress_bar.show()
        self.progress_bar.setMaximum(total)
        self.progress_bar.setValue(current)
        
    def _hideProgress(self):
        """Hide progress bar"""
        self.progress_bar.hide()
        self.progress_bar.setValue(0)
            
    def _showAbout(self):
        """Show about dialog"""
        QMessageBox.about(
            self,
            "About CV Master",
            """<h2>🎨 CV Master</h2>
            <p><b>Version:</b> 1.0.0</p>
            <p>A professional computer vision application built with:</p>
            <ul>
                <li>Python 3.12</li>
                <li>PySide6</li>
                <li>OpenCV</li>
            </ul>
            <p>© 2024 CV Master Team</p>
            """
        )
        
    def updateProcessedImage(self, cv_image: np.ndarray):
        """
        Update the processed image display
        Call this method from processing functions
        
        Args:
            cv_image: OpenCV image in BGR format or grayscale
        """
        self.processed_image = cv_image.copy()
        
        # Convert to RGB for display
        if len(cv_image.shape) == 3:
            rgb_image = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)
        else:
            rgb_image = cv_image
            
        pixmap = self._cvImageToQPixmap(rgb_image)
        self.image_viewer.setProcessedImage(pixmap)
        
        # Update histogram for processed image
        self._updateHistogram(cv_image)
        
    def _processBasicOperation(self, operation: str, params: dict):
        """
        Process basic point operations
        
        Args:
            operation: Operation name
            params: Operation parameters
        """
        if self.original_image is None:
            return
            
        image = self.original_image.copy()
        
        try:
            if operation == "original":
                result = image
            elif operation == "grayscale":
                result = point_ops.to_grayscale(image)
            elif operation == "negative":
                result = point_ops.negative(image)
            elif operation == "log":
                c = params.get("log_c", 46.0)
                result = point_ops.log_transform(image, c=c)
            elif operation == "gamma":
                gamma = params.get("gamma", 1.0)
                result = point_ops.power_law_transform(image, gamma=gamma)
            elif operation == "equalize":
                result = point_ops.histogram_equalization(image)
            else:
                result = image
                
            # Update display
            self.updateProcessedImage(result)
            
            # Update status
            op_names = {
                "original": "Original",
                "grayscale": "Grayscale",
                "negative": "Negative",
                "log": f"Log Transform (c={params.get('log_c', 46.0):.1f})",
                "gamma": f"Gamma Correction (γ={params.get('gamma', 1.0):.2f})",
                "equalize": "Histogram Equalization",
            }
            self._showStatus(f"Applied: {op_names.get(operation, operation)}")
            
        except Exception as e:
            self._showStatus(f"Error: {str(e)}")
            
    def _addNoise(self, noise_type: str, params: dict):
        """
        Add noise to the current image
        
        Args:
            noise_type: Type of noise ("gaussian" or "salt_pepper")
            params: Noise parameters
        """
        if self.original_image is None:
            return
            
        try:
            # Add noise to original image
            noisy_image = filter_ops.add_noise(self.original_image, noise_type, **params)
            
            # Store as working image for filters
            self.working_image = noisy_image.copy()
            
            # Update display
            self.updateProcessedImage(noisy_image)
            
            # Update status
            noise_names = {
                "gaussian": f"Gaussian Noise (σ={params.get('sigma', 25)})",
                "salt_pepper": f"Salt & Pepper Noise ({params.get('salt_prob', 0.02)*100:.1f}%)",
            }
            self._showStatus(f"Added: {noise_names.get(noise_type, noise_type)}")
            
        except Exception as e:
            self._showStatus(f"Error adding noise: {str(e)}")
            
    def _applyFilter(self, filter_type: str, params: dict):
        """
        Apply filter to the working image
        
        Args:
            filter_type: Type of filter
            params: Filter parameters
        """
        if self.working_image is None:
            if self.original_image is not None:
                self.working_image = self.original_image.copy()
            else:
                return
                
        try:
            if filter_type == "none":
                result = self.working_image.copy()
            else:
                kernel_size = params.get("kernel_size", 3)
                strength = params.get("strength", 1.0)
                
                result = filter_ops.apply_filter(
                    self.working_image, 
                    filter_type, 
                    kernel_size=kernel_size,
                    strength=strength
                )
            
            # Update display
            self.updateProcessedImage(result)
            
            # Update status
            filter_names = {
                "none": "No Filter",
                "mean": f"Mean Filter ({params.get('kernel_size', 3)}x{params.get('kernel_size', 3)})",
                "gaussian": f"Gaussian Blur ({params.get('kernel_size', 3)}x{params.get('kernel_size', 3)})",
                "median": f"Median Filter ({params.get('kernel_size', 3)}x{params.get('kernel_size', 3)})",
                "laplacian": f"Laplacian Sharpen (strength={params.get('strength', 1.0):.1f})",
                "unsharp": f"Unsharp Mask ({params.get('kernel_size', 3)}x{params.get('kernel_size', 3)})",
                "bilateral": f"Bilateral Filter (d={params.get('kernel_size', 9)})",
            }
            self._showStatus(f"Applied: {filter_names.get(filter_type, filter_type)}")
            
        except Exception as e:
            self._showStatus(f"Error applying filter: {str(e)}")
            
    def _updateHistogram(self, image: np.ndarray):
        """
        Update histogram display
        
        Args:
            image: OpenCV image (BGR or grayscale)
        """
        is_color = len(image.shape) == 3
        histogram_data = point_ops.calculate_histogram(image)
        self.control_panel.updateHistogram(histogram_data, is_color)
        
    def _applyMorphology(self, operation: str, params: dict):
        """
        Apply morphological operation
        
        Args:
            operation: Operation name (erosion, dilation, opening, closing, boundary, gradient)
            params: Operation parameters (shape, kernel_size, iterations, auto_binary)
        """
        if self.original_image is None:
            self._showStatus("No image loaded")
            return
            
        try:
            # Use processed image or original
            source_image = self.processed_image if self.processed_image is not None else self.original_image
            
            # Get parameters
            shape = params.get("shape", "rect")
            kernel_size = params.get("kernel_size", 3)
            iterations = params.get("iterations", 1)
            auto_binary = params.get("auto_binary", True)
            
            # Apply operation
            result = morph_ops.apply_morphology(
                source_image,
                operation=operation,
                kernel_size=kernel_size,
                shape=shape,
                auto_binary=auto_binary,
                iterations=iterations
            )
            
            # Update display
            self.updateProcessedImage(result)
            
            # Update status
            op_names = {
                "erosion": f"Erosion ({kernel_size}x{kernel_size}, iter={iterations})",
                "dilation": f"Dilation ({kernel_size}x{kernel_size}, iter={iterations})",
                "opening": f"Opening ({kernel_size}x{kernel_size})",
                "closing": f"Closing ({kernel_size}x{kernel_size})",
                "boundary": f"Boundary Extraction ({kernel_size}x{kernel_size})",
                "skeleton": "Skeleton (Medial Axis)",
                "gradient": f"Morphological Gradient ({kernel_size}x{kernel_size})",
            }
            self._showStatus(f"Applied: {op_names.get(operation, operation)}")
            
        except Exception as e:
            self._showStatus(f"Error applying morphology: {str(e)}")

    def _showSpectrum(self):
        """
        Show magnitude spectrum of the current image
        """
        if self.original_image is None:
            self._showStatus("No image loaded")
            return
            
        try:
            # Compute FFT
            fft_shift, magnitude_spectrum = freq_ops.compute_fft(self.original_image)
            
            # Store FFT for later filtering
            self._current_fft = fft_shift
            
            # Display magnitude spectrum
            self.updateProcessedImage(magnitude_spectrum)
            self._showStatus("Showing: Magnitude Spectrum (log scale)")
            
        except Exception as e:
            self._showStatus(f"Error computing FFT: {str(e)}")
            
    def _applyFrequencyFilter(self, filter_type: str, params: dict):
        """
        Apply frequency domain filter
        
        Args:
            filter_type: Type of frequency filter
            params: Filter parameters (d0, order)
        """
        if self.original_image is None:
            self._showStatus("No image loaded")
            return
            
        try:
            d0 = params.get("d0", 30)
            order = params.get("order", 2)
            
            # Apply filter
            filtered_image, filter_mask, filtered_spectrum = freq_ops.apply_frequency_filter(
                self.original_image,
                filter_type,
                d0=d0,
                order=order
            )
            
            # Store results for later viewing
            self._filtered_spectrum = filtered_spectrum
            self._filter_mask = filter_mask
            
            # Display filtered result
            self.updateProcessedImage(filtered_image)
            
            # Update status
            filter_names = {
                "ideal_lowpass": f"Ideal Lowpass (D₀={d0})",
                "ideal_highpass": f"Ideal Highpass (D₀={d0})",
                "gaussian_lowpass": f"Gaussian Lowpass (D₀={d0})",
                "gaussian_highpass": f"Gaussian Highpass (D₀={d0})",
                "butterworth_lowpass": f"Butterworth Lowpass (D₀={d0}, n={order})",
                "butterworth_highpass": f"Butterworth Highpass (D₀={d0}, n={order})",
            }
            self._showStatus(f"Applied: {filter_names.get(filter_type, filter_type)}")
            
        except Exception as e:
            self._showStatus(f"Error applying frequency filter: {str(e)}")
            
    def _showFilteredSpectrum(self):
        """
        Show the filtered spectrum
        """
        if not hasattr(self, '_filtered_spectrum') or self._filtered_spectrum is None:
            self._showStatus("No filtered spectrum available. Apply a filter first.")
            return
            
        try:
            self.updateProcessedImage(self._filtered_spectrum)
            self._showStatus("Showing: Filtered Magnitude Spectrum")
            
        except Exception as e:
            self._showStatus(f"Error showing filtered spectrum: {str(e)}")

    def _applyOtsu(self):
        """
        Apply Otsu's automatic thresholding
        """
        if self.original_image is None:
            self._showStatus("No image loaded")
            return
            
        try:
            # Apply Otsu thresholding
            binary, threshold = seg_ops.otsu_threshold(self.original_image)
            
            # Update display
            self.updateProcessedImage(binary)
            
            # Update the threshold value in UI
            self.control_panel.getSegmentationTab().updateOtsuResult(threshold)
            
            self._showStatus(f"Applied: Otsu Thresholding (threshold={threshold:.0f})")
            
        except Exception as e:
            self._showStatus(f"Error applying Otsu: {str(e)}")
            
    def _applyManualThreshold(self, threshold: int):
        """
        Apply manual thresholding
        
        Args:
            threshold: Threshold value (0-255)
        """
        if self.original_image is None:
            self._showStatus("No image loaded")
            return
            
        try:
            # Apply manual threshold
            result = seg_ops.manual_threshold(self.original_image, threshold)
            
            # Update display
            self.updateProcessedImage(result)
            
            self._showStatus(f"Applied: Manual Threshold ({threshold})")
            
        except Exception as e:
            self._showStatus(f"Error applying threshold: {str(e)}")
            
    def _applyKmeans(self, params: dict):
        """
        Apply K-Means clustering segmentation
        
        Args:
            params: Parameters dict with k and use_vibrant
        """
        if self.original_image is None:
            self._showStatus("No image loaded")
            return
            
        try:
            k = params.get("k", 4)
            use_vibrant = params.get("use_vibrant", True)
            
            # Apply K-Means
            segmented, labels, compactness = seg_ops.kmeans_with_custom_colors(
                self.original_image,
                k=k,
                use_vibrant_colors=use_vibrant
            )
            
            # Update display
            self.updateProcessedImage(segmented)
            
            # Update the result in UI
            self.control_panel.getSegmentationTab().updateKmeansResult(k, compactness)
            
            self._showStatus(f"Applied: K-Means Clustering (K={k})")
            
        except Exception as e:
            self._showStatus(f"Error applying K-Means: {str(e)}")

    def _loadFaceDataset(self):
        """
        Load face dataset for PCA Eigenfaces
        """
        # Open folder dialog
        folder_path = QFileDialog.getExistingDirectory(
            self,
            "Select Face Dataset Folder",
            str(Path.home()),
            QFileDialog.Option.ShowDirsOnly
        )
        
        if not folder_path:
            return
            
        try:
            # Create PCA recognizer
            self.pca_recognizer = PCAFaceRecognizer(image_size=(64, 64))
            
            # Load dataset
            n_faces, labels = self.pca_recognizer.load_dataset(folder_path)
            
            if n_faces < 2:
                QMessageBox.warning(
                    self,
                    "Warning",
                    "Need at least 2 face images for PCA."
                )
                return
                
            # Fit PCA
            self.pca_recognizer.fit()
            
            # Get max components
            max_components = min(n_faces, 64 * 64)
            
            # Update UI
            pca_tab = self.control_panel.getPCATab()
            pca_tab.updateDatasetInfo(n_faces, labels, max_components)
            
            # Display mean face
            mean_face = self.pca_recognizer.get_mean_face()
            pca_tab.setMeanFace(mean_face)
            
            # Display first 3 eigenfaces
            eigenfaces = []
            for i in range(min(3, n_faces - 1)):
                eigenfaces.append(self.pca_recognizer.get_eigenface(i))
            pca_tab.setEigenfaces(eigenfaces)
            
            # Display first face
            if n_faces > 0:
                first_face = self.pca_recognizer.get_face_by_index(0)
                pca_tab.setOriginalFace(first_face)
                
                # Initial reconstruction
                self._reconstructFace(0, min(10, max_components))
                
            self._showStatus(f"Loaded {n_faces} faces from {Path(folder_path).name}")
            
        except Exception as e:
            QMessageBox.critical(
                self,
                "Error",
                f"Failed to load face dataset:\n{str(e)}"
            )
            self._showStatus(f"Error: {str(e)}")
            
    def _reconstructFace(self, face_index: int, n_components: int):
        """
        Reconstruct a face using PCA
        
        Args:
            face_index: Index of face in dataset
            n_components: Number of principal components to use
        """
        if self.pca_recognizer is None or not self.pca_recognizer.is_fitted:
            self._showStatus("No PCA model fitted. Load a dataset first.")
            return
            
        try:
            # Get original face
            original = self.pca_recognizer.get_face_by_index(face_index)
            
            # Reconstruct
            reconstructed = self.pca_recognizer.reconstruct(original, n_components)
            
            # Get MSE
            mse = self.pca_recognizer.get_reconstruction_error(original, n_components)
            
            # Get variance explained
            cumulative_var = self.pca_recognizer.get_cumulative_variance()
            if n_components <= len(cumulative_var):
                variance = cumulative_var[n_components - 1]
            else:
                variance = 1.0
                
            # Update UI
            pca_tab = self.control_panel.getPCATab()
            pca_tab.setOriginalFace(original)
            pca_tab.setReconstructedFace(reconstructed)
            pca_tab.updateVariance(variance)
            pca_tab.updateMSE(mse)
            
            # Also update main viewer
            # Convert to larger image for main display
            display_size = (256, 256)
            original_large = cv2.resize(original, display_size)
            reconstructed_large = cv2.resize(reconstructed, display_size)
            
            # Show original in original panel
            original_rgb = cv2.cvtColor(
                cv2.cvtColor(original_large, cv2.COLOR_GRAY2BGR),
                cv2.COLOR_BGR2RGB
            )
            self.image_viewer.setOriginalImage(self._cvImageToQPixmap(original_rgb))
            
            # Show reconstructed in processed panel
            reconstructed_rgb = cv2.cvtColor(
                cv2.cvtColor(reconstructed_large, cv2.COLOR_GRAY2BGR),
                cv2.COLOR_BGR2RGB
            )
            self.image_viewer.setProcessedImage(self._cvImageToQPixmap(reconstructed_rgb))
            
            self._showStatus(
                f"Reconstructed face {face_index + 1} with K={n_components} "
                f"(Variance: {variance*100:.1f}%, MSE: {mse:.1f})"
            )
            
        except Exception as e:
            self._showStatus(f"Error reconstructing face: {str(e)}")

    # ================== Compression Handlers ==================
    
    def _onOriginalImageClicked(self, x: int, y: int):
        """
        Handle click on original image for compression block analysis
        
        Args:
            x: X coordinate (column) in original image
            y: Y coordinate (row) in original image
        """
        if self.original_image is None:
            return
            
        # Only process if Compression tab is active
        current_tab_index = self.control_panel.tab_widget.currentIndex()
        tab_text = self.control_panel.tab_widget.tabText(current_tab_index)
        
        if "Compress" not in tab_text:
            return
            
        self._analyzeCompressionBlock(x, y)
        
    def _analyzeCompressionBlock(self, x: int, y: int):
        """
        Analyze 8x8 block at clicked position
        
        Args:
            x: X coordinate in image
            y: Y coordinate in image
        """
        if self.original_image is None:
            return
            
        try:
            compress_tab = self.control_panel.getCompressionTab()
            quality = compress_tab.get_quality()
            
            # Get block position
            block_x, block_y = compress_ops.get_block_position(x, y)
            
            # Update block info display
            compress_tab.update_block_info(x, y, block_x, block_y)
            
            # Extract block
            block = compress_ops.extract_block(self.original_image, block_x, block_y)
            
            # Process block
            dct, quantized, zigzag, reconstructed, error = compress_ops.process_block(
                block, quality
            )
            
            # Display in UI
            compress_tab.display_original_block(block)
            compress_tab.display_dct_block(dct)
            compress_tab.display_quantized_block(quantized)
            compress_tab.display_zigzag(zigzag)
            
            # Update block statistics
            zeros_count = compress_ops.count_zeros(quantized)
            block_mse = np.mean(error ** 2)
            compress_tab.update_block_stats(zeros_count, block_mse)
            
            self._showStatus(
                f"Block at ({block_x}, {block_y}): "
                f"Quality={quality}, Zeros={zeros_count}/64, MSE={block_mse:.2f}"
            )
            
        except Exception as e:
            self._showStatus(f"Error analyzing block: {str(e)}")
            
    def _onCompressionQualityChanged(self, quality: int):
        """
        Handle compression quality slider change - re-analyze current block
        
        Args:
            quality: New quality value
        """
        # This will trigger re-analysis if there's a selected block
        # For now, we store the last clicked position and re-analyze
        pass
        
    def _compressFullImage(self, quality: int):
        """
        Compress the full image and display result
        
        Args:
            quality: JPEG quality (1-100)
        """
        if self.original_image is None:
            QMessageBox.warning(
                self,
                "Warning",
                "Please load an image first!"
            )
            return
            
        try:
            # Compress image
            compressed = compress_ops.compress_image(self.original_image, quality)
            
            # Calculate statistics
            stats = compress_ops.get_compression_stats(self.original_image, compressed)
            
            # Update UI
            compress_tab = self.control_panel.getCompressionTab()
            compress_tab.update_image_stats(stats['psnr'], stats['ssim'])
            
            # Store as processed image
            self.processed_image = compressed
            
            # Display
            # Convert to 3-channel for display if needed
            if len(compressed.shape) == 2:
                display_image = cv2.cvtColor(compressed, cv2.COLOR_GRAY2BGR)
            else:
                display_image = compressed
                
            rgb_image = cv2.cvtColor(display_image, cv2.COLOR_BGR2RGB)
            pixmap = self._cvImageToQPixmap(rgb_image)
            self.image_viewer.setProcessedImage(pixmap)
            
            self._showStatus(
                f"Compressed image: Quality={quality}, "
                f"PSNR={stats['psnr']:.2f} dB, SSIM={stats['ssim']:.4f}"
            )
            
            # Enable save button
            self.control_panel.enableSaveButton(True)
            
        except Exception as e:
            QMessageBox.critical(
                self,
                "Error",
                f"Failed to compress image:\n{str(e)}"
            )
            self._showStatus(f"Error: {str(e)}")

    # ================== Geometry Handlers ==================
    
    def _rotateImage(self, angle: float, keep_size: bool):
        """
        Rotate the image by given angle
        
        Args:
            angle: Rotation angle in degrees
            keep_size: Whether to keep original size
        """
        if self.original_image is None:
            QMessageBox.warning(self, "Warning", "Please load an image first!")
            return
            
        try:
            if keep_size:
                result = geom_ops.rotate_image_keep_size(self.original_image, angle)
            else:
                result = geom_ops.rotate_image(self.original_image, angle)
                
            self.processed_image = result
            self._displayProcessedImage(result)
            self._showStatus(f"Rotated: {angle}°")
            
        except Exception as e:
            self._showStatus(f"Error rotating: {str(e)}")
            
    def _scaleImage(self, scale_x: float, scale_y: float):
        """
        Scale the image by given factors
        
        Args:
            scale_x: Horizontal scale factor
            scale_y: Vertical scale factor
        """
        if self.original_image is None:
            QMessageBox.warning(self, "Warning", "Please load an image first!")
            return
            
        try:
            result = geom_ops.scale_image(self.original_image, scale_x, scale_y)
            self.processed_image = result
            self._displayProcessedImage(result)
            self._showStatus(f"Scaled: {scale_x:.2f}x, {scale_y:.2f}x")
            
        except Exception as e:
            self._showStatus(f"Error scaling: {str(e)}")
            
    def _resizeImage(self, width: int, height: int):
        """
        Resize the image to specific dimensions
        
        Args:
            width: Target width
            height: Target height
        """
        if self.original_image is None:
            QMessageBox.warning(self, "Warning", "Please load an image first!")
            return
            
        try:
            result = geom_ops.resize_image(self.original_image, width, height)
            self.processed_image = result
            self._displayProcessedImage(result)
            self._showStatus(f"Resized to: {width}x{height}")
            
        except Exception as e:
            self._showStatus(f"Error resizing: {str(e)}")
            
    def _flipImage(self, direction: str):
        """
        Flip the image
        
        Args:
            direction: 'horizontal', 'vertical', or 'both'
        """
        if self.original_image is None:
            QMessageBox.warning(self, "Warning", "Please load an image first!")
            return
            
        try:
            if direction == 'horizontal':
                result = geom_ops.flip_horizontal(self.original_image)
            elif direction == 'vertical':
                result = geom_ops.flip_vertical(self.original_image)
            else:  # both
                result = geom_ops.flip_both(self.original_image)
                
            self.processed_image = result
            self._displayProcessedImage(result)
            self._showStatus(f"Flipped: {direction}")
            
        except Exception as e:
            self._showStatus(f"Error flipping: {str(e)}")
            
    def _displayProcessedImage(self, cv_image: np.ndarray):
        """
        Display processed image in the viewer
        
        Args:
            cv_image: OpenCV image (BGR or grayscale)
        """
        if len(cv_image.shape) == 2:
            # Grayscale
            rgb_image = cv2.cvtColor(cv_image, cv2.COLOR_GRAY2RGB)
        else:
            rgb_image = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)
            
        pixmap = self._cvImageToQPixmap(rgb_image)
        self.image_viewer.setProcessedImage(pixmap)
        self.control_panel.enableSaveButton(True)

    # ============================================================
    # RESTORATION OPERATIONS
    # ============================================================
    
    def _applyRestoration(self, operation: str, params: dict):
        """
        Apply restoration operations
        
        Args:
            operation: Type of operation (add_uniform_noise, arithmetic_mean_filter, etc.)
            params: Operation parameters
        """
        if self.original_image is None:
            QMessageBox.warning(self, "Warning", "Please load an image first!")
            return
            
        try:
            source = self.working_image if self.working_image is not None else self.original_image
            result = None
            
            # Noise operations
            if operation == "add_uniform_noise":
                result = restore_ops.add_uniform_noise(
                    source, 
                    a=params.get("param_a", -50), 
                    b=params.get("param_b", 50)
                )
                self.working_image = result.copy()
                self._showStatus(f"Added Uniform noise (a={params.get('param_a')}, b={params.get('param_b')})")
                
            elif operation == "add_rayleigh_noise":
                result = restore_ops.add_rayleigh_noise(
                    source,
                    a=params.get("param_a", 0),
                    b=params.get("param_b", 50)
                )
                self.working_image = result.copy()
                self._showStatus(f"Added Rayleigh noise")
                
            elif operation == "add_exponential_noise":
                result = restore_ops.add_exponential_noise(
                    source,
                    scale=abs(params.get("param_b", 25))
                )
                self.working_image = result.copy()
                self._showStatus(f"Added Exponential noise")
                
            # Mean filters
            elif operation == "arithmetic_mean_filter":
                kernel_size = params.get("kernel_size", 3)
                result = restore_ops.arithmetic_mean_filter(source, kernel_size)
                self._showStatus(f"Applied Arithmetic Mean ({kernel_size}x{kernel_size})")
                
            elif operation == "geometric_mean_filter":
                kernel_size = params.get("kernel_size", 3)
                result = restore_ops.geometric_mean_filter(source, kernel_size)
                self._showStatus(f"Applied Geometric Mean ({kernel_size}x{kernel_size})")
                
            elif operation == "harmonic_mean_filter":
                kernel_size = params.get("kernel_size", 3)
                result = restore_ops.harmonic_mean_filter(source, kernel_size)
                self._showStatus(f"Applied Harmonic Mean ({kernel_size}x{kernel_size})")
                
            elif operation == "contraharmonic_mean_filter":
                kernel_size = params.get("kernel_size", 3)
                Q = params.get("q", 1.5)
                result = restore_ops.contraharmonic_mean_filter(source, kernel_size, Q)
                self._showStatus(f"Applied Contra-harmonic Mean (Q={Q})")
                
            # Order-statistics filters
            elif operation == "median_filter":
                kernel_size = params.get("kernel_size", 3)
                if kernel_size % 2 == 0:
                    kernel_size += 1
                result = cv2.medianBlur(source, kernel_size)
                self._showStatus(f"Applied Median Filter ({kernel_size}x{kernel_size})")
                
            elif operation == "max_filter":
                kernel_size = params.get("kernel_size", 3)
                result = restore_ops.max_filter(source, kernel_size)
                self._showStatus(f"Applied Max Filter ({kernel_size}x{kernel_size})")
                
            elif operation == "min_filter":
                kernel_size = params.get("kernel_size", 3)
                result = restore_ops.min_filter(source, kernel_size)
                self._showStatus(f"Applied Min Filter ({kernel_size}x{kernel_size})")
                
            elif operation == "midpoint_filter":
                kernel_size = params.get("kernel_size", 3)
                result = restore_ops.midpoint_filter(source, kernel_size)
                self._showStatus(f"Applied Midpoint Filter ({kernel_size}x{kernel_size})")
                
            elif operation == "alpha_trimmed_mean_filter":
                kernel_size = params.get("kernel_size", 5)
                d = params.get("d", 2)
                result = restore_ops.alpha_trimmed_mean_filter(source, kernel_size, d)
                self._showStatus(f"Applied Alpha-trimmed Mean (d={d})")
                
            # Adaptive filters
            elif operation == "adaptive_local_noise_reduction":
                kernel_size = params.get("kernel_size", 7)
                result = restore_ops.adaptive_local_noise_reduction(source, kernel_size)
                self._showStatus(f"Applied Adaptive Local Noise Reduction ({kernel_size}x{kernel_size})")
                
            elif operation == "adaptive_median_filter":
                kernel_size = params.get("kernel_size", 7)
                result = restore_ops.adaptive_median_filter(source, kernel_size)
                self._showStatus(f"Applied Adaptive Median Filter (max={kernel_size})")
                
            # Degradation models
            elif operation == "apply_motion_blur":
                length = params.get("length", 15)
                angle = params.get("angle", 0)
                result = restore_ops.apply_motion_blur(source, length, angle)
                self.working_image = result.copy()
                self._showStatus(f"Applied Motion blur (L={length}, θ={angle}°)")
                
            elif operation == "apply_atmospheric_blur":
                k = params.get("k", 0.001)
                result = restore_ops.apply_atmospheric_blur(source, k)
                self.working_image = result.copy()
                self._showStatus(f"Applied Atmospheric blur (k={k})")
                
            # Restoration
            elif operation == "restore_motion_blur":
                length = params.get("length", 15)
                angle = params.get("angle", 0)
                method = params.get("method", "wiener")
                K = params.get("K", 0.01)
                result = restore_ops.restore_motion_blur(source, length, angle, method, K)
                self._showStatus(f"Restored motion blur ({method})")
                
            elif operation == "restore_atmospheric_blur":
                k = params.get("k", 0.001)
                method = params.get("method", "wiener")
                K = params.get("K", 0.01)
                result = restore_ops.restore_atmospheric_blur(source, k, method, K)
                self._showStatus(f"Restored atmospheric blur ({method})")
                
            # Legacy operations (keep for compatibility)
            elif operation == "add_degradation":
                result = self._addDegradation(params)
            elif operation == "mean_filter":
                result = self._applyMeanFilter(source, params)
            elif operation == "order_filter":
                result = self._applyOrderFilter(source, params)
            elif operation == "adaptive_filter":
                result = self._applyAdaptiveFilter(source, params)
            elif operation == "restore_image":
                result = self._restoreImage(source, params)
            else:
                self._showStatus(f"Unknown operation: {operation}")
                return
                
            if result is not None:
                self.processed_image = result
                self._displayProcessedImage(result)
                
        except Exception as e:
            self._showStatus(f"Error: {str(e)}")
            
    def _addDegradation(self, params: dict) -> np.ndarray:
        """Add noise or blur degradation to image"""
        source = self.original_image.copy()
        deg_type = params.get("type", "gaussian")
        
        if deg_type == "gaussian":
            from core.filters import add_gaussian_noise
            result = add_gaussian_noise(
                source,
                mean=params.get("mean", 0),
                sigma=params.get("sigma", 25)
            )
            self._showStatus(f"Added Gaussian noise (σ={params.get('sigma', 25)})")
            
        elif deg_type == "salt_pepper":
            from core.filters import add_salt_pepper_noise
            result = add_salt_pepper_noise(
                source,
                salt_prob=params.get("salt_prob", 0.02),
                pepper_prob=params.get("pepper_prob", 0.02)
            )
            self._showStatus(f"Added Salt & Pepper noise")
            
        elif deg_type == "uniform":
            result = restore_ops.add_uniform_noise(
                source,
                a=params.get("a", -50),
                b=params.get("b", 50)
            )
            self._showStatus(f"Added Uniform noise")
            
        elif deg_type == "motion_blur":
            result = restore_ops.apply_motion_blur(
                source,
                length=params.get("length", 15),
                angle=params.get("angle", 0)
            )
            self._showStatus(f"Applied Motion blur (L={params.get('length', 15)}, θ={params.get('angle', 0)}°)")
            
        elif deg_type == "atmospheric":
            result = restore_ops.apply_atmospheric_blur(
                source,
                k=params.get("k", 0.001)
            )
            self._showStatus(f"Applied Atmospheric blur (k={params.get('k', 0.001)})")
        else:
            result = source
            
        # Store for filtering
        self.working_image = result.copy()
        return result
        
    def _applyMeanFilter(self, source: np.ndarray, params: dict) -> np.ndarray:
        """Apply mean-type filters"""
        filter_type = params.get("filter_type", "arithmetic")
        kernel_size = params.get("kernel_size", 3)
        
        if filter_type == "arithmetic":
            result = restore_ops.arithmetic_mean_filter(source, kernel_size)
            self._showStatus(f"Applied Arithmetic Mean ({kernel_size}x{kernel_size})")
            
        elif filter_type == "geometric":
            result = restore_ops.geometric_mean_filter(source, kernel_size)
            self._showStatus(f"Applied Geometric Mean ({kernel_size}x{kernel_size})")
            
        elif filter_type == "harmonic":
            result = restore_ops.harmonic_mean_filter(source, kernel_size)
            self._showStatus(f"Applied Harmonic Mean ({kernel_size}x{kernel_size})")
            
        elif filter_type == "contraharmonic":
            Q = params.get("Q", 1.5)
            result = restore_ops.contraharmonic_mean_filter(source, kernel_size, Q)
            self._showStatus(f"Applied Contra-harmonic Mean (Q={Q})")
        else:
            result = source
            
        return result
        
    def _applyOrderFilter(self, source: np.ndarray, params: dict) -> np.ndarray:
        """Apply order-statistics filters"""
        filter_type = params.get("filter_type", "median")
        kernel_size = params.get("kernel_size", 3)
        
        if filter_type == "median":
            result = cv2.medianBlur(source, kernel_size)
            self._showStatus(f"Applied Median Filter ({kernel_size}x{kernel_size})")
            
        elif filter_type == "max":
            result = restore_ops.max_filter(source, kernel_size)
            self._showStatus(f"Applied Max Filter ({kernel_size}x{kernel_size})")
            
        elif filter_type == "min":
            result = restore_ops.min_filter(source, kernel_size)
            self._showStatus(f"Applied Min Filter ({kernel_size}x{kernel_size})")
            
        elif filter_type == "midpoint":
            result = restore_ops.midpoint_filter(source, kernel_size)
            self._showStatus(f"Applied Midpoint Filter ({kernel_size}x{kernel_size})")
            
        elif filter_type == "alpha_trimmed":
            d = params.get("d", 4)
            result = restore_ops.alpha_trimmed_mean_filter(source, kernel_size, d)
            self._showStatus(f"Applied Alpha-trimmed Mean (d={d})")
        else:
            result = source
            
        return result
        
    def _applyAdaptiveFilter(self, source: np.ndarray, params: dict) -> np.ndarray:
        """Apply adaptive filters"""
        filter_type = params.get("filter_type", "local_noise")
        kernel_size = params.get("kernel_size", 7)
        
        if filter_type == "local_noise":
            result = restore_ops.adaptive_local_noise_reduction(source, kernel_size)
            self._showStatus(f"Applied Adaptive Local Noise Reduction ({kernel_size}x{kernel_size})")
            
        elif filter_type == "adaptive_median":
            result = restore_ops.adaptive_median_filter(source, kernel_size)
            self._showStatus(f"Applied Adaptive Median Filter (max={kernel_size})")
        else:
            result = source
            
        return result
        
    def _restoreImage(self, source: np.ndarray, params: dict) -> np.ndarray:
        """Apply deconvolution/restoration"""
        model = params.get("model", "motion")
        method = params.get("method", "wiener")
        
        if model == "motion":
            length = params.get("length", 15)
            angle = params.get("angle", 0)
            K = params.get("K", 0.01)
            
            result = restore_ops.restore_motion_blur(
                source, length, angle, method, K
            )
            self._showStatus(f"Restored motion blur ({method}, L={length}, θ={angle}°)")
            
        elif model == "atmospheric":
            k = params.get("k_atmos", 0.001)
            K = params.get("K", 0.01)
            
            result = restore_ops.restore_atmospheric_blur(
                source, k, method, K
            )
            self._showStatus(f"Restored atmospheric blur ({method}, k={k})")
        else:
            result = source
            
        return result
