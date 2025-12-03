"""
Worker Thread - Generic QThread for heavy operations
Supports progress reporting and cancellation
"""
from PySide6.QtCore import QThread, Signal, QMutex, QMutexLocker
from typing import Callable, Any, Optional
import traceback


class Worker(QThread):
    """
    Generic worker thread for running heavy operations
    
    Features:
    - Progress reporting
    - Cancellation support
    - Error handling
    - Result callback
    
    Usage:
        def heavy_task(progress_callback):
            for i in range(100):
                # Do work...
                progress_callback(i + 1, 100)  # Report progress
            return result
            
        worker = Worker(heavy_task)
        worker.progress.connect(update_progress_bar)
        worker.finished.connect(on_task_complete)
        worker.error.connect(on_error)
        worker.start()
    """
    
    # Signals
    started_work = Signal()  # Emitted when work starts
    progress = Signal(int, int)  # current, total
    progress_text = Signal(str)  # Status message
    finished_work = Signal(object)  # Result
    error = Signal(str, str)  # Error message, traceback
    
    def __init__(
        self,
        task: Callable,
        *args,
        task_name: str = "Processing",
        **kwargs
    ):
        """
        Initialize worker
        
        Args:
            task: Function to run. Should accept progress_callback as first arg
            *args: Arguments to pass to task
            task_name: Name for status messages
            **kwargs: Keyword arguments to pass to task
        """
        super().__init__()
        
        self._task = task
        self._args = args
        self._kwargs = kwargs
        self._task_name = task_name
        self._cancelled = False
        self._mutex = QMutex()
        self._result = None
        
    def run(self):
        """Run the task in thread"""
        self.started_work.emit()
        self.progress_text.emit(f"{self._task_name}...")
        
        try:
            # Create progress callback
            def progress_callback(current: int, total: int, message: str = None):
                if self._cancelled:
                    raise CancelledException("Task cancelled by user")
                self.progress.emit(current, total)
                if message:
                    self.progress_text.emit(message)
                    
            # Run the task with progress callback
            self._result = self._task(
                progress_callback,
                *self._args,
                **self._kwargs
            )
            
            if not self._cancelled:
                self.finished_work.emit(self._result)
                self.progress_text.emit(f"{self._task_name} completed")
                
        except CancelledException:
            self.progress_text.emit(f"{self._task_name} cancelled")
            
        except Exception as e:
            error_msg = str(e)
            tb = traceback.format_exc()
            self.error.emit(error_msg, tb)
            self.progress_text.emit(f"Error: {error_msg}")
            
    def cancel(self):
        """Request cancellation of the task"""
        with QMutexLocker(self._mutex):
            self._cancelled = True
            
    def is_cancelled(self) -> bool:
        """Check if cancellation was requested"""
        with QMutexLocker(self._mutex):
            return self._cancelled
            
    def get_result(self) -> Any:
        """Get the task result (after finished)"""
        return self._result


class CancelledException(Exception):
    """Exception raised when task is cancelled"""
    pass


class WorkerManager:
    """
    Manager for worker threads
    Ensures only one worker runs at a time
    """
    
    def __init__(self):
        self._current_worker: Optional[Worker] = None
        
    def run_task(
        self,
        task: Callable,
        *args,
        task_name: str = "Processing",
        on_progress: Callable[[int, int], None] = None,
        on_progress_text: Callable[[str], None] = None,
        on_finished: Callable[[Any], None] = None,
        on_error: Callable[[str, str], None] = None,
        **kwargs
    ) -> Worker:
        """
        Run a task in a worker thread
        
        Args:
            task: Function to run
            *args: Arguments for task
            task_name: Name for status messages
            on_progress: Progress callback (current, total)
            on_progress_text: Text status callback
            on_finished: Completion callback with result
            on_error: Error callback (message, traceback)
            **kwargs: Keyword arguments for task
            
        Returns:
            Worker instance
        """
        # Cancel any existing worker
        self.cancel_current()
        
        # Create new worker
        worker = Worker(task, *args, task_name=task_name, **kwargs)
        
        # Connect signals
        if on_progress:
            worker.progress.connect(on_progress)
        if on_progress_text:
            worker.progress_text.connect(on_progress_text)
        if on_finished:
            worker.finished_work.connect(on_finished)
        if on_error:
            worker.error.connect(on_error)
            
        # Store and start
        self._current_worker = worker
        worker.start()
        
        return worker
        
    def cancel_current(self):
        """Cancel current worker if running"""
        if self._current_worker and self._current_worker.isRunning():
            self._current_worker.cancel()
            self._current_worker.wait(3000)  # Wait up to 3 seconds
            
    def is_running(self) -> bool:
        """Check if a worker is currently running"""
        return self._current_worker is not None and self._current_worker.isRunning()


# ============================================================
# Task wrappers for common operations
# ============================================================

def create_kmeans_task(image, k: int, use_vibrant: bool):
    """
    Create a K-Means clustering task
    
    Returns:
        Task function for Worker
    """
    import cv2
    import numpy as np
    
    def task(progress_callback):
        progress_callback(0, 100, "Preparing data...")
        
        # Prepare data
        if len(image.shape) == 2:
            data = image.reshape((-1, 1)).astype(np.float32)
        else:
            data = image.reshape((-1, 3)).astype(np.float32)
            
        progress_callback(10, 100, "Running K-Means...")
        
        # K-Means criteria
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2)
        
        # Run K-Means
        _, labels, centers = cv2.kmeans(
            data, k, None, criteria, 10, cv2.KMEANS_PP_CENTERS
        )
        
        progress_callback(80, 100, "Creating output image...")
        
        # Create output
        if use_vibrant:
            # Generate vibrant colors
            np.random.seed(42)
            colors = []
            for i in range(k):
                hue = int(255 * i / k)
                color = np.array([[[hue, 255, 200]]], dtype=np.uint8)
                bgr = cv2.cvtColor(color, cv2.COLOR_HSV2BGR)[0, 0]
                colors.append(bgr)
            colors = np.array(colors, dtype=np.uint8)
        else:
            colors = centers.astype(np.uint8)
            
        result = colors[labels.flatten()].reshape(image.shape)
        
        progress_callback(100, 100, "K-Means completed")
        
        return result
        
    return task


def create_fft_task(image):
    """
    Create an FFT computation task
    
    Returns:
        Task function for Worker
    """
    import cv2
    import numpy as np
    
    def task(progress_callback):
        progress_callback(0, 100, "Converting to grayscale...")
        
        # Convert to grayscale if needed
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
            
        progress_callback(20, 100, "Computing FFT...")
        
        # Compute FFT
        f = np.fft.fft2(gray.astype(np.float32))
        fshift = np.fft.fftshift(f)
        
        progress_callback(60, 100, "Computing magnitude spectrum...")
        
        # Magnitude spectrum
        magnitude = np.abs(fshift)
        magnitude = np.log1p(magnitude)
        
        # Normalize for display
        magnitude = cv2.normalize(magnitude, None, 0, 255, cv2.NORM_MINMAX)
        spectrum = magnitude.astype(np.uint8)
        
        progress_callback(100, 100, "FFT completed")
        
        return {
            'spectrum': spectrum,
            'fshift': fshift,
            'shape': gray.shape
        }
        
    return task


def create_pca_fit_task(recognizer, images, labels):
    """
    Create a PCA fitting task
    
    Returns:
        Task function for Worker
    """
    def task(progress_callback):
        progress_callback(0, 100, "Loading images...")
        
        # Load dataset
        recognizer.faces = images
        recognizer.labels = labels
        recognizer.n_faces = len(images)
        
        progress_callback(20, 100, "Computing mean face...")
        
        # Compute mean
        data_matrix = recognizer._flatten_faces()
        recognizer.mean_face = np.mean(data_matrix, axis=0)
        
        progress_callback(40, 100, "Computing covariance...")
        
        # Center data
        centered = data_matrix - recognizer.mean_face
        
        progress_callback(50, 100, "Computing eigenfaces...")
        
        # PCA computation
        n_samples, n_features = centered.shape
        
        if n_samples < n_features:
            cov_matrix = centered @ centered.T / (n_samples - 1)
            eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)
            
            idx = np.argsort(eigenvalues)[::-1]
            eigenvalues = eigenvalues[idx]
            eigenvectors = eigenvectors[:, idx]
            
            recognizer.components = (centered.T @ eigenvectors).T
            
            for i in range(len(recognizer.components)):
                norm = np.linalg.norm(recognizer.components[i])
                if norm > 0:
                    recognizer.components[i] /= norm
        else:
            cov_matrix = centered.T @ centered / (n_samples - 1)
            eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)
            
            idx = np.argsort(eigenvalues)[::-1]
            eigenvalues = eigenvalues[idx]
            recognizer.components = eigenvectors[:, idx].T
            
        progress_callback(80, 100, "Computing variance...")
        
        # Variance
        total_var = np.sum(eigenvalues)
        recognizer.explained_variance = eigenvalues / total_var if total_var > 0 else eigenvalues
        recognizer.cumulative_variance = np.cumsum(recognizer.explained_variance)
        
        recognizer.is_fitted = True
        
        progress_callback(100, 100, "PCA fitting completed")
        
        return recognizer
        
    # Import numpy for the task
    import numpy as np
    
    return task


def create_compression_task(image, quality: int):
    """
    Create a full image compression task
    
    Returns:
        Task function for Worker
    """
    import cv2
    import numpy as np
    
    def task(progress_callback):
        progress_callback(0, 100, "Converting to grayscale...")
        
        # Convert to grayscale
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
            
        h, w = gray.shape
        
        progress_callback(5, 100, "Preparing blocks...")
        
        # Pad to multiple of 8
        pad_h = (8 - h % 8) % 8
        pad_w = (8 - w % 8) % 8
        padded = np.pad(gray, ((0, pad_h), (0, pad_w)), mode='edge')
        
        new_h, new_w = padded.shape
        result = np.zeros_like(padded, dtype=np.float32)
        
        # Count blocks
        total_blocks = (new_h // 8) * (new_w // 8)
        current_block = 0
        
        # Process blocks
        for y in range(0, new_h, 8):
            for x in range(0, new_w, 8):
                block = padded[y:y+8, x:x+8].astype(np.float32)
                
                # DCT
                shifted = block - 128.0
                dct = cv2.dct(shifted)
                
                # Quantize
                from core.compression import JPEG_QUANTIZATION_TABLE
                if quality < 50:
                    scale = 5000 / quality
                else:
                    scale = 200 - 2 * quality
                    
                q_table = np.floor((JPEG_QUANTIZATION_TABLE * scale + 50) / 100)
                q_table = np.clip(q_table, 1, 255)
                
                quantized = np.round(dct / q_table)
                dequantized = quantized * q_table
                
                # IDCT
                idct = cv2.idct(dequantized)
                reconstructed = idct + 128.0
                
                result[y:y+8, x:x+8] = np.clip(reconstructed, 0, 255)
                
                current_block += 1
                if current_block % 100 == 0:
                    progress = int(10 + 85 * current_block / total_blocks)
                    progress_callback(progress, 100, f"Processing block {current_block}/{total_blocks}")
                    
        progress_callback(95, 100, "Finalizing...")
        
        # Remove padding
        result = result[:h, :w].astype(np.uint8)
        
        progress_callback(100, 100, "Compression completed")
        
        return result
        
    return task
