"""
PCA Face Recognition - Eigenfaces
Load face dataset, compute eigenfaces, reconstruct faces
"""
import numpy as np
import cv2
from pathlib import Path
from typing import List, Tuple, Optional
import os


class PCAFaceRecognizer:
    """
    PCA-based Face Recognition using Eigenfaces
    """
    
    def __init__(self, image_size: Tuple[int, int] = (64, 64)):
        """
        Initialize PCA Face Recognizer
        
        Args:
            image_size: Target size (width, height) for face images
        """
        self.image_size = image_size
        self.mean_face = None
        self.eigenfaces = None
        self.face_matrix = None
        self.labels = []
        self.file_paths = []
        self.n_samples = 0
        self.n_features = image_size[0] * image_size[1]
        self.is_fitted = False
        
        # PCA components
        self.components = None
        self.explained_variance = None
        self.explained_variance_ratio = None
        
    def load_dataset(self, folder_path: str, extensions: List[str] = None) -> Tuple[int, List[str]]:
        """
        Load face images from a folder (including subfolders)
        
        Args:
            folder_path: Path to folder containing face images
            extensions: List of valid image extensions
            
        Returns:
            Tuple of (number of images loaded, list of labels)
        """
        if extensions is None:
            extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.pgm', '.ppm']
            
        folder = Path(folder_path)
        if not folder.exists():
            raise ValueError(f"Folder not found: {folder_path}")
            
        # Find all image files (including in subfolders)
        image_files = []
        for ext in extensions:
            image_files.extend(folder.glob(f"*{ext}"))
            image_files.extend(folder.glob(f"*{ext.upper()}"))
            # Also search in subfolders
            image_files.extend(folder.glob(f"**/*{ext}"))
            image_files.extend(folder.glob(f"**/*{ext.upper()}"))
        
        # Remove duplicates and sort
        image_files = sorted(set(image_files))
            
        if not image_files:
            raise ValueError(f"No images found in {folder_path}")
            
        # Load and preprocess images
        faces = []
        self.file_paths = []
        self.labels = []
        
        for i, img_path in enumerate(image_files):
            # Read image
            img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
            if img is None:
                continue
                
            # Resize to standard size
            img_resized = cv2.resize(img, self.image_size)
            
            # Flatten to 1D vector
            face_vector = img_resized.flatten().astype(np.float64)
            faces.append(face_vector)
            
            self.file_paths.append(str(img_path))
            # Use parent folder name as label (e.g., "s1", "s2")
            self.labels.append(img_path.parent.name)
            
        if not faces:
            raise ValueError("Could not load any valid images")
            
        # Create face matrix (n_samples x n_features)
        self.face_matrix = np.array(faces)
        self.n_samples = len(faces)
        
        return self.n_samples, self.labels
        
    def fit(self, n_components: Optional[int] = None):
        """
        Fit PCA on the loaded face dataset
        
        Args:
            n_components: Number of principal components to keep
                         If None, keeps all components
        """
        if self.face_matrix is None:
            raise ValueError("No dataset loaded. Call load_dataset() first.")
            
        # Determine number of components
        if n_components is None:
            n_components = min(self.n_samples, self.n_features)
        n_components = min(n_components, self.n_samples, self.n_features)
        
        # Compute mean face
        self.mean_face = np.mean(self.face_matrix, axis=0)
        
        # Center the data
        centered_faces = self.face_matrix - self.mean_face
        
        # Compute covariance matrix (use smaller dimension for efficiency)
        # If n_samples < n_features, use the trick: C = X @ X.T instead of X.T @ X
        if self.n_samples < self.n_features:
            # Small covariance matrix (n_samples x n_samples)
            cov_matrix = np.dot(centered_faces, centered_faces.T) / (self.n_samples - 1)
            
            # Compute eigenvalues and eigenvectors
            eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)
            
            # Sort by eigenvalue (descending)
            idx = np.argsort(eigenvalues)[::-1]
            eigenvalues = eigenvalues[idx]
            eigenvectors = eigenvectors[:, idx]
            
            # Convert to original space eigenvectors
            # v_original = X.T @ v_small / ||X.T @ v_small||
            eigenfaces = np.dot(centered_faces.T, eigenvectors)
            
            # Normalize
            norms = np.linalg.norm(eigenfaces, axis=0)
            eigenfaces = eigenfaces / (norms + 1e-10)
            
        else:
            # Standard PCA
            cov_matrix = np.dot(centered_faces.T, centered_faces) / (self.n_samples - 1)
            eigenvalues, eigenfaces = np.linalg.eigh(cov_matrix)
            
            # Sort by eigenvalue (descending)
            idx = np.argsort(eigenvalues)[::-1]
            eigenvalues = eigenvalues[idx]
            eigenfaces = eigenfaces[:, idx]
            
        # Keep only n_components
        self.eigenfaces = eigenfaces[:, :n_components].T  # Shape: (n_components, n_features)
        eigenvalues = eigenvalues[:n_components]
        
        # Compute explained variance
        total_variance = np.sum(np.var(self.face_matrix, axis=0))
        self.explained_variance = eigenvalues
        self.explained_variance_ratio = eigenvalues / (np.sum(eigenvalues) + 1e-10)
        
        # Store components for reconstruction
        self.components = self.eigenfaces
        
        self.is_fitted = True
        
        return self
        
    def project(self, face_image: np.ndarray) -> np.ndarray:
        """
        Project a face image onto eigenface space
        
        Args:
            face_image: Input face image (can be 2D or 1D)
            
        Returns:
            Weights/coefficients in eigenface space
        """
        if not self.is_fitted:
            raise ValueError("PCA not fitted. Call fit() first.")
            
        # Flatten if 2D
        if len(face_image.shape) == 2:
            face_vector = cv2.resize(face_image, self.image_size).flatten().astype(np.float64)
        else:
            face_vector = face_image.astype(np.float64)
            
        # Center the face
        centered = face_vector - self.mean_face
        
        # Project onto eigenfaces
        weights = np.dot(self.components, centered)
        
        return weights
        
    def reconstruct(self, face_image: np.ndarray, n_components: int) -> np.ndarray:
        """
        Reconstruct a face using k principal components
        
        Args:
            face_image: Original face image
            n_components: Number of components to use for reconstruction
            
        Returns:
            Reconstructed face image (2D array)
        """
        if not self.is_fitted:
            raise ValueError("PCA not fitted. Call fit() first.")
            
        # Limit n_components
        n_components = min(n_components, len(self.components))
        
        # Get projection weights
        weights = self.project(face_image)[:n_components]
        
        # Reconstruct from k components
        reconstructed = self.mean_face + np.dot(weights, self.components[:n_components])
        
        # Clip to valid range and reshape
        reconstructed = np.clip(reconstructed, 0, 255).astype(np.uint8)
        reconstructed = reconstructed.reshape(self.image_size[1], self.image_size[0])
        
        return reconstructed
        
    def get_mean_face(self) -> np.ndarray:
        """
        Get the mean face as 2D image
        
        Returns:
            Mean face image
        """
        if self.mean_face is None:
            raise ValueError("No mean face computed. Call fit() first.")
            
        mean_img = np.clip(self.mean_face, 0, 255).astype(np.uint8)
        return mean_img.reshape(self.image_size[1], self.image_size[0])
        
    def get_eigenface(self, index: int) -> np.ndarray:
        """
        Get a specific eigenface as 2D image
        
        Args:
            index: Index of eigenface
            
        Returns:
            Eigenface image (normalized for display)
        """
        if self.eigenfaces is None:
            raise ValueError("No eigenfaces computed. Call fit() first.")
            
        if index >= len(self.eigenfaces):
            raise ValueError(f"Index {index} out of range. Max: {len(self.eigenfaces)-1}")
            
        # Normalize for display
        eigenface = self.eigenfaces[index]
        eigenface_normalized = cv2.normalize(
            eigenface, None, 0, 255, cv2.NORM_MINMAX
        ).astype(np.uint8)
        
        return eigenface_normalized.reshape(self.image_size[1], self.image_size[0])
        
    def get_face_by_index(self, index: int) -> np.ndarray:
        """
        Get a face from the dataset by index
        
        Args:
            index: Index of face in dataset
            
        Returns:
            Face image (2D array)
        """
        if self.face_matrix is None:
            raise ValueError("No dataset loaded.")
            
        if index >= self.n_samples:
            raise ValueError(f"Index {index} out of range. Max: {self.n_samples-1}")
            
        face = self.face_matrix[index].astype(np.uint8)
        return face.reshape(self.image_size[1], self.image_size[0])
        
    def get_reconstruction_error(self, face_image: np.ndarray, n_components: int) -> float:
        """
        Compute reconstruction error for a face
        
        Args:
            face_image: Original face
            n_components: Number of components used
            
        Returns:
            Mean squared error
        """
        reconstructed = self.reconstruct(face_image, n_components)
        
        # Get original as same format
        if len(face_image.shape) == 2:
            original = cv2.resize(face_image, self.image_size).astype(np.float64)
        else:
            original = face_image.reshape(self.image_size[1], self.image_size[0]).astype(np.float64)
            
        mse = np.mean((original - reconstructed.astype(np.float64)) ** 2)
        return mse
        
    def get_cumulative_variance(self) -> np.ndarray:
        """
        Get cumulative explained variance ratio
        
        Returns:
            Array of cumulative variance ratios
        """
        if self.explained_variance_ratio is None:
            raise ValueError("PCA not fitted.")
            
        return np.cumsum(self.explained_variance_ratio)
        
    def get_n_components_for_variance(self, target_variance: float = 0.95) -> int:
        """
        Get number of components needed to explain target variance
        
        Args:
            target_variance: Target cumulative variance (0-1)
            
        Returns:
            Number of components needed
        """
        cumulative = self.get_cumulative_variance()
        n_components = np.argmax(cumulative >= target_variance) + 1
        return int(n_components)


def load_single_face(image_path: str, target_size: Tuple[int, int] = (64, 64)) -> np.ndarray:
    """
    Load and preprocess a single face image
    
    Args:
        image_path: Path to image
        target_size: Target size (width, height)
        
    Returns:
        Preprocessed grayscale face image
    """
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError(f"Cannot load image: {image_path}")
        
    return cv2.resize(img, target_size)
