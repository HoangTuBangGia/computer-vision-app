"""
Segmentation Operations
Otsu Thresholding, K-Means Clustering
"""
import numpy as np
import cv2
from typing import Tuple


def otsu_threshold(image: np.ndarray) -> Tuple[np.ndarray, float]:
    """
    Apply Otsu's thresholding to automatically find optimal threshold
    
    Args:
        image: Input image (color or grayscale)
        
    Returns:
        Tuple of (binary_image, threshold_value)
    """
    # Convert to grayscale if needed
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()
    
    # Apply Otsu's thresholding
    threshold_value, binary = cv2.threshold(
        gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )
    
    return binary, threshold_value


def adaptive_threshold(
    image: np.ndarray,
    method: str = "gaussian",
    block_size: int = 11,
    c: int = 2
) -> np.ndarray:
    """
    Apply adaptive thresholding
    
    Args:
        image: Input image
        method: "mean" or "gaussian"
        block_size: Size of neighborhood (odd number)
        c: Constant subtracted from mean
        
    Returns:
        Binary image
    """
    # Convert to grayscale if needed
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()
    
    # Ensure block_size is odd
    if block_size % 2 == 0:
        block_size += 1
    
    # Select method
    adaptive_method = (
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C if method == "gaussian"
        else cv2.ADAPTIVE_THRESH_MEAN_C
    )
    
    # Apply adaptive threshold
    binary = cv2.adaptiveThreshold(
        gray, 255, adaptive_method, cv2.THRESH_BINARY, block_size, c
    )
    
    return binary


def kmeans_segmentation(
    image: np.ndarray,
    k: int = 3,
    max_iter: int = 100,
    epsilon: float = 0.2
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Perform K-Means color clustering segmentation
    
    Args:
        image: Input image (color)
        k: Number of clusters
        max_iter: Maximum iterations
        epsilon: Convergence threshold
        
    Returns:
        Tuple of (segmented_image, labels, centers)
    """
    # Convert to RGB if grayscale
    if len(image.shape) == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    
    # Reshape image to 2D array of pixels
    pixels = image.reshape((-1, 3)).astype(np.float32)
    
    # Define criteria for k-means
    criteria = (
        cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER,
        max_iter,
        epsilon
    )
    
    # Apply k-means
    _, labels, centers = cv2.kmeans(
        pixels,
        k,
        None,
        criteria,
        10,  # Number of attempts
        cv2.KMEANS_PP_CENTERS  # Use k-means++ initialization
    )
    
    # Convert centers to uint8
    centers = np.uint8(centers)
    
    # Map labels to center colors
    segmented = centers[labels.flatten()]
    segmented_image = segmented.reshape(image.shape)
    
    # Reshape labels for return
    labels = labels.reshape(image.shape[:2])
    
    return segmented_image, labels, centers


def kmeans_with_custom_colors(
    image: np.ndarray,
    k: int = 3,
    use_vibrant_colors: bool = True
) -> Tuple[np.ndarray, np.ndarray, float]:
    """
    K-Means segmentation with visually appealing color mapping
    
    Args:
        image: Input image
        k: Number of clusters
        use_vibrant_colors: Use predefined vibrant colors instead of cluster means
        
    Returns:
        Tuple of (colored_segmentation, labels, compactness)
    """
    # Convert to RGB if grayscale
    if len(image.shape) == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    
    # Reshape image
    pixels = image.reshape((-1, 3)).astype(np.float32)
    
    # K-means criteria
    criteria = (
        cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER,
        100,
        0.2
    )
    
    # Apply k-means
    compactness, labels, centers = cv2.kmeans(
        pixels,
        k,
        None,
        criteria,
        10,
        cv2.KMEANS_PP_CENTERS
    )
    
    labels = labels.flatten()
    
    if use_vibrant_colors:
        # Predefined vibrant colors (BGR format)
        vibrant_colors = np.array([
            [86, 180, 233],    # Sky blue
            [230, 159, 0],     # Orange
            [0, 158, 115],     # Bluish green
            [240, 228, 66],    # Yellow
            [0, 114, 178],     # Blue
            [213, 94, 0],      # Vermillion
            [204, 121, 167],   # Reddish purple
            [0, 0, 0],         # Black
            [255, 255, 255],   # White
            [128, 128, 128],   # Gray
        ], dtype=np.uint8)
        
        # Use vibrant colors for each cluster
        colors_to_use = vibrant_colors[:k]
        segmented = colors_to_use[labels]
    else:
        # Use actual cluster centers
        centers = np.uint8(centers)
        segmented = centers[labels]
    
    segmented_image = segmented.reshape(image.shape)
    labels_2d = labels.reshape(image.shape[:2])
    
    return segmented_image, labels_2d, compactness


def manual_threshold(
    image: np.ndarray,
    threshold: int = 128,
    max_value: int = 255,
    threshold_type: str = "binary"
) -> np.ndarray:
    """
    Apply manual thresholding
    
    Args:
        image: Input image
        threshold: Threshold value (0-255)
        max_value: Maximum value for binary
        threshold_type: Type of thresholding
        
    Returns:
        Thresholded image
    """
    # Convert to grayscale if needed
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()
    
    # Select threshold type
    type_map = {
        "binary": cv2.THRESH_BINARY,
        "binary_inv": cv2.THRESH_BINARY_INV,
        "trunc": cv2.THRESH_TRUNC,
        "tozero": cv2.THRESH_TOZERO,
        "tozero_inv": cv2.THRESH_TOZERO_INV,
    }
    
    thresh_type = type_map.get(threshold_type, cv2.THRESH_BINARY)
    
    _, result = cv2.threshold(gray, threshold, max_value, thresh_type)
    
    return result


def watershed_segmentation(image: np.ndarray) -> np.ndarray:
    """
    Apply watershed segmentation
    
    Args:
        image: Input image
        
    Returns:
        Segmented image with colored regions
    """
    # Convert to grayscale
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image
        image = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    
    # Apply Otsu's thresholding
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    
    # Noise removal with morphological operations
    kernel = np.ones((3, 3), np.uint8)
    opening = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=2)
    
    # Sure background area (dilation)
    sure_bg = cv2.dilate(opening, kernel, iterations=3)
    
    # Sure foreground area (distance transform + threshold)
    dist_transform = cv2.distanceTransform(opening, cv2.DIST_L2, 5)
    _, sure_fg = cv2.threshold(dist_transform, 0.7 * dist_transform.max(), 255, 0)
    sure_fg = np.uint8(sure_fg)
    
    # Unknown region
    unknown = cv2.subtract(sure_bg, sure_fg)
    
    # Marker labeling
    _, markers = cv2.connectedComponents(sure_fg)
    markers = markers + 1
    markers[unknown == 255] = 0
    
    # Apply watershed
    markers = cv2.watershed(image, markers)
    
    # Create colored output
    result = image.copy()
    result[markers == -1] = [0, 0, 255]  # Mark boundaries in red
    
    return result


def region_growing(
    image: np.ndarray,
    seed_point: Tuple[int, int],
    threshold: int = 20
) -> np.ndarray:
    """
    Simple region growing segmentation
    
    Args:
        image: Input image
        seed_point: Starting point (x, y)
        threshold: Intensity difference threshold
        
    Returns:
        Binary mask of the grown region
    """
    # Convert to grayscale
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()
    
    h, w = gray.shape
    seed_x, seed_y = seed_point
    
    # Initialize mask
    mask = np.zeros((h, w), dtype=np.uint8)
    
    # Get seed intensity
    seed_intensity = int(gray[seed_y, seed_x])
    
    # Stack for pixels to check
    stack = [(seed_x, seed_y)]
    
    while stack:
        x, y = stack.pop()
        
        if x < 0 or x >= w or y < 0 or y >= h:
            continue
        if mask[y, x] == 255:
            continue
        
        # Check intensity difference
        if abs(int(gray[y, x]) - seed_intensity) <= threshold:
            mask[y, x] = 255
            # Add 4-connected neighbors
            stack.extend([(x+1, y), (x-1, y), (x, y+1), (x, y-1)])
    
    return mask
