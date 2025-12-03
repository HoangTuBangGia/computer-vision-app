"""
Morphological Operations - Binary image processing
Includes: Erosion, Dilation, Opening, Closing, Boundary Extraction
"""
import cv2
import numpy as np
from typing import Tuple, Literal
from enum import Enum


class StructuringElement(Enum):
    """Structuring element shapes"""
    RECT = cv2.MORPH_RECT
    CROSS = cv2.MORPH_CROSS
    ELLIPSE = cv2.MORPH_ELLIPSE


def to_binary(image: np.ndarray, threshold: int = 127, 
              method: str = "otsu") -> np.ndarray:
    """
    Convert image to binary (thresholding)
    
    Args:
        image: Input image (grayscale or color)
        threshold: Threshold value (used if method is "simple")
        method: Thresholding method ("simple", "otsu", "adaptive")
        
    Returns:
        Binary image
    """
    # Convert to grayscale if needed
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()
    
    if method == "otsu":
        # Otsu's automatic thresholding
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    elif method == "adaptive":
        # Adaptive thresholding
        binary = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
            cv2.THRESH_BINARY, 11, 2
        )
    else:
        # Simple thresholding
        _, binary = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY)
    
    return binary


def get_structuring_element(shape: str, size: int) -> np.ndarray:
    """
    Get structuring element (kernel) for morphological operations
    
    Args:
        shape: Shape of SE ("rect", "cross", "ellipse")
        size: Size of the kernel (will be size x size)
        
    Returns:
        Structuring element (numpy array)
    """
    shape_map = {
        "rect": cv2.MORPH_RECT,
        "cross": cv2.MORPH_CROSS,
        "ellipse": cv2.MORPH_ELLIPSE,
    }
    
    cv_shape = shape_map.get(shape.lower(), cv2.MORPH_RECT)
    return cv2.getStructuringElement(cv_shape, (size, size))


def erosion(image: np.ndarray, kernel_size: int = 3, 
            shape: str = "rect", iterations: int = 1,
            auto_binary: bool = True) -> np.ndarray:
    """
    Apply erosion to image
    
    Erosion shrinks bright regions and enlarges dark regions.
    Useful for: removing small noise, separating connected objects.
    
    Args:
        image: Input image
        kernel_size: Size of structuring element
        shape: Shape of SE ("rect", "cross", "ellipse")
        iterations: Number of times to apply erosion
        auto_binary: Automatically convert to binary first
        
    Returns:
        Eroded image
    """
    if auto_binary and len(image.shape) == 3:
        img = to_binary(image)
    else:
        img = image.copy()
        
    kernel = get_structuring_element(shape, kernel_size)
    return cv2.erode(img, kernel, iterations=iterations)


def dilation(image: np.ndarray, kernel_size: int = 3,
             shape: str = "rect", iterations: int = 1,
             auto_binary: bool = True) -> np.ndarray:
    """
    Apply dilation to image
    
    Dilation expands bright regions and shrinks dark regions.
    Useful for: filling small holes, connecting nearby objects.
    
    Args:
        image: Input image
        kernel_size: Size of structuring element
        shape: Shape of SE ("rect", "cross", "ellipse")
        iterations: Number of times to apply dilation
        auto_binary: Automatically convert to binary first
        
    Returns:
        Dilated image
    """
    if auto_binary and len(image.shape) == 3:
        img = to_binary(image)
    else:
        img = image.copy()
        
    kernel = get_structuring_element(shape, kernel_size)
    return cv2.dilate(img, kernel, iterations=iterations)


def opening(image: np.ndarray, kernel_size: int = 3,
            shape: str = "rect", auto_binary: bool = True) -> np.ndarray:
    """
    Apply opening operation (erosion followed by dilation)
    
    Opening removes small bright spots (noise) while preserving shape.
    Useful for: noise removal, smoothing contours.
    
    Args:
        image: Input image
        kernel_size: Size of structuring element
        shape: Shape of SE ("rect", "cross", "ellipse")
        auto_binary: Automatically convert to binary first
        
    Returns:
        Opened image
    """
    if auto_binary and len(image.shape) == 3:
        img = to_binary(image)
    else:
        img = image.copy()
        
    kernel = get_structuring_element(shape, kernel_size)
    return cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)


def closing(image: np.ndarray, kernel_size: int = 3,
            shape: str = "rect", auto_binary: bool = True) -> np.ndarray:
    """
    Apply closing operation (dilation followed by erosion)
    
    Closing fills small holes and gaps while preserving shape.
    Useful for: filling holes, connecting broken parts.
    
    Args:
        image: Input image
        kernel_size: Size of structuring element
        shape: Shape of SE ("rect", "cross", "ellipse")
        auto_binary: Automatically convert to binary first
        
    Returns:
        Closed image
    """
    if auto_binary and len(image.shape) == 3:
        img = to_binary(image)
    else:
        img = image.copy()
        
    kernel = get_structuring_element(shape, kernel_size)
    return cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)


def boundary_extraction(image: np.ndarray, kernel_size: int = 3,
                        shape: str = "rect", auto_binary: bool = True) -> np.ndarray:
    """
    Extract boundary using morphological operations
    
    Boundary = Original - Erosion
    
    Args:
        image: Input image
        kernel_size: Size of structuring element
        shape: Shape of SE ("rect", "cross", "ellipse")
        auto_binary: Automatically convert to binary first
        
    Returns:
        Boundary image
    """
    if auto_binary and len(image.shape) == 3:
        img = to_binary(image)
    else:
        img = image.copy()
        
    kernel = get_structuring_element(shape, kernel_size)
    eroded = cv2.erode(img, kernel, iterations=1)
    
    # Boundary = Original - Eroded
    boundary = cv2.subtract(img, eroded)
    return boundary


def gradient(image: np.ndarray, kernel_size: int = 3,
             shape: str = "rect", auto_binary: bool = True) -> np.ndarray:
    """
    Apply morphological gradient (Dilation - Erosion)
    
    Highlights edges/boundaries of objects.
    
    Args:
        image: Input image
        kernel_size: Size of structuring element
        shape: Shape of SE ("rect", "cross", "ellipse")
        auto_binary: Automatically convert to binary first
        
    Returns:
        Gradient image
    """
    if auto_binary and len(image.shape) == 3:
        img = to_binary(image)
    else:
        img = image.copy()
        
    kernel = get_structuring_element(shape, kernel_size)
    return cv2.morphologyEx(img, cv2.MORPH_GRADIENT, kernel)


def top_hat(image: np.ndarray, kernel_size: int = 3,
            shape: str = "rect", auto_binary: bool = True) -> np.ndarray:
    """
    Apply top-hat transform (Original - Opening)
    
    Extracts small bright elements on dark background.
    
    Args:
        image: Input image
        kernel_size: Size of structuring element
        shape: Shape of SE ("rect", "cross", "ellipse")
        auto_binary: Automatically convert to binary first
        
    Returns:
        Top-hat transformed image
    """
    if auto_binary and len(image.shape) == 3:
        img = to_binary(image)
    else:
        img = image.copy()
        
    kernel = get_structuring_element(shape, kernel_size)
    return cv2.morphologyEx(img, cv2.MORPH_TOPHAT, kernel)


def black_hat(image: np.ndarray, kernel_size: int = 3,
              shape: str = "rect", auto_binary: bool = True) -> np.ndarray:
    """
    Apply black-hat transform (Closing - Original)
    
    Extracts small dark elements on bright background.
    
    Args:
        image: Input image
        kernel_size: Size of structuring element
        shape: Shape of SE ("rect", "cross", "ellipse")
        auto_binary: Automatically convert to binary first
        
    Returns:
        Black-hat transformed image
    """
    if auto_binary and len(image.shape) == 3:
        img = to_binary(image)
    else:
        img = image.copy()
        
    kernel = get_structuring_element(shape, kernel_size)
    return cv2.morphologyEx(img, cv2.MORPH_BLACKHAT, kernel)


def apply_morphology(image: np.ndarray, operation: str, kernel_size: int = 3,
                     shape: str = "rect", auto_binary: bool = True,
                     iterations: int = 1) -> np.ndarray:
    """
    Apply morphological operation (wrapper function)
    
    Args:
        image: Input image
        operation: Operation name ("erosion", "dilation", "opening", 
                   "closing", "boundary", "gradient", "tophat", "blackhat")
        kernel_size: Size of structuring element
        shape: Shape of SE ("rect", "cross", "ellipse")
        auto_binary: Automatically convert to binary first
        iterations: Number of iterations (for erosion/dilation)
        
    Returns:
        Processed image
    """
    operations = {
        "erosion": lambda: erosion(image, kernel_size, shape, iterations, auto_binary),
        "dilation": lambda: dilation(image, kernel_size, shape, iterations, auto_binary),
        "opening": lambda: opening(image, kernel_size, shape, auto_binary),
        "closing": lambda: closing(image, kernel_size, shape, auto_binary),
        "boundary": lambda: boundary_extraction(image, kernel_size, shape, auto_binary),
        "gradient": lambda: gradient(image, kernel_size, shape, auto_binary),
        "tophat": lambda: top_hat(image, kernel_size, shape, auto_binary),
        "blackhat": lambda: black_hat(image, kernel_size, shape, auto_binary),
    }
    
    if operation in operations:
        return operations[operation]()
    else:
        return image
