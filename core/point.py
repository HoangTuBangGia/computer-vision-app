"""
Point Operations - Basic image processing operations
Includes: Grayscale, Negative, Log Transform, Power-law (Gamma), Histogram
"""
import cv2
import numpy as np
from typing import Tuple, Union


def to_grayscale(image: np.ndarray) -> np.ndarray:
    """
    Convert image to grayscale
    
    Args:
        image: Input image (BGR format)
        
    Returns:
        Grayscale image
    """
    if len(image.shape) == 2:
        return image
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


def negative(image: np.ndarray) -> np.ndarray:
    """
    Create negative of image (invert colors)
    
    s = L - 1 - r
    where L = 256 for 8-bit image
    
    Args:
        image: Input image
        
    Returns:
        Negative image
    """
    return 255 - image


def log_transform(image: np.ndarray, c: float = 1.0) -> np.ndarray:
    """
    Apply log transformation to image
    
    s = c * log(1 + r)
    
    This transformation expands dark pixels and compresses bright pixels.
    Useful for enhancing details in dark regions.
    
    Args:
        image: Input image
        c: Scaling constant (default: 1.0, will be auto-calculated if 1.0)
        
    Returns:
        Log transformed image
    """
    # Convert to float for calculation
    img_float = image.astype(np.float64)
    
    # Auto-calculate c to map output to [0, 255]
    if c == 1.0:
        c = 255 / np.log(1 + np.max(img_float))
    
    # Apply log transformation
    result = c * np.log(1 + img_float)
    
    # Clip and convert back to uint8
    result = np.clip(result, 0, 255).astype(np.uint8)
    
    return result


def power_law_transform(image: np.ndarray, gamma: float = 1.0, c: float = 1.0) -> np.ndarray:
    """
    Apply power-law (gamma) transformation to image
    
    s = c * r^gamma
    
    - gamma < 1: Brightens image (expands dark values)
    - gamma > 1: Darkens image (compresses dark values)
    - gamma = 1: No change
    
    Args:
        image: Input image
        gamma: Gamma value (0.1 to 5.0 typical range)
        c: Scaling constant (default: 1.0, auto-calculated)
        
    Returns:
        Gamma corrected image
    """
    # Normalize to [0, 1]
    img_normalized = image.astype(np.float64) / 255.0
    
    # Apply gamma correction
    result = c * np.power(img_normalized, gamma)
    
    # Scale back to [0, 255]
    result = (result * 255).clip(0, 255).astype(np.uint8)
    
    return result


def calculate_histogram(image: np.ndarray) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """
    Calculate histogram of image
    
    Args:
        image: Input image (grayscale or BGR)
        
    Returns:
        For grayscale: Single histogram array (256,)
        For color: Tuple of (B, G, R) histogram arrays
    """
    if len(image.shape) == 2:
        # Grayscale image
        hist = cv2.calcHist([image], [0], None, [256], [0, 256])
        return hist.flatten()
    else:
        # Color image - calculate histogram for each channel
        hist_b = cv2.calcHist([image], [0], None, [256], [0, 256]).flatten()
        hist_g = cv2.calcHist([image], [1], None, [256], [0, 256]).flatten()
        hist_r = cv2.calcHist([image], [2], None, [256], [0, 256]).flatten()
        return (hist_b, hist_g, hist_r)


def histogram_equalization(image: np.ndarray) -> np.ndarray:
    """
    Apply histogram equalization to enhance contrast
    
    Args:
        image: Input image
        
    Returns:
        Contrast enhanced image
    """
    if len(image.shape) == 2:
        # Grayscale
        return cv2.equalizeHist(image)
    else:
        # Color - convert to YCrCb and equalize Y channel
        ycrcb = cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)
        ycrcb[:, :, 0] = cv2.equalizeHist(ycrcb[:, :, 0])
        return cv2.cvtColor(ycrcb, cv2.COLOR_YCrCb2BGR)


def adjust_brightness(image: np.ndarray, value: int = 0) -> np.ndarray:
    """
    Adjust image brightness
    
    Args:
        image: Input image
        value: Brightness adjustment (-255 to 255)
        
    Returns:
        Brightness adjusted image
    """
    if value == 0:
        return image
        
    img_float = image.astype(np.float64)
    img_float += value
    return np.clip(img_float, 0, 255).astype(np.uint8)


def adjust_contrast(image: np.ndarray, factor: float = 1.0) -> np.ndarray:
    """
    Adjust image contrast
    
    Args:
        image: Input image
        factor: Contrast factor (0.0 to 3.0, 1.0 = no change)
        
    Returns:
        Contrast adjusted image
    """
    if factor == 1.0:
        return image
        
    img_float = image.astype(np.float64)
    # Apply contrast around middle gray (128)
    img_float = 128 + factor * (img_float - 128)
    return np.clip(img_float, 0, 255).astype(np.uint8)
