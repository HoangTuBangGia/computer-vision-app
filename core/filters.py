"""
Spatial Filters - Noise addition and filtering operations
Includes: Gaussian noise, Salt & Pepper noise, Mean, Gaussian, Median, Laplacian filters
"""
import cv2
import numpy as np
from typing import Tuple, Literal


# ============================================================
# NOISE FUNCTIONS
# ============================================================

def add_gaussian_noise(image: np.ndarray, mean: float = 0, sigma: float = 25) -> np.ndarray:
    """
    Add Gaussian noise to image
    
    Args:
        image: Input image
        mean: Mean of Gaussian distribution (default: 0)
        sigma: Standard deviation of Gaussian distribution (default: 25)
        
    Returns:
        Noisy image
    """
    # Generate Gaussian noise
    noise = np.random.normal(mean, sigma, image.shape).astype(np.float64)
    
    # Add noise to image
    noisy = image.astype(np.float64) + noise
    
    # Clip and convert back to uint8
    return np.clip(noisy, 0, 255).astype(np.uint8)


def add_salt_pepper_noise(image: np.ndarray, salt_prob: float = 0.02, 
                          pepper_prob: float = 0.02) -> np.ndarray:
    """
    Add Salt and Pepper noise to image
    
    Args:
        image: Input image
        salt_prob: Probability of salt (white) noise (default: 0.02)
        pepper_prob: Probability of pepper (black) noise (default: 0.02)
        
    Returns:
        Noisy image
    """
    noisy = image.copy()
    
    # Total number of pixels
    total_pixels = image.size
    
    # Add Salt (white) noise
    num_salt = int(total_pixels * salt_prob)
    salt_coords = [np.random.randint(0, i, num_salt) for i in image.shape[:2]]
    if len(image.shape) == 3:
        noisy[salt_coords[0], salt_coords[1], :] = 255
    else:
        noisy[salt_coords[0], salt_coords[1]] = 255
    
    # Add Pepper (black) noise
    num_pepper = int(total_pixels * pepper_prob)
    pepper_coords = [np.random.randint(0, i, num_pepper) for i in image.shape[:2]]
    if len(image.shape) == 3:
        noisy[pepper_coords[0], pepper_coords[1], :] = 0
    else:
        noisy[pepper_coords[0], pepper_coords[1]] = 0
    
    return noisy


def add_noise(image: np.ndarray, noise_type: str = "gaussian", **kwargs) -> np.ndarray:
    """
    Add noise to image (wrapper function)
    
    Args:
        image: Input image
        noise_type: Type of noise ("gaussian" or "salt_pepper")
        **kwargs: Additional parameters for noise function
        
    Returns:
        Noisy image
    """
    if noise_type == "gaussian":
        mean = kwargs.get("mean", 0)
        sigma = kwargs.get("sigma", 25)
        return add_gaussian_noise(image, mean, sigma)
    elif noise_type == "salt_pepper":
        salt_prob = kwargs.get("salt_prob", 0.02)
        pepper_prob = kwargs.get("pepper_prob", 0.02)
        return add_salt_pepper_noise(image, salt_prob, pepper_prob)
    else:
        return image


# ============================================================
# FILTER FUNCTIONS
# ============================================================

def mean_filter(image: np.ndarray, kernel_size: int = 3) -> np.ndarray:
    """
    Apply mean (average) filter to image
    
    This is a simple low-pass filter that replaces each pixel
    with the average of its neighbors.
    
    Args:
        image: Input image
        kernel_size: Size of the kernel (must be odd, default: 3)
        
    Returns:
        Filtered image
    """
    # Ensure kernel size is odd
    if kernel_size % 2 == 0:
        kernel_size += 1
        
    return cv2.blur(image, (kernel_size, kernel_size))


def gaussian_filter(image: np.ndarray, kernel_size: int = 3, sigma: float = 0) -> np.ndarray:
    """
    Apply Gaussian blur filter to image
    
    Gaussian blur is more effective than mean filter for noise reduction
    while preserving edges better.
    
    Args:
        image: Input image
        kernel_size: Size of the kernel (must be odd, default: 3)
        sigma: Standard deviation of Gaussian kernel (0 = auto-calculate)
        
    Returns:
        Filtered image
    """
    # Ensure kernel size is odd
    if kernel_size % 2 == 0:
        kernel_size += 1
        
    return cv2.GaussianBlur(image, (kernel_size, kernel_size), sigma)


def median_filter(image: np.ndarray, kernel_size: int = 3) -> np.ndarray:
    """
    Apply median filter to image
    
    Median filter is very effective for removing salt & pepper noise
    while preserving edges.
    
    Args:
        image: Input image
        kernel_size: Size of the kernel (must be odd, default: 3)
        
    Returns:
        Filtered image
    """
    # Ensure kernel size is odd
    if kernel_size % 2 == 0:
        kernel_size += 1
        
    return cv2.medianBlur(image, kernel_size)


def laplacian_sharpen(image: np.ndarray, strength: float = 1.0) -> np.ndarray:
    """
    Apply Laplacian sharpening to image
    
    Uses Laplacian operator to detect edges and enhance them.
    Formula: sharpened = original - strength * laplacian
    
    Args:
        image: Input image
        strength: Sharpening strength (default: 1.0)
        
    Returns:
        Sharpened image
    """
    # Convert to float for calculation
    img_float = image.astype(np.float64)
    
    # Apply Laplacian
    if len(image.shape) == 3:
        # For color images, apply to each channel
        laplacian = cv2.Laplacian(image, cv2.CV_64F)
    else:
        laplacian = cv2.Laplacian(image, cv2.CV_64F)
    
    # Sharpen: original - laplacian (subtract because Laplacian gives negative at edges)
    sharpened = img_float - strength * laplacian
    
    # Clip and convert back
    return np.clip(sharpened, 0, 255).astype(np.uint8)


def unsharp_mask(image: np.ndarray, kernel_size: int = 5, 
                 sigma: float = 1.0, strength: float = 1.5) -> np.ndarray:
    """
    Apply Unsharp Masking for image sharpening
    
    Creates a blurred version, subtracts it from original to get detail mask,
    then adds the mask back to original with given strength.
    
    Args:
        image: Input image
        kernel_size: Size of Gaussian kernel for blur
        sigma: Sigma for Gaussian blur
        strength: Sharpening strength (amount)
        
    Returns:
        Sharpened image
    """
    # Ensure kernel size is odd
    if kernel_size % 2 == 0:
        kernel_size += 1
        
    # Create blurred version
    blurred = cv2.GaussianBlur(image, (kernel_size, kernel_size), sigma)
    
    # Unsharp mask: original + strength * (original - blurred)
    img_float = image.astype(np.float64)
    blurred_float = blurred.astype(np.float64)
    
    sharpened = img_float + strength * (img_float - blurred_float)
    
    return np.clip(sharpened, 0, 255).astype(np.uint8)


def bilateral_filter(image: np.ndarray, d: int = 9, 
                     sigma_color: float = 75, sigma_space: float = 75) -> np.ndarray:
    """
    Apply bilateral filter to image
    
    Bilateral filter smooths images while keeping edges sharp.
    It's slower than other filters but produces better results for edge preservation.
    
    Args:
        image: Input image
        d: Diameter of each pixel neighborhood (default: 9)
        sigma_color: Filter sigma in the color space (default: 75)
        sigma_space: Filter sigma in the coordinate space (default: 75)
        
    Returns:
        Filtered image
    """
    return cv2.bilateralFilter(image, d, sigma_color, sigma_space)


def apply_filter(image: np.ndarray, filter_type: str, kernel_size: int = 3, 
                 **kwargs) -> np.ndarray:
    """
    Apply filter to image (wrapper function)
    
    Args:
        image: Input image
        filter_type: Type of filter ("mean", "gaussian", "median", "laplacian", 
                     "unsharp", "bilateral")
        kernel_size: Kernel size for applicable filters
        **kwargs: Additional parameters for specific filters
        
    Returns:
        Filtered image
    """
    if filter_type == "mean":
        return mean_filter(image, kernel_size)
    elif filter_type == "gaussian":
        sigma = kwargs.get("sigma", 0)
        return gaussian_filter(image, kernel_size, sigma)
    elif filter_type == "median":
        return median_filter(image, kernel_size)
    elif filter_type == "laplacian":
        strength = kwargs.get("strength", 1.0)
        return laplacian_sharpen(image, strength)
    elif filter_type == "unsharp":
        sigma = kwargs.get("sigma", 1.0)
        strength = kwargs.get("strength", 1.5)
        return unsharp_mask(image, kernel_size, sigma, strength)
    elif filter_type == "bilateral":
        sigma_color = kwargs.get("sigma_color", 75)
        sigma_space = kwargs.get("sigma_space", 75)
        return bilateral_filter(image, kernel_size, sigma_color, sigma_space)
    else:
        return image
