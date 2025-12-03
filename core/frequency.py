"""
Frequency Domain Operations
Fourier Transform, Frequency Filtering (Lowpass/Highpass)
"""
import numpy as np
import cv2
from typing import Tuple, Literal


def to_grayscale_if_needed(image: np.ndarray) -> np.ndarray:
    """Convert image to grayscale if it's color"""
    if len(image.shape) == 3:
        return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return image.copy()


def compute_fft(image: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute 2D FFT of an image
    
    Args:
        image: Input image (grayscale or color)
        
    Returns:
        Tuple of (fft_shift, magnitude_spectrum)
        - fft_shift: Shifted FFT result (complex)
        - magnitude_spectrum: Log-scaled magnitude spectrum for display
    """
    # Convert to grayscale if needed
    gray = to_grayscale_if_needed(image)
    
    # Convert to float32
    gray = np.float32(gray)
    
    # Compute 2D FFT
    fft = np.fft.fft2(gray)
    
    # Shift zero frequency to center
    fft_shift = np.fft.fftshift(fft)
    
    # Compute magnitude spectrum (log scale for visualization)
    magnitude = np.abs(fft_shift)
    magnitude_spectrum = np.log1p(magnitude)  # log(1 + magnitude) to avoid log(0)
    
    # Normalize to 0-255 for display
    magnitude_spectrum = cv2.normalize(
        magnitude_spectrum, None, 0, 255, cv2.NORM_MINMAX
    ).astype(np.uint8)
    
    return fft_shift, magnitude_spectrum


def compute_ifft(fft_shift: np.ndarray) -> np.ndarray:
    """
    Compute inverse FFT to get image back
    
    Args:
        fft_shift: Shifted FFT (complex)
        
    Returns:
        Reconstructed image
    """
    # Inverse shift
    fft_ishift = np.fft.ifftshift(fft_shift)
    
    # Inverse FFT
    img_back = np.fft.ifft2(fft_ishift)
    
    # Get magnitude (real part)
    img_back = np.abs(img_back)
    
    # Normalize to 0-255
    img_back = cv2.normalize(img_back, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    
    return img_back


def create_ideal_lowpass_filter(shape: Tuple[int, int], d0: float) -> np.ndarray:
    """
    Create Ideal Lowpass Filter
    
    H(u,v) = 1 if D(u,v) <= D0
             0 if D(u,v) > D0
    
    Args:
        shape: (rows, cols) of the filter
        d0: Cutoff frequency (radius)
        
    Returns:
        Filter mask
    """
    rows, cols = shape
    crow, ccol = rows // 2, cols // 2
    
    # Create meshgrid for distance calculation
    u = np.arange(rows)
    v = np.arange(cols)
    u, v = np.meshgrid(u - crow, v - ccol, indexing='ij')
    
    # Calculate distance from center
    d = np.sqrt(u**2 + v**2)
    
    # Create filter
    h = np.zeros(shape, dtype=np.float32)
    h[d <= d0] = 1
    
    return h


def create_ideal_highpass_filter(shape: Tuple[int, int], d0: float) -> np.ndarray:
    """
    Create Ideal Highpass Filter
    
    H(u,v) = 0 if D(u,v) <= D0
             1 if D(u,v) > D0
    
    Args:
        shape: (rows, cols) of the filter
        d0: Cutoff frequency (radius)
        
    Returns:
        Filter mask
    """
    return 1 - create_ideal_lowpass_filter(shape, d0)


def create_gaussian_lowpass_filter(shape: Tuple[int, int], d0: float) -> np.ndarray:
    """
    Create Gaussian Lowpass Filter
    
    H(u,v) = exp(-D(u,v)^2 / (2 * D0^2))
    
    Args:
        shape: (rows, cols) of the filter
        d0: Cutoff frequency (standard deviation)
        
    Returns:
        Filter mask
    """
    rows, cols = shape
    crow, ccol = rows // 2, cols // 2
    
    # Create meshgrid
    u = np.arange(rows)
    v = np.arange(cols)
    u, v = np.meshgrid(u - crow, v - ccol, indexing='ij')
    
    # Calculate distance
    d = np.sqrt(u**2 + v**2)
    
    # Gaussian filter
    h = np.exp(-(d**2) / (2 * (d0**2) + 1e-6))
    
    return h.astype(np.float32)


def create_gaussian_highpass_filter(shape: Tuple[int, int], d0: float) -> np.ndarray:
    """
    Create Gaussian Highpass Filter
    
    H(u,v) = 1 - exp(-D(u,v)^2 / (2 * D0^2))
    
    Args:
        shape: (rows, cols) of the filter
        d0: Cutoff frequency
        
    Returns:
        Filter mask
    """
    return 1 - create_gaussian_lowpass_filter(shape, d0)


def create_butterworth_lowpass_filter(
    shape: Tuple[int, int], 
    d0: float, 
    n: int = 2
) -> np.ndarray:
    """
    Create Butterworth Lowpass Filter
    
    H(u,v) = 1 / (1 + (D(u,v)/D0)^(2n))
    
    Args:
        shape: (rows, cols) of the filter
        d0: Cutoff frequency
        n: Order of the filter
        
    Returns:
        Filter mask
    """
    rows, cols = shape
    crow, ccol = rows // 2, cols // 2
    
    # Create meshgrid
    u = np.arange(rows)
    v = np.arange(cols)
    u, v = np.meshgrid(u - crow, v - ccol, indexing='ij')
    
    # Calculate distance
    d = np.sqrt(u**2 + v**2)
    
    # Butterworth filter
    h = 1 / (1 + (d / (d0 + 1e-6))**(2 * n))
    
    return h.astype(np.float32)


def create_butterworth_highpass_filter(
    shape: Tuple[int, int], 
    d0: float, 
    n: int = 2
) -> np.ndarray:
    """
    Create Butterworth Highpass Filter
    
    Args:
        shape: (rows, cols) of the filter
        d0: Cutoff frequency
        n: Order of the filter
        
    Returns:
        Filter mask
    """
    return 1 - create_butterworth_lowpass_filter(shape, d0, n)


def apply_frequency_filter(
    image: np.ndarray,
    filter_type: Literal["ideal_lowpass", "ideal_highpass", 
                         "gaussian_lowpass", "gaussian_highpass",
                         "butterworth_lowpass", "butterworth_highpass"],
    d0: float,
    order: int = 2
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Apply frequency domain filter to image
    
    Args:
        image: Input image
        filter_type: Type of filter to apply
        d0: Cutoff frequency
        order: Order for Butterworth filter
        
    Returns:
        Tuple of (filtered_image, filter_mask_display, filtered_spectrum)
    """
    # Convert to grayscale if needed
    gray = to_grayscale_if_needed(image)
    gray = np.float32(gray)
    
    # Compute FFT
    fft = np.fft.fft2(gray)
    fft_shift = np.fft.fftshift(fft)
    
    # Get shape
    shape = gray.shape
    
    # Create filter based on type
    filter_funcs = {
        "ideal_lowpass": lambda: create_ideal_lowpass_filter(shape, d0),
        "ideal_highpass": lambda: create_ideal_highpass_filter(shape, d0),
        "gaussian_lowpass": lambda: create_gaussian_lowpass_filter(shape, d0),
        "gaussian_highpass": lambda: create_gaussian_highpass_filter(shape, d0),
        "butterworth_lowpass": lambda: create_butterworth_lowpass_filter(shape, d0, order),
        "butterworth_highpass": lambda: create_butterworth_highpass_filter(shape, d0, order),
    }
    
    h = filter_funcs.get(filter_type, lambda: np.ones(shape, dtype=np.float32))()
    
    # Apply filter
    filtered_fft = fft_shift * h
    
    # Compute filtered spectrum for display
    magnitude = np.abs(filtered_fft)
    filtered_spectrum = np.log1p(magnitude)
    filtered_spectrum = cv2.normalize(
        filtered_spectrum, None, 0, 255, cv2.NORM_MINMAX
    ).astype(np.uint8)
    
    # Inverse FFT
    filtered_image = compute_ifft(filtered_fft)
    
    # Create filter mask display (normalized to 0-255)
    filter_display = cv2.normalize(h, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    
    return filtered_image, filter_display, filtered_spectrum


def get_filter_visualization(
    shape: Tuple[int, int],
    filter_type: str,
    d0: float,
    order: int = 2
) -> np.ndarray:
    """
    Get visualization of the filter mask
    
    Args:
        shape: (rows, cols)
        filter_type: Type of filter
        d0: Cutoff frequency
        order: Butterworth order
        
    Returns:
        Filter visualization image
    """
    filter_funcs = {
        "ideal_lowpass": lambda: create_ideal_lowpass_filter(shape, d0),
        "ideal_highpass": lambda: create_ideal_highpass_filter(shape, d0),
        "gaussian_lowpass": lambda: create_gaussian_lowpass_filter(shape, d0),
        "gaussian_highpass": lambda: create_gaussian_highpass_filter(shape, d0),
        "butterworth_lowpass": lambda: create_butterworth_lowpass_filter(shape, d0, order),
        "butterworth_highpass": lambda: create_butterworth_highpass_filter(shape, d0, order),
    }
    
    h = filter_funcs.get(filter_type, lambda: np.ones(shape, dtype=np.float32))()
    
    # Normalize to 0-255
    h_display = cv2.normalize(h, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    
    return h_display
