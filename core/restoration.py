"""
Image Restoration - Khôi phục ảnh
Includes: Noise models, Mean filters, Order-statistics filters, 
Adaptive filters, Motion blur, Inverse filtering
"""
import numpy as np
import cv2
from typing import Tuple, Optional
from scipy import ndimage


# ============================================================
# NOISE MODELS - Mô hình nhiễu
# ============================================================

def add_uniform_noise(image: np.ndarray, a: float = -50, b: float = 50) -> np.ndarray:
    """
    Add Uniform noise to image | Thêm nhiễu đồng nhất
    
    p(z) = 1/(b-a) if a <= z <= b, else 0
    mean = (a+b)/2, variance = (b-a)^2/12
    
    Args:
        image: Input image
        a: Lower bound (default: -50)
        b: Upper bound (default: 50)
        
    Returns:
        Noisy image
    """
    noise = np.random.uniform(a, b, image.shape)
    noisy = image.astype(np.float64) + noise
    return np.clip(noisy, 0, 255).astype(np.uint8)


def add_rayleigh_noise(image: np.ndarray, a: float = 0, b: float = 50) -> np.ndarray:
    """
    Add Rayleigh noise to image | Thêm nhiễu Rayleigh
    
    Args:
        image: Input image
        a: Location parameter
        b: Scale parameter
        
    Returns:
        Noisy image
    """
    noise = np.random.rayleigh(b, image.shape) + a
    noisy = image.astype(np.float64) + noise - np.mean(noise)
    return np.clip(noisy, 0, 255).astype(np.uint8)


def add_exponential_noise(image: np.ndarray, scale: float = 25) -> np.ndarray:
    """
    Add Exponential noise to image | Thêm nhiễu hàm mũ
    
    Args:
        image: Input image
        scale: Scale parameter (1/lambda)
        
    Returns:
        Noisy image
    """
    noise = np.random.exponential(scale, image.shape)
    noisy = image.astype(np.float64) + noise - np.mean(noise)
    return np.clip(noisy, 0, 255).astype(np.uint8)


# ============================================================
# MEAN FILTERS - Bộ lọc trung bình
# ============================================================

def arithmetic_mean_filter(image: np.ndarray, kernel_size: int = 3) -> np.ndarray:
    """
    Arithmetic Mean Filter | Bộ lọc trung bình số học
    
    f_hat(x,y) = (1/mn) * sum(g(s,t))
    
    Good for: Gaussian noise reduction
    
    Args:
        image: Input image
        kernel_size: Size of kernel (must be odd)
        
    Returns:
        Filtered image
    """
    if kernel_size % 2 == 0:
        kernel_size += 1
    return cv2.blur(image, (kernel_size, kernel_size))


def geometric_mean_filter(image: np.ndarray, kernel_size: int = 3) -> np.ndarray:
    """
    Geometric Mean Filter | Bộ lọc trung bình hình học
    
    f_hat(x,y) = [prod(g(s,t))]^(1/mn)
    
    Achieves smoothing comparable to arithmetic mean filter,
    but tends to lose less image detail.
    
    Args:
        image: Input image
        kernel_size: Size of kernel (must be odd)
        
    Returns:
        Filtered image
    """
    if kernel_size % 2 == 0:
        kernel_size += 1
    
    # Convert to float and avoid log(0)
    img_float = image.astype(np.float64) + 1e-10
    
    # Use log to compute geometric mean: exp(mean(log(x)))
    log_img = np.log(img_float)
    
    # Apply uniform filter to log values
    kernel = np.ones((kernel_size, kernel_size)) / (kernel_size ** 2)
    log_mean = cv2.filter2D(log_img, -1, kernel)
    
    # Convert back
    result = np.exp(log_mean)
    return np.clip(result, 0, 255).astype(np.uint8)


def harmonic_mean_filter(image: np.ndarray, kernel_size: int = 3) -> np.ndarray:
    """
    Harmonic Mean Filter | Bộ lọc trung bình điều hòa
    
    f_hat(x,y) = mn / sum(1/g(s,t))
    
    Good for: Salt noise (NOT pepper noise), Gaussian noise
    
    Args:
        image: Input image
        kernel_size: Size of kernel (must be odd)
        
    Returns:
        Filtered image
    """
    if kernel_size % 2 == 0:
        kernel_size += 1
    
    # Avoid division by zero
    img_float = image.astype(np.float64) + 1e-10
    
    # Compute 1/g
    inv_img = 1.0 / img_float
    
    # Sum of 1/g in neighborhood
    kernel = np.ones((kernel_size, kernel_size))
    sum_inv = cv2.filter2D(inv_img, -1, kernel)
    
    # mn / sum(1/g)
    mn = kernel_size ** 2
    result = mn / (sum_inv + 1e-10)
    
    return np.clip(result, 0, 255).astype(np.uint8)


def contraharmonic_mean_filter(image: np.ndarray, kernel_size: int = 3, 
                                Q: float = 1.5) -> np.ndarray:
    """
    Contra-harmonic Mean Filter | Bộ lọc trung bình nghịch điều hòa
    
    f_hat(x,y) = sum(g(s,t)^(Q+1)) / sum(g(s,t)^Q)
    
    - Q > 0: Eliminates pepper noise
    - Q < 0: Eliminates salt noise
    - Q = 0: Arithmetic mean
    - Q = -1: Harmonic mean
    
    Args:
        image: Input image
        kernel_size: Size of kernel (must be odd)
        Q: Order of the filter
        
    Returns:
        Filtered image
    """
    if kernel_size % 2 == 0:
        kernel_size += 1
    
    # Avoid issues with 0^negative
    img_float = image.astype(np.float64) + 1e-10
    
    kernel = np.ones((kernel_size, kernel_size))
    
    # Compute g^(Q+1) and g^Q
    numerator = cv2.filter2D(np.power(img_float, Q + 1), -1, kernel)
    denominator = cv2.filter2D(np.power(img_float, Q), -1, kernel)
    
    result = numerator / (denominator + 1e-10)
    
    return np.clip(result, 0, 255).astype(np.uint8)


# ============================================================
# ORDER-STATISTICS FILTERS - Bộ lọc thống kê theo thứ tự
# ============================================================

def max_filter(image: np.ndarray, kernel_size: int = 3) -> np.ndarray:
    """
    Max Filter | Bộ lọc giá trị lớn nhất
    
    f_hat(x,y) = max(g(s,t))
    
    Good for: Pepper noise (dark spots)
    
    Args:
        image: Input image
        kernel_size: Size of kernel (must be odd)
        
    Returns:
        Filtered image
    """
    if kernel_size % 2 == 0:
        kernel_size += 1
    
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_size, kernel_size))
    return cv2.dilate(image, kernel)


def min_filter(image: np.ndarray, kernel_size: int = 3) -> np.ndarray:
    """
    Min Filter | Bộ lọc giá trị nhỏ nhất
    
    f_hat(x,y) = min(g(s,t))
    
    Good for: Salt noise (bright spots)
    
    Args:
        image: Input image
        kernel_size: Size of kernel (must be odd)
        
    Returns:
        Filtered image
    """
    if kernel_size % 2 == 0:
        kernel_size += 1
    
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_size, kernel_size))
    return cv2.erode(image, kernel)


def midpoint_filter(image: np.ndarray, kernel_size: int = 3) -> np.ndarray:
    """
    Midpoint Filter | Bộ lọc trung điểm
    
    f_hat(x,y) = (max(g) + min(g)) / 2
    
    Good for: Gaussian and Uniform noise
    
    Args:
        image: Input image
        kernel_size: Size of kernel (must be odd)
        
    Returns:
        Filtered image
    """
    if kernel_size % 2 == 0:
        kernel_size += 1
    
    img_max = max_filter(image, kernel_size).astype(np.float64)
    img_min = min_filter(image, kernel_size).astype(np.float64)
    
    result = (img_max + img_min) / 2.0
    return result.astype(np.uint8)


def alpha_trimmed_mean_filter(image: np.ndarray, kernel_size: int = 5, 
                               d: int = 2) -> np.ndarray:
    """
    Alpha-trimmed Mean Filter | Bộ lọc trung bình cắt alpha
    
    f_hat(x,y) = (1/(mn-d)) * sum(g_r(s,t))
    
    Removes d/2 lowest and d/2 highest pixels, then computes mean.
    Good for: Combination of salt-and-pepper AND Gaussian noise
    
    Args:
        image: Input image
        kernel_size: Size of kernel (must be odd)
        d: Number of pixels to trim (d/2 from each end)
        
    Returns:
        Filtered image
    """
    if kernel_size % 2 == 0:
        kernel_size += 1
    
    # Ensure d is even and less than total pixels
    d = min(d, kernel_size ** 2 - 2)
    if d % 2 != 0:
        d -= 1
    
    img_float = image.astype(np.float64)
    pad = kernel_size // 2
    
    # Pad image
    padded = np.pad(img_float, pad, mode='reflect')
    
    result = np.zeros_like(img_float)
    h, w = img_float.shape[:2]
    
    for i in range(h):
        for j in range(w):
            # Extract neighborhood
            if len(image.shape) == 3:
                for c in range(image.shape[2]):
                    window = padded[i:i+kernel_size, j:j+kernel_size, c].flatten()
                    window = np.sort(window)
                    # Trim d/2 from each end
                    trimmed = window[d//2 : len(window) - d//2]
                    result[i, j, c] = np.mean(trimmed)
            else:
                window = padded[i:i+kernel_size, j:j+kernel_size].flatten()
                window = np.sort(window)
                trimmed = window[d//2 : len(window) - d//2]
                result[i, j] = np.mean(trimmed)
    
    return np.clip(result, 0, 255).astype(np.uint8)


# ============================================================
# ADAPTIVE FILTERS - Bộ lọc thích nghi
# ============================================================

def adaptive_local_noise_reduction(image: np.ndarray, kernel_size: int = 7,
                                    noise_variance: float = None) -> np.ndarray:
    """
    Adaptive Local Noise Reduction Filter | Bộ lọc giảm nhiễu cục bộ thích nghi
    
    f_hat(x,y) = g(x,y) - (sigma_n^2 / sigma_L^2) * (g(x,y) - m_L)
    
    Where:
    - sigma_n^2: Noise variance (estimated or provided)
    - sigma_L^2: Local variance
    - m_L: Local mean
    
    Args:
        image: Input image
        kernel_size: Size of local window
        noise_variance: Known noise variance (if None, estimated from image)
        
    Returns:
        Filtered image
    """
    if kernel_size % 2 == 0:
        kernel_size += 1
    
    img_float = image.astype(np.float64)
    
    # Estimate noise variance if not provided
    if noise_variance is None:
        # Use Median Absolute Deviation for robust estimation
        noise_variance = np.var(img_float) * 0.1  # Rough estimate
    
    # Compute local mean
    kernel = np.ones((kernel_size, kernel_size)) / (kernel_size ** 2)
    local_mean = cv2.filter2D(img_float, -1, kernel)
    
    # Compute local variance
    local_sq_mean = cv2.filter2D(img_float ** 2, -1, kernel)
    local_variance = local_sq_mean - local_mean ** 2
    local_variance = np.maximum(local_variance, 1e-10)  # Avoid division by zero
    
    # Compute ratio (clipped to [0, 1])
    ratio = noise_variance / local_variance
    ratio = np.clip(ratio, 0, 1)
    
    # Apply filter
    result = img_float - ratio * (img_float - local_mean)
    
    return np.clip(result, 0, 255).astype(np.uint8)


def adaptive_median_filter(image: np.ndarray, max_kernel_size: int = 7) -> np.ndarray:
    """
    Adaptive Median Filter | Bộ lọc trung vị thích nghi
    
    Adapts kernel size based on local statistics.
    Better at preserving detail than standard median filter.
    
    Args:
        image: Input image (grayscale)
        max_kernel_size: Maximum kernel size
        
    Returns:
        Filtered image
    """
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()
    
    img_float = gray.astype(np.float64)
    result = img_float.copy()
    h, w = img_float.shape
    
    for i in range(h):
        for j in range(w):
            kernel_size = 3
            while kernel_size <= max_kernel_size:
                pad = kernel_size // 2
                
                # Get window bounds
                y1 = max(0, i - pad)
                y2 = min(h, i + pad + 1)
                x1 = max(0, j - pad)
                x2 = min(w, j + pad + 1)
                
                window = img_float[y1:y2, x1:x2]
                
                z_min = np.min(window)
                z_max = np.max(window)
                z_med = np.median(window)
                z_xy = img_float[i, j]
                
                # Level A
                a1 = z_med - z_min
                a2 = z_med - z_max
                
                if a1 > 0 and a2 < 0:
                    # Level B
                    b1 = z_xy - z_min
                    b2 = z_xy - z_max
                    
                    if b1 > 0 and b2 < 0:
                        result[i, j] = z_xy
                    else:
                        result[i, j] = z_med
                    break
                else:
                    kernel_size += 2
            
            if kernel_size > max_kernel_size:
                result[i, j] = z_med
    
    return result.astype(np.uint8)


# ============================================================
# MOTION BLUR & DEGRADATION MODELS
# ============================================================

def create_motion_blur_kernel(length: int = 15, angle: float = 0) -> np.ndarray:
    """
    Create motion blur kernel | Tạo kernel nhòe chuyển động
    
    Args:
        length: Length of motion blur
        angle: Angle of motion in degrees
        
    Returns:
        Motion blur kernel
    """
    # Create horizontal kernel
    kernel = np.zeros((length, length))
    kernel[length // 2, :] = 1.0 / length
    
    # Rotate kernel
    center = (length // 2, length // 2)
    rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
    kernel = cv2.warpAffine(kernel, rotation_matrix, (length, length))
    
    # Normalize
    kernel = kernel / (kernel.sum() + 1e-10)
    
    return kernel


def apply_motion_blur(image: np.ndarray, length: int = 15, 
                       angle: float = 0) -> np.ndarray:
    """
    Apply motion blur to image | Áp dụng nhòe chuyển động
    
    Args:
        image: Input image
        length: Length of motion blur
        angle: Angle of motion in degrees
        
    Returns:
        Blurred image
    """
    kernel = create_motion_blur_kernel(length, angle)
    return cv2.filter2D(image, -1, kernel)


def create_atmospheric_turbulence_filter(shape: Tuple[int, int], 
                                          k: float = 0.001) -> np.ndarray:
    """
    Create Atmospheric Turbulence degradation filter
    
    H(u,v) = exp(-k * (u^2 + v^2)^(5/6))
    
    Args:
        shape: (rows, cols) of the filter
        k: Turbulence constant (higher = more blur)
        
    Returns:
        Degradation filter in frequency domain
    """
    rows, cols = shape
    crow, ccol = rows // 2, cols // 2
    
    u = np.arange(rows) - crow
    v = np.arange(cols) - ccol
    u, v = np.meshgrid(u, v, indexing='ij')
    
    d_squared = u**2 + v**2
    H = np.exp(-k * np.power(d_squared, 5/6))
    
    return H


def apply_atmospheric_blur(image: np.ndarray, k: float = 0.001) -> np.ndarray:
    """
    Apply atmospheric turbulence blur | Áp dụng nhòe do nhiễu loạn khí quyển
    
    Args:
        image: Input image
        k: Turbulence constant
        
    Returns:
        Degraded image
    """
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()
    
    gray = gray.astype(np.float64)
    
    # FFT
    fft = np.fft.fft2(gray)
    fft_shift = np.fft.fftshift(fft)
    
    # Apply degradation
    H = create_atmospheric_turbulence_filter(gray.shape, k)
    degraded_fft = fft_shift * H
    
    # Inverse FFT
    fft_ishift = np.fft.ifftshift(degraded_fft)
    result = np.fft.ifft2(fft_ishift)
    result = np.abs(result)
    
    return np.clip(result, 0, 255).astype(np.uint8)


# ============================================================
# INVERSE FILTERING - Lọc nghịch đảo
# ============================================================

def inverse_filter(image: np.ndarray, degradation_func: np.ndarray,
                   cutoff_ratio: float = 0.7) -> np.ndarray:
    """
    Inverse Filtering | Lọc nghịch đảo
    
    F_hat(u,v) = G(u,v) / H(u,v)
    
    Problem: H(u,v) may be 0 or very small
    Solution: Limit frequency around origin (cutoff)
    
    Args:
        image: Degraded image
        degradation_func: H(u,v) - degradation function in frequency domain
        cutoff_ratio: Ratio of frequencies to keep (0-1), limits restoration
        
    Returns:
        Restored image
    """
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()
    
    gray = gray.astype(np.float64)
    
    # FFT of degraded image
    G = np.fft.fft2(gray)
    G_shift = np.fft.fftshift(G)
    
    # Avoid division by zero
    H = degradation_func.copy()
    H[np.abs(H) < 1e-10] = 1e-10
    
    # Apply inverse filter
    F_hat = G_shift / H
    
    # Apply cutoff to limit high frequency noise amplification
    rows, cols = gray.shape
    crow, ccol = rows // 2, cols // 2
    
    # Create circular mask for cutoff
    u = np.arange(rows) - crow
    v = np.arange(cols) - ccol
    u, v = np.meshgrid(u, v, indexing='ij')
    d = np.sqrt(u**2 + v**2)
    
    max_freq = min(crow, ccol) * cutoff_ratio
    mask = d <= max_freq
    
    F_hat = F_hat * mask
    
    # Inverse FFT
    F_ishift = np.fft.ifftshift(F_hat)
    result = np.fft.ifft2(F_ishift)
    result = np.abs(result)
    
    # Normalize
    result = cv2.normalize(result, None, 0, 255, cv2.NORM_MINMAX)
    
    return result.astype(np.uint8)


def wiener_filter(image: np.ndarray, degradation_func: np.ndarray,
                  K: float = 0.01) -> np.ndarray:
    """
    Wiener Filter (Minimum Mean Square Error Filter)
    
    F_hat(u,v) = [H*(u,v) / (|H(u,v)|^2 + K)] * G(u,v)
    
    Where K ≈ NSR (Noise-to-Signal Ratio)
    
    Args:
        image: Degraded image
        degradation_func: H(u,v) - degradation function
        K: Noise-to-signal ratio estimate (regularization parameter)
        
    Returns:
        Restored image
    """
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()
    
    gray = gray.astype(np.float64)
    
    # FFT
    G = np.fft.fft2(gray)
    G_shift = np.fft.fftshift(G)
    
    H = degradation_func
    H_conj = np.conj(H)
    H_abs_sq = np.abs(H) ** 2
    
    # Wiener filter
    W = H_conj / (H_abs_sq + K)
    
    # Apply filter
    F_hat = W * G_shift
    
    # Inverse FFT
    F_ishift = np.fft.ifftshift(F_hat)
    result = np.fft.ifft2(F_ishift)
    result = np.abs(result)
    
    # Normalize
    result = cv2.normalize(result, None, 0, 255, cv2.NORM_MINMAX)
    
    return result.astype(np.uint8)


def restore_motion_blur(image: np.ndarray, length: int = 15, 
                         angle: float = 0, method: str = "wiener",
                         K: float = 0.01, cutoff_ratio: float = 0.7) -> np.ndarray:
    """
    Restore motion-blurred image | Khôi phục ảnh nhòe chuyển động
    
    Args:
        image: Motion-blurred image
        length: Estimated motion blur length
        angle: Estimated motion blur angle
        method: "inverse" or "wiener"
        K: Wiener filter parameter (for wiener method)
        cutoff_ratio: Frequency cutoff (for inverse method)
        
    Returns:
        Restored image
    """
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()
    
    # Create PSF (Point Spread Function) = motion blur kernel
    psf = create_motion_blur_kernel(length, angle)
    
    # Pad PSF to image size
    psf_padded = np.zeros(gray.shape)
    kh, kw = psf.shape
    cy, cx = gray.shape[0] // 2, gray.shape[1] // 2
    psf_padded[cy - kh//2:cy + kh//2 + kh%2, 
               cx - kw//2:cx + kw//2 + kw%2] = psf
    
    # FFT of PSF
    H = np.fft.fft2(np.fft.ifftshift(psf_padded))
    H = np.fft.fftshift(H)
    
    if method == "wiener":
        return wiener_filter(gray, H, K)
    else:
        return inverse_filter(gray, H, cutoff_ratio)


def restore_atmospheric_blur(image: np.ndarray, k: float = 0.001,
                              method: str = "wiener", K: float = 0.01,
                              cutoff_ratio: float = 0.7) -> np.ndarray:
    """
    Restore atmospheric turbulence blur | Khôi phục ảnh nhòe khí quyển
    
    Args:
        image: Blurred image
        k: Turbulence constant used in degradation
        method: "inverse" or "wiener"
        K: Wiener filter parameter
        cutoff_ratio: Frequency cutoff (for inverse method)
        
    Returns:
        Restored image
    """
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()
    
    # Create degradation function
    H = create_atmospheric_turbulence_filter(gray.shape, k)
    
    if method == "wiener":
        return wiener_filter(gray, H, K)
    else:
        return inverse_filter(gray, H, cutoff_ratio)


# ============================================================
# NOTCH FILTERS - Bộ lọc Notch
# ============================================================

def create_notch_reject_filter(shape: Tuple[int, int], 
                                centers: list,
                                d0: float = 10,
                                filter_type: str = "butterworth",
                                n: int = 2) -> np.ndarray:
    """
    Create Notch Reject Filter | Tạo bộ lọc chắn Notch
    
    Removes periodic noise at specific frequencies.
    
    Args:
        shape: (rows, cols) of the filter
        centers: List of (u, v) frequency centers to reject
        d0: Notch radius
        filter_type: "ideal", "butterworth", or "gaussian"
        n: Order for Butterworth filter
        
    Returns:
        Notch reject filter
    """
    rows, cols = shape
    crow, ccol = rows // 2, cols // 2
    
    u = np.arange(rows)
    v = np.arange(cols)
    u, v = np.meshgrid(u, v, indexing='ij')
    
    H = np.ones(shape, dtype=np.float64)
    
    for (u0, v0) in centers:
        # Distance to notch center and its conjugate
        d1 = np.sqrt((u - crow - u0)**2 + (v - ccol - v0)**2)
        d2 = np.sqrt((u - crow + u0)**2 + (v - ccol + v0)**2)
        
        if filter_type == "ideal":
            notch = np.ones(shape)
            notch[d1 <= d0] = 0
            notch[d2 <= d0] = 0
        elif filter_type == "butterworth":
            # Avoid division by zero
            d1 = np.maximum(d1, 1e-10)
            d2 = np.maximum(d2, 1e-10)
            notch = 1 / (1 + (d0 / d1)**(2*n)) * 1 / (1 + (d0 / d2)**(2*n))
        else:  # gaussian
            notch = (1 - np.exp(-(d1**2) / (2 * d0**2))) * \
                    (1 - np.exp(-(d2**2) / (2 * d0**2)))
        
        H = H * notch
    
    return H


def apply_notch_filter(image: np.ndarray, centers: list,
                        d0: float = 10, reject: bool = True) -> np.ndarray:
    """
    Apply Notch Filter to remove periodic noise
    
    Args:
        image: Input image with periodic noise
        centers: List of (u, v) frequency centers
        d0: Notch radius
        reject: True for notch reject, False for notch pass
        
    Returns:
        Filtered image
    """
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()
    
    gray = gray.astype(np.float64)
    
    # FFT
    fft = np.fft.fft2(gray)
    fft_shift = np.fft.fftshift(fft)
    
    # Create and apply notch filter
    H = create_notch_reject_filter(gray.shape, centers, d0)
    if not reject:
        H = 1 - H
    
    filtered_fft = fft_shift * H
    
    # Inverse FFT
    fft_ishift = np.fft.ifftshift(filtered_fft)
    result = np.fft.ifft2(fft_ishift)
    result = np.abs(result)
    
    return np.clip(result, 0, 255).astype(np.uint8)
