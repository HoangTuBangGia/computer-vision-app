"""
JPEG Compression Simulation
DCT, Quantization, Zig-zag encoding for educational purposes
"""
import numpy as np
import cv2
from typing import Tuple, List


# Standard JPEG luminance quantization table
JPEG_QUANTIZATION_TABLE = np.array([
    [16, 11, 10, 16, 24, 40, 51, 61],
    [12, 12, 14, 19, 26, 58, 60, 55],
    [14, 13, 16, 24, 40, 57, 69, 56],
    [14, 17, 22, 29, 51, 87, 80, 62],
    [18, 22, 37, 56, 68, 109, 103, 77],
    [24, 35, 55, 64, 81, 104, 113, 92],
    [49, 64, 78, 87, 103, 121, 120, 101],
    [72, 92, 95, 98, 112, 100, 103, 99]
], dtype=np.float32)

# Zig-zag order indices for 8x8 block
ZIGZAG_ORDER = np.array([
    0,  1,  8, 16,  9,  2,  3, 10,
    17, 24, 32, 25, 18, 11,  4,  5,
    12, 19, 26, 33, 40, 48, 41, 34,
    27, 20, 13,  6,  7, 14, 21, 28,
    35, 42, 49, 56, 57, 50, 43, 36,
    29, 22, 15, 23, 30, 37, 44, 51,
    58, 59, 52, 45, 38, 31, 39, 46,
    53, 60, 61, 54, 47, 55, 62, 63
])

# Inverse zig-zag order
ZIGZAG_INVERSE = np.argsort(ZIGZAG_ORDER)


def to_grayscale(image: np.ndarray) -> np.ndarray:
    """Convert image to grayscale if needed"""
    if len(image.shape) == 3:
        return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return image.copy()


def get_block_position(x: int, y: int) -> Tuple[int, int]:
    """
    Get the top-left corner of the 8x8 block containing point (x, y)
    
    Args:
        x: X coordinate (column)
        y: Y coordinate (row)
        
    Returns:
        Tuple of (block_x, block_y) - top-left corner
    """
    block_x = (x // 8) * 8
    block_y = (y // 8) * 8
    return block_x, block_y


def extract_block(image: np.ndarray, x: int, y: int) -> np.ndarray:
    """
    Extract 8x8 block from image at position (x, y)
    
    Args:
        image: Input image (grayscale)
        x: Top-left X coordinate
        y: Top-left Y coordinate
        
    Returns:
        8x8 block as numpy array
    """
    gray = to_grayscale(image)
    h, w = gray.shape
    
    # Handle boundary cases
    x = min(x, w - 8)
    y = min(y, h - 8)
    x = max(0, x)
    y = max(0, y)
    
    return gray[y:y+8, x:x+8].astype(np.float32)


def compute_dct(block: np.ndarray) -> np.ndarray:
    """
    Compute 2D DCT of an 8x8 block
    
    Args:
        block: 8x8 pixel block (float32)
        
    Returns:
        8x8 DCT coefficients
    """
    # Shift pixel values from [0, 255] to [-128, 127]
    shifted = block - 128.0
    
    # Compute 2D DCT
    dct = cv2.dct(shifted)
    
    return dct


def compute_idct(dct_block: np.ndarray) -> np.ndarray:
    """
    Compute inverse 2D DCT
    
    Args:
        dct_block: 8x8 DCT coefficients
        
    Returns:
        8x8 pixel block
    """
    # Inverse DCT
    idct = cv2.idct(dct_block)
    
    # Shift back to [0, 255]
    result = idct + 128.0
    
    # Clip to valid range
    return np.clip(result, 0, 255)


def quantize(dct_block: np.ndarray, quality: int = 50) -> np.ndarray:
    """
    Quantize DCT coefficients using JPEG quantization table
    
    Args:
        dct_block: 8x8 DCT coefficients
        quality: JPEG quality (1-100), lower = more compression
        
    Returns:
        8x8 quantized coefficients (integers)
    """
    # Compute quality factor
    if quality < 50:
        scale = 5000 / quality
    else:
        scale = 200 - 2 * quality
        
    # Scale quantization table
    q_table = np.floor((JPEG_QUANTIZATION_TABLE * scale + 50) / 100)
    q_table = np.clip(q_table, 1, 255)  # Ensure no division by zero
    
    # Quantize
    quantized = np.round(dct_block / q_table)
    
    return quantized.astype(np.int32)


def dequantize(quantized: np.ndarray, quality: int = 50) -> np.ndarray:
    """
    Dequantize coefficients
    
    Args:
        quantized: 8x8 quantized coefficients
        quality: JPEG quality used for quantization
        
    Returns:
        8x8 dequantized DCT coefficients
    """
    # Compute same quality factor
    if quality < 50:
        scale = 5000 / quality
    else:
        scale = 200 - 2 * quality
        
    # Scale quantization table
    q_table = np.floor((JPEG_QUANTIZATION_TABLE * scale + 50) / 100)
    q_table = np.clip(q_table, 1, 255)
    
    # Dequantize
    dequantized = quantized.astype(np.float32) * q_table
    
    return dequantized


def zigzag_encode(block: np.ndarray) -> np.ndarray:
    """
    Convert 8x8 block to 1D array using zig-zag order
    
    Args:
        block: 8x8 matrix
        
    Returns:
        1D array of 64 elements in zig-zag order
    """
    flat = block.flatten()
    return flat[ZIGZAG_ORDER]


def zigzag_decode(zigzag: np.ndarray) -> np.ndarray:
    """
    Convert 1D zig-zag array back to 8x8 block
    
    Args:
        zigzag: 1D array of 64 elements
        
    Returns:
        8x8 matrix
    """
    flat = np.zeros(64, dtype=zigzag.dtype)
    flat[ZIGZAG_ORDER] = zigzag
    return flat.reshape(8, 8)


def process_block(
    block: np.ndarray,
    quality: int = 50
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Process a single 8x8 block through JPEG compression pipeline
    
    Args:
        block: 8x8 pixel block
        quality: JPEG quality (1-100)
        
    Returns:
        Tuple of (dct, quantized, zigzag, reconstructed, error)
    """
    # Step 1: DCT
    dct = compute_dct(block)
    
    # Step 2: Quantization
    quantized = quantize(dct, quality)
    
    # Step 3: Zig-zag encoding
    zigzag = zigzag_encode(quantized)
    
    # Step 4: Reconstruction (for visualization)
    dequantized = dequantize(quantized, quality)
    reconstructed = compute_idct(dequantized)
    
    # Step 5: Compute error
    error = np.abs(block - reconstructed)
    
    return dct, quantized, zigzag, reconstructed, error


def compress_image(image: np.ndarray, quality: int = 50) -> np.ndarray:
    """
    Compress entire image using JPEG-like compression
    
    Args:
        image: Input image
        quality: JPEG quality (1-100)
        
    Returns:
        Compressed/reconstructed image
    """
    gray = to_grayscale(image)
    h, w = gray.shape
    
    # Pad image to multiple of 8
    pad_h = (8 - h % 8) % 8
    pad_w = (8 - w % 8) % 8
    padded = np.pad(gray, ((0, pad_h), (0, pad_w)), mode='edge')
    
    new_h, new_w = padded.shape
    result = np.zeros_like(padded, dtype=np.float32)
    
    # Process each 8x8 block
    for y in range(0, new_h, 8):
        for x in range(0, new_w, 8):
            block = padded[y:y+8, x:x+8].astype(np.float32)
            
            # Compress and decompress
            dct = compute_dct(block)
            quantized = quantize(dct, quality)
            dequantized = dequantize(quantized, quality)
            reconstructed = compute_idct(dequantized)
            
            result[y:y+8, x:x+8] = reconstructed
            
    # Remove padding
    result = result[:h, :w]
    
    return np.clip(result, 0, 255).astype(np.uint8)


def get_compression_stats(original: np.ndarray, compressed: np.ndarray) -> dict:
    """
    Calculate compression statistics
    
    Args:
        original: Original image
        compressed: Compressed image
        
    Returns:
        Dictionary with MSE, PSNR, etc.
    """
    original_gray = to_grayscale(original).astype(np.float32)
    compressed_gray = to_grayscale(compressed).astype(np.float32)
    
    # MSE
    mse = np.mean((original_gray - compressed_gray) ** 2)
    
    # PSNR
    if mse == 0:
        psnr = float('inf')
    else:
        psnr = 10 * np.log10(255**2 / mse)
        
    # SSIM-like simple metric
    mean_orig = np.mean(original_gray)
    mean_comp = np.mean(compressed_gray)
    var_orig = np.var(original_gray)
    var_comp = np.var(compressed_gray)
    covar = np.mean((original_gray - mean_orig) * (compressed_gray - mean_comp))
    
    c1 = (0.01 * 255) ** 2
    c2 = (0.03 * 255) ** 2
    
    ssim = ((2 * mean_orig * mean_comp + c1) * (2 * covar + c2)) / \
           ((mean_orig**2 + mean_comp**2 + c1) * (var_orig + var_comp + c2))
    
    return {
        'mse': mse,
        'psnr': psnr,
        'ssim': ssim,
    }


def format_matrix_for_display(matrix: np.ndarray, precision: int = 1) -> str:
    """
    Format a matrix as a string for display
    
    Args:
        matrix: 2D numpy array
        precision: Decimal places for floats
        
    Returns:
        Formatted string
    """
    rows = []
    for row in matrix:
        if matrix.dtype in [np.int32, np.int64]:
            row_str = ' '.join(f'{int(v):4d}' for v in row)
        else:
            row_str = ' '.join(f'{v:6.{precision}f}' for v in row)
        rows.append(row_str)
    return '\n'.join(rows)


def count_zeros(quantized: np.ndarray) -> int:
    """Count zeros in quantized block"""
    return int(np.sum(quantized == 0))


def count_trailing_zeros(zigzag: np.ndarray) -> int:
    """Count trailing zeros in zig-zag array"""
    count = 0
    for i in range(len(zigzag) - 1, -1, -1):
        if zigzag[i] == 0:
            count += 1
        else:
            break
    return count
