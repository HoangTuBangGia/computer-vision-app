"""
Geometry Operations
Rotate, Scale, Flip transformations
"""
import numpy as np
import cv2
from typing import Tuple, Optional


def rotate_image(
    image: np.ndarray,
    angle: float,
    center: Optional[Tuple[int, int]] = None,
    scale: float = 1.0,
    border_mode: int = cv2.BORDER_CONSTANT,
    border_value: Tuple[int, ...] = (0, 0, 0)
) -> np.ndarray:
    """
    Rotate image by given angle
    
    Args:
        image: Input image
        angle: Rotation angle in degrees (positive = counter-clockwise)
        center: Rotation center (default: image center)
        scale: Optional scaling factor
        border_mode: Border interpolation mode
        border_value: Border fill value
        
    Returns:
        Rotated image
    """
    h, w = image.shape[:2]
    
    if center is None:
        center = (w // 2, h // 2)
        
    # Get rotation matrix
    rotation_matrix = cv2.getRotationMatrix2D(center, angle, scale)
    
    # Calculate new image bounds to fit entire rotated image
    cos = np.abs(rotation_matrix[0, 0])
    sin = np.abs(rotation_matrix[0, 1])
    
    new_w = int((h * sin) + (w * cos))
    new_h = int((h * cos) + (w * sin))
    
    # Adjust rotation matrix for new bounds
    rotation_matrix[0, 2] += (new_w / 2) - center[0]
    rotation_matrix[1, 2] += (new_h / 2) - center[1]
    
    # Apply rotation
    rotated = cv2.warpAffine(
        image,
        rotation_matrix,
        (new_w, new_h),
        flags=cv2.INTER_LINEAR,
        borderMode=border_mode,
        borderValue=border_value
    )
    
    return rotated


def rotate_image_keep_size(
    image: np.ndarray,
    angle: float,
    border_value: Tuple[int, ...] = (0, 0, 0)
) -> np.ndarray:
    """
    Rotate image by given angle, keeping original size
    
    Args:
        image: Input image
        angle: Rotation angle in degrees
        border_value: Border fill value
        
    Returns:
        Rotated image with same size as input
    """
    h, w = image.shape[:2]
    center = (w // 2, h // 2)
    
    rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
    
    rotated = cv2.warpAffine(
        image,
        rotation_matrix,
        (w, h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=border_value
    )
    
    return rotated


def scale_image(
    image: np.ndarray,
    scale_x: float = 1.0,
    scale_y: float = 1.0,
    interpolation: int = cv2.INTER_LINEAR
) -> np.ndarray:
    """
    Scale image by given factors
    
    Args:
        image: Input image
        scale_x: Horizontal scale factor
        scale_y: Vertical scale factor
        interpolation: Interpolation method
        
    Returns:
        Scaled image
    """
    h, w = image.shape[:2]
    new_w = int(w * scale_x)
    new_h = int(h * scale_y)
    
    # Ensure minimum size
    new_w = max(1, new_w)
    new_h = max(1, new_h)
    
    scaled = cv2.resize(image, (new_w, new_h), interpolation=interpolation)
    
    return scaled


def resize_image(
    image: np.ndarray,
    width: Optional[int] = None,
    height: Optional[int] = None,
    interpolation: int = cv2.INTER_LINEAR
) -> np.ndarray:
    """
    Resize image to specific dimensions
    
    Args:
        image: Input image
        width: Target width (None to calculate from height)
        height: Target height (None to calculate from width)
        interpolation: Interpolation method
        
    Returns:
        Resized image
    """
    h, w = image.shape[:2]
    
    if width is None and height is None:
        return image.copy()
        
    if width is None:
        # Calculate width from height
        aspect = w / h
        width = int(height * aspect)
    elif height is None:
        # Calculate height from width
        aspect = h / w
        height = int(width * aspect)
        
    # Ensure minimum size
    width = max(1, width)
    height = max(1, height)
    
    resized = cv2.resize(image, (width, height), interpolation=interpolation)
    
    return resized


def flip_horizontal(image: np.ndarray) -> np.ndarray:
    """
    Flip image horizontally (mirror)
    
    Args:
        image: Input image
        
    Returns:
        Horizontally flipped image
    """
    return cv2.flip(image, 1)


def flip_vertical(image: np.ndarray) -> np.ndarray:
    """
    Flip image vertically
    
    Args:
        image: Input image
        
    Returns:
        Vertically flipped image
    """
    return cv2.flip(image, 0)


def flip_both(image: np.ndarray) -> np.ndarray:
    """
    Flip image both horizontally and vertically (180 degree rotation)
    
    Args:
        image: Input image
        
    Returns:
        Flipped image
    """
    return cv2.flip(image, -1)


def crop_image(
    image: np.ndarray,
    x: int,
    y: int,
    width: int,
    height: int
) -> np.ndarray:
    """
    Crop a region from image
    
    Args:
        image: Input image
        x: Left coordinate
        y: Top coordinate
        width: Crop width
        height: Crop height
        
    Returns:
        Cropped image
    """
    h, w = image.shape[:2]
    
    # Clamp coordinates
    x = max(0, min(x, w - 1))
    y = max(0, min(y, h - 1))
    x2 = min(x + width, w)
    y2 = min(y + height, h)
    
    return image[y:y2, x:x2].copy()


def translate_image(
    image: np.ndarray,
    dx: int,
    dy: int,
    border_value: Tuple[int, ...] = (0, 0, 0)
) -> np.ndarray:
    """
    Translate (shift) image by given amounts
    
    Args:
        image: Input image
        dx: Horizontal shift (positive = right)
        dy: Vertical shift (positive = down)
        border_value: Border fill value
        
    Returns:
        Translated image
    """
    h, w = image.shape[:2]
    
    # Translation matrix
    translation_matrix = np.float32([
        [1, 0, dx],
        [0, 1, dy]
    ])
    
    translated = cv2.warpAffine(
        image,
        translation_matrix,
        (w, h),
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=border_value
    )
    
    return translated


def shear_image(
    image: np.ndarray,
    shear_x: float = 0.0,
    shear_y: float = 0.0,
    border_value: Tuple[int, ...] = (0, 0, 0)
) -> np.ndarray:
    """
    Apply shear transformation
    
    Args:
        image: Input image
        shear_x: Horizontal shear factor
        shear_y: Vertical shear factor
        border_value: Border fill value
        
    Returns:
        Sheared image
    """
    h, w = image.shape[:2]
    
    # Shear matrix
    shear_matrix = np.float32([
        [1, shear_x, 0],
        [shear_y, 1, 0]
    ])
    
    # Calculate output size
    corners = np.array([
        [0, 0, 1],
        [w, 0, 1],
        [0, h, 1],
        [w, h, 1]
    ], dtype=np.float32)
    
    new_corners = corners @ shear_matrix.T
    
    min_x = int(np.floor(new_corners[:, 0].min()))
    max_x = int(np.ceil(new_corners[:, 0].max()))
    min_y = int(np.floor(new_corners[:, 1].min()))
    max_y = int(np.ceil(new_corners[:, 1].max()))
    
    # Adjust matrix
    shear_matrix[0, 2] = -min_x
    shear_matrix[1, 2] = -min_y
    
    new_w = max_x - min_x
    new_h = max_y - min_y
    
    sheared = cv2.warpAffine(
        image,
        shear_matrix,
        (new_w, new_h),
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=border_value
    )
    
    return sheared


def apply_geometry(
    image: np.ndarray,
    operation: str,
    params: dict
) -> np.ndarray:
    """
    Apply geometry operation based on name and parameters
    
    Args:
        image: Input image
        operation: Operation name
        params: Operation parameters
        
    Returns:
        Transformed image
    """
    if operation == "rotate":
        angle = params.get("angle", 0)
        keep_size = params.get("keep_size", True)
        if keep_size:
            return rotate_image_keep_size(image, angle)
        else:
            return rotate_image(image, angle)
            
    elif operation == "scale":
        scale_x = params.get("scale_x", 1.0)
        scale_y = params.get("scale_y", 1.0)
        return scale_image(image, scale_x, scale_y)
        
    elif operation == "resize":
        width = params.get("width")
        height = params.get("height")
        return resize_image(image, width, height)
        
    elif operation == "flip_h":
        return flip_horizontal(image)
        
    elif operation == "flip_v":
        return flip_vertical(image)
        
    elif operation == "flip_both":
        return flip_both(image)
        
    elif operation == "translate":
        dx = params.get("dx", 0)
        dy = params.get("dy", 0)
        return translate_image(image, dx, dy)
        
    elif operation == "shear":
        shear_x = params.get("shear_x", 0)
        shear_y = params.get("shear_y", 0)
        return shear_image(image, shear_x, shear_y)
        
    else:
        return image.copy()
