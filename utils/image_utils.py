"""Image loading and processing utilities."""

from __future__ import annotations

import io
from typing import Union

import numpy as np
from PIL import Image


def load_image_as_array(image_data: Union[bytes, str, np.ndarray]) -> np.ndarray:
    """Load an image and return as RGB float32 array normalized to [0, 1].
    
    Args:
        image_data: Image bytes, file path, or existing numpy array.
        
    Returns:
        RGB image as float32 array with shape (H, W, 3) and values in [0, 1].
    """
    if isinstance(image_data, np.ndarray):
        # Already an array, ensure correct format
        if image_data.dtype == np.uint8:
            return image_data.astype(np.float32) / 255.0
        elif image_data.max() > 1.0:
            return image_data.astype(np.float32) / 255.0
        return image_data.astype(np.float32)
    
    if isinstance(image_data, bytes):
        img = Image.open(io.BytesIO(image_data))
    else:
        img = Image.open(image_data)
    
    # Convert to RGB if needed
    if img.mode != "RGB":
        img = img.convert("RGB")
    
    # Convert to numpy array and normalize
    arr = np.array(img, dtype=np.float32) / 255.0
    return arr


def crop_face_region(
    image: np.ndarray,
    x: int,
    y: int,
    width: int,
    height: int,
    margin: float = 0.2,
) -> np.ndarray:
    """Crop a face region from an image with optional margin.
    
    Args:
        image: Source image as numpy array (H, W, 3).
        x, y: Top-left corner of face bounding box.
        width, height: Size of face bounding box.
        margin: Relative margin to add around the face (0.2 = 20%).
        
    Returns:
        Cropped face region as numpy array.
    """
    img_h, img_w = image.shape[:2]
    
    # Add margin
    margin_x = int(width * margin)
    margin_y = int(height * margin)
    
    x1 = max(0, x - margin_x)
    y1 = max(0, y - margin_y)
    x2 = min(img_w, x + width + margin_x)
    y2 = min(img_h, y + height + margin_y)
    
    return image[y1:y2, x1:x2]


def resize_image(image: np.ndarray, target_size: tuple[int, int]) -> np.ndarray:
    """Resize an image to target size.
    
    Args:
        image: Source image as numpy array.
        target_size: (width, height) tuple.
        
    Returns:
        Resized image as numpy array.
    """
    # Convert to PIL, resize, convert back
    if image.max() <= 1.0:
        pil_img = Image.fromarray((image * 255).astype(np.uint8))
    else:
        pil_img = Image.fromarray(image.astype(np.uint8))
    
    resized = pil_img.resize(target_size, Image.Resampling.LANCZOS)
    
    return np.array(resized, dtype=np.float32) / 255.0
