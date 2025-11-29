"""Utility helpers for decoding and preparing images."""

from __future__ import annotations

import io
from typing import Optional

import numpy as np
from PIL import Image


def load_image_as_array(image_bytes: bytes) -> np.ndarray:
    """Return an RGB float32 numpy array in the [0, 1] range."""
    with Image.open(io.BytesIO(image_bytes)) as img:
        rgb_image = img.convert("RGB")
        array = np.asarray(rgb_image, dtype=np.float32)
    if array.max() > 1.0:
        array /= 255.0
    return array


def ensure_batch_dimension(image: np.ndarray) -> np.ndarray:
    """Guarantee a batch dimension for downstream models."""
    if image.ndim == 3:
        return image[None, ...]
    if image.ndim == 4:
        return image
    raise ValueError("Expected image with rank 3 or 4")


def resize_with_aspect_ratio(image: np.ndarray, size: int) -> np.ndarray:
    """Resize an image so the shortest side matches ``size`` while keeping aspect."""
    with Image.fromarray((image * 255).astype("uint8")) as pil_image:
        ratio = size / min(pil_image.size)
        resized = pil_image.resize(
            (int(round(pil_image.width * ratio)), int(round(pil_image.height * ratio)))
        )
        array = np.asarray(resized, dtype=np.float32) / 255.0
    return array
