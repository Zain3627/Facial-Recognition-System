"""Face detection using MediaPipe."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

import numpy as np

try:
    import mediapipe as mp
    MEDIAPIPE_AVAILABLE = True
except ImportError:
    MEDIAPIPE_AVAILABLE = False
    mp = None


@dataclass
class FaceRegion:
    """Represents a detected face region."""
    x: int  # Top-left x coordinate
    y: int  # Top-left y coordinate
    width: int
    height: int
    confidence: float
    
    @property
    def center(self) -> tuple[int, int]:
        """Return center point of the face region."""
        return (self.x + self.width // 2, self.y + self.height // 2)
    
    @property
    def box(self) -> tuple[int, int, int, int]:
        """Return (x1, y1, x2, y2) bounding box."""
        return (self.x, self.y, self.x + self.width, self.y + self.height)


class FaceDetector:
    """Face detector using MediaPipe Face Detection."""
    
    def __init__(
        self,
        min_confidence: float = 0.5,
        min_face_size: int = 20,
        margin: float = 0.2,
        model_selection: int = 0,
    ) -> None:
        """Initialize the face detector.
        
        Args:
            min_confidence: Minimum confidence threshold for detections.
            min_face_size: Minimum face size in pixels to consider.
            margin: Relative margin to add around detected faces.
            model_selection: 0 for short-range (2m), 1 for full-range (5m).
        """
        self.min_confidence = min_confidence
        self.min_face_size = min_face_size
        self.margin = margin
        
        if not MEDIAPIPE_AVAILABLE:
            raise ImportError("MediaPipe is required for face detection. Install with: pip install mediapipe")
        
        self._face_detection = mp.solutions.face_detection.FaceDetection(
            min_detection_confidence=min_confidence,
            model_selection=model_selection,
        )
    
    def detect_faces(self, image: np.ndarray) -> List[FaceRegion]:
        """Detect faces in an image.
        
        Args:
            image: RGB image as numpy array (H, W, 3). 
                   Can be float32 [0,1] or uint8 [0,255].
                   
        Returns:
            List of FaceRegion objects for detected faces.
        """
        # Convert to uint8 if needed
        if image.dtype == np.float32 or image.dtype == np.float64:
            if image.max() <= 1.0:
                image_uint8 = (image * 255).astype(np.uint8)
            else:
                image_uint8 = image.astype(np.uint8)
        else:
            image_uint8 = image
        
        img_h, img_w = image_uint8.shape[:2]
        
        # Run detection
        results = self._face_detection.process(image_uint8)
        
        faces: List[FaceRegion] = []
        
        if results.detections:
            for detection in results.detections:
                bbox = detection.location_data.relative_bounding_box
                
                # Convert relative coordinates to absolute
                x = int(bbox.xmin * img_w)
                y = int(bbox.ymin * img_h)
                width = int(bbox.width * img_w)
                height = int(bbox.height * img_h)
                
                # Skip faces that are too small
                if width < self.min_face_size or height < self.min_face_size:
                    continue
                
                confidence = detection.score[0] if detection.score else 0.0
                
                faces.append(FaceRegion(
                    x=x,
                    y=y,
                    width=width,
                    height=height,
                    confidence=confidence,
                ))
        
        return faces
    
    def crop_face(
        self,
        image: np.ndarray,
        face: FaceRegion,
        target_size: Optional[tuple[int, int]] = None,
    ) -> np.ndarray:
        """Crop a face region from an image.
        
        Args:
            image: Source image as numpy array (H, W, 3).
            face: FaceRegion to crop.
            target_size: Optional (width, height) to resize the crop.
            
        Returns:
            Cropped face image as numpy array.
        """
        img_h, img_w = image.shape[:2]
        
        # Add margin
        margin_x = int(face.width * self.margin)
        margin_y = int(face.height * self.margin)
        
        x1 = max(0, face.x - margin_x)
        y1 = max(0, face.y - margin_y)
        x2 = min(img_w, face.x + face.width + margin_x)
        y2 = min(img_h, face.y + face.height + margin_y)
        
        crop = image[y1:y2, x1:x2]
        
        if target_size is not None:
            from PIL import Image as PILImage
            
            # Convert to PIL for resizing
            if crop.dtype == np.float32 and crop.max() <= 1.0:
                pil_crop = PILImage.fromarray((crop * 255).astype(np.uint8))
            else:
                pil_crop = PILImage.fromarray(crop.astype(np.uint8))
            
            pil_crop = pil_crop.resize(target_size, PILImage.Resampling.LANCZOS)
            
            crop = np.array(pil_crop, dtype=np.float32) / 255.0
        
        return crop
    
    def __del__(self):
        """Clean up MediaPipe resources."""
        if hasattr(self, '_face_detection'):
            self._face_detection.close()
