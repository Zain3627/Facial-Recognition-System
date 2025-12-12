#!/usr/bin/env python3
"""
Generate embeddings from LFW dataset using MediaPipe for face detection
and the transfer learning FaceNet model.

This script:
1. Iterates through the LFW dataset
2. Uses MediaPipe to detect and crop faces
3. Generates embeddings using the transfer learning model
4. Saves embeddings in the same format as embeddings_store.json

Usage:
    python generate_lfw_embeddings.py --output emb.json
    python generate_lfw_embeddings.py --max-persons 100 --max-images 5
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from PIL import Image
from tqdm import tqdm

# Add project directory to path
PROJECT_DIR = Path(__file__).resolve().parent
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

# Import after path setup
try:
    import mediapipe as mp
    MEDIAPIPE_AVAILABLE = True
except ImportError:
    MEDIAPIPE_AVAILABLE = False
    print("ERROR: MediaPipe is required. Install with: pip install mediapipe")
    sys.exit(1)

try:
    import tensorflow as tf
except ImportError:
    print("ERROR: TensorFlow is required. Install with: pip install tensorflow")
    sys.exit(1)


# Default paths
DEFAULT_LFW_DIR = PROJECT_DIR / "LFW" / "lfw-deepfunneled" / "lfw-deepfunneled"
DEFAULT_OUTPUT_PATH = PROJECT_DIR / "emb.json"
TRANSFER_MODEL_DIR = PROJECT_DIR / "deployment_model" / "facenet_transfer_model"
BACKBONE_MODEL_DIR = PROJECT_DIR / "models" / "facenet"


class FaceDetector:
    """MediaPipe-based face detector for LFW processing."""
    
    def __init__(
        self,
        min_confidence: float = 0.5,
        margin: float = 0.2,
        model_selection: int = 1,  # Full-range model for LFW images
    ) -> None:
        self.min_confidence = min_confidence
        self.margin = margin
        
        self._face_detection = mp.solutions.face_detection.FaceDetection(
            min_detection_confidence=min_confidence,
            model_selection=model_selection,
        )
    
    def detect_and_crop(
        self,
        image: np.ndarray,
        target_size: Tuple[int, int] = (160, 160),
    ) -> Optional[np.ndarray]:
        """Detect face in image and return cropped face.
        
        Args:
            image: RGB image as numpy array (H, W, 3), uint8 [0,255]
            target_size: Output size (width, height)
            
        Returns:
            Cropped face as float32 [0,1] or None if no face detected.
        """
        img_h, img_w = image.shape[:2]
        
        # Run detection
        results = self._face_detection.process(image)
        
        if not results.detections:
            return None
        
        # Take the first (usually most confident) detection
        detection = results.detections[0]
        bbox = detection.location_data.relative_bounding_box
        
        # Convert relative to absolute coordinates
        x = int(bbox.xmin * img_w)
        y = int(bbox.ymin * img_h)
        width = int(bbox.width * img_w)
        height = int(bbox.height * img_h)
        
        # Add margin
        margin_x = int(width * self.margin)
        margin_y = int(height * self.margin)
        
        x1 = max(0, x - margin_x)
        y1 = max(0, y - margin_y)
        x2 = min(img_w, x + width + margin_x)
        y2 = min(img_h, y + height + margin_y)
        
        # Crop
        crop = image[y1:y2, x1:x2]
        
        # Resize using PIL
        pil_crop = Image.fromarray(crop)
        pil_crop = pil_crop.resize(target_size, Image.Resampling.LANCZOS)
        
        # Convert to float32 [0, 1]
        return np.array(pil_crop, dtype=np.float32) / 255.0
    
    def close(self):
        """Release resources."""
        self._face_detection.close()


class EmbeddingGenerator:
    """Generate embeddings using the FaceNet backbone model."""
    
    def __init__(self, model_dir: Path, batch_size: int = 32) -> None:
        if not model_dir.exists():
            raise FileNotFoundError(f"Model directory not found: {model_dir}")
        
        print(f"Loading model from {model_dir}...")
        self._model = tf.saved_model.load(str(model_dir))
        self._infer = self._model.signatures["serving_default"]
        self._batch_size = batch_size
        
        # Find embedding output key
        self._embedding_key = self._find_embedding_key()
        print(f"Using output key: {self._embedding_key}")
    
    def _find_embedding_key(self) -> str:
        """Find the embedding output key from model outputs."""
        preferred_keys = ("Bottleneck_BatchNorm", "embeddings", "output_0")
        for key in preferred_keys:
            if key in self._infer.structured_outputs:
                return key
        return next(iter(self._infer.structured_outputs))
    
    @staticmethod
    def _preprocess(images: tf.Tensor) -> tf.Tensor:
        """Preprocess images for FaceNet: resize and scale to [-1, 1]."""
        resized = tf.image.resize(images, (160, 160))
        scaled = (resized * 2.0) - 1.0
        return scaled
    
    @staticmethod
    def _normalize(embeddings: np.ndarray) -> np.ndarray:
        """L2 normalize embeddings."""
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        norms = np.maximum(norms, 1e-12)
        return embeddings / norms
    
    def generate(self, images: List[np.ndarray]) -> np.ndarray:
        """Generate embeddings for a list of images.
        
        Args:
            images: List of face images as float32 [0,1] arrays (H, W, 3)
            
        Returns:
            L2-normalized embeddings as (N, embedding_dim) array.
        """
        if not images:
            return np.array([])
        
        # Stack images
        stacked = np.stack(images, axis=0)
        
        all_embeddings = []
        
        for start in range(0, len(stacked), self._batch_size):
            end = start + self._batch_size
            batch = stacked[start:end]
            
            # Convert and preprocess
            batch_tensor = tf.convert_to_tensor(batch, dtype=tf.float32)
            batch_tensor = self._preprocess(batch_tensor)
            
            # Get embeddings
            outputs = self._infer(batch_tensor)
            embeddings = outputs.get(self._embedding_key, next(iter(outputs.values())))
            all_embeddings.append(embeddings.numpy())
        
        merged = np.vstack(all_embeddings)
        return self._normalize(merged)


def get_lfw_persons(lfw_dir: Path) -> Dict[str, List[Path]]:
    """Get all persons and their image paths from LFW directory.
    
    Returns:
        Dictionary mapping person name to list of image paths.
    """
    persons = {}
    
    for person_dir in sorted(lfw_dir.iterdir()):
        if not person_dir.is_dir():
            continue
        
        # Get all jpg images
        images = sorted(person_dir.glob("*.jpg"))
        if images:
            persons[person_dir.name] = images
    
    return persons


def load_image(path: Path) -> np.ndarray:
    """Load image as RGB uint8 array."""
    img = Image.open(path).convert("RGB")
    return np.array(img)


def generate_embeddings(
    lfw_dir: Path,
    output_path: Path,
    model_dir: Path,
    min_images: int = 1,
    max_persons: Optional[int] = None,
    max_images_per_person: Optional[int] = None,
    min_confidence: float = 0.5,
) -> None:
    """Generate embeddings from LFW dataset.
    
    Args:
        lfw_dir: Path to LFW dataset directory.
        output_path: Path to save output JSON.
        model_dir: Path to FaceNet model directory.
        min_images: Minimum images required per person.
        max_persons: Maximum number of persons to process.
        max_images_per_person: Maximum images to use per person.
        min_confidence: Minimum face detection confidence.
    """
    print("=" * 60)
    print("LFW Embeddings Generator")
    print("=" * 60)
    print(f"LFW Directory: {lfw_dir}")
    print(f"Output Path: {output_path}")
    print(f"Model Directory: {model_dir}")
    print(f"Min Images per Person: {min_images}")
    print(f"Max Persons: {max_persons or 'All'}")
    print(f"Max Images per Person: {max_images_per_person or 'All'}")
    print("=" * 60)
    
    # Initialize components
    face_detector = FaceDetector(min_confidence=min_confidence)
    embedding_generator = EmbeddingGenerator(model_dir)
    
    # Get all persons
    print("\nScanning LFW directory...")
    persons = get_lfw_persons(lfw_dir)
    print(f"Found {len(persons)} persons in LFW dataset")
    
    # Filter by minimum images
    persons = {k: v for k, v in persons.items() if len(v) >= min_images}
    print(f"Persons with >= {min_images} images: {len(persons)}")
    
    # Limit persons if specified
    if max_persons:
        person_names = list(persons.keys())[:max_persons]
        persons = {k: persons[k] for k in person_names}
        print(f"Limited to {len(persons)} persons")
    
    # Process each person
    users = []
    stats = {"processed": 0, "skipped": 0, "no_face": 0}
    
    print("\nProcessing persons...")
    for person_name, image_paths in tqdm(persons.items(), desc="Persons"):
        # Limit images per person
        if max_images_per_person:
            image_paths = image_paths[:max_images_per_person]
        
        # Process images
        cropped_faces = []
        
        for img_path in image_paths:
            try:
                # Load image
                image = load_image(img_path)
                
                # Detect and crop face
                face = face_detector.detect_and_crop(image)
                
                if face is not None:
                    cropped_faces.append(face)
                else:
                    stats["no_face"] += 1
                    
            except Exception as e:
                tqdm.write(f"Error processing {img_path}: {e}")
                stats["skipped"] += 1
        
        if not cropped_faces:
            stats["skipped"] += 1
            continue
        
        # Generate embeddings
        embeddings = embedding_generator.generate(cropped_faces)
        
        # Format name (replace underscores with spaces)
        display_name = person_name.replace("_", " ")
        
        # Create user entry in same format as embeddings_store.json
        now = datetime.utcnow().isoformat()
        user_entry = {
            "label": display_name,
            "embeddings": embeddings.tolist(),
            "created_at": now,
            "updated_at": now,
        }
        
        users.append(user_entry)
        stats["processed"] += 1
    
    # Close face detector
    face_detector.close()
    
    # Save to JSON
    print("\nSaving embeddings...")
    output_data = {"users": users}
    
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output_data, f, indent=2)
    
    # Print statistics
    print("\n" + "=" * 60)
    print("COMPLETED")
    print("=" * 60)
    print(f"Persons processed: {stats['processed']}")
    print(f"Persons skipped (no faces): {stats['skipped']}")
    print(f"Images with no face detected: {stats['no_face']}")
    print(f"Total embeddings: {sum(len(u['embeddings']) for u in users)}")
    print(f"Output saved to: {output_path}")
    print("=" * 60)


def main():
    parser = argparse.ArgumentParser(
        description="Generate embeddings from LFW dataset using MediaPipe and FaceNet.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    
    parser.add_argument(
        "--lfw-dir",
        type=Path,
        default=DEFAULT_LFW_DIR,
        help="Path to LFW dataset directory",
    )
    
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT_PATH,
        help="Output JSON file path",
    )
    
    parser.add_argument(
        "--model-dir",
        type=Path,
        default=BACKBONE_MODEL_DIR,
        help="Path to FaceNet model directory",
    )
    
    parser.add_argument(
        "--min-images",
        type=int,
        default=1,
        help="Minimum images required per person",
    )
    
    parser.add_argument(
        "--max-persons",
        type=int,
        default=None,
        help="Maximum number of persons to process (None = all)",
    )
    
    parser.add_argument(
        "--max-images-per-person",
        type=int,
        default=None,
        help="Maximum images to use per person (None = all)",
    )
    
    parser.add_argument(
        "--min-confidence",
        type=float,
        default=0.5,
        help="Minimum face detection confidence",
    )
    
    args = parser.parse_args()
    
    # Validate paths
    if not args.lfw_dir.exists():
        print(f"ERROR: LFW directory not found: {args.lfw_dir}")
        sys.exit(1)
    
    if not args.model_dir.exists():
        print(f"ERROR: Model directory not found: {args.model_dir}")
        sys.exit(1)
    
    # Run generation
    generate_embeddings(
        lfw_dir=args.lfw_dir,
        output_path=args.output,
        model_dir=args.model_dir,
        min_images=args.min_images,
        max_persons=args.max_persons,
        max_images_per_person=args.max_images_per_person,
        min_confidence=args.min_confidence,
    )


if __name__ == "__main__":
    main()
