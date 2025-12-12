"""
Generate embeddings for LFW dataset using FaceNet backbone and optional transfer head.

This script processes the LFW (Labeled Faces in the Wild) dataset and generates
embeddings using the FaceNet model. Optionally, it can also apply the trained
transfer head for classification on degraded images.

Usage:
    python generate_lfw_embeddings.py --output embeddings_output.json
    python generate_lfw_embeddings.py --output embeddings_output.json --max-per-person 5
    python generate_lfw_embeddings.py --output embeddings_output.json --apply-degradation
"""

from __future__ import annotations

import argparse
import json
import os
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import tensorflow as tf

# Force CPU to avoid GPU memory issues
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'


class FaceNetEmbedder:
    """FaceNet embedding extractor using the pre-trained backbone."""
    
    def __init__(
        self,
        backbone_dir: Path,
        transfer_head_path: Optional[Path] = None,
        embedding_dim: int = 128,
        batch_size: int = 32
    ):
        """
        Initialize the FaceNet embedder.
        
        Args:
            backbone_dir: Path to the FaceNet SavedModel directory
            transfer_head_path: Optional path to the trained transfer head (.keras)
            embedding_dim: Dimension of embeddings (default 128)
            batch_size: Batch size for processing
        """
        self.backbone_dir = backbone_dir
        self.embedding_dim = embedding_dim
        self.batch_size = batch_size
        
        # Load the FaceNet backbone
        print(f"Loading FaceNet backbone from {backbone_dir}...")
        self.backbone = tf.saved_model.load(str(backbone_dir))
        self._infer = self.backbone.signatures["serving_default"]
        self._embedding_key = "Bottleneck_BatchNorm"
        
        # Load transfer head if provided
        self.transfer_head = None
        if transfer_head_path and transfer_head_path.exists():
            print(f"Loading transfer head from {transfer_head_path}...")
            self.transfer_head = tf.keras.models.load_model(str(transfer_head_path))
            
            # Load metadata if available
            metadata_path = transfer_head_path.parent / "metadata.json"
            if metadata_path.exists():
                with open(metadata_path, "r") as f:
                    self.metadata = json.load(f)
                    self.label_mapping = self.metadata.get("label_mapping", {})
            else:
                self.metadata = {}
                self.label_mapping = {}
        
        print("Model(s) loaded successfully!")
    
    def _preprocess(self, image: np.ndarray) -> np.ndarray:
        """
        Preprocess image for FaceNet.
        
        Args:
            image: BGR image (OpenCV format)
            
        Returns:
            Preprocessed image tensor
        """
        # Convert BGR to RGB
        if len(image.shape) == 3 and image.shape[2] == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Resize to 160x160
        image = cv2.resize(image, (160, 160))
        
        # Scale to [-1, 1]
        image = image.astype(np.float32)
        image = (image - 127.5) / 127.5
        
        return image
    
    def _normalise(self, embeddings: np.ndarray) -> np.ndarray:
        """L2 normalize embeddings."""
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        norms = np.maximum(norms, 1e-10)  # Avoid division by zero
        return embeddings / norms
    
    def apply_degradation(
        self,
        image: np.ndarray,
        gamma: float = 2.5,
        brightness: float = 0.5,
        blur_kernel: int = 5
    ) -> np.ndarray:
        """
        Apply degradation transformations to simulate low-quality images.
        
        Args:
            image: Input image
            gamma: Gamma value for power transform (>1 darkens)
            brightness: Brightness multiplier (<1 darkens)
            blur_kernel: Kernel size for Gaussian blur
            
        Returns:
            Degraded image
        """
        # Gamma correction (power transform)
        img = image.astype(np.float32) / 255.0
        img = np.power(img, gamma)
        
        # Reduce brightness
        img = img * brightness
        
        # Clip to valid range and convert back
        img = np.clip(img * 255, 0, 255).astype(np.uint8)
        
        # Apply blur
        if blur_kernel > 0:
            img = cv2.GaussianBlur(img, (blur_kernel, blur_kernel), 0)
        
        return img
    
    def embed_single(self, image: np.ndarray, apply_degradation: bool = False) -> np.ndarray:
        """
        Get embedding for a single image.
        
        Args:
            image: BGR image
            apply_degradation: Whether to apply degradation before embedding
            
        Returns:
            Normalized embedding vector
        """
        if apply_degradation:
            image = self.apply_degradation(image)
        
        preprocessed = self._preprocess(image)
        batch = np.expand_dims(preprocessed, axis=0)
        
        tensor = tf.constant(batch, dtype=tf.float32)
        output = self._infer(tensor)
        embedding = output[self._embedding_key].numpy()
        
        return self._normalise(embedding)[0]
    
    def embed_batch(
        self,
        images: List[np.ndarray],
        apply_degradation: bool = False
    ) -> np.ndarray:
        """
        Get embeddings for a batch of images.
        
        Args:
            images: List of BGR images
            apply_degradation: Whether to apply degradation before embedding
            
        Returns:
            Array of normalized embeddings
        """
        if not images:
            return np.array([])
        
        # Apply degradation if requested
        if apply_degradation:
            images = [self.apply_degradation(img) for img in images]
        
        # Preprocess all images
        preprocessed = np.array([self._preprocess(img) for img in images])
        
        # Process in batches
        all_embeddings = []
        for i in range(0, len(preprocessed), self.batch_size):
            batch = preprocessed[i:i + self.batch_size]
            tensor = tf.constant(batch, dtype=tf.float32)
            output = self._infer(tensor)
            embeddings = output[self._embedding_key].numpy()
            all_embeddings.append(embeddings)
        
        embeddings = np.vstack(all_embeddings)
        return self._normalise(embeddings)
    
    def predict_class(self, embedding: np.ndarray) -> Tuple[int, float, str]:
        """
        Predict class using transfer head.
        
        Args:
            embedding: Normalized embedding vector
            
        Returns:
            Tuple of (class_index, confidence, class_name)
        """
        if self.transfer_head is None:
            raise ValueError("Transfer head not loaded")
        
        embedding = np.expand_dims(embedding, axis=0)
        predictions = self.transfer_head.predict(embedding, verbose=0)
        class_idx = np.argmax(predictions[0])
        confidence = float(predictions[0][class_idx])
        
        # Get class name from mapping
        class_name = self.label_mapping.get(str(class_idx), f"class_{class_idx}")
        
        return int(class_idx), confidence, class_name


def collect_lfw_images(
    lfw_dir: Path,
    max_per_person: Optional[int] = None,
    min_images: int = 1
) -> Dict[str, List[Path]]:
    """
    Collect image paths from LFW dataset.
    
    Args:
        lfw_dir: Path to LFW directory (containing person subdirectories)
        max_per_person: Maximum images per person (None for all)
        min_images: Minimum images required per person
        
    Returns:
        Dictionary mapping person name to list of image paths
    """
    person_images: Dict[str, List[Path]] = {}
    
    for person_dir in sorted(lfw_dir.iterdir()):
        if not person_dir.is_dir():
            continue
        
        images = sorted(person_dir.glob("*.jpg"))
        
        if len(images) < min_images:
            continue
        
        if max_per_person:
            images = images[:max_per_person]
        
        person_images[person_dir.name] = images
    
    return person_images


def main():
    parser = argparse.ArgumentParser(
        description="Generate embeddings for LFW dataset using FaceNet"
    )
    parser.add_argument(
        "--lfw-dir",
        type=Path,
        default=Path(__file__).parent.parent / "LFW" / "lfw-deepfunneled" / "lfw-deepfunneled",
        help="Path to LFW deepfunneled directory"
    )
    parser.add_argument(
        "--backbone-dir",
        type=Path,
        default=Path(__file__).parent.parent / "models" / "facenet",
        help="Path to FaceNet SavedModel directory"
    )
    parser.add_argument(
        "--transfer-head",
        type=Path,
        default=None,
        help="Path to trained transfer head (.keras file)"
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).parent / "lfw_embeddings.json",
        help="Output JSON file path"
    )
    parser.add_argument(
        "--max-per-person",
        type=int,
        default=None,
        help="Maximum images per person"
    )
    parser.add_argument(
        "--min-images",
        type=int,
        default=1,
        help="Minimum images required per person"
    )
    parser.add_argument(
        "--apply-degradation",
        action="store_true",
        help="Apply image degradation before embedding"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Batch size for processing"
    )
    
    args = parser.parse_args()
    
    # Validate paths
    if not args.lfw_dir.exists():
        print(f"Error: LFW directory not found: {args.lfw_dir}")
        return 1
    
    if not args.backbone_dir.exists():
        print(f"Error: FaceNet backbone not found: {args.backbone_dir}")
        return 1
    
    # Initialize embedder
    embedder = FaceNetEmbedder(
        backbone_dir=args.backbone_dir,
        transfer_head_path=args.transfer_head,
        batch_size=args.batch_size
    )
    
    # Collect images
    print(f"\nCollecting images from {args.lfw_dir}...")
    person_images = collect_lfw_images(
        args.lfw_dir,
        max_per_person=args.max_per_person,
        min_images=args.min_images
    )
    
    total_people = len(person_images)
    total_images = sum(len(imgs) for imgs in person_images.values())
    print(f"Found {total_people} people with {total_images} total images")
    
    # Generate embeddings
    print(f"\nGenerating embeddings (degradation={'enabled' if args.apply_degradation else 'disabled'})...")
    
    # Use the same format as EmbeddingStore
    results = {
        "users": []
    }
    
    for idx, (person_name, image_paths) in enumerate(person_images.items()):
        print(f"  [{idx+1}/{total_people}] Processing {person_name} ({len(image_paths)} images)...")
        
        # Load images
        images = []
        valid_paths = []
        for img_path in image_paths:
            img = cv2.imread(str(img_path))
            if img is not None:
                images.append(img)
                valid_paths.append(img_path)
        
        if not images:
            print(f"    Warning: Could not load any images for {person_name}")
            continue
        
        # Generate embeddings
        embeddings = embedder.embed_batch(images, apply_degradation=args.apply_degradation)
        
        # Store results in EmbeddingStore format
        now = datetime.utcnow().isoformat()
        person_data = {
            "label": person_name,
            "embeddings": [emb.tolist() for emb in embeddings],
            "created_at": now,
            "updated_at": now
        }
        
        results["users"].append(person_data)
    
    # Save results
    print(f"\nSaving embeddings to {args.output}...")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    
    print(f"\nDone! Generated embeddings for {len(results['users'])} people.")
    return 0


if __name__ == "__main__":
    exit(main())
