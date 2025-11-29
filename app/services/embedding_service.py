"""Service for loading FaceNet and producing normalized embeddings."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Sequence

import numpy as np
import tensorflow as tf


@dataclass
class EmbeddingBatch:
    """Container for embeddings and their source indexes."""

    embeddings: np.ndarray
    indexes: List[int]


class FaceNetEmbeddingService:
    """Wraps the FaceNet SavedModel to generate L2-normalised embeddings."""

    def __init__(
        self,
        backbone_dir: Path,
        embedding_dim: int,
        batch_size: int = 32,
    ) -> None:
        if not backbone_dir.exists():
            raise FileNotFoundError(f"FaceNet backbone directory not found: {backbone_dir}")
        self._batch_size = batch_size
        self._backbone = tf.saved_model.load(str(backbone_dir))
        self._infer = self._backbone.signatures["serving_default"]
        self._embedding_key = self._resolve_embedding_key()
        self.embedding_dim = embedding_dim

    def _resolve_embedding_key(self) -> str:
        preferred_keys = ("Bottleneck_BatchNorm", "embeddings")
        for key in preferred_keys:
            if key in self._infer.structured_outputs:
                return key
        return next(iter(self._infer.structured_outputs))

    @staticmethod
    def _preprocess(images: tf.Tensor) -> tf.Tensor:
        resized = tf.image.resize(images, (160, 160))
        scaled = (resized * 2.0) - 1.0
        return scaled

    @staticmethod
    def _normalise(embeddings: np.ndarray) -> np.ndarray:
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        norms = np.maximum(norms, 1e-12)
        return embeddings / norms

    def embed_single(self, image: np.ndarray) -> np.ndarray:
        batch = self.embed_batch([image])
        return batch.embeddings[0]

    def embed_batch(self, images: Sequence[np.ndarray]) -> EmbeddingBatch:
        stacks = [np.asarray(img, dtype=np.float32) for img in images]
        if not stacks:
            raise ValueError("No images provided for embedding")
        stacked = np.stack(stacks, axis=0)
        embeddings = []
        indexes: List[int] = []
        for start in range(0, len(stacked), self._batch_size):
            end = start + self._batch_size
            batch = stacked[start:end]
            batch_tensor = tf.convert_to_tensor(batch, dtype=tf.float32)
            batch_tensor = self._preprocess(batch_tensor)
            outputs = self._infer(batch_tensor)
            batch_embeddings = outputs.get(self._embedding_key, next(iter(outputs.values())))
            embeddings.append(batch_embeddings.numpy())
            indexes.extend(range(start, min(end, len(stacked))))
        merged = np.vstack(embeddings)
        normalised = self._normalise(merged)
        return EmbeddingBatch(embeddings=normalised, indexes=indexes)
