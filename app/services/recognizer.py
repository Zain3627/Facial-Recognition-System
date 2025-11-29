"""High level recognition service built on top of the embedding store."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np

from app.services.embedding_service import FaceNetEmbeddingService
from app.storage.embedding_store import EmbeddingStore


@dataclass
class RecognitionResult:
    """Encapsulates a recognition attempt."""

    label: Optional[str]
    similarity: float
    embedding: np.ndarray


class EmbeddingRecognizer:
    """Performs nearest-neighbour search against stored embeddings."""

    def __init__(
        self,
        embedding_service: FaceNetEmbeddingService,
        embedding_store: EmbeddingStore,
        similarity_threshold: float,
    ) -> None:
        self._embedding_service = embedding_service
        self._embedding_store = embedding_store
        self._similarity_threshold = similarity_threshold

    def identify(self, image: np.ndarray) -> RecognitionResult:
        query_embedding = self._embedding_service.embed_single(image)
        label, similarity = self._embedding_store.find_best_match(query_embedding)
        if label is None or similarity < self._similarity_threshold:
            return RecognitionResult(label=None, similarity=similarity, embedding=query_embedding)
        return RecognitionResult(label=label, similarity=similarity, embedding=query_embedding)
