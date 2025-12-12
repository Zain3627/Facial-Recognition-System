"""Persistent storage of user embeddings for the web app."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np


@dataclass
class StoredIdentity:
    """Internal representation of an enrolled identity."""

    label: str
    embeddings: List[np.ndarray] = field(default_factory=list)
    created_at: datetime = field(default_factory=datetime.utcnow)
    updated_at: datetime = field(default_factory=datetime.utcnow)

    def as_dict(self) -> Dict:
        return {
            "label": self.label,
            "embeddings": [embedding.tolist() for embedding in self.embeddings],
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
        }

    @classmethod
    def from_dict(cls, payload: Dict) -> "StoredIdentity":
        embeddings = [np.asarray(vec, dtype=np.float32) for vec in payload.get("embeddings", [])]
        created_at = datetime.fromisoformat(payload.get("created_at")) if payload.get("created_at") else datetime.utcnow()
        updated_at = datetime.fromisoformat(payload.get("updated_at")) if payload.get("updated_at") else datetime.utcnow()
        return cls(
            label=payload["label"],
            embeddings=embeddings,
            created_at=created_at,
            updated_at=updated_at,
        )


class EmbeddingStore:
    """Simple JSON backed store for embeddings."""

    def __init__(self, store_path: Path) -> None:
        self._store_path = store_path
        self._identities: Dict[str, StoredIdentity] = {}
        self._load()

    def _load(self) -> None:
        if not self._store_path.exists():
            self._identities = {}
            return
        with self._store_path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
        identities = {}
        for entry in payload.get("users", []):
            identity = StoredIdentity.from_dict(entry)
            identities[identity.label.lower()] = identity
        self._identities = identities

    def _flush(self) -> None:
        serialised = {"users": [identity.as_dict() for identity in self._identities.values()]}
        with self._store_path.open("w", encoding="utf-8") as handle:
            json.dump(serialised, handle, indent=2)

    def list_labels(self) -> List[str]:
        return [identity.label for identity in self._identities.values()]

    def upsert_embeddings(self, label: str, embeddings: Sequence[np.ndarray]) -> None:
        if len(embeddings) == 0:
            raise ValueError("Cannot store empty embedding list")
        key = label.strip().lower()
        if not key:
            raise ValueError("Label must not be empty")
        vectors = [np.asarray(vector, dtype=np.float32) for vector in embeddings]
        identity = self._identities.get(key)
        now = datetime.utcnow()
        if identity:
            identity.embeddings = vectors
            identity.updated_at = now
        else:
            identity = StoredIdentity(label=label.strip(), embeddings=vectors, created_at=now, updated_at=now)
            self._identities[key] = identity
        self._flush()

    def find_best_match(self, query_embedding: np.ndarray) -> Tuple[Optional[str], float]:
        if not self._identities:
            return None, 0.0
        best_label: Optional[str] = None
        best_similarity = -1.0
        for identity in self._identities.values():
            representative = self._average_embedding(identity.embeddings)
            similarity = float(np.dot(query_embedding, representative))
            if similarity > best_similarity:
                best_similarity = similarity
                best_label = identity.label
        return best_label, best_similarity

    @staticmethod
    def _average_embedding(embeddings: Sequence[np.ndarray]) -> np.ndarray:
        stacked = np.stack(embeddings, axis=0)
        mean_vector = stacked.mean(axis=0)
        norm = np.linalg.norm(mean_vector)
        if norm < 1e-12:
            return mean_vector
        return mean_vector / norm
