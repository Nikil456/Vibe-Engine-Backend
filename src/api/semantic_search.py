"""Semantic search using cosine similarity on pre-computed embeddings."""

from __future__ import annotations

from typing import Any

import numpy as np
from sentence_transformers import SentenceTransformer


class SemanticSearcher:
    def __init__(
        self,
        profiles: list[dict[str, Any]],
        model_name: str = "all-MiniLM-L6-v2",
    ):
        self.profiles = {p["business_id"]: p for p in profiles}
        self.ids = [p["business_id"] for p in profiles]

        self.embeddings = np.array(
            [p.get("embedding", [0] * 384) for p in profiles], dtype=np.float32
        )

        norms = np.linalg.norm(self.embeddings, axis=1, keepdims=True)
        norms[norms == 0] = 1
        self.embeddings = self.embeddings / norms

        self.model = SentenceTransformer(model_name)

    def search(
        self, query: str, limit: int = 20, min_score: float = 0.0
    ) -> list[dict[str, Any]]:
        query_emb = self.model.encode(
            [query], normalize_embeddings=True, convert_to_numpy=True
        )

        scores = np.dot(self.embeddings, query_emb.T).flatten()

        top_indices = np.argsort(scores)[::-1][:limit]

        results = []
        for idx in top_indices:
            if scores[idx] >= min_score:
                profile = dict(self.profiles[self.ids[idx]])
                profile["relevance_score"] = float(scores[idx])
                results.append(profile)

        return results
