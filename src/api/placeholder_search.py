"""Keyword ranking until Nikil replaces it with vector similarity."""

from __future__ import annotations

import re
from typing import Any


def _search_blob(profile: dict[str, Any]) -> str:
    parts = [
        str(profile.get("restaurant_name") or ""),
        str(profile.get("business_id") or ""),
    ]
    vibes = profile.get("aggregated_vibes")
    if isinstance(vibes, dict):
        for aspect, cell in vibes.items():
            parts.append(aspect)
            if isinstance(cell, dict) and cell.get("dominant_value") is not None:
                parts.append(str(cell["dominant_value"]))
    return " ".join(parts).lower()


def keyword_rank(
    query: str,
    profiles: list[dict[str, Any]],
    limit: int,
) -> list[dict[str, Any]]:
    if not profiles:
        return []

    tokens = [t for t in re.findall(r"[a-z0-9]+", query.lower()) if len(t) > 1]
    indexed = [(p, _search_blob(p)) for p in profiles]

    if not tokens:
        return [p for p, _ in indexed[:limit]]

    scored: list[tuple[float, dict[str, Any]]] = []
    q = query.lower().strip()

    for prof, blob in indexed:
        score = sum(blob.count(tok) for tok in tokens)
        if score == 0 and q and q in blob:
            score = 0.5
        scored.append((score, prof))

    scored.sort(key=lambda x: x[0], reverse=True)
    winners = [p for s, p in scored if s > 0]
    if winners:
        return winners[:limit]
    return [p for _, p in scored[:limit]]
