"""Loads `restaurant_profiles.json` or `.jsonl` produced by aggregate_restaurants."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from src.utils.io import read_jsonl

logger = logging.getLogger(__name__)


class ProfileStore:
    def __init__(self, items: list[dict[str, Any]]):
        self._items = items
        self._by_id = {
            str(r.get("business_id", "")): r for r in items if r.get("business_id")
        }

    @classmethod
    def load(cls, path: Path) -> ProfileStore:
        if not path.is_file():
            logger.warning("No profile file at %s — API returns empty lists", path)
            return cls([])

        if path.suffix.lower() == ".jsonl":
            return cls(list(read_jsonl(path)))

        data = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(data, list):
            raise ValueError(f"{path} must be a JSON array of restaurant objects")
        return cls(data)

    def all(self) -> list[dict[str, Any]]:
        return list(self._items)

    def get(self, business_id: str) -> dict[str, Any] | None:
        return self._by_id.get(business_id)
