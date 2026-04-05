"""Settings from environment variables (pipeline + shared paths)."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


def _int(key: str, default: int) -> int:
    raw = os.environ.get(key, "").strip()
    if not raw:
        return default
    try:
        return int(raw)
    except ValueError:
        return default


def _float(key: str, default: float) -> float:
    raw = os.environ.get(key, "").strip()
    if not raw:
        return default
    try:
        return float(raw)
    except ValueError:
        return default


def _optional_device() -> int | None:
    raw = os.environ.get("VIBE_DEVICE", "").strip()
    if not raw:
        return None
    try:
        return int(raw)
    except ValueError:
        return None


@dataclass
class Settings:
    project_root: Path
    zero_shot_model: str
    csv_chunk_rows: int
    jsonl_chunk_rows: int
    mention_threshold: float
    max_review_chars: int
    device: int | None


def load_settings() -> Settings:
    root = Path(__file__).resolve().parent.parent
    return Settings(
        project_root=root,
        zero_shot_model=os.environ.get("VIBE_ZERO_SHOT_MODEL", "facebook/bart-large-mnli"),
        csv_chunk_rows=_int("VIBE_CSV_CHUNK", 512),
        jsonl_chunk_rows=_int("VIBE_JSONL_CHUNK", 512),
        mention_threshold=_float("VIBE_MENTION_THRESHOLD", 0.35),
        max_review_chars=_int("VIBE_MAX_REVIEW_CHARS", 1500),
        device=_optional_device(),
    )
