"""Stream reviews from CSV or JSONL without loading the whole file into memory."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Iterator

import pandas as pd

logger = logging.getLogger(__name__)

# ETL outputs sometimes use different column names — normalize to these keys.
ALIASES: dict[str, list[str]] = {
    "business_id": ["business_id", "businessId"],
    "review_id": ["review_id", "reviewId"],
    "review_text": ["review_text", "text", "review"],
    "restaurant_name": ["restaurant_name", "name", "business_name"],
    "latitude": ["latitude", "lat"],
    "longitude": ["longitude", "lng", "lon"],
    "stars": ["stars", "review_stars"],
}


def _rename_columns(df: pd.DataFrame) -> pd.DataFrame:
    rename: dict[str, str] = {}
    for canonical, options in ALIASES.items():
        if canonical in df.columns:
            continue
        for opt in options:
            if opt in df.columns:
                rename[opt] = canonical
                break
    return df.rename(columns=rename) if rename else df


def _normalize_keys(row: dict[str, Any]) -> dict[str, Any]:
    lower = {str(k).lower(): k for k in row}
    out = dict(row)
    for canonical, options in ALIASES.items():
        if canonical in out:
            continue
        for opt in options:
            lk = opt.lower()
            if lk in lower:
                orig = lower[lk]
                out[canonical] = out.pop(orig)
                break
    return out


def iter_csv_rows(path: Path, chunk_size: int = 512) -> Iterator[dict[str, Any]]:
    reader = pd.read_csv(path, chunksize=chunk_size, dtype=str, keep_default_na=False)
    for chunk in reader:
        chunk = _rename_columns(chunk)
        if "review_text" not in chunk.columns:
            raise ValueError(f"Need a review text column; got {list(chunk.columns)}")
        # Faster than iterrows() for large chunks.
        yield from chunk.to_dict(orient="records")


def iter_jsonl_batches(path: Path, batch_size: int = 512) -> Iterator[list[dict[str, Any]]]:
    batch: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as e:
                logger.warning("Skipping invalid JSON line in %s: %s", path, e)
                continue
            if isinstance(obj, dict):
                batch.append(_normalize_keys(obj))
            if len(batch) >= batch_size:
                yield batch
                batch = []
        if batch:
            yield batch


def input_kind(path: Path) -> str:
    ext = path.suffix.lower()
    if ext == ".csv":
        return "csv"
    if ext in (".jsonl", ".ndjson", ".json"):
        return "jsonl"
    raise ValueError(f"Expected .csv or .jsonl, got {ext}")
