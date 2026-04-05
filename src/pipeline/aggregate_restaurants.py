"""CLI: review_vibes.jsonl → restaurant_profiles.json (or .jsonl)."""

from __future__ import annotations

import argparse
import json
import logging
from collections import defaultdict
from typing import Any

from src.config import load_settings
from src.models.vibe_schema import AggregatedAspectProfile, RestaurantProfile
from src.utils.io import read_jsonl, resolve_path

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def _coerce_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def dominant_value_and_mass(pairs: list[tuple[Any, float]]) -> tuple[Any | None, float, int]:
    """Sum confidence per extracted value; return winner and how many reviews contributed."""
    if not pairs:
        return None, 0.0, 0
    totals: dict[Any, float] = defaultdict(float)
    for val, conf in pairs:
        totals[val] += float(conf)
    best = max(totals, key=lambda k: totals[k])
    return best, float(totals[best]), len(pairs)


def build_profile(business_id: str, rows: list[dict[str, Any]], total_reviews: int) -> RestaurantProfile:
    name = None
    lat: float | None = None
    lon: float | None = None
    by_aspect: dict[str, list[tuple[Any, float]]] = defaultdict(list)

    for r in rows:
        if name is None:
            name = r.get("restaurant_name")
        if lat is None:
            lat = _coerce_float(r.get("latitude"))
        if lon is None:
            lon = _coerce_float(r.get("longitude"))

        aspects = r.get("aspects") or {}
        if not isinstance(aspects, dict):
            continue
        for key, cell in aspects.items():
            if not isinstance(cell, dict):
                continue
            conf = cell.get("confidence")
            if conf is None:
                continue
            cf = _coerce_float(conf)
            if cf is None:
                continue
            by_aspect[key].append((cell.get("value"), cf))

    vibes: dict[str, AggregatedAspectProfile] = {}
    for key, pairs in by_aspect.items():
        dom, mass, support = dominant_value_and_mass(pairs)
        score = mass / max(1, total_reviews)
        vibes[key] = AggregatedAspectProfile(
            dominant_value=dom,
            score=min(1.0, max(0.0, score)),
            support_reviews=support,
        )

    return RestaurantProfile(
        business_id=business_id,
        restaurant_name=name,
        latitude=lat,
        longitude=lon,
        aggregated_vibes=vibes,
        review_count=total_reviews,
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Aggregate review JSONL into restaurant profiles.")
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args(argv)

    settings = load_settings()
    in_path = resolve_path(args.input, base=settings.project_root)
    out_path = resolve_path(args.output, base=settings.project_root)

    if not in_path.is_file():
        logger.error("Input not found: %s", in_path)
        return 1

    by_business: defaultdict[str, list[dict[str, Any]]] = defaultdict(list)
    review_counts: defaultdict[str, int] = defaultdict(int)

    for rec in read_jsonl(in_path):
        bid = str(rec.get("business_id") or "").strip()
        if not bid:
            continue
        review_counts[bid] += 1
        if rec.get("error"):
            continue
        by_business[bid].append(rec)

    profiles = [
        build_profile(bid, rows, review_counts[bid]) for bid, rows in by_business.items()
    ]

    out_path.parent.mkdir(parents=True, exist_ok=True)
    if out_path.suffix.lower() == ".jsonl":
        with out_path.open("w", encoding="utf-8") as f:
            for prof in profiles:
                f.write(json.dumps(prof.model_dump(mode="json"), ensure_ascii=False) + "\n")
    else:
        payload = [p.model_dump(mode="json") for p in profiles]
        out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    logger.info("%s restaurants → %s", len(profiles), out_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
