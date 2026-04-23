"""Generate semantic embeddings for restaurant profiles."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from sentence_transformers import SentenceTransformer
from tqdm import tqdm

from src.config import load_settings
from src.utils.io import resolve_path


def create_profile_text(profile: dict) -> str:
    """Convert profile to searchable text representation."""
    parts = [profile.get("restaurant_name", "")]

    vibes = profile.get("aggregated_vibes", {})
    for aspect, data in vibes.items():
        if data.get("dominant_value"):
            parts.append(f"{aspect}: {data['dominant_value']}")

    return " | ".join(parts)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Generate embeddings for restaurant profiles"
    )
    parser.add_argument("--input", required=True, help="Input restaurant_profiles.json")
    parser.add_argument("--output", required=True, help="Output file with embeddings")
    parser.add_argument(
        "--model", default="all-MiniLM-L6-v2", help="Sentence transformer model"
    )
    args = parser.parse_args(argv)

    settings = load_settings()
    in_path = resolve_path(args.input, base=settings.project_root)
    out_path = resolve_path(args.output, base=settings.project_root)

    if not in_path.is_file():
        parser.error(f"Input not found: {in_path}")

    profiles = json.loads(in_path.read_text(encoding="utf-8"))

    print(f"Loading model: {args.model}")
    model = SentenceTransformer(args.model)

    texts = [create_profile_text(p) for p in profiles]
    print(f"Generating embeddings for {len(texts)} restaurants...")

    embeddings = model.encode(texts, show_progress_bar=True, convert_to_numpy=True)

    for i, emb in enumerate(embeddings):
        profiles[i]["embedding"] = emb.tolist()

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        json.dumps(profiles, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"Saved {len(profiles)} embeddings to {out_path}")
    return 0


if __name__ == "__main__":
    import sys

    raise SystemExit(main(sys.argv[1:]))
