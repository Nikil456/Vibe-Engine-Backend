"""CLI: cleaned reviews → review_vibes.jsonl (one JSON object per line)."""

from __future__ import annotations

import argparse
import logging
from dataclasses import replace
from pathlib import Path
from typing import Any, Iterator

from tqdm import tqdm

from src.config import load_settings
from src.data.load_data import input_kind, iter_csv_rows, iter_jsonl_batches
from src.models.extractor import VibeExtractor
from src.utils.io import append_jsonl_record, ensure_parent, resolve_path

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def iter_all_rows(path: Path, kind: str, csv_chunk: int, jsonl_chunk: int) -> Iterator[dict[str, Any]]:
    if kind == "csv":
        yield from iter_csv_rows(path, chunk_size=csv_chunk)
    else:
        for batch in iter_jsonl_batches(path, batch_size=jsonl_chunk):
            yield from batch


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Extract vibe aspects from reviews.")
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--max-reviews", type=int, default=None, help="Stop after N rows (smoke tests).")
    parser.add_argument("--model", default=None, help="Hugging Face model id.")
    parser.add_argument("--mention-threshold", type=float, default=None)
    parser.add_argument("--max-chars", type=int, default=None)
    parser.add_argument("--device", type=int, default=None, help="Device: -1 CPU, 0 GPU, etc.")
    parser.add_argument("--append", action="store_true")
    args = parser.parse_args(argv)

    settings = load_settings()
    if args.model:
        settings = replace(settings, zero_shot_model=args.model)
    if args.mention_threshold is not None:
        settings = replace(settings, mention_threshold=args.mention_threshold)
    if args.max_chars is not None:
        settings = replace(settings, max_review_chars=args.max_chars)
    if args.device is not None:
        settings = replace(settings, device=args.device)

    root = settings.project_root
    in_path = resolve_path(args.input, base=root)
    out_path = resolve_path(args.output, base=root)

    if not in_path.is_file():
        logger.error("Input not found: %s", in_path)
        return 1

    ensure_parent(out_path)
    if not args.append and out_path.exists():
        out_path.unlink()

    extractor = VibeExtractor(settings)
    kind = input_kind(in_path)
    stream = iter_all_rows(
        in_path,
        kind,
        csv_chunk=settings.csv_chunk_rows,
        jsonl_chunk=settings.jsonl_chunk_rows,
    )

    n = 0
    bar = tqdm(desc="reviews", unit="rev")
    try:
        for row in stream:
            if args.max_reviews is not None and n >= args.max_reviews:
                break
            try:
                rec = extractor.extract_row(row)
                append_jsonl_record(out_path, rec.model_dump(mode="json"))
            except Exception as e:
                logger.exception("Failed on review_id=%s", row.get("review_id"))
                append_jsonl_record(
                    out_path,
                    {
                        "business_id": str(row.get("business_id") or ""),
                        "review_id": str(row.get("review_id") or ""),
                        "error": str(e),
                        "aspects": {},
                    },
                )
            n += 1
            bar.update(1)
    finally:
        bar.close()

    logger.info("Wrote %s lines → %s", n, out_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
