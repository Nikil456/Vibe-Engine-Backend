"""CLI: compare predicted aspect sentiments to a hand-labeled JSONL gold file."""

from __future__ import annotations

import argparse
import logging
from collections import defaultdict
from pathlib import Path
from typing import Any

from sklearn.metrics import classification_report, precision_recall_fscore_support

from src.config import load_settings
from src.utils.io import read_jsonl, resolve_path

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)

VALID_SENTIMENT = frozenset({"positive", "negative", "neutral"})


def normalize_sentiment(raw: Any) -> str | None:
    if raw is None:
        return None
    s = str(raw).strip().lower()
    return s if s in VALID_SENTIMENT else None


def index_by_review_id(path: Path) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    for row in read_jsonl(path):
        rid = str(row.get("review_id") or "").strip()
        if rid:
            out[rid] = row
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Precision / recall / F1 vs gold labels.")
    parser.add_argument("--pred", required=True)
    parser.add_argument("--gold", required=True)
    parser.add_argument("--labels", default="positive,negative,neutral")
    args = parser.parse_args(argv)

    settings = load_settings()
    pred_path = resolve_path(args.pred, base=settings.project_root)
    gold_path = resolve_path(args.gold, base=settings.project_root)
    if not pred_path.is_file() or not gold_path.is_file():
        logger.error("Missing --pred or --gold file.")
        return 1

    label_order = [x.strip().lower() for x in args.labels.split(",") if x.strip()]
    if not label_order or not set(label_order) <= VALID_SENTIMENT:
        logger.error("--labels must be a subset of positive,negative,neutral")
        return 1

    gold = index_by_review_id(gold_path)
    pred = index_by_review_id(pred_path)

    # aspect -> parallel lists for sklearn
    per_aspect: dict[str, tuple[list[str], list[str]]] = defaultdict(lambda: ([], []))

    for rid, gold_row in gold.items():
        if rid not in pred:
            continue
        g_asp = gold_row.get("aspects") or {}
        p_asp = pred[rid].get("aspects") or {}
        if not isinstance(g_asp, dict) or not isinstance(p_asp, dict):
            continue
        for aspect, g_cell in g_asp.items():
            if not isinstance(g_cell, dict):
                continue
            y_true = normalize_sentiment(g_cell.get("sentiment"))
            if y_true is None:
                continue
            p_cell = p_asp.get(aspect)
            y_pred = normalize_sentiment(p_cell.get("sentiment")) if isinstance(p_cell, dict) else None
            if y_pred is None:
                y_pred = "neutral"
            ys, yp = per_aspect[aspect]
            ys.append(y_true)
            yp.append(y_pred)

    if not per_aspect:
        logger.warning("No overlapping labels; check review_id values match.")
        return 0

    pooled_true: list[str] = []
    pooled_pred: list[str] = []

    print("=== Per aspect ===")
    for aspect in sorted(per_aspect):
        y_true, y_pred = per_aspect[aspect]
        if not y_true:
            continue
        p, r, f1, _ = precision_recall_fscore_support(
            y_true, y_pred, labels=label_order, average="macro", zero_division=0
        )
        print(f"\n{aspect} (n={len(y_true)})  P={p:.3f} R={r:.3f} F1={f1:.3f}")
        print(classification_report(y_true, y_pred, labels=label_order, zero_division=0))
        pooled_true.extend(y_true)
        pooled_pred.extend(y_pred)

    if pooled_true:
        p, r, f1, _ = precision_recall_fscore_support(
            pooled_true, pooled_pred, labels=label_order, average="macro", zero_division=0
        )
        print("\n=== All aspects pooled ===")
        print(f"macro P={p:.3f} R={r:.3f} F1={f1:.3f}")
        print(classification_report(pooled_true, pooled_pred, labels=label_order, zero_division=0))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
