# Zero-shot vibe extraction (Hugging Face). Per aspect: mentioned? → value bucket → sentiment.
# Swap `classify()` for a fine-tuned head later without changing the JSON schema.

from __future__ import annotations

import logging
import math
from typing import Any

import src.env_setup  # noqa: F401  # local HF cache before first download
import torch
from transformers import pipeline

from src.config import Settings, load_settings
from src.models.vibe_schema import (
    DEFAULT_ASPECT_SPECS,
    AspectPrediction,
    AspectSpec,
    ReviewVibeRecord,
    SentimentLabel,
)

logger = logging.getLogger(__name__)

H_MENTION = "This restaurant review is {}."
H_VALUE = "The review characterizes the restaurant as {}."
H_SENTIMENT = "This review contains {}."

SENTIMENT_TOPIC: dict[str, str] = {
    "noise_level": "the noise level or sound environment",
    "lighting": "the lighting or brightness",
    "ambience": "the atmosphere or vibe",
    "romantic": "romantic or date-night suitability",
    "study_friendly": "studying or working there",
    "group_friendly": "large groups or parties",
    "service": "the service or staff",
    "food_quality": "the food quality or taste",
    "late_night": "late-night dining or hours",
}


class VibeExtractor:
    def __init__(self, settings: Settings, aspects: dict[str, AspectSpec] | None = None):
        self.settings = settings
        self.aspects = aspects or DEFAULT_ASPECT_SPECS

        device = settings.device
        if device is None:
            device = 0 if torch.cuda.is_available() else -1

        logger.info("Loading %s (device %s)", settings.zero_shot_model, device)
        self.classifier = pipeline(
            "zero-shot-classification",
            model=settings.zero_shot_model,
            device=device,
        )
        self.model_id = settings.zero_shot_model

    def _clip(self, text: str) -> str:
        text = (text or "").strip()
        cap = self.settings.max_review_chars
        return text if len(text) <= cap else text[:cap]

    def classify(self, text: str, labels: list[str], hypothesis_template: str) -> tuple[str, float]:
        if not text.strip():
            return labels[-1], 0.0
        out = self.classifier(
            text,
            candidate_labels=labels,
            hypothesis_template=hypothesis_template,
            multi_label=False,
        )
        return out["labels"][0], float(out["scores"][0])

    def aspect_mentioned(self, spec: AspectSpec, review: str) -> tuple[bool, float]:
        labels = spec.mention_labels
        not_mentioned = labels[-1]
        top, score = self.classify(review, labels, H_MENTION)
        if top == not_mentioned:
            return False, score
        return score >= self.settings.mention_threshold, score

    def aspect_value(self, spec: AspectSpec, review: str) -> tuple[str | bool | None, float]:
        top, score = self.classify(review, spec.value_labels, H_VALUE)
        return spec.value_map.get(top), score

    def aspect_sentiment(self, aspect_key: str, review: str) -> tuple[SentimentLabel, float]:
        topic = SENTIMENT_TOPIC.get(aspect_key, aspect_key.replace("_", " "))
        labels = [
            f"praise or liking regarding {topic}",
            f"complaint or dislike regarding {topic}",
            f"neutral or purely factual statements regarding {topic}",
        ]
        top, score = self.classify(review, labels, H_SENTIMENT)
        if top == labels[0]:
            return SentimentLabel.positive, score
        if top == labels[1]:
            return SentimentLabel.negative, score
        return SentimentLabel.neutral, score

    def extract_row(self, row: dict[str, Any]) -> ReviewVibeRecord:
        review = self._clip(str(row.get("review_text") or ""))
        record = ReviewVibeRecord(
            business_id=str(row.get("business_id") or "").strip(),
            review_id=str(row.get("review_id") or "").strip(),
            review_text=review,
            restaurant_name=_str_or_none(row.get("restaurant_name")),
            latitude=_float_or_none(row.get("latitude")),
            longitude=_float_or_none(row.get("longitude")),
            stars=_float_or_none(row.get("stars")),
            model_version=self.model_id,
        )

        if not review:
            record.error = "empty_review_text"
            return record

        aspects_out: dict[str, AspectPrediction] = {}
        for key, spec in self.aspects.items():
            try:
                mentioned, mention_score = self.aspect_mentioned(spec, review)
                if not mentioned:
                    continue
                value, v_score = self.aspect_value(spec, review)
                sentiment, s_score = self.aspect_sentiment(key, review)
                conf = math.sqrt(v_score * s_score)
                aspects_out[key] = AspectPrediction(
                    value=value,
                    sentiment=sentiment,
                    confidence=min(1.0, max(0.0, conf)),
                    mention_score=mention_score,
                )
            except Exception as err:
                logger.debug("Skipping aspect %s: %s", key, err)

        record.aspects = aspects_out
        return record


def _str_or_none(v: Any) -> str | None:
    if v is None:
        return None
    if isinstance(v, float) and math.isnan(v):
        return None
    s = str(v).strip()
    return s or None


def _float_or_none(v: Any) -> float | None:
    if v is None or v == "":
        return None
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def make_extractor(settings: Settings | None = None) -> VibeExtractor:
    return VibeExtractor(settings or load_settings())
