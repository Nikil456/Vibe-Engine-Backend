"""Output shapes + the label lists we feed into the zero-shot model."""

from __future__ import annotations

from enum import Enum
from typing import Any, Dict, List, Optional, Union

from pydantic import BaseModel, ConfigDict, Field, field_validator


class SentimentLabel(str, Enum):
    positive = "positive"
    negative = "negative"
    neutral = "neutral"


class AspectSpec(BaseModel):
    # First label ≈ "review talks about this", last ≈ "doesn't talk about this"
    mention_labels: List[str]
    value_labels: List[str]
    # Maps the winning value_label string to what we store in JSON (quiet, True, etc.)
    value_map: Dict[str, Union[bool, str]] = Field(default_factory=dict)


def default_aspects() -> Dict[str, AspectSpec]:
    return {
        "noise_level": AspectSpec(
            mention_labels=[
                "about how loud, quiet, peaceful, or noisy the restaurant is",
                "not about noise level, volume, sound, or quietness",
            ],
            value_labels=[
                "quiet or calm",
                "loud or noisy",
                "average or moderate noise",
            ],
            value_map={
                "quiet or calm": "quiet",
                "loud or noisy": "loud",
                "average or moderate noise": "moderate",
            },
        ),
        "lighting": AspectSpec(
            mention_labels=[
                "about lighting, brightness, or how dim the place is",
                "not about lighting or brightness",
            ],
            value_labels=[
                "dim or moody lighting",
                "bright or well-lit",
                "average lighting",
            ],
            value_map={
                "dim or moody lighting": "dim",
                "bright or well-lit": "bright",
                "average lighting": "average",
            },
        ),
        "ambience": AspectSpec(
            mention_labels=[
                "about overall atmosphere, vibe, decor, or aesthetic of the place",
                "not about atmosphere, vibe, or decor",
            ],
            value_labels=[
                "cozy or intimate atmosphere",
                "energetic or lively atmosphere",
                "casual or plain atmosphere",
            ],
            value_map={
                "cozy or intimate atmosphere": "cozy",
                "energetic or lively atmosphere": "lively",
                "casual or plain atmosphere": "casual",
            },
        ),
        "romantic": AspectSpec(
            mention_labels=[
                "about romantic dates, couples, or intimate dining",
                "not about romance or date-night suitability",
            ],
            value_labels=[
                "romantic or good for dates",
                "not romantic or poor for dates",
                "unclear for romance",
            ],
            value_map={
                "romantic or good for dates": True,
                "not romantic or poor for dates": False,
                "unclear for romance": False,
            },
        ),
        "study_friendly": AspectSpec(
            mention_labels=[
                "about studying, working on a laptop, wifi, or staying long to work",
                "not about studying or working onsite",
            ],
            value_labels=[
                "good for studying or working",
                "bad for studying or working",
                "unclear for studying",
            ],
            value_map={
                "good for studying or working": True,
                "bad for studying or working": False,
                "unclear for studying": False,
            },
        ),
        "group_friendly": AspectSpec(
            mention_labels=[
                "about large groups, parties, sharing tables, or gatherings",
                "not about group size or parties",
            ],
            value_labels=[
                "good for groups or parties",
                "bad for groups or cramped for parties",
                "unclear for groups",
            ],
            value_map={
                "good for groups or parties": True,
                "bad for groups or cramped for parties": False,
                "unclear for groups": False,
            },
        ),
        "service": AspectSpec(
            mention_labels=[
                "about staff, service speed, or attentiveness",
                "not about service or staff",
            ],
            value_labels=[
                "fast or attentive service",
                "slow or poor service",
                "average service",
            ],
            value_map={
                "fast or attentive service": "good",
                "slow or poor service": "slow",
                "average service": "average",
            },
        ),
        "food_quality": AspectSpec(
            mention_labels=[
                "about food taste, quality, freshness, or presentation",
                "not about food quality",
            ],
            value_labels=[
                "excellent or delicious food",
                "mediocre or bad food",
                "average food quality",
            ],
            value_map={
                "excellent or delicious food": "high",
                "mediocre or bad food": "low",
                "average food quality": "average",
            },
        ),
        "late_night": AspectSpec(
            mention_labels=[
                "about late hours, open late, or nighttime dining",
                "not about late-night hours",
            ],
            value_labels=[
                "good late-night or open late option",
                "not a late-night place or closes early",
                "unclear late-night hours",
            ],
            value_map={
                "good late-night or open late option": True,
                "not a late-night place or closes early": False,
                "unclear late-night hours": False,
            },
        ),
    }


DEFAULT_ASPECT_SPECS = default_aspects()


class AspectPrediction(BaseModel):
    model_config = ConfigDict(extra="ignore")

    value: Optional[Union[str, bool]] = None
    sentiment: SentimentLabel
    confidence: float = Field(ge=0.0, le=1.0)
    mention_score: Optional[float] = None


class ReviewVibeRecord(BaseModel):
    business_id: str
    review_id: str
    review_text: Optional[str] = None
    restaurant_name: Optional[str] = None
    latitude: Optional[float] = None
    longitude: Optional[float] = None
    stars: Optional[float] = None
    aspects: Dict[str, AspectPrediction] = Field(default_factory=dict)
    model_version: Optional[str] = None
    error: Optional[str] = None

    @field_validator("business_id", "review_id", mode="before")
    @classmethod
    def strip_ids(cls, v: Any) -> str:
        if v is None:
            return ""
        return str(v).strip()


class AggregatedAspectProfile(BaseModel):
    dominant_value: Optional[Union[str, bool]] = None
    score: float = Field(ge=0.0, le=1.0)
    support_reviews: Optional[int] = None


class RestaurantProfile(BaseModel):
    model_config = ConfigDict(extra="ignore")

    business_id: str
    restaurant_name: Optional[str] = None
    latitude: Optional[float] = None
    longitude: Optional[float] = None
    aggregated_vibes: Dict[str, AggregatedAspectProfile] = Field(default_factory=dict)
    review_count: int = 0
