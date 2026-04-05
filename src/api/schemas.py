"""Pydantic models for HTTP JSON bodies."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class SearchRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=500)
    limit: int = Field(default=20, ge=1, le=200)


class SearchResponse(BaseModel):
    query: str
    results: list[dict[str, Any]]
    count: int
    # Stable contract: "placeholder_keyword" until embedding search ships (see api/constants.py).
    search_mode: str
    message: str | None = None


class RestaurantListResponse(BaseModel):
    restaurants: list[dict[str, Any]]
    count: int
