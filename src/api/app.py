"""
REST API for restaurant profiles and search.

Run from repo root:
  uvicorn src.api.app:app --reload --host 0.0.0.0 --port 8000
"""

from __future__ import annotations

import logging
import os
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Annotated

from fastapi import APIRouter, Depends, FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware

from src import __version__ as package_version
from src.api.constants import (
    API_V1_PREFIX,
    DEFAULT_CORS_ORIGINS,
    SEARCH_MODE_KEYWORD,
    SEARCH_MODE_SEMANTIC,
    SEARCH_PLACEHOLDER_NOTE,
)
from src.api.placeholder_search import keyword_rank
from src.api.profile_store import ProfileStore
from src.api.schemas import RestaurantListResponse, SearchRequest, SearchResponse
from src.api.semantic_search import SemanticSearcher
from src.config import load_settings

logger = logging.getLogger(__name__)


def cors_allow_origins() -> list[str]:
    raw = os.environ.get("VIBE_CORS_ORIGINS", "").strip()
    if raw:
        return [o.strip() for o in raw.split(",") if o.strip()]
    return list(DEFAULT_CORS_ORIGINS)


def profiles_file_path() -> Path:
    settings = load_settings()
    rel = os.environ.get("VIBE_PROFILES_PATH", "data/restaurant_profiles.json").strip()
    path = Path(rel)
    return path if path.is_absolute() else settings.project_root / path


def profile_store_dep(request: Request) -> ProfileStore:
    return request.app.state.profile_store


Store = Annotated[ProfileStore, Depends(profile_store_dep)]


@asynccontextmanager
async def lifespan(app: FastAPI):
    path = profiles_file_path()
    app.state.profile_store = ProfileStore.load(path)
    app.state.profiles_path = path

    if app.state.profile_store.has_embeddings():
        logger.info("Initializing semantic searcher...")
        app.state.semantic_searcher = SemanticSearcher(
            app.state.profile_store.all()
        )
    else:
        app.state.semantic_searcher = None

    logger.info("Loaded %s restaurants from %s", len(app.state.profile_store.all()), path)
    yield


v1 = APIRouter(prefix=API_V1_PREFIX, tags=["v1"])


@v1.get("/restaurants", response_model=RestaurantListResponse)
def list_restaurants(store: Store) -> RestaurantListResponse:
    rows = store.all()
    return RestaurantListResponse(restaurants=rows, count=len(rows))


@v1.get("/restaurants/{business_id}")
def get_restaurant(business_id: str, store: Store) -> dict:
    row = store.get(business_id)
    if row is None:
        raise HTTPException(status_code=404, detail="Unknown business_id")
    return row


@v1.post("/search", response_model=SearchResponse)
def search(body: SearchRequest, store: Store, request: Request) -> SearchResponse:
    rows = store.all()

    semantic_searcher: SemanticSearcher | None = getattr(
        request.app.state, "semantic_searcher", None
    )

    if semantic_searcher and body.query.strip():
        ranked = semantic_searcher.search(body.query, body.limit)
        return SearchResponse(
            query=body.query,
            results=ranked,
            count=len(ranked),
            search_mode=SEARCH_MODE_SEMANTIC,
        )

    ranked = keyword_rank(body.query, rows, body.limit)
    return SearchResponse(
        query=body.query,
        results=ranked,
        count=len(ranked),
        search_mode=SEARCH_MODE_KEYWORD,
        message=SEARCH_PLACEHOLDER_NOTE if not semantic_searcher else None,
    )


def create_app() -> FastAPI:
    app = FastAPI(
        title="Restaurant Vibe Engine API",
        description="Profiles from ABSA pipeline + placeholder search for the map UI.",
        version=package_version,
        lifespan=lifespan,
    )
    app.add_middleware(
        CORSMiddleware,
        allow_origins=cors_allow_origins(),
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.get("/health")
    def health(request: Request) -> dict:
        store: ProfileStore = request.app.state.profile_store
        return {
            "status": "ok",
            "version": package_version,
            "restaurant_count": len(store.all()),
            "profiles_path": str(request.app.state.profiles_path),
        }

    app.include_router(v1)
    return app


app = create_app()


if __name__ == "__main__":
    import uvicorn

    port = int(os.environ.get("VIBE_API_PORT", os.environ.get("PORT", "8000")))
    uvicorn.run("src.api.app:app", host="0.0.0.0", port=port, reload=True)
