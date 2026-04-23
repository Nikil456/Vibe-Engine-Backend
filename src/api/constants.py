"""Central place for API strings the frontend and Nikil's search can rely on."""

API_V1_PREFIX = "/api/v1"

# Browser origins allowed when VIBE_CORS_ORIGINS is unset (local Vite + CRA).
DEFAULT_CORS_ORIGINS = (
    "http://localhost:5173",
    "http://127.0.0.1:5173",
    "http://localhost:3000",
    "http://127.0.0.1:3000",
)

# SearchResponse.search_mode — swap to e.g. "semantic" when embeddings are live.
SEARCH_MODE_KEYWORD = "placeholder_keyword"
SEARCH_MODE_SEMANTIC = "semantic"

SEARCH_PLACEHOLDER_NOTE = (
    "Semantic search not wired yet — Nikil will replace this with embedding similarity."
)
