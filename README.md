# Restaurant Vibe Engine (NLP)

1. **Aspect-oriented vibe extraction** from review text (ABSA-style, production-shaped code).
2. **Restaurant-level aggregation** of review signals into structured profiles for downstream embedding / search.
3. **Simple evaluation** against a hand-labeled JSONL slice (precision / recall / F1 per aspect).

---

## Modeling approach (why this hybrid)

We use a **hybrid zero-shot pipeline** (all stages share the same NLI-style zero-shot classifier by default):

1. **Mention gate** — For each aspect, the model chooses between *“this review is about …”* vs *“not about …”*. Low scores drop the aspect (reduces false positives on huge Yelp data).
2. **Value head** — If mentioned, a second pass picks among **small, human-readable label sets** (e.g. quiet / loud / moderate for `noise_level`). Labels map to structured values in code (`value_map`).
3. **Sentiment head** — A third pass uses **praise / complaint / neutral** phrasing tied to a short natural hint per aspect (e.g. service, lighting).

**Why not one generative JSON prompt?** For a course project, zero-shot classification is easier to **explain**, **evaluate**, and **swap** later: you can replace the `classify()` calls in `extractor.py` with a fine-tuned model without changing aggregation or the JSON shape.

**Default model**: `facebook/bart-large-mnli` — reliable public NLI backbone for zero-shot. For faster (lighter) runs, try `typeform/distilbert-base-uncased-mnli` via `VIBE_ZERO_SHOT_MODEL` or `--model`.

---

**Endpoints**

| Method | Path | Purpose |
|--------|------|---------|
| `GET` | `/health` | Liveness + how many restaurants are loaded |
| `GET` | `/api/v1/restaurants` | All profiles (`business_id`, `restaurant_name`, `latitude`, `longitude`, `aggregated_vibes`, `review_count`) |
| `GET` | `/api/v1/restaurants/{business_id}` | One profile |
| `POST` | `/api/v1/search` | Body: `{ "query": "...", "limit": 20 }` → `{ results, search_mode, ... }` |

**Run the API** (after `aggregate_restaurants` has written the default profile file):

```bash
cd Vibe-Engine-Backend
source .venv/bin/activate
uvicorn src.api.app:app --reload --host 0.0.0.0 --port 8000
```

---

## Setup

A root-level **`data/`** directory is **gitignored** (so `data/review_vibes.jsonl` etc. never land in git). Create it locally for ETL outputs. Use **`examples/`** for the tiny tracked smoke-test CSV/JSONL.

**Requirements**: Python **3.11+**, ~4GB+ disk for PyTorch + transformers (varies by platform).

```bash
cd Vibe-Engine-Backend
python3.11 -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

**Hugging Face cache**: On first run, models download into `Vibe-Engine-Backend/.cache/huggingface` (see `src/env_setup.py`, imported by `process_reviews`). Override with the standard `HF_HOME` environment variable if you prefer another location.

Optional environment variables:

| Variable | Purpose |
|----------|---------|
| `VIBE_ZERO_SHOT_MODEL` | Hugging Face model id (default: `facebook/bart-large-mnli`) |
| `VIBE_MENTION_THRESHOLD` | Mention gate cutoff (default `0.35`) |
| `VIBE_MAX_REVIEW_CHARS` | Truncate reviews (default `1500`) |
| `VIBE_CSV_CHUNK` / `VIBE_JSONL_CHUNK` | Reader batch sizes |
| `VIBE_DEVICE` | Integer device id if you want to force GPU index |
| `VIBE_PROFILES_PATH` | JSON / JSONL path for the API (default `data/restaurant_profiles.json`) |
| `VIBE_CORS_ORIGINS` | Comma-separated allowed browser origins for the API |
| `PORT` / `VIBE_API_PORT` | API listen port for `python -m src.api.app` (default `8000`) |

---

## CLI examples

**1) Process reviews** → `review_vibes.jsonl`

```bash
python -m src.pipeline.process_reviews \
  --input data/clean_reviews.csv \
  --output data/review_vibes.jsonl
```

Quick demo on the bundled sample (1 row):

```bash
mkdir -p data
python -m src.pipeline.process_reviews \
  --input examples/sample_clean_reviews.csv \
  --output data/review_vibes.jsonl \
  --max-reviews 1
```

**2) Aggregate restaurants** → `restaurant_profiles.json`

```bash
python -m src.pipeline.aggregate_restaurants \
  --input data/review_vibes.jsonl \
  --output data/restaurant_profiles.json
```

Use `.jsonl` extension on `--output` if you prefer **one restaurant per line**.

**3) Evaluate** against hand labels

```bash
python -m src.eval.evaluate_absa \
  --pred data/review_vibes.jsonl \
  --gold examples/gold_labels.example.jsonl
```

---

## Input formats

### Cleaned reviews (CSV or JSONL)

Expected columns (aliases accepted — see `src/data/load_data.py`):

| Column | Required |
|--------|----------|
| `business_id` | yes |
| `review_id` | yes |
| `review_text` | yes |
| `restaurant_name`, `latitude`, `longitude`, `stars` | optional (carried through) |

### Review-level output (`review_vibes.jsonl`)

Each line is a JSON object: Pydantic `ReviewVibeRecord` with `aspects` keyed by schema name. Example shape:

```json
{
  "business_id": "…",
  "review_id": "…",
  "review_text": "…",
  "aspects": {
    "noise_level": {
      "value": "quiet",
      "sentiment": "positive",
      "confidence": 0.91,
      "mention_score": 0.88
    }
  },
  "model_version": "MoritzLaurer/DeBERTa-v3-small-mnli-fever-anli"
}
```

### Restaurant profile output

```json
{
  "business_id": "…",
  "restaurant_name": "…",
  "latitude": 37.77,
  "longitude": -122.42,
  "review_count": 42,
  "aggregated_vibes": {
    "noise_level": {
      "dominant_value": "quiet",
      "score": 0.81,
      "support_reviews": 30
    },
    "study_friendly": {
      "dominant_value": true,
      "score": 0.55,
      "support_reviews": 18
    }
  }
}
```

`score` = (sum of confidences at the dominant value) ÷ (total reviews for that business), clamped to `[0,1]`.

### Gold labels for evaluation

JSONL with **at least** `review_id` and `aspects.<name>.sentiment` ∈ {`positive`,`negative`,`neutral`}. See `examples/gold_labels.example.jsonl`.

---

## What to run first

1. **Install** dependencies (see Setup).
2. Run **`process_reviews`** on `examples/sample_clean_reviews.csv` with `--max-reviews 1` to verify the HF model downloads and the pipeline end-to-end.
3. Run **`aggregate_restaurants`** on the produced JSONL.
4. Optionally run **`evaluate_absa`** with your own gold file.
