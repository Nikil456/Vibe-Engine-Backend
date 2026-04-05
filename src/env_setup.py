# Put Hugging Face downloads in this repo so you don't depend on ~/.cache (helpful in CI).
import os
from pathlib import Path

_root = Path(__file__).resolve().parent.parent
_cache = _root / ".cache" / "huggingface"
_cache.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("HF_HOME", str(_cache))
