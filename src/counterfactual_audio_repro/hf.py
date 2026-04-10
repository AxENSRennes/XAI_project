from __future__ import annotations

import os
from pathlib import Path


def configure_hf_cache(root: str | Path | None = None) -> Path:
    cache_root = Path(root or "/home/axel/XAI/.hf-cache").resolve()
    cache_root.mkdir(parents=True, exist_ok=True)
    os.environ["HF_HOME"] = str(cache_root)
    os.environ["HF_HUB_CACHE"] = str(cache_root / "hub")
    os.environ["HUGGINGFACE_HUB_CACHE"] = str(cache_root / "hub")
    os.environ["TRANSFORMERS_CACHE"] = str(cache_root / "transformers")
    return cache_root
