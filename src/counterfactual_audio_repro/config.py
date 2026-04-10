from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import yaml


def load_config(path: str | Path | None) -> dict[str, Any]:
    if path is None:
        return {}
    config_path = Path(path)
    with config_path.open("r", encoding="utf-8") as handle:
        if config_path.suffix in {".yaml", ".yml"}:
            return yaml.safe_load(handle) or {}
        if config_path.suffix == ".json":
            return json.load(handle)
    raise ValueError(f"Unsupported config format: {config_path}")

