from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Iterable

from .counterfactuals import generate_counterfactual


def read_manifest(path: str | Path) -> list[dict]:
    manifest_path = Path(path)
    if manifest_path.suffix == ".jsonl":
        rows = []
        with manifest_path.open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if line:
                    rows.append(json.loads(line))
        return rows
    if manifest_path.suffix == ".json":
        with manifest_path.open("r", encoding="utf-8") as handle:
            data = json.load(handle)
        if isinstance(data, list):
            return data
        raise ValueError("JSON manifest must contain a top-level list.")
    if manifest_path.suffix == ".csv":
        with manifest_path.open("r", encoding="utf-8") as handle:
            return list(csv.DictReader(handle))
    raise ValueError(f"Unsupported manifest format: {manifest_path}")


def write_jsonl(path: str | Path, rows: Iterable[dict]) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=True) + "\n")


def flatten_release_json(
    release_json: str | Path,
    dataset_root: str | Path | None = None,
    dataset_name: str | None = None,
    generate_missing_counterfactuals: bool = False,
) -> list[dict]:
    release_path = Path(release_json)
    with release_path.open("r", encoding="utf-8") as handle:
        records = json.load(handle)

    base_root = Path(dataset_root) if dataset_root else None
    flattened: list[dict] = []
    default_name = dataset_name or release_path.stem.replace("-counterfactual", "")

    for record in records:
        captions = record.get("captions") or []
        counterfactuals = record.get("captions_counterfactual") or []
        if generate_missing_counterfactuals and (
            not counterfactuals or counterfactuals == captions
        ):
            counterfactuals = [
                generate_counterfactual(caption).counterfactual_caption
                for caption in captions
            ]
        elif not counterfactuals:
            counterfactuals = list(captions)

        limit = min(len(captions), len(counterfactuals))
        audio_rel_path = record["path"]
        audio_path = str((base_root / audio_rel_path).resolve()) if base_root else audio_rel_path

        for index in range(limit):
            flattened.append(
                {
                    "dataset": default_name,
                    "split": record.get("split"),
                    "path": audio_rel_path,
                    "audio_path": audio_path,
                    "caption_index": index,
                    "caption": captions[index],
                    "counterfactual_caption": counterfactuals[index],
                    "samplerate": record.get("samplerate"),
                    "duration": record.get("duration"),
                    "channels": record.get("channels"),
                }
            )

    return flattened
