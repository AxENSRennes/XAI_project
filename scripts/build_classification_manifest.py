from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from counterfactual_audio_repro.counterfactuals import generate_counterfactual
from counterfactual_audio_repro.manifests import write_jsonl


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", required=True)
    parser.add_argument("--audio-root", required=True)
    parser.add_argument("--path-column", required=True)
    parser.add_argument("--label-column", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--dataset-name", required=True)
    parser.add_argument("--path-prefix", default="")
    parser.add_argument("--caption-template", default="this is a sound of {label}")
    args = parser.parse_args()

    audio_root = Path(args.audio_root)
    rows = []

    with Path(args.csv).open("r", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for record in reader:
            relative_path = f"{args.path_prefix}{record[args.path_column]}"
            label = record[args.label_column]
            caption = args.caption_template.format(label=label)
            counterfactual = generate_counterfactual(caption).counterfactual_caption
            rows.append(
                {
                    "dataset": args.dataset_name,
                    "split": record.get("fold") or record.get("esc10") or record.get("folds"),
                    "path": relative_path,
                    "audio_path": str((audio_root / relative_path).resolve()),
                    "caption_index": 0,
                    "caption": caption,
                    "counterfactual_caption": counterfactual,
                    "label": label,
                }
            )

    write_jsonl(args.output, rows)
    print(f"wrote {len(rows)} rows to {args.output}")


if __name__ == "__main__":
    main()

