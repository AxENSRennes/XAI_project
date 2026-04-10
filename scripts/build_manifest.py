from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from counterfactual_audio_repro.manifests import flatten_release_json, write_jsonl


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--release-json", required=True)
    parser.add_argument("--dataset-root")
    parser.add_argument("--dataset-name")
    parser.add_argument("--output", required=True)
    parser.add_argument("--generate-missing-counterfactuals", action="store_true")
    args = parser.parse_args()

    rows = flatten_release_json(
        release_json=args.release_json,
        dataset_root=args.dataset_root,
        dataset_name=args.dataset_name,
        generate_missing_counterfactuals=args.generate_missing_counterfactuals,
    )
    write_jsonl(args.output, rows)
    print(f"wrote {len(rows)} rows to {args.output}")


if __name__ == "__main__":
    main()

