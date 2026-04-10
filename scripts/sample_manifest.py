from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from counterfactual_audio_repro.manifests import read_manifest, write_jsonl


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--limit", type=int, required=True)
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()

    rows = read_manifest(args.input)
    rng = random.Random(args.seed)
    sampled = rows if len(rows) <= args.limit else rng.sample(rows, args.limit)
    write_jsonl(args.output, sampled)
    print(f"wrote {len(sampled)} rows to {args.output}")


if __name__ == "__main__":
    main()
