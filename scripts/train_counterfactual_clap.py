from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from counterfactual_audio_repro.config import load_config
from counterfactual_audio_repro.train import train_model


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--train-manifest", required=True)
    parser.add_argument("--eval-manifest")
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()

    config = load_config(args.config)
    metrics = train_model(
        config=config,
        train_manifest=args.train_manifest,
        eval_manifest=args.eval_manifest,
        output_dir=args.output_dir,
    )
    print(json.dumps(metrics, indent=2, ensure_ascii=True))


if __name__ == "__main__":
    main()

