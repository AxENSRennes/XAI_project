from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from counterfactual_audio_repro.evaluation import evaluate_retrieval
from counterfactual_audio_repro.hf import configure_hf_cache
from counterfactual_audio_repro.train import load_model_from_checkpoint


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--num-workers", type=int)
    args = parser.parse_args()

    configure_hf_cache()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, config = load_model_from_checkpoint(args.checkpoint, device)

    from transformers import AutoProcessor

    processor = AutoProcessor.from_pretrained(config["model_name"])
    metrics = evaluate_retrieval(
        model=model,
        processor=processor,
        manifest_path=args.manifest,
        sample_rate=int(config["sample_rate"]),
        clip_duration_s=float(config["clip_duration_s"]),
        batch_size=args.batch_size or int(config["batch_size"]),
        num_workers=args.num_workers or int(config.get("num_workers", 0)),
        device=device,
    )
    print(json.dumps(metrics, indent=2, ensure_ascii=True))


if __name__ == "__main__":
    main()
