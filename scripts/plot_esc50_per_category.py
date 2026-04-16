from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import seaborn as sns
import torch

ROOT = Path(__file__).resolve().parents[1]
SRC  = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from counterfactual_audio_repro.evaluation import evaluate_zero_shot
from counterfactual_audio_repro.hf import configure_hf_cache
from counterfactual_audio_repro.train import load_model_from_checkpoint


LABEL_TEMPLATE = "this is a sound of {}"


def get_per_label_acc(checkpoint: str, manifest: str, device: torch.device) -> dict[str, float]:
    model, config = load_model_from_checkpoint(checkpoint, device)
    from transformers import AutoProcessor
    processor = AutoProcessor.from_pretrained(config["model_name"])
    metrics = evaluate_zero_shot(
        model=model,
        processor=processor,
        manifest_path=manifest,
        sample_rate=int(config["sample_rate"]),
        clip_duration_s=float(config["clip_duration_s"]),
        batch_size=int(config.get("batch_size", 32)),
        num_workers=int(config.get("num_workers", 0)),
        device=device,
        label_template=LABEL_TEMPLATE,
    )
    # Keep only per-label entries (keys starting with "acc_")
    return {k[4:]: v for k, v in metrics.items() if k.startswith("acc_")}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", required=True)
    parser.add_argument("--no-cf",    required=True, dest="no_cf")
    parser.add_argument("--full",     required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--out", default="figures/esc50_per_category.png")
    args = parser.parse_args()

    configure_hf_cache()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    print("Evaluating baseline...")
    acc_baseline = get_per_label_acc(args.baseline, args.manifest, device)
    print("Evaluating no-CF model...")
    acc_nocf     = get_per_label_acc(args.no_cf,    args.manifest, device)
    print("Evaluating full model...")
    acc_full     = get_per_label_acc(args.full,     args.manifest, device)

    # Sort categories by improvement (full - baseline), largest first
    labels = sorted(acc_baseline.keys(), key=lambda l: acc_full.get(l, 0) - acc_baseline.get(l, 0), reverse=True)

    n = len(labels)
    x = np.arange(n)
    width = 0.28

    sns.set_theme(style="ticks", context="paper", font_scale=0.85)
    plt.rcParams.update({"figure.dpi": 150, "savefig.dpi": 300})

    palette = sns.color_palette("tab10")
    fig, ax = plt.subplots(figsize=(14, 5))

    b1 = ax.bar(x - width, [acc_baseline.get(l, 0) for l in labels], width,
                label="Pre-trained CLAP", color=palette[0], alpha=0.85)
    b2 = ax.bar(x,          [acc_nocf.get(l, 0) for l in labels],     width,
                label="Fine-tuned (no CF)", color=palette[1], alpha=0.85)
    b3 = ax.bar(x + width,  [acc_full.get(l, 0) for l in labels],     width,
                label="Full model", color=palette[2], alpha=0.85)

    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=7)
    ax.set_ylabel("Top-1 accuracy", fontsize=10)
    ax.set_title("Per-category zero-shot accuracy on ESC-50", fontsize=11, fontweight="bold")
    ax.set_ylim(0, 1.05)
    ax.axhline(y=1.0, color="gray", linewidth=0.5, linestyle="--", alpha=0.5)
    ax.legend(fontsize=9, framealpha=0.7)
    sns.despine(ax=ax)

    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    print(f"Saved → {out_path}")

    # Print summary of biggest gains and losses
    diffs = {l: acc_full.get(l, 0) - acc_baseline.get(l, 0) for l in labels}
    print("\nTop 5 categories improved by full model vs baseline:")
    for l in sorted(diffs, key=diffs.get, reverse=True)[:5]:
        print(f"  {l:<30} {diffs[l]:+.3f}  (baseline={acc_baseline.get(l,0):.2f}, full={acc_full.get(l,0):.2f})")
    print("\nTop 5 categories hurt by full model vs baseline:")
    for l in sorted(diffs, key=diffs.get)[:5]:
        print(f"  {l:<30} {diffs[l]:+.3f}  (baseline={acc_baseline.get(l,0):.2f}, full={acc_full.get(l,0):.2f})")


if __name__ == "__main__":
    main()
