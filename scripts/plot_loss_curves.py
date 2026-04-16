"""
Plot training loss curves and eval Top-1 for the 4 ablation configs.
Produces: figures/loss_curves.png

Usage:
  python scripts/plot_loss_curves.py [--output-dir outputs] [--out figures/loss_curves.png]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns


CONFIGS = {
    "No CF losses":    "ablation_no_cf",
    "Angle loss only": "ablation_angle_only",
    "Factual loss only": "ablation_factual_only",
    "Full model":      "ablation_full",
}

# One distinct colour per config
PALETTE = sns.color_palette("tab10", n_colors=len(CONFIGS))


def load_metrics(run_dir: Path) -> list[dict]:
    path = run_dir / "metrics.jsonl"
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", default="outputs")
    parser.add_argument("--out", default="figures/loss_curves.png")
    args = parser.parse_args()

    output_root = Path(args.output_dir)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    sns.set_theme(style="ticks", context="paper", font_scale=1.1)
    plt.rcParams.update({"figure.dpi": 150, "savefig.dpi": 300})

    fig, axes = plt.subplots(1, 2, figsize=(9, 3.8))
    ax_loss, ax_top1 = axes

    for color, (label, folder) in zip(PALETTE, CONFIGS.items()):
        rows = load_metrics(output_root / folder)
        if not rows:
            print(f"  [skip] no metrics.jsonl found for {folder}")
            continue
        epochs    = [r["epoch"] for r in rows]
        train_loss = [r.get("train_loss", float("nan")) for r in rows]
        eval_top1  = [r.get("eval_top1", float("nan")) for r in rows]

        ax_loss.plot(epochs, train_loss, marker="o", color=color, label=label, linewidth=1.8)
        ax_top1.plot(epochs, eval_top1,  marker="o", color=color, label=label, linewidth=1.8)

    # --- left panel: training loss ---
    ax_loss.set_title("Training loss", fontsize=11, fontweight="bold", pad=8)
    ax_loss.set_xlabel("Epoch", fontsize=10)
    ax_loss.set_ylabel("Total train loss", fontsize=10)
    ax_loss.xaxis.set_major_locator(mticker.MaxNLocator(integer=True))
    ax_loss.legend(fontsize=8, framealpha=0.7)
    sns.despine(ax=ax_loss)

    # --- right panel: eval Top-1 ---
    ax_top1.set_title("Retrieval Top-1 (Clotho val)", fontsize=11, fontweight="bold", pad=8)
    ax_top1.set_xlabel("Epoch", fontsize=10)
    ax_top1.set_ylabel("Top-1 accuracy", fontsize=10)
    ax_top1.xaxis.set_major_locator(mticker.MaxNLocator(integer=True))
    ax_top1.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.3f"))
    ax_top1.legend(fontsize=8, framealpha=0.7)
    sns.despine(ax=ax_top1)

    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    print(f"Saved → {out_path}")


if __name__ == "__main__":
    main()
