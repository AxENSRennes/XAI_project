from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
import torch

ROOT = Path(__file__).resolve().parents[1]
SRC  = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from counterfactual_audio_repro.evaluation import collect_retrieval_embeddings
from counterfactual_audio_repro.hf import configure_hf_cache
from counterfactual_audio_repro.train import load_model_from_checkpoint


CONFIGS = {
    "Pre-trained CLAP":    "pretrained_baseline",
    "No CF losses":        "ablation_no_cf",
    "Angle loss only":     "ablation_angle_only",
    "Factual loss only":   "ablation_factual_only",
    "Full model":          "ablation_full",
}

K_MAX = 10
PALETTE = sns.color_palette("tab10", n_colors=len(CONFIGS))


@torch.no_grad()
def recall_at_k(checkpoint: str, manifest: str, device: torch.device) -> list[float]:
    """Return Recall@k for k = 1 .. K_MAX."""
    model, config = load_model_from_checkpoint(checkpoint, device)
    from transformers import AutoProcessor
    processor = AutoProcessor.from_pretrained(config["model_name"])

    audio_paths, audio_emb, text_paths, _, text_emb, _ = collect_retrieval_embeddings(
        model=model,
        processor=processor,
        manifest_path=manifest,
        sample_rate=int(config["sample_rate"]),
        clip_duration_s=float(config["clip_duration_s"]),
        batch_size=int(config.get("batch_size", 32)),
        num_workers=int(config.get("num_workers", 0)),
        device=device,
    )

    audio_index = {p: i for i, p in enumerate(audio_paths)}
    similarity  = text_emb @ audio_emb.T          # (n_queries, n_audio)
    rankings    = torch.argsort(similarity, dim=1, descending=True)  # (n_queries, n_audio)

    hits = [0] * K_MAX
    for q_idx, q_path in enumerate(text_paths):
        target = audio_index[q_path]
        top_k  = rankings[q_idx, :K_MAX].tolist()
        for k in range(K_MAX):
            if target in top_k[: k + 1]:
                hits[k] += 1

    n = len(text_paths)
    return [h / n for h in hits]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest",   required=True)
    parser.add_argument("--output-dir", default="outputs")
    parser.add_argument("--out",        default="figures/recall_at_k.png")
    args = parser.parse_args()

    configure_hf_cache()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    output_root = Path(args.output_dir)

    sns.set_theme(style="ticks", context="paper", font_scale=1.1)
    plt.rcParams.update({"figure.dpi": 150, "savefig.dpi": 300})

    fig, ax = plt.subplots(figsize=(6, 4.2))
    k_values = list(range(1, K_MAX + 1))

    for color, (label, folder) in zip(PALETTE, CONFIGS.items()):
        ckpt = output_root / folder / "best.pt"
        if not ckpt.exists():
            print(f"  [skip] checkpoint not found: {ckpt}")
            continue
        print(f"Computing Recall@k for: {label}")
        recalls = recall_at_k(str(ckpt), args.manifest, device)

        # Distinguish pre-trained baseline visually
        ls = "--" if "Pre-trained" in label else "-"
        marker = "x" if "Pre-trained" in label else "o"
        ax.plot(k_values, recalls, linestyle=ls, marker=marker,
                color=color, label=label, linewidth=1.8, markersize=4)

    ax.set_xlabel("k", fontsize=10)
    ax.set_ylabel("Recall@k", fontsize=10)
    ax.set_title("Text-to-audio Recall@k on Clotho validation set",
                 fontsize=11, fontweight="bold", pad=10)
    ax.set_xticks(k_values)
    ax.xaxis.set_major_formatter(mticker.FormatStrFormatter("%d"))
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.2f"))
    ax.legend(fontsize=8.5, framealpha=0.7, loc="lower right")
    sns.despine(ax=ax)

    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    print(f"Saved → {out_path}")


if __name__ == "__main__":
    main()
