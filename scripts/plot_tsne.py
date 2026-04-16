"""
t-SNE visualisation of audio, factual-text and counterfactual-text embeddings,
before (pretrained baseline) and after (full model) training.
Produces: figures/tsne_before_after.png

Usage:
  python scripts/plot_tsne.py \
      --baseline  outputs/pretrained_baseline/best.pt \
      --full      outputs/ablation_full/best.pt \
      --manifest  data/manifests/clotho_val.jsonl \
      [--n-samples 300] [--out figures/tsne_before_after.png]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
from sklearn.manifold import TSNE

ROOT = Path(__file__).resolve().parents[1]
SRC  = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from counterfactual_audio_repro.evaluation import collect_retrieval_embeddings
from counterfactual_audio_repro.hf import configure_hf_cache
from counterfactual_audio_repro.train import load_model_from_checkpoint


# Colour and marker settings
AUDIO_COLOR = "#4878CF"         # muted blue
FACTUAL_COLOR = "#6ACC65"       # muted green
CF_COLOR = "#D65F5F"            # muted red
POINT_SIZE = 12
POINT_ALPHA = 0.65


def get_embeddings(
    checkpoint: str,
    manifest: str,
    n_samples: int,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (audio, factual_text, counterfactual_text) embeddings, subsampled."""
    model, config = load_model_from_checkpoint(checkpoint, device)
    from transformers import AutoProcessor
    processor = AutoProcessor.from_pretrained(config["model_name"])

    audio_paths, audio_emb, text_paths, _, text_emb, cf_emb = collect_retrieval_embeddings(
        model=model,
        processor=processor,
        manifest_path=manifest,
        sample_rate=int(config["sample_rate"]),
        clip_duration_s=float(config["clip_duration_s"]),
        batch_size=int(config.get("batch_size", 32)),
        num_workers=int(config.get("num_workers", 0)),
        device=device,
    )

    # One audio embedding per unique audio file; subsample consistently
    rng = np.random.default_rng(42)
    n = min(n_samples, len(text_paths))
    idx_text = rng.choice(len(text_paths), size=n, replace=False)

    # Match audio embeddings to the sampled text entries
    audio_index = {p: i for i, p in enumerate(audio_paths)}
    sampled_audio = torch.stack([audio_emb[audio_index[text_paths[i]]] for i in idx_text])
    sampled_text  = text_emb[idx_text]
    sampled_cf    = cf_emb[idx_text]

    return sampled_audio.numpy(), sampled_text.numpy(), sampled_cf.numpy()


def panel(
    ax: plt.Axes,
    audio: np.ndarray,
    factual: np.ndarray,
    cf: np.ndarray,
    title: str,
) -> None:
    n = len(audio)
    all_emb = np.concatenate([audio, factual, cf], axis=0)

    print(f"  Running t-SNE on {all_emb.shape[0]} points …")
    coords = TSNE(
        n_components=2,
        perplexity=30,
        learning_rate="auto",
        init="pca",
        random_state=42,
        n_jobs=1,
    ).fit_transform(all_emb)

    ax.scatter(
        coords[:n, 0], coords[:n, 1],
        s=POINT_SIZE, alpha=POINT_ALPHA, color=AUDIO_COLOR,
        label="Audio", rasterized=True,
    )
    ax.scatter(
        coords[n:2*n, 0], coords[n:2*n, 1],
        s=POINT_SIZE, alpha=POINT_ALPHA, color=FACTUAL_COLOR,
        label="Factual text", rasterized=True,
    )
    ax.scatter(
        coords[2*n:, 0], coords[2*n:, 1],
        s=POINT_SIZE, alpha=POINT_ALPHA, color=CF_COLOR,
        label="Counterfactual text", rasterized=True,
    )

    ax.set_title(title, fontsize=11, fontweight="bold", pad=10)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.legend(fontsize=8, markerscale=1.8, framealpha=0.7, loc="lower right")
    sns.despine(ax=ax, left=True, bottom=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline",  required=True)
    parser.add_argument("--full",      required=True)
    parser.add_argument("--manifest",  required=True)
    parser.add_argument("--n-samples", type=int, default=300)
    parser.add_argument("--out",       default="figures/tsne_before_after.png")
    args = parser.parse_args()

    configure_hf_cache()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    sns.set_theme(style="ticks", context="paper", font_scale=1.1)
    plt.rcParams.update({"figure.dpi": 150, "savefig.dpi": 300})

    fig, axes = plt.subplots(1, 2, figsize=(9, 4.2))

    checkpoints = [
        (args.baseline, "Pre-trained CLAP (no fine-tuning)"),
        (args.full,     r"Full model ($w_1=1,\ w_2=100$)"),
    ]

    for ax, (ckpt, title) in zip(axes, checkpoints):
        print(f"\nLoading: {ckpt}")
        audio, factual, cf = get_embeddings(ckpt, args.manifest, args.n_samples, device)
        panel(ax, audio, factual, cf, title)

    fig.suptitle(
        "t-SNE of audio, factual and counterfactual text embeddings",
        fontsize=12,
        fontweight="bold",
        y=1.01,
    )
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    print(f"\nSaved → {out_path}")


if __name__ == "__main__":
    main()
