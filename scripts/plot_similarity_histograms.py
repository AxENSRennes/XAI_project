from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import seaborn as sns
import torch

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from counterfactual_audio_repro.evaluation import collect_retrieval_embeddings
from counterfactual_audio_repro.hf import configure_hf_cache
from counterfactual_audio_repro.train import load_model_from_checkpoint


# ── colour palette ────────────────────────────────────────────────────────────
FACTUAL_COLOR       = "#4C72B0"   # muted blue
COUNTERFACTUAL_COLOR = "#DD8452"  # warm orange
FILL_ALPHA          = 0.35
LINE_WIDTH          = 2.0


def compute_similarities(
    checkpoint: str,
    manifest: str,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray]:
    """Return arrays of per-pair cosine similarities (factual, counterfactual)."""
    model, config = load_model_from_checkpoint(checkpoint, device)
    from transformers import AutoProcessor
    processor = AutoProcessor.from_pretrained(config["model_name"])

    (
        audio_paths,
        audio_embeddings,
        text_paths,
        _captions,
        text_embeddings,
        counterfactual_embeddings,
    ) = collect_retrieval_embeddings(
        model=model,
        processor=processor,
        manifest_path=manifest,
        sample_rate=int(config["sample_rate"]),
        clip_duration_s=float(config["clip_duration_s"]),
        batch_size=int(config.get("batch_size", 32)),
        num_workers=int(config.get("num_workers", 0)),
        device=device,
    )

    audio_index = {path: idx for idx, path in enumerate(audio_paths)}
    matched_audio = torch.stack(
        [audio_embeddings[audio_index[p]] for p in text_paths]
    )

    sim_f  = torch.sum(text_embeddings          * matched_audio, dim=-1).numpy()
    sim_cf = torch.sum(counterfactual_embeddings * matched_audio, dim=-1).numpy()
    return sim_f, sim_cf


def panel(
    ax: plt.Axes,
    sim_f: np.ndarray,
    sim_cf: np.ndarray,
    title: str,
) -> None:
    """Draw one histogram panel onto ax."""

    # KDE + histogram via seaborn
    sns.histplot(
        sim_f,
        ax=ax,
        color=FACTUAL_COLOR,
        kde=True,
        stat="density",
        bins=50,
        alpha=FILL_ALPHA,
        label=r"$\mathrm{sim}(x,\,y)$  — factual",
        line_kws={"linewidth": LINE_WIDTH},
        edgecolor="none",
    )
    sns.histplot(
        sim_cf,
        ax=ax,
        color=COUNTERFACTUAL_COLOR,
        kde=True,
        stat="density",
        bins=50,
        alpha=FILL_ALPHA,
        label=r"$\mathrm{sim}(x,\,y^*)$ — counterfactual",
        line_kws={"linewidth": LINE_WIDTH},
        edgecolor="none",
    )

    # vertical mean lines
    mean_f  = sim_f.mean()
    mean_cf = sim_cf.mean()
    ax.axvline(mean_f,  color=FACTUAL_COLOR,       linewidth=1.6, linestyle="--", alpha=0.9)
    ax.axvline(mean_cf, color=COUNTERFACTUAL_COLOR, linewidth=1.6, linestyle="--", alpha=0.9)

    # gap annotation
    gap = mean_f - mean_cf
    ymax = ax.get_ylim()[1]
    ax.annotate(
        "",
        xy=(mean_f, ymax * 0.72),
        xytext=(mean_cf, ymax * 0.72),
        arrowprops=dict(arrowstyle="<->", color="#444444", lw=1.4),
    )
    ax.text(
        (mean_f + mean_cf) / 2,
        ymax * 0.75,
        rf"$\Delta={gap:+.3f}$",
        ha="center",
        va="bottom",
        fontsize=9,
        color="#444444",
    )

    # cosmetics
    ax.set_title(title, fontsize=11, fontweight="bold", pad=8)
    ax.set_xlabel("Cosine similarity", fontsize=10)
    ax.set_ylabel("Density", fontsize=10)
    ax.xaxis.set_major_formatter(mticker.FormatStrFormatter("%.2f"))
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(labelsize=9)
    ax.legend(fontsize=9, framealpha=0.6)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", required=True, help="Path to pre-trained checkpoint")
    parser.add_argument("--full",     required=True, help="Path to full-model checkpoint")
    parser.add_argument("--manifest", required=True, help="Path to clotho_val.jsonl")
    parser.add_argument("--out",      default="figures/similarity_histograms.png")
    parser.add_argument("--batch-size", type=int)
    args = parser.parse_args()

    configure_hf_cache()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)

    # ── global style ──────────────────────────────────────────────────────────
    sns.set_theme(style="whitegrid", font="DejaVu Sans")
    plt.rcParams.update({
        "figure.dpi": 150,
        "savefig.dpi": 300,
        "axes.grid": True,
        "grid.alpha": 0.35,
        "grid.linestyle": "--",
    })

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), sharey=False)
    fig.suptitle(
        r"Distribution of cosine similarities: $\mathrm{sim}(x,y)$ vs $\mathrm{sim}(x,y^*)$",
        fontsize=12,
        fontweight="bold",
        y=1.01,
    )

    checkpoints = [
        (args.baseline, "Pre-trained CLAP\n(no fine-tuning)"),
        (args.full,     "Full model\n$(w_1=1,\\ w_2=100)$"),
    ]

    for ax, (ckpt, title) in zip(axes, checkpoints):
        print(f"Computing similarities for: {ckpt}")
        sim_f, sim_cf = compute_similarities(ckpt, args.manifest, device)
        panel(ax, sim_f, sim_cf, title)

    fig.tight_layout()
    fig.savefig(args.out, bbox_inches="tight")
    print(f"Saved → {args.out}")


if __name__ == "__main__":
    main()
