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
    "Pre-trained CLAP": "pretrained_baseline",
    "No CF losses":     "ablation_no_cf",
    "Full model":       "ablation_full",
}

K_MAX   = 10
PALETTE = ["#888888", "#F28E2B", "#4878CF"]


@torch.no_grad()
def compute_both_directions(
    checkpoint: str,
    manifest: str,
    device: torch.device,
) -> tuple[list[float], list[float]]:
    """
    Returns (t2a_recalls, a2t_recalls) where each is a list of length K_MAX.

    T2A: for each text query i, rank audio clips by similarity and check
         whether the correct audio clip appears in the top k.

    A2T: for each unique audio clip j, rank text queries by similarity and
         check whether at least one correct caption appears in the top k.
         "Correct captions" are all rows i where text_paths[i] == audio_paths[j].
    """
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

    # Similarity matrix: (n_queries, n_audio)
    sim = text_emb @ audio_emb.T

    # Index maps
    audio_index = {p: i for i, p in enumerate(audio_paths)}   # path -> col index

    t2a_rankings = torch.argsort(sim, dim=1, descending=True)  # (n_queries, n_audio)
    t2a_hits = [0] * K_MAX
    for q_idx, q_path in enumerate(text_paths):
        target = audio_index[q_path]
        top_k  = t2a_rankings[q_idx, :K_MAX].tolist()
        for k in range(K_MAX):
            if target in top_k[: k + 1]:
                t2a_hits[k] += 1
    n_queries = len(text_paths)
    t2a_recalls = [h / n_queries for h in t2a_hits]

    # Build map: audio_path -> list of text row indices (its captions)
    from collections import defaultdict
    audio_to_text_rows: dict[str, list[int]] = defaultdict(list)
    for q_idx, q_path in enumerate(text_paths):
        audio_to_text_rows[q_path].append(q_idx)

    a2t_rankings = torch.argsort(sim, dim=0, descending=True)  # (n_queries, n_audio)
    a2t_hits = [0] * K_MAX
    for a_idx, a_path in enumerate(audio_paths):
        correct_rows = set(audio_to_text_rows.get(a_path, []))
        if not correct_rows:
            continue
        top_k = a2t_rankings[:K_MAX, a_idx].tolist()  # top-k text row indices for audio j
        for k in range(K_MAX):
            if correct_rows & set(top_k[: k + 1]):
                a2t_hits[k] += 1
    n_audio = len(audio_paths)
    a2t_recalls = [h / n_audio for h in a2t_hits]

    return t2a_recalls, a2t_recalls


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest",   required=True)
    parser.add_argument("--output-dir", default="outputs")
    parser.add_argument("--out",        default="figures/audio_to_text_retrieval.png")
    args = parser.parse_args()

    configure_hf_cache()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    output_root = Path(args.output_dir)

    sns.set_theme(style="ticks", context="paper", font_scale=1.1)
    plt.rcParams.update({"figure.dpi": 150, "savefig.dpi": 300})

    k_values = list(range(1, K_MAX + 1))

    fig, (ax_t2a, ax_a2t) = plt.subplots(1, 2, figsize=(10, 4.2), sharey=False)

    print("\n{'Model':<22} {'R@1 T→A':>10} {'R@10 T→A':>11} {'R@1 A→T':>10} {'R@10 A→T':>11}")
    print("-" * 68)

    for color, (label, folder) in zip(PALETTE, CONFIGS.items()):
        ckpt = output_root / folder / "best.pt"
        if not ckpt.exists():
            print(f"  [skip] checkpoint not found: {ckpt}")
            continue
        print(f"Computing both-direction Recall@k: {label}")
        t2a, a2t = compute_both_directions(str(ckpt), args.manifest, device)

        print(
            f"  {label:<22} R@1 T→A={t2a[0]:.3f}  R@10 T→A={t2a[-1]:.3f}"
            f"  R@1 A→T={a2t[0]:.3f}  R@10 A→T={a2t[-1]:.3f}"
        )

        ls     = "--" if "Pre-trained" in label else "-"
        marker = "x"  if "Pre-trained" in label else "o"
        kw = dict(linestyle=ls, marker=marker, color=color, label=label,
                  linewidth=1.8, markersize=4)

        ax_t2a.plot(k_values, t2a, **kw)
        ax_a2t.plot(k_values, a2t, **kw)

    for ax, title in [
        (ax_t2a, "Text → Audio"),
        (ax_a2t, "Audio → Text"),
    ]:
        ax.set_xlabel("k", fontsize=10)
        ax.set_ylabel("Recall@k", fontsize=10)
        ax.set_title(title, fontsize=11, fontweight="bold", pad=8)
        ax.set_xticks(k_values)
        ax.xaxis.set_major_formatter(mticker.FormatStrFormatter("%d"))
        ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.2f"))
        ax.legend(fontsize=8, framealpha=0.7, loc="lower right")
        sns.despine(ax=ax)

    fig.suptitle(
        "Symmetric retrieval Recall@k on Clotho validation set",
        fontsize=11, fontweight="bold", y=1.02,
    )
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    print(f"\nSaved → {out_path}")


if __name__ == "__main__":
    main()
