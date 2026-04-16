from __future__ import annotations

import argparse
import random
import sys
import textwrap
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import torch

ROOT = Path(__file__).resolve().parents[1]
SRC  = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from counterfactual_audio_repro.evaluation import collect_retrieval_embeddings
from counterfactual_audio_repro.hf import configure_hf_cache
from counterfactual_audio_repro.train import load_model_from_checkpoint


WRAP = 42          # characters per line for caption wrapping
ROW_HEIGHT = 1.0   # inches per example row
COL_WIDTHS = [0.30, 0.17, 0.30]  # fractions: query | correct clip | retrieved


def find_failures(
    checkpoint: str,
    manifest: str,
    device: torch.device,
    n_examples: int,
    seed: int = 0,
) -> list[dict]:
    """
    Run retrieval and collect Top-1 failures.
    Returns a list of dicts with keys:
      query_caption, correct_path, retrieved_path, retrieved_caption,
      correct_rank (1-based), similarity_correct, similarity_retrieved
    """
    model, config = load_model_from_checkpoint(checkpoint, device)
    from transformers import AutoProcessor
    processor = AutoProcessor.from_pretrained(config["model_name"])

    audio_paths, audio_emb, text_paths, captions, text_emb, _ = collect_retrieval_embeddings(
        model=model,
        processor=processor,
        manifest_path=manifest,
        sample_rate=int(config["sample_rate"]),
        clip_duration_s=float(config["clip_duration_s"]),
        batch_size=int(config.get("batch_size", 32)),
        num_workers=int(config.get("num_workers", 0)),
        device=device,
    )

    # Build a map from audio path → one representative caption
    path_to_caption: dict[str, str] = {}
    for path, caption in zip(text_paths, captions):
        if path not in path_to_caption:
            path_to_caption[path] = caption

    audio_index = {p: i for i, p in enumerate(audio_paths)}
    similarity  = text_emb @ audio_emb.T   # (n_queries, n_audio)

    failures = []
    for q_idx, (q_path, q_caption) in enumerate(zip(text_paths, captions)):
        ranking      = torch.argsort(similarity[q_idx], descending=True)
        target_idx   = audio_index[q_path]
        retrieved_idx = int(ranking[0].item())

        if retrieved_idx == target_idx:
            continue   # Top-1 correct — skip

        correct_rank = int((ranking == target_idx).nonzero(as_tuple=True)[0].item()) + 1
        retrieved_path = audio_paths[retrieved_idx]

        failures.append({
            "query_caption":    q_caption,
            "correct_path":     Path(q_path).name,
            "retrieved_path":   Path(retrieved_path).name,
            "retrieved_caption": path_to_caption.get(retrieved_path, "—"),
            "correct_rank":     correct_rank,
            "sim_retrieved":    float(similarity[q_idx, retrieved_idx].item()),
            "sim_correct":      float(similarity[q_idx, target_idx].item()),
        })

    # Sample a diverse set: prefer cases where the retrieved caption looks
    # semantically close (small similarity gap) — these are the most interesting
    failures.sort(key=lambda r: r["sim_retrieved"] - r["sim_correct"])
    # Take from the top (smallest gap = most confusing failures)
    pool = failures[:max(n_examples * 4, 40)]
    random.seed(seed)
    return random.sample(pool, min(n_examples, len(pool)))


def wrap(text: str, width: int = WRAP) -> str:
    return "\n".join(textwrap.wrap(text, width))


def draw_figure(examples: list[dict], out_path: Path) -> None:
    n = len(examples)
    fig_h = ROW_HEIGHT * n + 1.0   # +1 for header
    fig_w = 10.0

    sns.set_theme(style="white", context="paper", font_scale=1.0)
    plt.rcParams.update({"figure.dpi": 150, "savefig.dpi": 300})

    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, n + 1)
    ax.axis("off")

    # Column header positions (centre of each column)
    col_x  = [0.01, 0.49, 0.65]
    headers = ["Query caption", "Correct clip", "Retrieved clip  (caption)"]
    header_bg = "#2C3E50"

    for x, hdr in zip(col_x, headers):
        ax.text(
            x, n + 0.62, hdr,
            fontsize=9, fontweight="bold", color="white",
            va="center", ha="left",
        )
    ax.add_patch(mpatches.FancyBboxPatch(
        (0, n + 0.25), 1.0, 0.62,
        boxstyle="square,pad=0", facecolor=header_bg, zorder=0,
    ))

    # Row colours alternating
    row_colors = ["#F7F9FC", "#FFFFFF"]

    for i, ex in enumerate(examples):
        row_y_bottom = i
        row_y_centre = i + 0.5

        # Background stripe
        ax.add_patch(mpatches.FancyBboxPatch(
            (0, row_y_bottom), 1.0, ROW_HEIGHT,
            boxstyle="square,pad=0",
            facecolor=row_colors[i % 2],
            zorder=0,
        ))

        # Query caption
        ax.text(
            col_x[0], row_y_centre,
            wrap(ex["query_caption"]),
            fontsize=7.5, va="center", ha="left",
            color="#1A1A1A",
        )

        # Correct clip filename + rank info
        rank_color = "#27AE60" if ex["correct_rank"] <= 5 else "#E74C3C"
        ax.text(
            col_x[1], row_y_centre + 0.15,
            ex["correct_path"],
            fontsize=6.8, va="center", ha="left",
            color="#1A1A1A", style="italic",
        )
        ax.text(
            col_x[1], row_y_centre - 0.18,
            f"rank {ex['correct_rank']}  (sim={ex['sim_correct']:.3f})",
            fontsize=6.5, va="center", ha="left",
            color=rank_color,
        )

        # Retrieved clip filename + caption
        ax.text(
            col_x[2], row_y_centre + 0.15,
            ex["retrieved_path"],
            fontsize=6.8, va="center", ha="left",
            color="#1A1A1A", style="italic",
        )
        ax.text(
            col_x[2], row_y_centre - 0.18,
            wrap(ex["retrieved_caption"], width=38),
            fontsize=6.5, va="center", ha="left",
            color="#555555",
        )

        # Horizontal separator
        ax.axhline(y=row_y_bottom, color="#CCCCCC", linewidth=0.5, zorder=1)

    # Vertical separators
    for x in [0.47, 0.63]:
        ax.axvline(x=x, ymin=0, ymax=1, color="#CCCCCC", linewidth=0.5, zorder=1)

    fig.tight_layout(pad=0.2)
    fig.savefig(out_path, bbox_inches="tight")
    print(f"Saved → {out_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--manifest",   required=True)
    parser.add_argument("--n-examples", type=int, default=5)
    parser.add_argument("--out",        default="figures/retrieval_failures.png")
    args = parser.parse_args()

    configure_hf_cache()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    print("Running retrieval to find failures …")
    examples = find_failures(args.checkpoint, args.manifest, device, args.n_examples)
    print(f"Found {len(examples)} failure examples to display.")
    draw_figure(examples, out_path)


if __name__ == "__main__":
    main()
