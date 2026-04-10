from __future__ import annotations

import json
import random
from pathlib import Path

import numpy as np
import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader

from .data import CounterfactualCollator, ManifestAudioTextDataset
from .evaluation import evaluate_retrieval
from .hf import configure_hf_cache
from .model import CounterfactualClap


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _move_to_device(inputs: dict[str, torch.Tensor], device: torch.device) -> dict[str, torch.Tensor]:
    return {
        key: value.to(device) if isinstance(value, torch.Tensor) else value
        for key, value in inputs.items()
    }


def save_checkpoint(
    output_path: str | Path,
    model: CounterfactualClap,
    config: dict,
    epoch: int,
    metrics: dict | None,
) -> None:
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "state_dict": model.state_dict(),
            "config": config,
            "epoch": epoch,
            "metrics": metrics or {},
        },
        output,
    )


def load_model_from_checkpoint(checkpoint_path: str | Path, device: torch.device) -> tuple[CounterfactualClap, dict]:
    configure_hf_cache()
    checkpoint = torch.load(checkpoint_path, map_location=device)
    config = checkpoint["config"]
    model = CounterfactualClap(
        model_name=config["model_name"],
        freeze_text=config.get("freeze_text", True),
        freeze_audio=config.get("freeze_audio", False),
        clip_loss_weight=config.get("clip_loss_weight", 1.0),
        angle_loss_weight=config.get("angle_loss_weight", 1.0),
        factual_loss_weight=config.get("factual_loss_weight", 100.0),
        margin=config.get("margin", 0.1),
    )
    model.load_state_dict(checkpoint["state_dict"])
    model.to(device)
    model.eval()
    return model, config


def train_model(
    config: dict,
    train_manifest: str,
    eval_manifest: str | None,
    output_dir: str,
) -> dict:
    from transformers import AutoProcessor

    configure_hf_cache()
    set_seed(int(config.get("seed", 7)))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    processor = AutoProcessor.from_pretrained(config["model_name"])

    train_dataset = ManifestAudioTextDataset(
        manifest_path=train_manifest,
        sample_rate=int(config["sample_rate"]),
        clip_duration_s=float(config["clip_duration_s"]),
        random_crop=True,
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=int(config["batch_size"]),
        shuffle=True,
        num_workers=int(config.get("num_workers", 0)),
        collate_fn=CounterfactualCollator(
            processor=processor,
            sample_rate=int(config["sample_rate"]),
        ),
    )

    model = CounterfactualClap(
        model_name=config["model_name"],
        freeze_text=bool(config.get("freeze_text", True)),
        freeze_audio=bool(config.get("freeze_audio", False)),
        clip_loss_weight=float(config.get("clip_loss_weight", 1.0)),
        angle_loss_weight=float(config.get("angle_loss_weight", 1.0)),
        factual_loss_weight=float(config.get("factual_loss_weight", 100.0)),
        margin=float(config.get("margin", 0.1)),
    ).to(device)

    optimizer = AdamW(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        lr=float(config.get("learning_rate", 1e-5)),
        weight_decay=float(config.get("weight_decay", 1e-4)),
    )

    output_root = Path(output_dir)
    output_root.mkdir(parents=True, exist_ok=True)

    best_metrics: dict = {}
    best_top1 = float("-inf")

    for epoch in range(1, int(config.get("epochs", 1)) + 1):
        model.train()
        running = {"loss": 0.0, "clip_loss": 0.0, "angle_loss": 0.0, "factual_loss": 0.0}

        for batch in train_loader:
            optimizer.zero_grad(set_to_none=True)
            outputs = model(
                audio_inputs=_move_to_device(batch["audio_inputs"], device),
                factual_text_inputs=_move_to_device(batch["factual_text_inputs"], device),
                counterfactual_text_inputs=_move_to_device(batch["counterfactual_text_inputs"], device),
            )
            outputs.loss.backward()
            optimizer.step()

            running["loss"] += float(outputs.loss.item())
            running["clip_loss"] += float(outputs.clip_loss.item())
            running["angle_loss"] += float(outputs.angle_loss.item())
            running["factual_loss"] += float(outputs.factual_loss.item())

        num_batches = max(len(train_loader), 1)
        epoch_metrics = {
            "epoch": epoch,
            "train_loss": running["loss"] / num_batches,
            "train_clip_loss": running["clip_loss"] / num_batches,
            "train_angle_loss": running["angle_loss"] / num_batches,
            "train_factual_loss": running["factual_loss"] / num_batches,
        }

        if eval_manifest:
            model.eval()
            retrieval_metrics = evaluate_retrieval(
                model=model,
                processor=processor,
                manifest_path=eval_manifest,
                sample_rate=int(config["sample_rate"]),
                clip_duration_s=float(config["clip_duration_s"]),
                batch_size=int(config["batch_size"]),
                num_workers=int(config.get("num_workers", 0)),
                device=device,
            )
            epoch_metrics.update({f"eval_{k}": v for k, v in retrieval_metrics.items()})
            if retrieval_metrics["top1"] > best_top1:
                best_top1 = retrieval_metrics["top1"]
                best_metrics = epoch_metrics
                save_checkpoint(output_root / "best.pt", model, config, epoch, epoch_metrics)
        else:
            best_metrics = epoch_metrics
            save_checkpoint(output_root / "best.pt", model, config, epoch, epoch_metrics)

        print(json.dumps(epoch_metrics, ensure_ascii=True), flush=True)
        save_checkpoint(output_root / "last.pt", model, config, epoch, epoch_metrics)
        with (output_root / "metrics.jsonl").open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(epoch_metrics, ensure_ascii=True) + "\n")

    return best_metrics
