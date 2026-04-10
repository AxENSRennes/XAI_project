from __future__ import annotations

from collections import defaultdict

import numpy as np
import torch
from torch.utils.data import DataLoader

from .data import CounterfactualCollator, ManifestAudioTextDataset


def _move_to_device(batch_inputs: dict[str, torch.Tensor], device: torch.device) -> dict[str, torch.Tensor]:
    return {
        key: value.to(device) if isinstance(value, torch.Tensor) else value
        for key, value in batch_inputs.items()
    }


@torch.no_grad()
def collect_retrieval_embeddings(
    model,
    processor,
    manifest_path: str,
    sample_rate: int,
    clip_duration_s: float,
    batch_size: int,
    num_workers: int,
    device: torch.device,
) -> tuple[list[str], torch.Tensor, list[str], list[str], torch.Tensor, torch.Tensor]:
    dataset = ManifestAudioTextDataset(
        manifest_path=manifest_path,
        sample_rate=sample_rate,
        clip_duration_s=clip_duration_s,
        random_crop=False,
    )
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=CounterfactualCollator(processor=processor, sample_rate=sample_rate),
    )

    audio_embeddings_by_path: dict[str, torch.Tensor] = {}
    text_paths: list[str] = []
    text_captions: list[str] = []
    text_embeddings: list[torch.Tensor] = []
    counterfactual_text_embeddings: list[torch.Tensor] = []

    for batch in loader:
        audio_inputs = _move_to_device(batch["audio_inputs"], device)
        factual_inputs = _move_to_device(batch["factual_text_inputs"], device)
        counterfactual_inputs = _move_to_device(batch["counterfactual_text_inputs"], device)
        audio_batch = model.encode_audio(audio_inputs).cpu()
        text_batch = model.encode_text(factual_inputs).cpu()
        counterfactual_batch = model.encode_text(counterfactual_inputs).cpu()

        for path, audio_embedding in zip(batch["audio_paths"], audio_batch):
            audio_embeddings_by_path[path] = audio_embedding
        text_paths.extend(batch["audio_paths"])
        text_captions.extend(batch["captions"])
        text_embeddings.extend(list(text_batch))
        counterfactual_text_embeddings.extend(list(counterfactual_batch))

    audio_paths = sorted(audio_embeddings_by_path.keys())
    stacked_audio = torch.stack([audio_embeddings_by_path[path] for path in audio_paths])
    stacked_text = torch.stack(text_embeddings)
    stacked_counterfactual = torch.stack(counterfactual_text_embeddings)
    return audio_paths, stacked_audio, text_paths, text_captions, stacked_text, stacked_counterfactual


@torch.no_grad()
def evaluate_retrieval(
    model,
    processor,
    manifest_path: str,
    sample_rate: int,
    clip_duration_s: float,
    batch_size: int,
    num_workers: int,
    device: torch.device,
) -> dict[str, float]:
    (
        audio_paths,
        audio_embeddings,
        text_paths,
        _text_captions,
        text_embeddings,
        counterfactual_embeddings,
    ) = collect_retrieval_embeddings(
        model=model,
        processor=processor,
        manifest_path=manifest_path,
        sample_rate=sample_rate,
        clip_duration_s=clip_duration_s,
        batch_size=batch_size,
        num_workers=num_workers,
        device=device,
    )

    similarity = text_embeddings @ audio_embeddings.T
    audio_index = {path: idx for idx, path in enumerate(audio_paths)}

    top1_hits = 0
    top10_hits = 0
    factual_similarity = torch.sum(text_embeddings * torch.stack([audio_embeddings[audio_index[path]] for path in text_paths]), dim=-1)
    counterfactual_similarity = torch.sum(counterfactual_embeddings * torch.stack([audio_embeddings[audio_index[path]] for path in text_paths]), dim=-1)
    factual_better_than_counterfactual = int((factual_similarity > counterfactual_similarity).sum().item())

    for row_index, path in enumerate(text_paths):
        ranking = torch.argsort(similarity[row_index], descending=True)
        target_index = audio_index[path]
        if ranking[0].item() == target_index:
            top1_hits += 1
        if target_index in ranking[:10].tolist():
            top10_hits += 1

    top1 = top1_hits / len(text_paths)
    top10 = top10_hits / len(text_paths)

    return {
        "top1": top1,
        "top10": top10,
        "num_queries": float(len(text_paths)),
        "audio_items": float(len(audio_paths)),
        "sim_xy_gt_sim_xy_star_count": float(factual_better_than_counterfactual),
    }


@torch.no_grad()
def evaluate_zero_shot(
    model,
    processor,
    manifest_path: str,
    sample_rate: int,
    clip_duration_s: float,
    batch_size: int,
    num_workers: int,
    device: torch.device,
    label_template: str,
) -> dict[str, float]:
    dataset = ManifestAudioTextDataset(
        manifest_path=manifest_path,
        sample_rate=sample_rate,
        clip_duration_s=clip_duration_s,
        random_crop=False,
    )
    labels = sorted({row["label"] for row in dataset.rows if row.get("label")})
    if not labels:
        raise ValueError("Zero-shot evaluation manifest must contain a 'label' column.")

    tokenizer = getattr(processor, "tokenizer", processor)
    text_batch = tokenizer(
        [label_template.format(label) for label in labels],
        return_tensors="pt",
        padding=True,
        truncation=True,
    )
    text_batch = _move_to_device(dict(text_batch), device)
    class_embeddings = model.encode_text(text_batch).cpu()
    label_to_index = {label: idx for idx, label in enumerate(labels)}

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=CounterfactualCollator(processor=processor, sample_rate=sample_rate),
    )

    correct = 0
    total = 0
    per_label_total: dict[str, int] = defaultdict(int)
    per_label_correct: dict[str, int] = defaultdict(int)

    for batch in loader:
        audio_inputs = _move_to_device(batch["audio_inputs"], device)
        audio_embeddings = model.encode_audio(audio_inputs).cpu()
        scores = audio_embeddings @ class_embeddings.T
        predictions = torch.argmax(scores, dim=-1).tolist()

        for predicted_index, gold_label in zip(predictions, batch["labels"]):
            if gold_label is None:
                continue
            total += 1
            per_label_total[gold_label] += 1
            if predicted_index == label_to_index[gold_label]:
                correct += 1
                per_label_correct[gold_label] += 1

    metrics = {
        "top1_accuracy": correct / max(total, 1),
        "num_examples": float(total),
        "num_labels": float(len(labels)),
    }
    for label, total_count in per_label_total.items():
        metrics[f"acc_{label}"] = per_label_correct[label] / max(total_count, 1)
    return metrics
