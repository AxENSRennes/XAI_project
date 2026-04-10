from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
import torchaudio
from torch.utils.data import Dataset

from .manifests import read_manifest


def load_audio_segment(
    audio_path: str,
    sample_rate: int,
    clip_duration_s: float,
    random_crop: bool,
) -> np.ndarray:
    waveform, source_sr = sf.read(audio_path, always_2d=True)
    waveform = waveform.mean(axis=1)

    if source_sr != sample_rate:
        wave_tensor = torch.from_numpy(waveform).float().unsqueeze(0)
        waveform = torchaudio.functional.resample(
            wave_tensor,
            source_sr,
            sample_rate,
        ).squeeze(0).numpy()

    target_length = int(sample_rate * clip_duration_s)
    if len(waveform) < target_length:
        padded = np.zeros(target_length, dtype=np.float32)
        padded[: len(waveform)] = waveform.astype(np.float32)
        return padded

    if len(waveform) == target_length:
        return waveform.astype(np.float32)

    if random_crop:
        start = random.randint(0, len(waveform) - target_length)
    else:
        start = (len(waveform) - target_length) // 2
    return waveform[start : start + target_length].astype(np.float32)


class ManifestAudioTextDataset(Dataset):
    def __init__(
        self,
        manifest_path: str,
        sample_rate: int,
        clip_duration_s: float,
        random_crop: bool,
    ) -> None:
        self.rows = read_manifest(manifest_path)
        self.sample_rate = sample_rate
        self.clip_duration_s = clip_duration_s
        self.random_crop = random_crop

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict:
        row = self.rows[index]
        audio_path = row.get("audio_path") or row.get("path")
        if not audio_path:
            raise ValueError(f"Missing audio path in row {index}: {row}")
        audio = load_audio_segment(
            audio_path=audio_path,
            sample_rate=self.sample_rate,
            clip_duration_s=self.clip_duration_s,
            random_crop=self.random_crop,
        )
        return {
            "audio": audio,
            "audio_path": audio_path,
            "caption": row.get("caption", ""),
            "counterfactual_caption": row.get("counterfactual_caption", row.get("caption", "")),
            "label": row.get("label"),
            "dataset": row.get("dataset"),
        }


@dataclass
class CounterfactualCollator:
    processor: object
    sample_rate: int

    def __call__(self, batch: list[dict]) -> dict:
        waveforms = [item["audio"] for item in batch]
        captions = [item["caption"] for item in batch]
        counterfactuals = [item["counterfactual_caption"] for item in batch]

        feature_extractor = getattr(self.processor, "feature_extractor", self.processor)
        tokenizer = getattr(self.processor, "tokenizer", self.processor)

        audio_inputs = feature_extractor(
            waveforms,
            sampling_rate=self.sample_rate,
            return_tensors="pt",
            padding=True,
        )
        factual_text_inputs = tokenizer(
            captions,
            return_tensors="pt",
            padding=True,
            truncation=True,
        )
        counterfactual_text_inputs = tokenizer(
            counterfactuals,
            return_tensors="pt",
            padding=True,
            truncation=True,
        )

        return {
            "audio_inputs": dict(audio_inputs),
            "factual_text_inputs": dict(factual_text_inputs),
            "counterfactual_text_inputs": dict(counterfactual_text_inputs),
            "audio_paths": [item["audio_path"] for item in batch],
            "captions": captions,
            "counterfactual_captions": counterfactuals,
            "labels": [item["label"] for item in batch],
        }

