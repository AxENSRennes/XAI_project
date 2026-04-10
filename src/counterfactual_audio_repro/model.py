from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import nn


@dataclass
class ForwardOutput:
    loss: torch.Tensor
    clip_loss: torch.Tensor
    angle_loss: torch.Tensor
    factual_loss: torch.Tensor
    audio_embeddings: torch.Tensor
    factual_text_embeddings: torch.Tensor
    counterfactual_text_embeddings: torch.Tensor


class CounterfactualClap(nn.Module):
    def __init__(
        self,
        model_name: str,
        freeze_text: bool = True,
        freeze_audio: bool = False,
        clip_loss_weight: float = 1.0,
        angle_loss_weight: float = 1.0,
        factual_loss_weight: float = 100.0,
        margin: float = 0.1,
    ) -> None:
        super().__init__()
        from transformers import ClapModel

        self.backbone = ClapModel.from_pretrained(model_name)
        projection_dim = int(self.backbone.config.projection_dim)
        self.audio_adapter = nn.Linear(projection_dim, projection_dim, bias=False)
        self.text_adapter = nn.Linear(projection_dim, projection_dim, bias=False)
        self.clip_loss_weight = clip_loss_weight
        self.angle_loss_weight = angle_loss_weight
        self.factual_loss_weight = factual_loss_weight
        self.margin = margin
        self.logit_scale = nn.Parameter(torch.tensor(2.6592))
        nn.init.eye_(self.audio_adapter.weight)
        nn.init.eye_(self.text_adapter.weight)

        self.freeze_text = freeze_text
        self.freeze_audio = freeze_audio

        if freeze_text:
            self._freeze_module(getattr(self.backbone, "text_model", None))
            self._freeze_module(getattr(self.backbone, "text_projection", None))
        if freeze_audio:
            self._freeze_module(getattr(self.backbone, "audio_model", None))
            self._freeze_module(getattr(self.backbone, "audio_projection", None))

    @staticmethod
    def _freeze_module(module: nn.Module | None) -> None:
        if module is None:
            return
        for parameter in module.parameters():
            parameter.requires_grad = False

    def train(self, mode: bool = True):
        super().train(mode)
        if self.freeze_audio:
            audio_model = getattr(self.backbone, "audio_model", None)
            if audio_model is not None:
                audio_model.eval()
        if self.freeze_text:
            text_model = getattr(self.backbone, "text_model", None)
            if text_model is not None:
                text_model.eval()
        return self

    @staticmethod
    def _move_inputs(inputs: dict[str, torch.Tensor], device: torch.device) -> dict[str, torch.Tensor]:
        return {
            key: value.to(device)
            if isinstance(value, torch.Tensor)
            else value
            for key, value in inputs.items()
        }

    def encode_audio(self, audio_inputs: dict[str, torch.Tensor]) -> torch.Tensor:
        if self.freeze_audio:
            with torch.no_grad():
                features = self.backbone.get_audio_features(**audio_inputs)
        else:
            features = self.backbone.get_audio_features(**audio_inputs)
        return F.normalize(self.audio_adapter(features), dim=-1)

    def encode_text(self, text_inputs: dict[str, torch.Tensor]) -> torch.Tensor:
        if self.freeze_text:
            with torch.no_grad():
                features = self.backbone.get_text_features(**text_inputs)
        else:
            features = self.backbone.get_text_features(**text_inputs)
        return F.normalize(self.text_adapter(features), dim=-1)

    def compute_similarity(self, audio_embeddings: torch.Tensor, text_embeddings: torch.Tensor) -> torch.Tensor:
        scale = self.logit_scale.exp()
        return scale * text_embeddings @ audio_embeddings.T

    def forward(
        self,
        audio_inputs: dict[str, torch.Tensor],
        factual_text_inputs: dict[str, torch.Tensor],
        counterfactual_text_inputs: dict[str, torch.Tensor],
    ) -> ForwardOutput:
        audio_embeddings = self.encode_audio(audio_inputs)
        factual_embeddings = self.encode_text(factual_text_inputs)
        counterfactual_embeddings = self.encode_text(counterfactual_text_inputs)

        logits = self.compute_similarity(audio_embeddings, factual_embeddings)
        labels = torch.arange(logits.size(0), device=logits.device)
        clip_loss = 0.5 * (
            F.cross_entropy(logits, labels) + F.cross_entropy(logits.T, labels)
        )

        cosine_factual = F.cosine_similarity(audio_embeddings, factual_embeddings)
        cosine_counterfactual = F.cosine_similarity(audio_embeddings, counterfactual_embeddings)
        angle_loss = torch.relu(cosine_counterfactual - cosine_factual + self.margin).mean()
        factual_loss = F.mse_loss(audio_embeddings, factual_embeddings)

        loss = (
            self.clip_loss_weight * clip_loss
            + self.angle_loss_weight * angle_loss
            + self.factual_loss_weight * factual_loss
        )

        return ForwardOutput(
            loss=loss,
            clip_loss=clip_loss,
            angle_loss=angle_loss,
            factual_loss=factual_loss,
            audio_embeddings=audio_embeddings,
            factual_text_embeddings=factual_embeddings,
            counterfactual_text_embeddings=counterfactual_embeddings,
        )
