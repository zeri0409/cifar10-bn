"""
Loss functions used for CIFAR-10 experiments.
"""

from __future__ import annotations

import torch
from torch import nn


class FocalLoss(nn.Module):
    def __init__(self, gamma: float = 2.0, label_smoothing: float = 0.0) -> None:
        super().__init__()
        self.gamma = gamma
        self.ce = nn.CrossEntropyLoss(label_smoothing=label_smoothing, reduction="none")

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        ce_loss = self.ce(logits, targets)
        probabilities = torch.softmax(logits, dim=1)
        target_prob = probabilities.gather(1, targets.unsqueeze(1)).squeeze(1)
        focal_weight = (1.0 - target_prob).pow(self.gamma)
        return (focal_weight * ce_loss).mean()


def build_loss(name: str, label_smoothing: float = 0.0, **loss_kwargs) -> nn.Module:
    normalized_name = name.lower()
    if normalized_name == "cross_entropy":
        return nn.CrossEntropyLoss()
    if normalized_name == "label_smoothing":
        return nn.CrossEntropyLoss(label_smoothing=label_smoothing)
    if normalized_name == "focal":
        gamma = float(loss_kwargs.get("gamma", 2.0))
        return FocalLoss(gamma=gamma, label_smoothing=label_smoothing)
    raise ValueError(f"Unknown loss function: {name}")
