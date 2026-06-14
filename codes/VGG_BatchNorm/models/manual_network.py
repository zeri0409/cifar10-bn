"""
Manually implemented CNN using only tensor operations.
"""

from __future__ import annotations

import math
from collections import OrderedDict

import torch


def kaiming_tensor(*shape: int, device: torch.device | str = "cpu") -> torch.Tensor:
    fan_in = 1
    if len(shape) >= 2:
        fan_in = shape[1]
    if len(shape) == 4:
        fan_in *= shape[2] * shape[3]
    std = math.sqrt(2.0 / fan_in)
    return (torch.randn(*shape, device=device) * std).requires_grad_(True)


def bias_tensor(size: int, device: torch.device | str = "cpu") -> torch.Tensor:
    return torch.zeros(size, device=device, requires_grad=True)


def zero_pad2d(x: torch.Tensor, padding: int) -> torch.Tensor:
    if padding == 0:
        return x
    batch, channels, height, width = x.shape
    padded = torch.zeros(
        batch,
        channels,
        height + 2 * padding,
        width + 2 * padding,
        dtype=x.dtype,
        device=x.device,
    )
    padded[:, :, padding:padding + height, padding:padding + width] = x
    return padded


def manual_conv2d(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    stride: int = 1,
    padding: int = 0,
) -> torch.Tensor:
    kernel_h, kernel_w = weight.shape[-2:]
    x_padded = zero_pad2d(x, padding)
    patches = x_padded.unfold(2, kernel_h, stride).unfold(3, kernel_w, stride)
    return torch.einsum("bchwkl,ockl->bohw", patches, weight) + bias.view(1, -1, 1, 1)


def manual_relu(x: torch.Tensor) -> torch.Tensor:
    return torch.clamp_min(x, 0.0)


def manual_leaky_relu(x: torch.Tensor, negative_slope: float = 0.1) -> torch.Tensor:
    return torch.where(x >= 0.0, x, x * negative_slope)


def manual_max_pool2d(x: torch.Tensor, kernel_size: int = 2, stride: int = 2) -> torch.Tensor:
    patches = x.unfold(2, kernel_size, stride).unfold(3, kernel_size, stride)
    return patches.amax(dim=-1).amax(dim=-1)


def manual_linear(x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor) -> torch.Tensor:
    return x @ weight.t() + bias


class ManualCifarNet:
    def __init__(
        self,
        num_classes: int = 10,
        device: torch.device | str = "cpu",
        widths: tuple[int, int, int] = (32, 64, 128),
        hidden_dim: int = 256,
        activation: str = "relu",
    ) -> None:
        self.device = torch.device(device)
        self.widths = widths
        self.hidden_dim = hidden_dim
        self.activation = activation
        self.params = OrderedDict(
            conv1_weight=kaiming_tensor(widths[0], 3, 3, 3, device=self.device),
            conv1_bias=bias_tensor(widths[0], device=self.device),
            conv2_weight=kaiming_tensor(widths[1], widths[0], 3, 3, device=self.device),
            conv2_bias=bias_tensor(widths[1], device=self.device),
            conv3_weight=kaiming_tensor(widths[2], widths[1], 3, 3, device=self.device),
            conv3_bias=bias_tensor(widths[2], device=self.device),
            fc1_weight=kaiming_tensor(hidden_dim, widths[2] * 4 * 4, device=self.device),
            fc1_bias=bias_tensor(hidden_dim, device=self.device),
            fc2_weight=kaiming_tensor(num_classes, hidden_dim, device=self.device),
            fc2_bias=bias_tensor(num_classes, device=self.device),
        )

    def to(self, device: torch.device | str):
        device = torch.device(device)
        self.device = device
        for name, value in self.params.items():
            self.params[name] = value.detach().to(device).requires_grad_(True)
        return self

    def parameters(self):
        return list(self.params.values())

    def named_parameters(self):
        return list(self.params.items())

    def train(self):
        return self

    def eval(self):
        return self

    def _activate(self, x: torch.Tensor) -> torch.Tensor:
        if self.activation == "relu":
            return manual_relu(x)
        if self.activation == "leaky_relu":
            return manual_leaky_relu(x)
        raise ValueError(f"Unsupported manual activation: {self.activation}")

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        p = self.params
        x = manual_conv2d(x, p["conv1_weight"], p["conv1_bias"], padding=1)
        x = self._activate(x)
        x = manual_max_pool2d(x, 2, 2)
        x = manual_conv2d(x, p["conv2_weight"], p["conv2_bias"], padding=1)
        x = self._activate(x)
        x = manual_max_pool2d(x, 2, 2)
        x = manual_conv2d(x, p["conv3_weight"], p["conv3_bias"], padding=1)
        x = self._activate(x)
        x = manual_max_pool2d(x, 2, 2)
        x = x.reshape(x.size(0), -1)
        x = manual_linear(x, p["fc1_weight"], p["fc1_bias"])
        x = self._activate(x)
        x = manual_linear(x, p["fc2_weight"], p["fc2_bias"])
        return x
