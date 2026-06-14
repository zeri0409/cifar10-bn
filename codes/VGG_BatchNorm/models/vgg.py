"""
Model definitions used throughout Project 2.
"""

from __future__ import annotations

import numpy as np
import torch
from torch import nn

from utils.nn import init_weights_


def get_number_of_parameters(model: nn.Module) -> int:
    return int(sum(np.prod(parameter.shape).item() for parameter in model.parameters()))


def _get_activation(name: str) -> nn.Module:
    activations = {
        "relu": nn.ReLU(inplace=True),
        "leaky_relu": nn.LeakyReLU(negative_slope=0.1, inplace=True),
        "elu": nn.ELU(inplace=True),
        "gelu": nn.GELU(),
        "silu": nn.SiLU(inplace=True),
    }
    if name not in activations:
        raise ValueError(f"Unsupported activation: {name}")
    return activations[name]


class ConvBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        activation: str = "relu",
        use_bn: bool = False,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        layers: list[nn.Module] = [
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=not use_bn),
        ]
        if use_bn:
            layers.append(nn.BatchNorm2d(out_channels))
        layers.append(_get_activation(activation))
        if dropout > 0.0:
            layers.append(nn.Dropout2d(dropout))
        self.block = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class ResidualBlock(nn.Module):
    def __init__(self, channels: int, activation: str = "relu", use_bn: bool = True) -> None:
        super().__init__()
        self.conv1 = ConvBlock(channels, channels, activation=activation, use_bn=use_bn)
        self.conv2 = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=not use_bn),
            nn.BatchNorm2d(channels) if use_bn else nn.Identity(),
        )
        self.activation = _get_activation(activation)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        out = self.conv1(x)
        out = self.conv2(out)
        return self.activation(out + residual)


class SEBlock(nn.Module):
    def __init__(self, channels: int, reduction: int = 16) -> None:
        super().__init__()
        hidden = max(channels // reduction, 4)
        self.layers = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(channels, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, channels),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        weights = self.layers(x).view(x.size(0), x.size(1), 1, 1)
        return x * weights


class VGG_A(nn.Module):
    """
    VGG-A adapted to CIFAR-10.
    """

    def __init__(
        self,
        inp_ch: int = 3,
        num_classes: int = 10,
        init_weights: bool = True,
        activation: str = "relu",
        base_width: int = 64,
        use_bn: bool = False,
        dropout: float = 0.0,
        use_residual: bool = False,
        use_se: bool = False,
    ) -> None:
        super().__init__()
        widths = [
            base_width,
            base_width * 2,
            base_width * 4,
            base_width * 8,
            base_width * 8,
        ]

        features: list[nn.Module] = []
        in_channels = inp_ch
        stage_depths = [1, 1, 2, 2, 2]
        for stage_index, (width, depth) in enumerate(zip(widths, stage_depths)):
            for block_index in range(depth):
                features.append(
                    ConvBlock(
                        in_channels=in_channels,
                        out_channels=width,
                        activation=activation,
                        use_bn=use_bn,
                        dropout=dropout if stage_index >= 2 else 0.0,
                    )
                )
                in_channels = width
                if use_residual and depth > 1 and block_index == depth - 1:
                    features.append(ResidualBlock(width, activation=activation, use_bn=use_bn))
            if use_se:
                features.append(SEBlock(width))
            features.append(nn.MaxPool2d(kernel_size=2, stride=2))

        self.features = nn.Sequential(*features)
        classifier_layers: list[nn.Module] = []
        if dropout > 0.0:
            classifier_layers.append(nn.Dropout(dropout))
        classifier_layers.extend([
            nn.Linear(widths[-1], 512),
            _get_activation(activation),
        ])
        if dropout > 0.0:
            classifier_layers.append(nn.Dropout(dropout))
        classifier_layers.extend([
            nn.Linear(512, 512),
            _get_activation(activation),
            nn.Linear(512, num_classes),
        ])
        self.classifier = nn.Sequential(*classifier_layers)

        if init_weights:
            self._init_weights()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = x.view(x.size(0), -1)
        return self.classifier(x)

    def _init_weights(self) -> None:
        for module in self.modules():
            init_weights_(module)


class VGG_A_BatchNorm(VGG_A):
    def __init__(self, **kwargs) -> None:
        kwargs.setdefault("use_bn", True)
        super().__init__(**kwargs)


class VGG_A_Dropout(VGG_A):
    def __init__(self, dropout: float = 0.3, **kwargs) -> None:
        kwargs.setdefault("dropout", dropout)
        super().__init__(**kwargs)


class VGG_A_Residual(VGG_A):
    def __init__(self, **kwargs) -> None:
        kwargs.setdefault("use_residual", True)
        kwargs.setdefault("use_bn", True)
        super().__init__(**kwargs)


class VGG_A_SE(VGG_A):
    def __init__(self, **kwargs) -> None:
        kwargs.setdefault("use_se", True)
        kwargs.setdefault("use_bn", True)
        super().__init__(**kwargs)


class SmallCifarCNN(nn.Module):
    def __init__(
        self,
        num_classes: int = 10,
        widths: tuple[int, int, int] = (32, 64, 128),
        activation: str = "relu",
        use_bn: bool = True,
        dropout: float = 0.25,
        use_residual: bool = False,
    ) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        in_channels = 3
        for index, width in enumerate(widths):
            layers.append(ConvBlock(in_channels, width, activation=activation, use_bn=use_bn, dropout=dropout))
            if use_residual and index > 0:
                layers.append(ResidualBlock(width, activation=activation, use_bn=use_bn))
            layers.append(nn.MaxPool2d(2))
            in_channels = width
        self.features = nn.Sequential(*layers)
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(widths[-1], 256),
            _get_activation(activation),
            nn.Dropout(dropout),
            nn.Linear(256, num_classes),
        )
        self.apply(init_weights_)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.features(x))


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_channels: int, out_channels: int, stride: int = 1, activation: str = "relu", dropout: float = 0.0) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.act = _get_activation(activation)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.dropout = nn.Dropout2d(dropout) if dropout > 0.0 else nn.Identity()
        self.shortcut = nn.Identity()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.act(self.bn1(self.conv1(x)))
        out = self.dropout(out)
        out = self.bn2(self.conv2(out))
        out = out + self.shortcut(x)
        return self.act(out)


class SmallResNet(nn.Module):
    def __init__(
        self,
        num_classes: int = 10,
        base_width: int = 64,
        activation: str = "relu",
        dropout: float = 0.0,
        blocks_per_stage: tuple[int, int, int] = (2, 2, 2),
    ) -> None:
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(3, base_width, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(base_width),
            _get_activation(activation),
        )
        widths = [base_width, base_width * 2, base_width * 4]
        in_channels = base_width
        stages: list[nn.Module] = []
        for stage_index, (width, num_blocks) in enumerate(zip(widths, blocks_per_stage)):
            stride = 1 if stage_index == 0 else 2
            blocks = [BasicBlock(in_channels, width, stride=stride, activation=activation, dropout=dropout)]
            in_channels = width
            for _ in range(1, num_blocks):
                blocks.append(BasicBlock(in_channels, width, stride=1, activation=activation, dropout=dropout))
            stages.append(nn.Sequential(*blocks))
        self.stages = nn.Sequential(*stages)
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Dropout(dropout) if dropout > 0.0 else nn.Identity(),
            nn.Linear(widths[-1], num_classes),
        )
        self.apply(init_weights_)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        x = self.stages(x)
        return self.head(x)


if __name__ == "__main__":
    models = {
        "VGG_A": VGG_A(),
        "VGG_A_BatchNorm": VGG_A_BatchNorm(),
        "VGG_A_Dropout": VGG_A_Dropout(),
        "VGG_A_Residual": VGG_A_Residual(),
        "VGG_A_SE": VGG_A_SE(),
        "SmallCifarCNN": SmallCifarCNN(),
        "SmallResNet": SmallResNet(),
    }
    for name, model in models.items():
        print(name, get_number_of_parameters(model))
