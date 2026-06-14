"""
End-to-end experiment runner for the course project.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch

from data.loaders import get_cifar_loader, get_train_val_loaders, save_sample_grid
from models.manual_network import ManualCifarNet
from models.vgg import (
    SmallResNet,
    SmallCifarCNN,
    VGG_A,
    VGG_A_BatchNorm,
    VGG_A_Dropout,
    VGG_A_Residual,
    VGG_A_SE,
    get_number_of_parameters,
)
from utils.losses import build_loss
from utils.optimizers import ManualAdam, ManualSGD
from utils.trainer import (
    analyze_prediction_error,
    analyze_gradient_predictiveness,
    evaluate,
    fit,
    get_device,
    load_history,
    load_model_state,
    save_feature_maps,
    save_confusion_matrix,
    save_filter_visualization,
    save_gradient_analysis_plot,
    save_loss_landscape_plot,
    save_misclassified_examples,
    save_per_class_accuracy,
    save_saliency_map,
    save_scatter_plot,
    set_random_seeds,
    compute_loss_landscape_curves,
)


@dataclass
class ExperimentConfig:
    name: str
    epochs: int
    lr: float
    loss_name: str = "cross_entropy"
    label_smoothing: float = 0.0
    weight_decay: float = 0.0
    optimizer_name: str = "adam"
    model_kind: str = "small_cnn"
    activation: str = "relu"
    width_scale: int = 32
    scheduler_name: str = "none"
    scheduler_kwargs: dict = field(default_factory=dict)
    augmentation: str = "basic"
    batch_aug: str = "none"
    optimizer_kwargs: dict = field(default_factory=dict)
    loss_kwargs: dict = field(default_factory=dict)
    model_kwargs: dict = field(default_factory=dict)


def build_model(model_kind: str, activation: str = "relu", width_scale: int = 32, **model_kwargs):
    if model_kind == "small_cnn":
        widths = model_kwargs.pop("widths", (width_scale, width_scale * 2, width_scale * 4))
        return SmallCifarCNN(widths=widths, activation=activation, **model_kwargs)
    if model_kind == "small_resnet":
        model_kwargs.pop("widths", None)
        blocks = model_kwargs.pop("blocks", model_kwargs.pop("blocks_per_stage", (2, 2, 2)))
        dropout = model_kwargs.pop("dropout", 0.0)
        return SmallResNet(base_width=width_scale, blocks_per_stage=blocks, activation=activation, dropout=dropout, **model_kwargs)
    if model_kind == "vgg_a":
        return VGG_A(activation=activation, base_width=max(width_scale, 32), **model_kwargs)
    if model_kind == "vgg_bn":
        return VGG_A_BatchNorm(activation=activation, base_width=max(width_scale, 32), **model_kwargs)
    if model_kind == "vgg_dropout":
        dropout = model_kwargs.pop("dropout", 0.3)
        return VGG_A_Dropout(activation=activation, base_width=max(width_scale, 32), dropout=dropout, **model_kwargs)
    if model_kind == "vgg_residual":
        return VGG_A_Residual(activation=activation, base_width=max(width_scale, 32), **model_kwargs)
    if model_kind == "vgg_other":
        return VGG_A_SE(activation=activation, base_width=max(width_scale, 32), **model_kwargs)
    raise ValueError(f"Unknown model kind: {model_kind}")


def build_optimizer(name: str, params, lr: float, weight_decay: float, **optimizer_kwargs):
    normalized_name = name.lower()
    if normalized_name == "sgd":
        defaults = {"momentum": 0.9, "nesterov": False}
        defaults.update(optimizer_kwargs)
        return torch.optim.SGD(params, lr=lr, weight_decay=weight_decay, **defaults)
    if normalized_name == "adam":
        return torch.optim.Adam(params, lr=lr, weight_decay=weight_decay, **optimizer_kwargs)
    if normalized_name == "adamw":
        return torch.optim.AdamW(params, lr=lr, weight_decay=weight_decay, **optimizer_kwargs)
    if normalized_name == "rmsprop":
        defaults = {"momentum": 0.9}
        defaults.update(optimizer_kwargs)
        return torch.optim.RMSprop(params, lr=lr, weight_decay=weight_decay, **defaults)
    raise ValueError(f"Unknown optimizer: {name}")


def build_scheduler(name: str, optimizer, epochs: int, **scheduler_kwargs):
    normalized_name = name.lower()
    if normalized_name == "none":
        return None
    if normalized_name == "step":
        defaults = {"step_size": max(1, epochs // 3), "gamma": 0.2}
        defaults.update(scheduler_kwargs)
        return torch.optim.lr_scheduler.StepLR(optimizer, **defaults)
    if normalized_name == "cosine":
        defaults = {"T_max": max(1, epochs), "eta_min": 1e-6}
        defaults.update(scheduler_kwargs)
        return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, **defaults)
    raise ValueError(f"Unknown scheduler: {name}")


def _one_hot(targets: torch.Tensor, num_classes: int = 10) -> torch.Tensor:
    return torch.nn.functional.one_hot(targets, num_classes=num_classes).float()


def make_batch_transform(name: str):
    normalized_name = name.lower()
    if normalized_name == "none":
        return None

    def mixup_transform(inputs: torch.Tensor, targets: torch.Tensor, alpha: float) -> tuple[torch.Tensor, torch.Tensor]:
        lam = float(np.random.beta(alpha, alpha)) if alpha > 0 else 1.0
        index = torch.randperm(inputs.size(0), device=inputs.device)
        mixed_inputs = lam * inputs + (1.0 - lam) * inputs[index]
        mixed_targets = lam * _one_hot(targets) + (1.0 - lam) * _one_hot(targets[index])
        return mixed_inputs, mixed_targets

    def cutmix_transform(inputs: torch.Tensor, targets: torch.Tensor, alpha: float) -> tuple[torch.Tensor, torch.Tensor]:
        lam = float(np.random.beta(alpha, alpha)) if alpha > 0 else 1.0
        index = torch.randperm(inputs.size(0), device=inputs.device)
        _, _, height, width = inputs.shape
        cut_ratio = np.sqrt(1.0 - lam)
        cut_w = int(width * cut_ratio)
        cut_h = int(height * cut_ratio)
        cx = np.random.randint(width)
        cy = np.random.randint(height)
        x1 = np.clip(cx - cut_w // 2, 0, width)
        x2 = np.clip(cx + cut_w // 2, 0, width)
        y1 = np.clip(cy - cut_h // 2, 0, height)
        y2 = np.clip(cy + cut_h // 2, 0, height)
        mixed_inputs = inputs.clone()
        mixed_inputs[:, :, y1:y2, x1:x2] = inputs[index, :, y1:y2, x1:x2]
        area = max(1, (x2 - x1) * (y2 - y1))
        lam_adjusted = 1.0 - area / float(width * height)
        mixed_targets = lam_adjusted * _one_hot(targets) + (1.0 - lam_adjusted) * _one_hot(targets[index])
        return mixed_inputs, mixed_targets

    if normalized_name == "mixup":
        return lambda inputs, targets: mixup_transform(inputs, targets, alpha=0.2)
    if normalized_name == "mixup_strong":
        return lambda inputs, targets: mixup_transform(inputs, targets, alpha=1.0)
    if normalized_name == "cutmix":
        return lambda inputs, targets: cutmix_transform(inputs, targets, alpha=1.0)
    raise ValueError(f"Unknown batch augmentation: {name}")


def scaled_epochs(base_epochs: int, epoch_scale: float) -> int:
    return max(1, int(round(base_epochs * epoch_scale)))


def load_summary(summary_path: Path) -> dict:
    with summary_path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def run_single_experiment(
    config: ExperimentConfig,
    train_loader,
    val_loader,
    test_loader,
    output_root: Path,
    device: torch.device,
    resume: bool = False,
    skip_completed: bool = False,
):
    experiment_dir = output_root / config.name
    summary_path = experiment_dir / "summary.json"
    history_path = experiment_dir / "history.json"
    model_path = experiment_dir / "best_model.pt"

    model = build_model(
        config.model_kind,
        activation=config.activation,
        width_scale=config.width_scale,
        **config.model_kwargs,
    )
    if skip_completed and summary_path.exists() and history_path.exists() and model_path.exists():
        print(f"Skipping completed experiment: {config.name}")
        load_model_state(model, model_path, device)
        return model, load_history(history_path), load_summary(summary_path)

    criterion = build_loss(config.loss_name, label_smoothing=config.label_smoothing, **config.loss_kwargs)
    optimizer = build_optimizer(
        config.optimizer_name,
        model.parameters(),
        lr=config.lr,
        weight_decay=config.weight_decay,
        **config.optimizer_kwargs,
    )
    scheduler = build_scheduler(config.scheduler_name, optimizer, config.epochs, **config.scheduler_kwargs)
    history = fit(
        model,
        train_loader,
        val_loader,
        criterion,
        optimizer,
        device=device,
        epochs=config.epochs,
        output_dir=experiment_dir,
        scheduler=scheduler,
        train_batch_transform=make_batch_transform(config.batch_aug),
        resume=resume,
    )
    test_stats = evaluate(model, test_loader, criterion, device=device)
    summary = {
        "name": config.name,
        "model_kind": config.model_kind,
        "activation": config.activation,
        "width_scale": config.width_scale,
        "optimizer": config.optimizer_name,
        "loss_name": config.loss_name,
        "lr": config.lr,
        "weight_decay": config.weight_decay,
        "label_smoothing": config.label_smoothing,
        "scheduler": config.scheduler_name,
        "augmentation": config.augmentation,
        "batch_aug": config.batch_aug,
        "parameters": get_number_of_parameters(model),
        "best_val_accuracy": max(history.val_accuracy),
        "test_accuracy": test_stats.accuracy,
        "test_error": 1.0 - test_stats.accuracy,
        "average_epoch_time": float(np.mean(history.epoch_times)) if history.epoch_times else 0.0,
        "total_train_time": float(np.sum(history.epoch_times)) if history.epoch_times else 0.0,
        "checkpoint_path": str(model_path),
        "experiment_dir": str(experiment_dir),
    }
    with summary_path.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
    return model, history, summary


def run_architecture_ablation(train_loader, val_loader, test_loader, output_root: Path, device: torch.device, epoch_scale: float, resume: bool, skip_completed: bool):
    variants = [
        ExperimentConfig(name="tinycnn", epochs=scaled_epochs(20, epoch_scale), lr=1e-3, model_kind="small_cnn", width_scale=16),
        ExperimentConfig(name="basiccnn", epochs=scaled_epochs(20, epoch_scale), lr=1e-3, model_kind="small_cnn", width_scale=32),
        ExperimentConfig(name="cnn_bn", epochs=scaled_epochs(20, epoch_scale), lr=1e-3, model_kind="vgg_bn", width_scale=32),
        ExperimentConfig(name="cnn_bn_dropout", epochs=scaled_epochs(20, epoch_scale), lr=1e-3, model_kind="vgg_dropout", width_scale=32, model_kwargs={"dropout": 0.3}),
        ExperimentConfig(name="smallresnet", epochs=scaled_epochs(24, epoch_scale), lr=1e-3, model_kind="small_resnet", width_scale=32),
        ExperimentConfig(name="widesmallresnet", epochs=scaled_epochs(24, epoch_scale), lr=1e-3, model_kind="small_resnet", width_scale=48, model_kwargs={"dropout": 0.1}),
        ExperimentConfig(name="baseline_small_cnn", epochs=scaled_epochs(20, epoch_scale), lr=1e-3, model_kind="small_cnn"),
        ExperimentConfig(name="vgg_a", epochs=scaled_epochs(20, epoch_scale), lr=1e-3, model_kind="vgg_a"),
        ExperimentConfig(name="vgg_bn", epochs=scaled_epochs(20, epoch_scale), lr=1e-3, model_kind="vgg_bn"),
        ExperimentConfig(name="vgg_dropout", epochs=scaled_epochs(20, epoch_scale), lr=1e-3, model_kind="vgg_dropout"),
        ExperimentConfig(name="vgg_residual", epochs=scaled_epochs(20, epoch_scale), lr=1e-3, model_kind="vgg_residual"),
        ExperimentConfig(name="vgg_other_se", epochs=scaled_epochs(20, epoch_scale), lr=1e-3, model_kind="vgg_other"),
    ]
    results = []
    for config in variants:
        _, _, summary = run_single_experiment(config, train_loader, val_loader, test_loader, output_root, device, resume=resume, skip_completed=skip_completed)
        results.append(summary)
    return results


def run_filter_activation_loss_ablations(train_loader, val_loader, test_loader, output_root: Path, device: torch.device, epoch_scale: float, resume: bool, skip_completed: bool):
    results = []
    for width_scale in [16, 24, 32, 48, 64]:
        config = ExperimentConfig(
            name=f"filters_{width_scale}",
            epochs=scaled_epochs(15, epoch_scale),
            lr=1e-3,
            model_kind="small_cnn",
            width_scale=width_scale,
        )
        _, _, summary = run_single_experiment(config, train_loader, val_loader, test_loader, output_root, device, resume=resume, skip_completed=skip_completed)
        results.append(summary)

    for activation in ["relu", "leaky_relu", "elu", "gelu", "silu"]:
        config = ExperimentConfig(
            name=f"activation_{activation}",
            epochs=scaled_epochs(15, epoch_scale),
            lr=1e-3,
            model_kind="small_cnn",
            activation=activation,
        )
        _, _, summary = run_single_experiment(config, train_loader, val_loader, test_loader, output_root, device, resume=resume, skip_completed=skip_completed)
        results.append(summary)

    loss_settings = [
        ("loss_ce_wd0", "cross_entropy", 0.0, 0.0, {}),
        ("loss_ce_wd5e4", "cross_entropy", 0.0, 5e-4, {}),
        ("loss_ce_wd1e3", "cross_entropy", 0.0, 1e-3, {}),
        ("loss_ls_0p05_wd5e4", "label_smoothing", 0.05, 5e-4, {}),
        ("loss_ls_0p10_wd5e4", "label_smoothing", 0.10, 5e-4, {}),
        ("loss_focal_g1", "focal", 0.0, 1e-4, {"gamma": 1.0}),
        ("loss_focal_g2", "focal", 0.05, 1e-4, {"gamma": 2.0}),
    ]
    for config_name, loss_name, smoothing, weight_decay, loss_kwargs in loss_settings:
        config = ExperimentConfig(
            name=config_name,
            epochs=scaled_epochs(15, epoch_scale),
            lr=1e-3,
            model_kind="small_cnn",
            loss_name=loss_name,
            label_smoothing=smoothing,
            weight_decay=weight_decay,
            loss_kwargs=loss_kwargs,
        )
        _, _, summary = run_single_experiment(config, train_loader, val_loader, test_loader, output_root, device, resume=resume, skip_completed=skip_completed)
        results.append(summary)
    return results


def run_optimizer_ablation(train_loader, val_loader, test_loader, output_root: Path, device: torch.device, epoch_scale: float, resume: bool, skip_completed: bool):
    results = []
    optimizer_settings = [
        ExperimentConfig(
            name="optimizer_sgd_lr0p05",
            epochs=scaled_epochs(15, epoch_scale),
            lr=5e-2,
            model_kind="small_cnn",
            optimizer_name="sgd",
            weight_decay=5e-4,
            optimizer_kwargs={"momentum": 0.9, "nesterov": False},
        ),
        ExperimentConfig(
            name="optimizer_sgd_nesterov",
            epochs=scaled_epochs(15, epoch_scale),
            lr=0.05,
            model_kind="small_cnn",
            optimizer_name="sgd",
            weight_decay=5e-4,
            optimizer_kwargs={"momentum": 0.9, "nesterov": True},
        ),
        ExperimentConfig(
            name="optimizer_adam_lr1e3",
            epochs=scaled_epochs(15, epoch_scale),
            lr=1e-3,
            model_kind="small_cnn",
            optimizer_name="adam",
            weight_decay=5e-4,
        ),
        ExperimentConfig(
            name="optimizer_adam_lr5e4",
            epochs=scaled_epochs(15, epoch_scale),
            lr=5e-4,
            model_kind="small_cnn",
            optimizer_name="adam",
            weight_decay=5e-4,
        ),
        ExperimentConfig(
            name="optimizer_adamw_lr1e3",
            epochs=scaled_epochs(15, epoch_scale),
            lr=1e-3,
            model_kind="small_cnn",
            optimizer_name="adamw",
            weight_decay=5e-4,
        ),
        ExperimentConfig(
            name="optimizer_adamw_lr3e4",
            epochs=scaled_epochs(15, epoch_scale),
            lr=3e-4,
            model_kind="small_cnn",
            optimizer_name="adamw",
            weight_decay=1e-3,
        ),
        ExperimentConfig(
            name="optimizer_rmsprop_lr1e3",
            epochs=scaled_epochs(15, epoch_scale),
            lr=1e-3,
            model_kind="small_cnn",
            optimizer_name="rmsprop",
            weight_decay=5e-4,
        ),
    ]
    for config in optimizer_settings:
        _, _, summary = run_single_experiment(config, train_loader, val_loader, test_loader, output_root, device, resume=resume, skip_completed=skip_completed)
        results.append(summary)
    return results


def run_regularization_ablation(data_root: str, batch_size: int, num_workers: int, train_items: int, val_items: int, test_items: int, seed: int, output_root: Path, device: torch.device, epoch_scale: float, resume: bool, skip_completed: bool):
    results = []

    dropout_settings = [0.0, 0.1, 0.3, 0.5]
    for dropout in dropout_settings:
        train_loader, val_loader = get_train_val_loaders(
            root=data_root,
            batch_size=batch_size,
            num_workers=num_workers,
            train_items=train_items,
            val_items=val_items,
            seed=seed,
            augmentation="basic",
        )
        test_loader = get_cifar_loader(root=data_root, batch_size=batch_size, train=False, shuffle=False, num_workers=num_workers, n_items=test_items, use_augmentation=False)
        config = ExperimentConfig(
            name=f"dropout_{str(dropout).replace('.', 'p')}",
            epochs=scaled_epochs(20, epoch_scale),
            lr=1e-3,
            model_kind="small_resnet",
            width_scale=32,
            model_kwargs={"dropout": dropout},
        )
        _, _, summary = run_single_experiment(config, train_loader, val_loader, test_loader, output_root, device, resume=resume, skip_completed=skip_completed)
        results.append(summary)

    for augmentation in ["basic", "colorjitter", "random_erasing", "autoaugment", "randaugment"]:
        train_loader, val_loader = get_train_val_loaders(
            root=data_root,
            batch_size=batch_size,
            num_workers=num_workers,
            train_items=train_items,
            val_items=val_items,
            seed=seed,
            augmentation=augmentation,
        )
        test_loader = get_cifar_loader(root=data_root, batch_size=batch_size, train=False, shuffle=False, num_workers=num_workers, n_items=test_items, use_augmentation=False)
        config = ExperimentConfig(
            name=f"augmentation_{augmentation}",
            epochs=scaled_epochs(20, epoch_scale),
            lr=1e-3,
            model_kind="small_resnet",
            width_scale=32,
            augmentation=augmentation,
        )
        _, _, summary = run_single_experiment(config, train_loader, val_loader, test_loader, output_root, device, resume=resume, skip_completed=skip_completed)
        results.append(summary)

    batch_aug_settings = [
        ("batchaug_mixup_a0p2", "mixup", "basic"),
        ("batchaug_mixup_a1p0", "mixup_strong", "basic"),
        ("batchaug_cutmix_a1p0", "cutmix", "basic"),
    ]
    for name, batch_aug, augmentation in batch_aug_settings:
        train_loader, val_loader = get_train_val_loaders(
            root=data_root,
            batch_size=batch_size,
            num_workers=num_workers,
            train_items=train_items,
            val_items=val_items,
            seed=seed,
            augmentation=augmentation,
        )
        test_loader = get_cifar_loader(root=data_root, batch_size=batch_size, train=False, shuffle=False, num_workers=num_workers, n_items=test_items, use_augmentation=False)
        config = ExperimentConfig(
            name=name,
            epochs=scaled_epochs(22, epoch_scale),
            lr=1e-3,
            model_kind="small_resnet",
            width_scale=32,
            loss_name="label_smoothing",
            label_smoothing=0.1,
            batch_aug=batch_aug,
            augmentation=augmentation,
        )
        _, _, summary = run_single_experiment(config, train_loader, val_loader, test_loader, output_root, device, resume=resume, skip_completed=skip_completed)
        results.append(summary)

    return results


def run_scheduler_ablation(train_loader, val_loader, test_loader, output_root: Path, device: torch.device, epoch_scale: float, resume: bool, skip_completed: bool):
    results = []
    configs = [
        ExperimentConfig(name="scheduler_step_lr0p01", epochs=scaled_epochs(20, epoch_scale), lr=0.01, model_kind="small_resnet", optimizer_name="sgd", weight_decay=5e-4, scheduler_name="step"),
        ExperimentConfig(name="scheduler_cosine_lr0p01", epochs=scaled_epochs(20, epoch_scale), lr=0.01, model_kind="small_resnet", optimizer_name="sgd", weight_decay=5e-4, scheduler_name="cosine"),
        ExperimentConfig(name="scheduler_cosine_lr0p05", epochs=scaled_epochs(20, epoch_scale), lr=0.05, model_kind="small_resnet", optimizer_name="sgd", weight_decay=5e-4, scheduler_name="cosine"),
        ExperimentConfig(name="scheduler_cosine_lr0p1", epochs=scaled_epochs(20, epoch_scale), lr=0.1, model_kind="small_resnet", optimizer_name="sgd", weight_decay=5e-4, scheduler_name="cosine"),
        ExperimentConfig(name="scheduler_cosine_lr0p2", epochs=scaled_epochs(20, epoch_scale), lr=0.2, model_kind="small_resnet", optimizer_name="sgd", weight_decay=5e-4, scheduler_name="cosine"),
        ExperimentConfig(name="scheduler_adamw_cosine", epochs=scaled_epochs(20, epoch_scale), lr=3e-4, model_kind="small_resnet", optimizer_name="adamw", weight_decay=1e-3, scheduler_name="cosine"),
    ]
    for config in configs:
        _, _, summary = run_single_experiment(config, train_loader, val_loader, test_loader, output_root, device, resume=resume, skip_completed=skip_completed)
        results.append(summary)
    return results


def run_manual_network_with_torch_optim(train_loader, val_loader, test_loader, output_root: Path, device: torch.device, epoch_scale: float, resume: bool, skip_completed: bool):
    results = []
    manual_settings = [
        ("manual_network_torch_optim", {"widths": (32, 64, 128), "hidden_dim": 256, "activation": "relu"}, 1e-3, 20),
        ("manual_network_torch_optim_large", {"widths": (48, 96, 192), "hidden_dim": 384, "activation": "relu"}, 7e-4, 20),
        ("manual_network_torch_optim_leaky", {"widths": (32, 64, 128), "hidden_dim": 256, "activation": "leaky_relu"}, 1e-3, 20),
    ]

    for experiment_name, model_kwargs, lr, base_epochs in manual_settings:
        experiment_dir = output_root / experiment_name
        summary_path = experiment_dir / "summary.json"
        if skip_completed and summary_path.exists():
            print(f"Skipping completed experiment: {experiment_name}")
            results.append(load_summary(summary_path))
            continue

        model = ManualCifarNet(device=device, **model_kwargs)
        criterion = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        history = fit(
            model,
            train_loader,
            val_loader,
            criterion,
            optimizer,
            device=device,
            epochs=scaled_epochs(base_epochs, epoch_scale),
            output_dir=experiment_dir,
            resume=resume,
        )
        test_stats = evaluate(model, test_loader, criterion, device=device)
        summary = {
            "name": experiment_name,
            "model_kind": "manual_cifar",
            "activation": model_kwargs["activation"],
            "width_scale": model_kwargs["widths"][0],
            "optimizer": "adam",
            "loss_name": "cross_entropy",
            "lr": lr,
            "weight_decay": 0.0,
            "label_smoothing": 0.0,
            "parameters": sum(parameter.numel() for parameter in model.parameters()),
            "test_accuracy": test_stats.accuracy,
            "test_error": 1.0 - test_stats.accuracy,
            "best_val_accuracy": max(history.val_accuracy),
            "average_epoch_time": float(np.mean(history.epoch_times)) if history.epoch_times else 0.0,
            "total_train_time": float(np.sum(history.epoch_times)) if history.epoch_times else 0.0,
            "checkpoint_path": str(experiment_dir / "best_model.pt"),
            "experiment_dir": str(experiment_dir),
        }
        with summary_path.open("w", encoding="utf-8") as handle:
            json.dump(summary, handle, indent=2)
        results.append(summary)
    return results


def run_manual_full_optimizer(train_loader, val_loader, test_loader, output_root: Path, device: torch.device, epoch_scale: float, resume: bool, skip_completed: bool):
    results = []
    for optimizer_name, optimizer_cls, lr, optimizer_kwargs in [
        ("manual_sgd", ManualSGD, 2e-2, {"momentum": 0.9}),
        ("manual_sgd_fast", ManualSGD, 5e-2, {"momentum": 0.9}),
        ("manual_adam", ManualAdam, 1e-3, {}),
        ("manual_adam_lr5e4", ManualAdam, 5e-4, {}),
    ]:
        experiment_dir = output_root / optimizer_name
        summary_path = experiment_dir / "summary.json"
        if skip_completed and summary_path.exists():
            print(f"Skipping completed experiment: {optimizer_name}")
            results.append(load_summary(summary_path))
            continue

        model = ManualCifarNet(device=device, widths=(32, 64, 128), hidden_dim=256, activation="relu")
        criterion = torch.nn.CrossEntropyLoss()
        optimizer = optimizer_cls(model.parameters(), lr=lr, weight_decay=1e-4, **optimizer_kwargs)
        history = fit(model, train_loader, val_loader, criterion, optimizer, device=device, epochs=scaled_epochs(20, epoch_scale), output_dir=experiment_dir, resume=resume)
        test_stats = evaluate(model, test_loader, criterion, device=device)
        summary = {
            "name": optimizer_name,
            "model_kind": "manual_cifar",
            "activation": "relu",
            "width_scale": 32,
            "optimizer": optimizer_name,
            "loss_name": "cross_entropy",
            "lr": lr,
            "weight_decay": 1e-4,
            "label_smoothing": 0.0,
            "parameters": sum(parameter.numel() for parameter in model.parameters()),
            "test_accuracy": test_stats.accuracy,
            "test_error": 1.0 - test_stats.accuracy,
            "best_val_accuracy": max(history.val_accuracy),
            "average_epoch_time": float(np.mean(history.epoch_times)) if history.epoch_times else 0.0,
            "total_train_time": float(np.sum(history.epoch_times)) if history.epoch_times else 0.0,
            "checkpoint_path": str(experiment_dir / "best_model.pt"),
            "experiment_dir": str(experiment_dir),
        }
        with summary_path.open("w", encoding="utf-8") as handle:
            json.dump(summary, handle, indent=2)
        results.append(summary)
    return results


def run_high_performance_search(train_loader, val_loader, test_loader, output_root: Path, device: torch.device, epoch_scale: float, resume: bool, skip_completed: bool):
    configs = [
        ExperimentConfig(
            name="vgg_bn_tuned_adamw",
            epochs=scaled_epochs(30, epoch_scale),
            lr=3e-4,
            weight_decay=5e-4,
            optimizer_name="adamw",
            model_kind="vgg_bn",
            label_smoothing=0.05,
            loss_name="label_smoothing",
            width_scale=48,
        ),
        ExperimentConfig(
            name="vgg_bn_tuned_adam",
            epochs=scaled_epochs(30, epoch_scale),
            lr=8e-4,
            weight_decay=5e-4,
            optimizer_name="adam",
            model_kind="vgg_bn",
            width_scale=48,
        ),
        ExperimentConfig(
            name="vgg_bn_wide64",
            epochs=scaled_epochs(30, epoch_scale),
            lr=5e-4,
            weight_decay=5e-4,
            optimizer_name="adamw",
            model_kind="vgg_bn",
            loss_name="label_smoothing",
            label_smoothing=0.05,
            width_scale=64,
        ),
        ExperimentConfig(
            name="vgg_other_se_tuned",
            epochs=scaled_epochs(30, epoch_scale),
            lr=3e-4,
            weight_decay=5e-4,
            optimizer_name="adamw",
            model_kind="vgg_other",
            loss_name="label_smoothing",
            label_smoothing=0.05,
            width_scale=48,
        ),
    ]
    results = []
    for config in configs:
        _, _, summary = run_single_experiment(config, train_loader, val_loader, test_loader, output_root, device, resume=resume, skip_completed=skip_completed)
        results.append(summary)
    return results


def run_bn_analysis(train_loader, val_loader, test_loader, output_root: Path, device: torch.device, epoch_scale: float, resume: bool, skip_completed: bool):
    learning_rates = [1e-4, 5e-4, 1e-3, 2e-3]
    baseline_losses = {}
    bn_losses = {}
    baseline_stats = None
    bn_stats = None

    for lr in learning_rates:
        for model_kind, target in [("vgg_a", baseline_losses), ("vgg_bn", bn_losses)]:
            config = ExperimentConfig(
                name=f"{model_kind}_lr_{lr:.0e}",
                epochs=scaled_epochs(8, epoch_scale),
                lr=lr,
                model_kind=model_kind,
            )
            model, history, _ = run_single_experiment(config, train_loader, val_loader, test_loader, output_root / "bn_landscape", device, resume=resume, skip_completed=skip_completed)
            target[f"{lr:.0e}"] = history.step_losses
            if lr == learning_rates[-1]:
                stats = analyze_gradient_predictiveness(model, train_loader, torch.nn.CrossEntropyLoss(), device=device, max_batches=12)
                stats.update(
                    analyze_prediction_error(
                        model,
                        train_loader,
                        torch.nn.CrossEntropyLoss(),
                        device=device,
                        step_sizes=learning_rates,
                        max_batches=4,
                    )
                )
                if model_kind == "vgg_a":
                    baseline_stats = stats
                else:
                    bn_stats = stats

    baseline_curves = compute_loss_landscape_curves(baseline_losses)
    bn_curves = compute_loss_landscape_curves(bn_losses)
    save_loss_landscape_plot(baseline_curves, bn_curves, output_root / "figures" / "loss_landscape_comparison.png")
    if baseline_stats is not None and bn_stats is not None:
        save_gradient_analysis_plot(
            baseline_stats,
            bn_stats,
            output_root / "figures" / "gradient_analysis.png",
        )
    with (output_root / "bn_analysis_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(
            {
                "learning_rates": learning_rates,
                "baseline_curves": baseline_curves,
                "bn_curves": bn_curves,
                "baseline_gradient_stats": baseline_stats,
                "bn_gradient_stats": bn_stats,
            },
            handle,
            indent=2,
        )


def export_ranked_table(results: list[dict], output_path: Path):
    if not results:
        return
    ranked = sorted(results, key=lambda item: item.get("test_accuracy", 0.0), reverse=True)
    header = []
    for row in ranked:
        for key in row.keys():
            if key not in header:
                header.append(key)
    lines = [",".join(header)]
    for row in ranked:
        lines.append(",".join(str(row.get(column, "")) for column in header))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines), encoding="utf-8")


def load_best_result_model(results: list[dict], device: torch.device):
    model_results = [result for result in results if result.get("checkpoint_path") and result.get("model_kind") in {"small_cnn", "small_resnet", "vgg_a", "vgg_bn", "vgg_dropout", "vgg_residual", "vgg_other"}]
    if not model_results:
        return build_model("vgg_bn"), None
    best_result = max(model_results, key=lambda item: item.get("test_accuracy", 0.0))
    model = build_model(
        best_result["model_kind"],
        activation=best_result.get("activation", "relu"),
        width_scale=int(best_result.get("width_scale", 32)),
    )
    load_model_state(model, best_result["checkpoint_path"], device)
    return model, best_result


def main():
    parser = argparse.ArgumentParser(description="Run the full Project 2 experiment suite.")
    parser.add_argument("--output-root", default="outputs", type=str)
    parser.add_argument("--data-root", default="data", type=str)
    parser.add_argument("--batch-size", default=128, type=int)
    parser.add_argument("--num-workers", default=4, type=int)
    parser.add_argument("--train-items", default=-1, type=int)
    parser.add_argument("--val-items", default=-1, type=int)
    parser.add_argument("--test-items", default=-1, type=int)
    parser.add_argument("--seed", default=2026, type=int)
    parser.add_argument("--device", default="auto", type=str)
    parser.add_argument("--epoch-scale", default=1.0, type=float)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--skip-completed", action="store_true")
    args = parser.parse_args()

    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    device = get_device(args.device)
    set_random_seeds(args.seed, device=device)
    print(f"Using device: {device}")

    save_sample_grid(output_root / "figures" / "cifar10_samples.png", root=args.data_root)
    train_loader, val_loader = get_train_val_loaders(
        root=args.data_root,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        train_items=args.train_items,
        val_items=args.val_items,
        seed=args.seed,
        augmentation="basic",
    )
    test_loader = get_cifar_loader(
        root=args.data_root,
        batch_size=args.batch_size,
        train=False,
        shuffle=False,
        num_workers=args.num_workers,
        n_items=args.test_items,
        use_augmentation=False,
    )

    architecture_results = run_architecture_ablation(train_loader, val_loader, test_loader, output_root / "architectures", device, args.epoch_scale, args.resume, args.skip_completed)
    ablation_results = run_filter_activation_loss_ablations(train_loader, val_loader, test_loader, output_root / "ablations", device, args.epoch_scale, args.resume, args.skip_completed)
    regularization_results = run_regularization_ablation(args.data_root, args.batch_size, args.num_workers, args.train_items, args.val_items, args.test_items, args.seed, output_root / "regularization", device, args.epoch_scale, args.resume, args.skip_completed)
    optimizer_results = run_optimizer_ablation(train_loader, val_loader, test_loader, output_root / "optimizers", device, args.epoch_scale, args.resume, args.skip_completed)
    scheduler_results = run_scheduler_ablation(train_loader, val_loader, test_loader, output_root / "schedulers", device, args.epoch_scale, args.resume, args.skip_completed)
    manual_torch_results = run_manual_network_with_torch_optim(train_loader, val_loader, test_loader, output_root / "manual", device, args.epoch_scale, args.resume, args.skip_completed)
    manual_optimizer_results = run_manual_full_optimizer(train_loader, val_loader, test_loader, output_root / "manual", device, args.epoch_scale, args.resume, args.skip_completed)
    run_bn_analysis(train_loader, val_loader, test_loader, output_root / "bn_analysis", device, args.epoch_scale, args.resume, args.skip_completed)
    high_performance_results = run_high_performance_search(train_loader, val_loader, test_loader, output_root / "advanced", device, args.epoch_scale, args.resume, args.skip_completed)

    all_results = architecture_results + ablation_results + regularization_results + optimizer_results + scheduler_results + manual_torch_results + manual_optimizer_results + high_performance_results
    best_model, best_result = load_best_result_model(all_results, device)
    criterion = torch.nn.CrossEntropyLoss()
    if best_result is None:
        best_optimizer = torch.optim.Adam(best_model.parameters(), lr=1e-3, weight_decay=5e-4)
        fit(best_model, train_loader, val_loader, criterion, best_optimizer, device=device, epochs=scaled_epochs(20, args.epoch_scale), output_dir=output_root / "best_model", resume=args.resume)
    class_names = test_loader.dataset.classes if hasattr(test_loader.dataset, "classes") else test_loader.dataset.dataset.classes
    save_filter_visualization(best_model, output_root / "figures" / "first_layer_filters.png")
    save_confusion_matrix(best_model, test_loader, class_names, output_root / "figures" / "confusion_matrix.png", device=device)
    save_per_class_accuracy(best_model, test_loader, class_names, output_root / "figures" / "per_class_accuracy.png", device=device)
    save_misclassified_examples(best_model, test_loader, class_names, output_root / "figures" / "misclassified_examples.png", device=device)
    save_feature_maps(best_model, test_loader, output_root / "figures" / "feature_maps.png", device=device)
    save_saliency_map(best_model, test_loader, output_root / "figures" / "saliency_map.png", device=device)
    save_scatter_plot(all_results, x_key="parameters", y_key="test_accuracy", output_path=output_root / "figures" / "params_vs_accuracy.png", title="Parameters vs Test Accuracy", xlabel="Parameters", ylabel="Test Accuracy")
    save_scatter_plot(all_results, x_key="best_val_accuracy", y_key="test_accuracy", output_path=output_root / "figures" / "val_vs_test_accuracy.png", title="Validation vs Test Accuracy", xlabel="Best Validation Accuracy", ylabel="Test Accuracy")
    save_scatter_plot(all_results, x_key="average_epoch_time", y_key="test_accuracy", output_path=output_root / "figures" / "speed_vs_accuracy.png", title="Training Speed vs Test Accuracy", xlabel="Average Epoch Time (s)", ylabel="Test Accuracy")

    export_ranked_table(all_results, output_root / "all_results.csv")

    summary = {
        "architecture_results": architecture_results,
        "ablation_results": ablation_results,
        "regularization_results": regularization_results,
        "optimizer_results": optimizer_results,
        "scheduler_results": scheduler_results,
        "manual_network_torch_optim_results": manual_torch_results,
        "manual_optimizer_results": manual_optimizer_results,
        "high_performance_results": high_performance_results,
    }
    with (output_root / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
    print("Finished all experiments.")


if __name__ == "__main__":
    main()
