"""
Training, evaluation, visualization, and BN-analysis helpers.
"""

from __future__ import annotations

import json
import math
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn

mpl.use("Agg")


@dataclass
class EpochStats:
    loss: float
    accuracy: float


@dataclass
class TrainHistory:
    train_loss: list[float]
    train_accuracy: list[float]
    val_loss: list[float]
    val_accuracy: list[float]
    step_losses: list[float]
    step_grad_norms: list[float]
    epoch_times: list[float]


def history_from_dict(data: dict[str, list[float]]) -> TrainHistory:
    return TrainHistory(
        train_loss=data.get("train_loss", []),
        train_accuracy=data.get("train_accuracy", []),
        val_loss=data.get("val_loss", []),
        val_accuracy=data.get("val_accuracy", []),
        step_losses=data.get("step_losses", []),
        step_grad_norms=data.get("step_grad_norms", []),
        epoch_times=data.get("epoch_times", []),
    )


def set_random_seeds(seed_value: int = 42, device: str | torch.device = "cpu") -> None:
    random.seed(seed_value)
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    if str(device) != "cpu" and torch.cuda.is_available():
        torch.cuda.manual_seed(seed_value)
        torch.cuda.manual_seed_all(seed_value)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def get_device(preferred: str = "auto") -> torch.device:
    def _validate_device(candidate: str) -> torch.device:
        device = torch.device(candidate)
        if device.type == "cuda":
            try:
                torch.zeros(1).to(device)
            except Exception as error:
                print(f"CUDA unavailable for '{candidate}', falling back to CPU: {error}")
                return torch.device("cpu")
        return device

    if preferred != "auto":
        return _validate_device(preferred)

    if torch.cuda.is_available():
        return _validate_device("cuda")
    return torch.device("cpu")


def get_accuracy(logits: torch.Tensor, targets: torch.Tensor) -> float:
    if targets.ndim > 1:
        targets = targets.argmax(dim=1)
    predictions = logits.argmax(dim=1)
    return (predictions == targets).float().mean().item()


def compute_loss_value(criterion, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    if targets.ndim > 1:
        log_probs = torch.log_softmax(logits, dim=1)
        return -(targets * log_probs).sum(dim=1).mean()
    return criterion(logits, targets)


def _compute_grad_norm(model: Any) -> float:
    total = 0.0
    for parameter in model.parameters():
        if parameter.grad is not None:
            total += parameter.grad.detach().pow(2).sum().item()
    return math.sqrt(total)


def run_epoch(
    model: Any,
    loader,
    criterion,
    optimizer=None,
    device: torch.device | str = "cpu",
    batch_transform=None,
) -> tuple[EpochStats, list[float], list[float]]:
    is_train = optimizer is not None
    if hasattr(model, "train"):
        model.train() if is_train else model.eval()

    total_loss = 0.0
    total_correct = 0
    total_count = 0
    step_losses: list[float] = []
    step_grad_norms: list[float] = []

    for inputs, targets in loader:
        inputs = inputs.to(device)
        targets = targets.to(device)
        if batch_transform is not None:
            inputs, targets = batch_transform(inputs, targets)
        if is_train:
            optimizer.zero_grad()

        logits = model(inputs)
        loss = compute_loss_value(criterion, logits, targets)

        if is_train:
            loss.backward()
            step_grad_norms.append(_compute_grad_norm(model))
            optimizer.step()
        else:
            step_grad_norms.append(0.0)

        batch_size = inputs.size(0)
        total_loss += loss.item() * batch_size
        eval_targets = targets.argmax(dim=1) if targets.ndim > 1 else targets
        total_correct += (logits.argmax(dim=1) == eval_targets).sum().item()
        total_count += batch_size
        step_losses.append(loss.item())

    return (
        EpochStats(loss=total_loss / total_count, accuracy=total_correct / total_count),
        step_losses,
        step_grad_norms,
    )


def evaluate(model: Any, loader, criterion, device: torch.device | str = "cpu") -> EpochStats:
    with torch.no_grad():
        stats, _, _ = run_epoch(model, loader, criterion, optimizer=None, device=device)
    return stats


def clone_model_state(model: Any) -> dict[str, torch.Tensor]:
    if isinstance(model, nn.Module):
        return {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
    return {name: value.detach().cpu().clone() for name, value in model.named_parameters()}


def restore_model_state(model: Any, state: dict[str, torch.Tensor], device: torch.device | str) -> None:
    if isinstance(model, nn.Module):
        model.load_state_dict(state)
        model.to(device)
        return
    for name, value in state.items():
        model.params[name] = value.to(device).detach().requires_grad_(True)


def save_model_state(model: Any, output_path: str | Path) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if isinstance(model, nn.Module):
        torch.save(model.state_dict(), output_path)
        return
    manual_state = {
        name: value.detach().cpu()
        for name, value in model.named_parameters()
    }
    torch.save(manual_state, output_path)


def load_model_state(model: Any, model_path: str | Path, device: torch.device | str) -> None:
    state = torch.load(model_path, map_location="cpu")
    restore_model_state(model, state, device)


def load_history(history_path: str | Path) -> TrainHistory:
    with Path(history_path).open("r", encoding="utf-8") as handle:
        return history_from_dict(json.load(handle))


def save_checkpoint(
    model: Any,
    optimizer,
    history: TrainHistory,
    epoch: int,
    best_val_accuracy: float,
    best_state: dict[str, torch.Tensor],
    output_dir: str | Path,
) -> None:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    checkpoint = {
        "epoch": epoch,
        "history": asdict(history),
        "best_val_accuracy": best_val_accuracy,
        "best_state": {key: value.detach().cpu() for key, value in best_state.items()},
        "current_state": clone_model_state(model),
        "optimizer_state": optimizer.state_dict() if hasattr(optimizer, "state_dict") else None,
    }
    torch.save(checkpoint, output_path / "latest_checkpoint.pt")
    with (output_path / "history.json").open("w", encoding="utf-8") as handle:
        json.dump(asdict(history), handle, indent=2)
    save_training_curves(history, output_path / "training_curves.png")


def try_resume_from_checkpoint(
    model: Any,
    optimizer,
    output_dir: str | Path,
    device: torch.device | str,
) -> tuple[int, TrainHistory, float, dict[str, torch.Tensor]] | None:
    checkpoint_path = Path(output_dir) / "latest_checkpoint.pt"
    if not checkpoint_path.exists():
        return None

    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    restore_model_state(model, checkpoint["current_state"], device)
    if checkpoint.get("optimizer_state") is not None and hasattr(optimizer, "load_state_dict"):
        optimizer.load_state_dict(checkpoint["optimizer_state"])
    history = history_from_dict(checkpoint["history"])
    best_state = {key: value.detach().cpu() for key, value in checkpoint["best_state"].items()}
    return checkpoint["epoch"], history, checkpoint["best_val_accuracy"], best_state


def fit(
    model: Any,
    train_loader,
    val_loader,
    criterion,
    optimizer,
    device: torch.device | str = "cpu",
    epochs: int = 10,
    scheduler=None,
    output_dir: str | Path | None = None,
    resume: bool = False,
    train_batch_transform=None,
) -> TrainHistory:
    device = torch.device(device)
    if isinstance(model, nn.Module):
        model.to(device)
    else:
        model.to(device)

    history = TrainHistory([], [], [], [], [], [], [])
    best_val_accuracy = -1.0
    best_state = clone_model_state(model)
    start_epoch = 0

    if resume and output_dir is not None:
        resumed_state = try_resume_from_checkpoint(model, optimizer, output_dir, device)
        if resumed_state is not None:
            start_epoch, history, best_val_accuracy, best_state = resumed_state
            print(f"Resuming from epoch {start_epoch + 1:02d}/{epochs:02d} in {output_dir}")

    for epoch in range(start_epoch, epochs):
        epoch_start = time.perf_counter()
        train_stats, step_losses, step_grad_norms = run_epoch(
            model,
            train_loader,
            criterion,
            optimizer=optimizer,
            device=device,
            batch_transform=train_batch_transform,
        )
        val_stats = evaluate(model, val_loader, criterion, device=device)
        if scheduler is not None:
            scheduler.step()
        epoch_time = time.perf_counter() - epoch_start

        history.train_loss.append(train_stats.loss)
        history.train_accuracy.append(train_stats.accuracy)
        history.val_loss.append(val_stats.loss)
        history.val_accuracy.append(val_stats.accuracy)
        history.step_losses.extend(step_losses)
        history.step_grad_norms.extend(step_grad_norms)
        history.epoch_times.append(epoch_time)

        if val_stats.accuracy > best_val_accuracy:
            best_val_accuracy = val_stats.accuracy
            best_state = clone_model_state(model)

        print(
            f"Epoch {epoch + 1:02d}/{epochs:02d} "
            f"train_loss={train_stats.loss:.4f} train_acc={train_stats.accuracy:.4f} "
            f"val_loss={val_stats.loss:.4f} val_acc={val_stats.accuracy:.4f} "
            f"time={epoch_time:.2f}s"
        )

        if output_dir is not None:
            save_checkpoint(model, optimizer, history, epoch + 1, best_val_accuracy, best_state, output_dir)

    restore_model_state(model, best_state, device)

    if output_dir is not None:
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        with (output_path / "history.json").open("w", encoding="utf-8") as handle:
            json.dump(asdict(history), handle, indent=2)
        save_training_curves(history, output_path / "training_curves.png")
        save_model_state(model, output_path / "best_model.pt")
    return history


def save_training_curves(history: TrainHistory, output_path: str | Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].plot(history.train_loss, label="train")
    axes[0].plot(history.val_loss, label="val")
    axes[0].set_title("Loss")
    axes[0].legend()
    axes[1].plot(history.train_accuracy, label="train")
    axes[1].plot(history.val_accuracy, label="val")
    axes[1].set_title("Accuracy")
    axes[1].legend()
    fig.tight_layout()
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def save_filter_visualization(model: nn.Module, output_path: str | Path, max_filters: int = 16) -> None:
    first_conv = next(module for module in model.modules() if isinstance(module, nn.Conv2d))
    filters = first_conv.weight.detach().cpu()
    filters = (filters - filters.min()) / (filters.max() - filters.min() + 1e-8)
    n_filters = min(filters.size(0), max_filters)
    cols = 4
    rows = math.ceil(n_filters / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 3, rows * 3))
    axes = np.atleast_2d(axes)
    for index in range(rows * cols):
        axis = axes.flat[index]
        axis.axis("off")
        if index < n_filters:
            image = filters[index].permute(1, 2, 0).numpy()
            axis.imshow(image)
            axis.set_title(f"Filter {index}")
    fig.tight_layout()
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def _collect_logits(model: nn.Module, loader, device: torch.device | str):
    logits_list = []
    targets_list = []
    with torch.no_grad():
        model.eval()
        for inputs, targets in loader:
            inputs = inputs.to(device)
            logits = model(inputs)
            logits_list.append(logits.cpu())
            targets_list.append(targets.cpu())
    return torch.cat(logits_list), torch.cat(targets_list)


def save_per_class_accuracy(
    model: nn.Module,
    loader,
    class_names: list[str],
    output_path: str | Path,
    device: torch.device | str,
) -> dict[str, float]:
    logits, targets = _collect_logits(model, loader, device)
    predictions = logits.argmax(dim=1)
    result = {}
    for index, class_name in enumerate(class_names):
        mask = targets == index
        correct = (predictions[mask] == targets[mask]).float().mean().item() if mask.any() else 0.0
        result[class_name] = correct
    fig, axis = plt.subplots(figsize=(10, 4))
    axis.bar(range(len(class_names)), list(result.values()), color="#0072b2")
    axis.set_xticks(range(len(class_names)), class_names, rotation=45, ha="right")
    axis.set_ylim(0.0, 1.0)
    axis.set_ylabel("Accuracy")
    axis.set_title("Per-class Accuracy")
    fig.tight_layout()
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200)
    plt.close(fig)
    return result


def save_misclassified_examples(
    model: nn.Module,
    loader,
    class_names: list[str],
    output_path: str | Path,
    device: torch.device | str,
    max_items: int = 16,
) -> None:
    model.eval()
    collected = []
    with torch.no_grad():
        for inputs, targets in loader:
            inputs = inputs.to(device)
            logits = model(inputs)
            predictions = logits.argmax(dim=1).cpu()
            wrong = predictions != targets
            for image, target, pred in zip(inputs.cpu(), targets.cpu(), predictions):
                if pred != target:
                    collected.append((image, int(target), int(pred)))
                if len(collected) >= max_items:
                    break
            if len(collected) >= max_items:
                break
    cols = 4
    rows = max(1, math.ceil(len(collected) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 3, rows * 3))
    axes = np.atleast_2d(axes)
    mean = np.array([0.4914, 0.4822, 0.4465])
    std = np.array([0.2023, 0.1994, 0.2010])
    for index in range(rows * cols):
        axis = axes.flat[index]
        axis.axis("off")
        if index < len(collected):
            image, target, pred = collected[index]
            image = image.permute(1, 2, 0).numpy()
            image = np.clip(image * std + mean, 0.0, 1.0)
            axis.imshow(image)
            axis.set_title(f"T:{class_names[target]}\nP:{class_names[pred]}", fontsize=8)
    fig.tight_layout()
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def save_feature_maps(
    model: nn.Module,
    loader,
    output_path: str | Path,
    device: torch.device | str,
    max_maps: int = 16,
) -> None:
    model.eval()
    inputs, _ = next(iter(loader))
    inputs = inputs[:1].to(device)
    first_feature = None
    for module in model.modules():
        if isinstance(module, nn.Conv2d):
            first_feature = module
            break
    captured = {}
    def hook(_, __, output):
        captured["maps"] = output.detach().cpu()
    handle = first_feature.register_forward_hook(hook)
    with torch.no_grad():
        model(inputs)
    handle.remove()
    feature_maps = captured["maps"][0]
    cols = 4
    n_maps = min(max_maps, feature_maps.size(0))
    rows = math.ceil(n_maps / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 3, rows * 3))
    axes = np.atleast_2d(axes)
    for index in range(rows * cols):
        axis = axes.flat[index]
        axis.axis("off")
        if index < n_maps:
            axis.imshow(feature_maps[index].numpy(), cmap="viridis")
            axis.set_title(f"Map {index}")
    fig.tight_layout()
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def save_confusion_matrix(
    model: nn.Module,
    loader,
    class_names: list[str],
    output_path: str | Path,
    device: torch.device | str,
) -> np.ndarray:
    logits, targets = _collect_logits(model, loader, device)
    predictions = logits.argmax(dim=1)
    n_classes = len(class_names)
    matrix = torch.zeros(n_classes, n_classes, dtype=torch.int64)
    for target, prediction in zip(targets, predictions):
        matrix[target.long(), prediction.long()] += 1

    fig, axis = plt.subplots(figsize=(8, 6))
    image = axis.imshow(matrix.numpy(), cmap="Blues")
    axis.set_xticks(range(n_classes), class_names, rotation=45, ha="right")
    axis.set_yticks(range(n_classes), class_names)
    axis.set_xlabel("Predicted")
    axis.set_ylabel("True")
    axis.set_title("Confusion Matrix")
    fig.colorbar(image, ax=axis, shrink=0.8)
    fig.tight_layout()
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200)
    plt.close(fig)
    return matrix.numpy()


def save_saliency_map(
    model: nn.Module,
    loader,
    output_path: str | Path,
    device: torch.device | str,
) -> None:
    model.eval()
    inputs, targets = next(iter(loader))
    inputs = inputs[:1].to(device)
    targets = targets[:1].to(device)
    inputs.requires_grad_(True)

    logits = model(inputs)
    logits[0, targets.item()].backward()
    saliency = inputs.grad.abs().max(dim=1)[0].detach().cpu().squeeze(0).numpy()
    image = inputs.detach().cpu().squeeze(0).permute(1, 2, 0).numpy()
    mean = np.array([0.4914, 0.4822, 0.4465])
    std = np.array([0.2023, 0.1994, 0.2010])
    image = np.clip(image * std + mean, 0.0, 1.0)

    fig, axes = plt.subplots(1, 2, figsize=(8, 4))
    axes[0].imshow(image)
    axes[0].set_title("Input")
    axes[0].axis("off")
    axes[1].imshow(saliency, cmap="hot")
    axes[1].set_title("Saliency")
    axes[1].axis("off")
    fig.tight_layout()
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def compute_loss_landscape_curves(step_losses_by_lr: dict[str, list[float]]) -> dict[str, list[float]]:
    max_length = max(len(losses) for losses in step_losses_by_lr.values())
    min_curve = []
    max_curve = []
    mean_curve = []
    for index in range(max_length):
        values = [losses[index] for losses in step_losses_by_lr.values() if index < len(losses)]
        min_curve.append(float(np.min(values)))
        max_curve.append(float(np.max(values)))
        mean_curve.append(float(np.mean(values)))
    return {"min_curve": min_curve, "max_curve": max_curve, "mean_curve": mean_curve}


def save_loss_landscape_plot(
    baseline_curves: dict[str, list[float]],
    bn_curves: dict[str, list[float]],
    output_path: str | Path,
) -> None:
    fig, axis = plt.subplots(figsize=(10, 5))
    x_baseline = np.arange(len(baseline_curves["mean_curve"]))
    x_bn = np.arange(len(bn_curves["mean_curve"]))

    axis.plot(x_baseline, baseline_curves["mean_curve"], label="VGG-A", color="#d55e00")
    axis.fill_between(x_baseline, baseline_curves["min_curve"], baseline_curves["max_curve"], alpha=0.25, color="#d55e00")
    axis.plot(x_bn, bn_curves["mean_curve"], label="VGG-A + BN", color="#0072b2")
    axis.fill_between(x_bn, bn_curves["min_curve"], bn_curves["max_curve"], alpha=0.25, color="#0072b2")
    axis.set_xlabel("Training step")
    axis.set_ylabel("Loss")
    axis.set_title("Loss landscape band across learning rates")
    axis.legend()
    fig.tight_layout()
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def cosine_similarity(values_a: torch.Tensor, values_b: torch.Tensor) -> float:
    return torch.nn.functional.cosine_similarity(values_a, values_b, dim=0).item()


def _flatten_gradients(model: Any) -> torch.Tensor:
    grads = []
    for parameter in model.parameters():
        if parameter.grad is None:
            grads.append(torch.zeros_like(parameter).reshape(-1))
        else:
            grads.append(parameter.grad.detach().reshape(-1))
    return torch.cat(grads)


def analyze_gradient_predictiveness(
    model: Any,
    loader,
    criterion,
    device: torch.device | str,
    max_batches: int = 10,
) -> dict[str, list[float] | float]:
    model_state = clone_model_state(model)
    gradient_vectors: list[torch.Tensor] = []

    for batch_index, (inputs, targets) in enumerate(loader):
        if batch_index >= max_batches:
            break
        if hasattr(model, "train"):
            model.train()
        inputs = inputs.to(device)
        targets = targets.to(device)
        for parameter in model.parameters():
            if parameter.grad is not None:
                parameter.grad.zero_()
        logits = model(inputs)
        loss = criterion(logits, targets)
        loss.backward()
        gradient_vectors.append(_flatten_gradients(model).cpu())

    restore_model_state(model, model_state, device)
    cosine_values = []
    l2_values = []
    for left, right in zip(gradient_vectors[:-1], gradient_vectors[1:]):
        cosine_values.append(cosine_similarity(left, right))
        l2_values.append(torch.norm(right - left, p=2).item())

    return {
        "gradient_cosine": cosine_values,
        "gradient_l2_diff": l2_values,
        "max_gradient_difference": max(l2_values) if l2_values else 0.0,
    }


def _clone_parameter_tensors(model: Any) -> list[torch.Tensor]:
    return [parameter.detach().clone() for parameter in model.parameters()]


def _restore_parameter_tensors(model: Any, tensors: list[torch.Tensor]) -> None:
    with torch.no_grad():
        for parameter, tensor in zip(model.parameters(), tensors):
            parameter.copy_(tensor.to(parameter.device))


def analyze_prediction_error(
    model: Any,
    loader,
    criterion,
    device: torch.device | str,
    step_sizes: list[float],
    max_batches: int = 4,
) -> dict[str, list[float] | float]:
    device = torch.device(device)
    prediction_errors = {f"{step_size:.0e}": [] for step_size in step_sizes}
    original_params = _clone_parameter_tensors(model)

    for batch_index, (inputs, targets) in enumerate(loader):
        if batch_index >= max_batches:
            break
        if hasattr(model, "train"):
            model.train()
        inputs = inputs.to(device)
        targets = targets.to(device)
        for parameter in model.parameters():
            if parameter.grad is not None:
                parameter.grad.zero_()

        logits = model(inputs)
        loss = criterion(logits, targets)
        loss.backward()
        gradient_vector = _flatten_gradients(model)
        grad_norm = torch.norm(gradient_vector, p=2).item()
        if grad_norm <= 1e-12:
            continue
        base_loss = loss.item()
        saved_params = _clone_parameter_tensors(model)

        for step_size in step_sizes:
            with torch.no_grad():
                for parameter in model.parameters():
                    if parameter.grad is not None:
                        parameter.add_(parameter.grad, alpha=-step_size / (grad_norm + 1e-12))
            with torch.no_grad():
                perturbed_logits = model(inputs)
                perturbed_loss = criterion(perturbed_logits, targets).item()
            predicted_loss = base_loss - step_size * grad_norm
            prediction_errors[f"{step_size:.0e}"].append(abs(perturbed_loss - predicted_loss))
            _restore_parameter_tensors(model, saved_params)

    _restore_parameter_tensors(model, original_params)
    mean_prediction_errors = {
        key: float(np.mean(values)) if values else 0.0
        for key, values in prediction_errors.items()
    }
    return {
        "prediction_error_curves": prediction_errors,
        "mean_prediction_error": list(mean_prediction_errors.values()),
        "prediction_error_step_sizes": list(mean_prediction_errors.keys()),
        "max_prediction_error": max(mean_prediction_errors.values()) if mean_prediction_errors else 0.0,
    }


def save_gradient_analysis_plot(
    baseline_stats: dict[str, list[float] | float],
    bn_stats: dict[str, list[float] | float],
    output_path: str | Path,
) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(18, 4))
    axes[0].plot(baseline_stats["gradient_cosine"], label="VGG-A", color="#d55e00")
    axes[0].plot(bn_stats["gradient_cosine"], label="VGG-A + BN", color="#0072b2")
    axes[0].set_title("Gradient predictiveness")
    axes[0].set_ylabel("Cosine similarity")
    axes[0].set_xlabel("Step")
    axes[0].legend()

    axes[1].plot(baseline_stats["gradient_l2_diff"], label="VGG-A", color="#d55e00")
    axes[1].plot(bn_stats["gradient_l2_diff"], label="VGG-A + BN", color="#0072b2")
    axes[1].set_title("Gradient difference over distance")
    axes[1].set_ylabel("L2 difference")
    axes[1].set_xlabel("Step")
    axes[1].legend()

    if "prediction_error_step_sizes" in baseline_stats and "prediction_error_step_sizes" in bn_stats:
        x = np.arange(len(baseline_stats["prediction_error_step_sizes"]))
        axes[2].plot(x, baseline_stats["mean_prediction_error"], marker="o", label="VGG-A", color="#d55e00")
        axes[2].plot(x, bn_stats["mean_prediction_error"], marker="o", label="VGG-A + BN", color="#0072b2")
        axes[2].set_xticks(x, baseline_stats["prediction_error_step_sizes"])
        axes[2].set_title("First-order prediction error")
        axes[2].set_ylabel("Absolute error")
        axes[2].set_xlabel("Step size")
        axes[2].legend()
    fig.tight_layout()
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def save_scatter_plot(
    results: list[dict],
    x_key: str,
    y_key: str,
    output_path: str | Path,
    title: str,
    xlabel: str,
    ylabel: str,
) -> None:
    xs = [item[x_key] for item in results if x_key in item and y_key in item]
    ys = [item[y_key] for item in results if x_key in item and y_key in item]
    labels = [item["name"] for item in results if x_key in item and y_key in item]
    fig, axis = plt.subplots(figsize=(8, 6))
    axis.scatter(xs, ys, color="#0072b2")
    for x, y, label in zip(xs, ys, labels):
        axis.annotate(label, (x, y), fontsize=7, alpha=0.8)
    axis.set_title(title)
    axis.set_xlabel(xlabel)
    axis.set_ylabel(ylabel)
    fig.tight_layout()
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200)
    plt.close(fig)
