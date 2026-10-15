"""Train and evaluate the baseline or improved 3D U-Net on HipMRI."""

from __future__ import annotations

import argparse
import json
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

import torch
import torch.nn.functional as functional
from torch import Tensor, nn
from torch.utils.data import DataLoader

from dataset import DEFAULT_DATA_ROOT, LABEL_NAMES, build_datasets
from modules import build_model, count_trainable_parameters


NUM_CLASSES = len(LABEL_NAMES)


@dataclass(frozen=True)
class TrainingConfig:
    """Serializable experiment settings stored with every checkpoint."""

    model: str
    data_root: str
    output_dir: str
    epochs: int
    batch_size: int
    learning_rate: float
    weight_decay: float
    base_channels: int
    target_shape: tuple[int, int, int]
    num_workers: int
    seed: int
    patience: int
    amp: bool


class SegmentationMeter:
    """Accumulate class-wise overlap statistics without storing predictions."""

    def __init__(self, num_classes: int) -> None:
        self.num_classes = num_classes
        self.intersection = torch.zeros(num_classes, dtype=torch.float64)
        self.prediction_size = torch.zeros(num_classes, dtype=torch.float64)
        self.target_size = torch.zeros(num_classes, dtype=torch.float64)

    @torch.no_grad()
    def update(self, logits: Tensor, target: Tensor) -> None:
        prediction = logits.argmax(dim=1)
        for class_id in range(self.num_classes):
            predicted_class = prediction == class_id
            target_class = target == class_id
            self.intersection[class_id] += torch.count_nonzero(
                predicted_class & target_class
            ).cpu()
            self.prediction_size[class_id] += torch.count_nonzero(
                predicted_class
            ).cpu()
            self.target_size[class_id] += torch.count_nonzero(target_class).cpu()

    def compute(self) -> dict[str, Any]:
        epsilon = 1e-7
        dice_denominator = self.prediction_size + self.target_size
        union = self.prediction_size + self.target_size - self.intersection
        dice = (2.0 * self.intersection + epsilon) / (dice_denominator + epsilon)
        iou = (self.intersection + epsilon) / (union + epsilon)

        per_class = {
            LABEL_NAMES[class_id]: {
                "dice": float(dice[class_id]),
                "iou": float(iou[class_id]),
            }
            for class_id in range(self.num_classes)
        }
        return {
            "mean_dice": float(dice.mean()),
            "mean_iou": float(iou.mean()),
            "mean_foreground_dice": float(dice[1:].mean()),
            "mean_foreground_iou": float(iou[1:].mean()),
            "per_class": per_class,
        }


def soft_dice_loss(logits: Tensor, target: Tensor) -> Tensor:
    """Compute multiclass soft Dice loss while excluding background."""
    probabilities = torch.softmax(logits, dim=1)
    one_hot_target = functional.one_hot(target, num_classes=NUM_CLASSES)
    one_hot_target = one_hot_target.movedim(-1, 1).to(probabilities.dtype)
    reduction_dimensions = (0, 2, 3, 4)
    intersection = torch.sum(
        probabilities * one_hot_target, dim=reduction_dimensions
    )
    denominator = torch.sum(
        probabilities + one_hot_target, dim=reduction_dimensions
    )
    class_dice = (2.0 * intersection + 1e-5) / (denominator + 1e-5)
    return 1.0 - class_dice[1:].mean()


def segmentation_loss(logits: Tensor, target: Tensor) -> Tensor:
    """Balance voxel-wise classification with region-overlap optimisation."""
    cross_entropy = functional.cross_entropy(logits, target)
    dice = soft_dice_loss(logits, target)
    return 0.5 * cross_entropy + 0.5 * dice


def supervised_loss(outputs: Tensor | tuple[Tensor, ...], target: Tensor) -> Tensor:
    """Apply decreasing weights to the main and deep-supervision outputs."""
    if isinstance(outputs, Tensor):
        return segmentation_loss(outputs, target)

    weights = (1.0, 0.5, 0.25)
    active_weights = weights[: len(outputs)]
    weighted_losses = [
        weight * segmentation_loss(output, target)
        for weight, output in zip(active_weights, outputs)
    ]
    return sum(weighted_losses) / sum(active_weights)


def primary_logits(outputs: Tensor | tuple[Tensor, ...]) -> Tensor:
    """Select the full-resolution output used for metrics and inference."""
    return outputs[0] if isinstance(outputs, tuple) else outputs


def set_reproducibility(seed: int) -> None:
    """Seed Python and PyTorch and choose deterministic CUDA behaviour."""
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def select_device(requested_device: str) -> torch.device:
    if requested_device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(requested_device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available in this job.")
    return device


def build_loaders(
    config: TrainingConfig,
    device: torch.device,
) -> dict[str, DataLoader]:
    """Create deterministic DataLoaders for all patient-level partitions."""
    datasets = build_datasets(
        data_root=config.data_root,
        target_shape=config.target_shape,
        seed=config.seed,
    )
    generator = torch.Generator().manual_seed(config.seed)
    common_options = {
        "batch_size": config.batch_size,
        "num_workers": config.num_workers,
        "pin_memory": device.type == "cuda",
        "persistent_workers": config.num_workers > 0,
    }
    return {
        "train": DataLoader(
            datasets["train"],
            shuffle=True,
            generator=generator,
            **common_options,
        ),
        "validation": DataLoader(
            datasets["validation"], shuffle=False, **common_options
        ),
        "test": DataLoader(datasets["test"], shuffle=False, **common_options),
    }


def _limited_batches(
    loader: DataLoader,
    maximum_batches: int | None,
) -> Iterable[dict[str, Any]]:
    for batch_index, batch in enumerate(loader):
        if maximum_batches is not None and batch_index >= maximum_batches:
            break
        yield batch


def train_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scaler: torch.amp.GradScaler,
    device: torch.device,
    amp_enabled: bool,
    maximum_batches: int | None = None,
) -> tuple[float, dict[str, Any]]:
    model.train()
    meter = SegmentationMeter(NUM_CLASSES)
    running_loss = 0.0
    batch_count = 0

    for batch in _limited_batches(loader, maximum_batches):
        image = batch["image"].to(device, non_blocking=True)
        target = batch["label"].to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)

        with torch.autocast(
            device_type=device.type,
            dtype=torch.float16,
            enabled=amp_enabled,
        ):
            outputs = model(image)
            loss = supervised_loss(outputs, target)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        logits = primary_logits(outputs)
        meter.update(logits.detach(), target)
        running_loss += float(loss.detach())
        batch_count += 1

    if batch_count == 0:
        raise RuntimeError("The training DataLoader produced no batches.")
    return running_loss / batch_count, meter.compute()


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    amp_enabled: bool,
    maximum_batches: int | None = None,
) -> tuple[float, dict[str, Any]]:
    model.eval()
    meter = SegmentationMeter(NUM_CLASSES)
    running_loss = 0.0
    batch_count = 0

    for batch in _limited_batches(loader, maximum_batches):
        image = batch["image"].to(device, non_blocking=True)
        target = batch["label"].to(device, non_blocking=True)
        with torch.autocast(
            device_type=device.type,
            dtype=torch.float16,
            enabled=amp_enabled,
        ):
            outputs = model(image)
            loss = supervised_loss(outputs, target)
        logits = primary_logits(outputs)
        meter.update(logits, target)
        running_loss += float(loss)
        batch_count += 1

    if batch_count == 0:
        raise RuntimeError("The evaluation DataLoader produced no batches.")
    return running_loss / batch_count, meter.compute()


def save_checkpoint(
    path: Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    validation_metrics: dict[str, Any],
    config: TrainingConfig,
) -> None:
    """Save all state required to reproduce or resume the best model."""
    torch.save(
        {
            "epoch": epoch,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "validation_metrics": validation_metrics,
            "config": asdict(config),
            "label_names": LABEL_NAMES,
        },
        path,
    )


def save_json(path: Path, contents: Any) -> None:
    path.write_text(json.dumps(contents, indent=2), encoding="utf-8")


def plot_history(history: list[dict[str, float]], output_path: Path) -> None:
    """Save loss and foreground-Dice curves required for the report."""
    try:
        import matplotlib.pyplot as pyplot
    except ImportError:
        print("matplotlib is not installed; skipping the training-curve plot.")
        return

    epochs = [record["epoch"] for record in history]
    figure, axes = pyplot.subplots(1, 2, figsize=(11, 4))
    axes[0].plot(epochs, [record["train_loss"] for record in history], label="train")
    axes[0].plot(
        epochs, [record["validation_loss"] for record in history], label="validation"
    )
    axes[0].set(title="Loss", xlabel="Epoch", ylabel="Dice + cross-entropy")
    axes[0].legend()

    axes[1].plot(
        epochs,
        [record["train_foreground_dice"] for record in history],
        label="train",
    )
    axes[1].plot(
        epochs,
        [record["validation_foreground_dice"] for record in history],
        label="validation",
    )
    axes[1].set(title="Mean foreground Dice", xlabel="Epoch", ylabel="Dice")
    axes[1].legend()
    figure.tight_layout()
    figure.savefig(output_path, dpi=180)
    pyplot.close(figure)


def run_training(
    config: TrainingConfig,
    device: torch.device,
    maximum_train_batches: int | None,
    maximum_validation_batches: int | None,
    maximum_test_batches: int | None,
) -> dict[str, Any]:
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    save_json(output_dir / "config.json", asdict(config))

    loaders = build_loaders(config, device)
    model = build_model(
        config.model,
        num_classes=NUM_CLASSES,
        base_channels=config.base_channels,
        deep_supervision=True,
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=5
    )
    amp_enabled = config.amp and device.type == "cuda"
    scaler = torch.amp.GradScaler(device.type, enabled=amp_enabled)

    checkpoint_path = output_dir / "best_model.pt"
    history: list[dict[str, float]] = []
    best_validation_dice = -1.0
    epochs_without_improvement = 0

    print(f"Device: {device}")
    print(f"Model: {config.model}")
    print(f"Trainable parameters: {count_trainable_parameters(model):,}")
    print(f"AMP enabled: {amp_enabled}")

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    for epoch in range(1, config.epochs + 1):
        epoch_start = time.perf_counter()
        train_loss, train_metrics = train_epoch(
            model,
            loaders["train"],
            optimizer,
            scaler,
            device,
            amp_enabled,
            maximum_batches=maximum_train_batches,
        )
        validation_loss, validation_metrics = evaluate(
            model,
            loaders["validation"],
            device,
            amp_enabled,
            maximum_batches=maximum_validation_batches,
        )
        validation_dice = validation_metrics["mean_foreground_dice"]
        scheduler.step(validation_dice)
        epoch_seconds = time.perf_counter() - epoch_start

        history.append(
            {
                "epoch": float(epoch),
                "train_loss": train_loss,
                "validation_loss": validation_loss,
                "train_foreground_dice": train_metrics["mean_foreground_dice"],
                "validation_foreground_dice": validation_dice,
                "learning_rate": optimizer.param_groups[0]["lr"],
                "seconds": epoch_seconds,
            }
        )
        save_json(output_dir / "history.json", history)
        plot_history(history, output_dir / "training_curves.png")
        print(
            f"Epoch {epoch:03d}/{config.epochs:03d} | "
            f"train loss {train_loss:.4f} dice "
            f"{train_metrics['mean_foreground_dice']:.4f} | "
            f"val loss {validation_loss:.4f} dice {validation_dice:.4f} | "
            f"{epoch_seconds:.1f}s"
        )

        if validation_dice > best_validation_dice:
            best_validation_dice = validation_dice
            epochs_without_improvement = 0
            save_checkpoint(
                checkpoint_path,
                model,
                optimizer,
                epoch,
                validation_metrics,
                config,
            )
            print(f"  Saved new best checkpoint: {checkpoint_path}")
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= config.patience:
                print(f"Early stopping after {config.patience} unimproved epochs.")
                break

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state"])
    test_loss, test_metrics = evaluate(
        model,
        loaders["test"],
        device,
        amp_enabled,
        maximum_batches=maximum_test_batches,
    )
    peak_vram_mb = (
        torch.cuda.max_memory_allocated(device) / (1024**2)
        if device.type == "cuda"
        else 0.0
    )
    results = {
        "best_epoch": checkpoint["epoch"],
        "best_validation_metrics": checkpoint["validation_metrics"],
        "test_loss": test_loss,
        "test_metrics": test_metrics,
        "trainable_parameters": count_trainable_parameters(model),
        "peak_vram_mb": peak_vram_mb,
    }
    save_json(output_dir / "results.json", results)
    print("\nTest metrics:")
    print(json.dumps(test_metrics, indent=2))
    print(f"Peak allocated GPU memory: {peak_vram_mb:.1f} MB")
    return results


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train a baseline or improved 3D U-Net on HipMRI."
    )
    parser.add_argument("--model", choices=("baseline", "improved"), default="improved")
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Result directory (default: results/<model>).",
    )
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--base-channels", type=int, default=16)
    parser.add_argument(
        "--target-shape",
        type=int,
        nargs=3,
        default=(64, 128, 128),
        metavar=("DEPTH", "HEIGHT", "WIDTH"),
    )
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=3710)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda, or cuda:N")
    parser.add_argument(
        "--amp",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use CUDA automatic mixed precision when a GPU is available.",
    )
    parser.add_argument(
        "--max-train-batches",
        type=int,
        default=None,
        help="Debug option: limit training batches per epoch.",
    )
    parser.add_argument(
        "--max-validation-batches",
        type=int,
        default=None,
        help="Debug option: limit validation batches per epoch.",
    )
    parser.add_argument(
        "--max-test-batches",
        type=int,
        default=None,
        help="Debug option: limit test batches.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    if args.epochs <= 0 or args.batch_size <= 0 or args.base_channels <= 0:
        raise ValueError("epochs, batch-size, and base-channels must be positive.")
    if args.patience <= 0:
        raise ValueError("patience must be positive.")

    config = TrainingConfig(
        model=args.model,
        data_root=str(args.data_root),
        output_dir=str(args.output_dir or Path("results") / args.model),
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        base_channels=args.base_channels,
        target_shape=tuple(args.target_shape),
        num_workers=args.num_workers,
        seed=args.seed,
        patience=args.patience,
        amp=args.amp,
    )
    set_reproducibility(config.seed)
    device = select_device(args.device)
    run_training(
        config,
        device,
        maximum_train_batches=args.max_train_batches,
        maximum_validation_batches=args.max_validation_batches,
        maximum_test_batches=args.max_test_batches,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

