"""Evaluate saved 2D or 3D segmentation models and create report outputs."""

from __future__ import annotations

import argparse
import csv
import json
import statistics
import time
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch
import torch.nn.functional as functional

from dataset import DEFAULT_DATA_ROOT, LABEL_NAMES, build_datasets
from modules import UNet2D, build_model


PROSTATE_CLASS = 5


def select_device(requested_device: str) -> torch.device:
    """Resolve a user device request and fail clearly when CUDA is unavailable."""
    if requested_device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(requested_device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested, but torch.cuda.is_available() is False.")
    return device


def load_checkpoint_model(
    checkpoint_path: Path,
    device: torch.device,
) -> tuple[torch.nn.Module, dict[str, Any]]:
    """Rebuild a model from the training configuration stored in a checkpoint."""
    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
        weights_only=False,
    )
    if "model_state" not in checkpoint or "config" not in checkpoint:
        raise ValueError(
            f"Checkpoint {checkpoint_path} does not contain model_state and config."
        )

    config = checkpoint["config"]
    model = build_model(
        config["model"],
        num_classes=len(LABEL_NAMES),
        base_channels=int(config["base_channels"]),
        deep_supervision=True,
    ).to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    return model, checkpoint


def _synchronise(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def predict_volume(
    model: torch.nn.Module,
    image: torch.Tensor,
    device: torch.device,
    amp_enabled: bool,
) -> tuple[torch.Tensor, float]:
    """Return a CPU label map and model-only inference time in milliseconds."""
    if isinstance(model, UNet2D):
        model_input = image.permute(1, 0, 2, 3).to(device, non_blocking=True)
    else:
        model_input = image.unsqueeze(0).to(device, non_blocking=True)
    _synchronise(device)
    start = time.perf_counter()
    with torch.inference_mode(), torch.autocast(
        device_type=device.type,
        dtype=torch.float16,
        enabled=amp_enabled,
    ):
        outputs = model(model_input)
        logits = outputs[0] if isinstance(outputs, tuple) else outputs
        prediction = logits.argmax(dim=1)
    _synchronise(device)
    elapsed_ms = (time.perf_counter() - start) * 1000.0
    if not isinstance(model, UNet2D):
        prediction = prediction[0]
    return prediction.cpu(), elapsed_ms


def warm_up_model(
    model: torch.nn.Module,
    image: torch.Tensor,
    device: torch.device,
    amp_enabled: bool,
    repetitions: int = 2,
) -> None:
    """Exclude one-time CUDA initialisation from reported latency."""
    if isinstance(model, UNet2D):
        # A small ordered slice batch is sufficient to initialise 2D CUDA kernels.
        model_input = image[:, : min(8, image.shape[1])].permute(1, 0, 2, 3)
        model_input = model_input.to(device)
    else:
        model_input = image.unsqueeze(0).to(device)
    with torch.inference_mode():
        for _ in range(repetitions):
            with torch.autocast(
                device_type=device.type,
                dtype=torch.float16,
                enabled=amp_enabled,
            ):
                model(model_input)
    _synchronise(device)


def binary_overlap(
    prediction: torch.Tensor,
    target: torch.Tensor,
    class_id: int,
) -> tuple[float | None, float | None]:
    """Calculate Dice and IoU, returning None if a class is absent in both maps."""
    predicted_class = prediction == class_id
    target_class = target == class_id
    predicted_count = int(predicted_class.sum())
    target_count = int(target_class.sum())
    if predicted_count + target_count == 0:
        return None, None

    intersection = int(torch.logical_and(predicted_class, target_class).sum())
    dice = (2.0 * intersection) / (predicted_count + target_count)
    union = predicted_count + target_count - intersection
    iou = intersection / union if union else 1.0
    return float(dice), float(iou)


def _mean_present(values: Sequence[float | None]) -> float | None:
    present = [float(value) for value in values if value is not None]
    return float(statistics.fmean(present)) if present else None


def calculate_case_metrics(
    prediction: torch.Tensor,
    target: torch.Tensor,
) -> dict[str, Any]:
    """Calculate classwise and anatomical-end prostate metrics for one case."""
    per_class: dict[str, dict[str, float | None]] = {}
    for class_id, class_name in LABEL_NAMES.items():
        dice, iou = binary_overlap(prediction, target, class_id)
        per_class[class_name] = {"dice": dice, "iou": iou}

    foreground_dice = [
        per_class[LABEL_NAMES[class_id]]["dice"]
        for class_id in LABEL_NAMES
        if class_id != 0
    ]
    foreground_iou = [
        per_class[LABEL_NAMES[class_id]]["iou"]
        for class_id in LABEL_NAMES
        if class_id != 0
    ]
    all_dice = [per_class[name]["dice"] for name in LABEL_NAMES.values()]
    all_iou = [per_class[name]["iou"] for name in LABEL_NAMES.values()]

    prostate_by_depth = torch.any(target == PROSTATE_CLASS, dim=(1, 2))
    prostate_depths = torch.where(prostate_by_depth)[0]
    apex_dice: float | None = None
    base_dice: float | None = None
    if len(prostate_depths) > 0:
        first_depth = int(prostate_depths[0])
        last_depth = int(prostate_depths[-1])
        band_width = max(1, min(3, (last_depth - first_depth + 1) // 3))
        apex_slice = slice(first_depth, first_depth + band_width)
        base_slice = slice(last_depth - band_width + 1, last_depth + 1)
        apex_dice, _ = binary_overlap(
            prediction[apex_slice], target[apex_slice], PROSTATE_CLASS
        )
        base_dice, _ = binary_overlap(
            prediction[base_slice], target[base_slice], PROSTATE_CLASS
        )

    return {
        "mean_dice": _mean_present(all_dice),
        "mean_iou": _mean_present(all_iou),
        "mean_foreground_dice": _mean_present(foreground_dice),
        "mean_foreground_iou": _mean_present(foreground_iou),
        "prostate_apex_dice": apex_dice,
        "prostate_base_dice": base_dice,
        "per_class": per_class,
    }


def save_prediction_nifti(
    prediction: torch.Tensor,
    source_image_path: str,
    output_path: Path,
) -> None:
    """Restore a prediction to the canonical source grid and save it as NIfTI."""
    import nibabel as nib

    source = nib.as_closest_canonical(nib.load(source_image_path))
    target_dhw = tuple(int(value) for value in source.shape[::-1])
    restored_dhw = functional.interpolate(
        prediction[None, None].to(dtype=torch.float32),
        size=target_dhw,
        mode="nearest",
    )[0, 0].to(dtype=torch.uint8)
    restored_xyz = np.transpose(restored_dhw.numpy(), (2, 1, 0))
    header = source.header.copy()
    header.set_data_dtype(np.uint8)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    nib.save(
        nib.Nifti1Image(restored_xyz, source.affine, header),
        str(output_path),
    )


def _rgba_overlay(labels: np.ndarray, alpha: float = 0.55) -> np.ndarray:
    colours = {
        1: (0.12, 0.47, 0.71),
        2: (1.00, 0.50, 0.05),
        3: (0.17, 0.63, 0.17),
        4: (0.84, 0.15, 0.16),
        5: (0.58, 0.40, 0.74),
    }
    overlay = np.zeros((*labels.shape, 4), dtype=np.float32)
    for class_id, colour in colours.items():
        selected = labels == class_id
        overlay[selected, :3] = colour
        overlay[selected, 3] = alpha
    return overlay


def prostate_slice_indices(target: np.ndarray) -> list[tuple[str, int]]:
    """Return inferior/apex, middle, and superior/base prostate slices."""
    depths = np.where(np.any(target == PROSTATE_CLASS, axis=(1, 2)))[0]
    if len(depths) == 0:
        total_depth = target.shape[0]
        return [
            ("Apex", total_depth // 4),
            ("Middle", total_depth // 2),
            ("Base", (3 * total_depth) // 4),
        ]
    return [
        ("Apex", int(depths[0])),
        ("Middle", int(depths[len(depths) // 2])),
        ("Base", int(depths[-1])),
    ]


def save_case_figure(
    image: torch.Tensor,
    target: torch.Tensor,
    prediction: torch.Tensor,
    case_id: str,
    output_path: Path,
) -> None:
    """Create a three-level comparison of MRI, reference, and prediction."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as pyplot
    from matplotlib.patches import Patch

    image_array = image[0].numpy()
    target_array = target.numpy()
    prediction_array = prediction.numpy()
    slices = prostate_slice_indices(target_array)
    finite_values = image_array[np.isfinite(image_array)]
    lower, upper = np.percentile(finite_values, (1, 99))

    figure, axes = pyplot.subplots(3, 3, figsize=(11, 10))
    for row, (level_name, depth) in enumerate(slices):
        mri_slice = image_array[depth]
        target_slice = target_array[depth]
        prediction_slice = prediction_array[depth]
        columns = (
            ("MRI", None),
            ("Ground truth", target_slice),
            ("Prediction", prediction_slice),
        )
        for column, (title, overlay) in enumerate(columns):
            axis = axes[row, column]
            axis.imshow(
                mri_slice,
                cmap="gray",
                vmin=lower,
                vmax=upper,
                origin="lower",
            )
            if overlay is not None:
                axis.imshow(_rgba_overlay(overlay), origin="lower")
            axis.set_title(f"{level_name} - {title}")
            axis.axis("off")

    legend_colours = {
        1: (0.12, 0.47, 0.71),
        2: (1.00, 0.50, 0.05),
        3: (0.17, 0.63, 0.17),
        4: (0.84, 0.15, 0.16),
        5: (0.58, 0.40, 0.74),
    }
    handles = [
        Patch(color=legend_colours[class_id], label=LABEL_NAMES[class_id])
        for class_id in range(1, len(LABEL_NAMES))
    ]
    figure.legend(handles=handles, loc="lower center", ncol=5)
    figure.suptitle(f"{case_id}: prostate apex, middle, and base", fontsize=14)
    figure.tight_layout(rect=(0, 0.05, 1, 0.96))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=180, bbox_inches="tight")
    pyplot.close(figure)


def save_case_csv(records: Sequence[dict[str, Any]], output_path: Path) -> None:
    """Save flat per-case metrics for analysis in a spreadsheet."""
    fieldnames = [
        "case_id",
        "patient_id",
        "inference_ms",
        "mean_dice",
        "mean_iou",
        "mean_foreground_dice",
        "mean_foreground_iou",
        "prostate_apex_dice",
        "prostate_base_dice",
    ]
    for class_name in LABEL_NAMES.values():
        fieldnames.extend((f"dice_{class_name}", f"iou_{class_name}"))

    with output_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        for record in records:
            flat = {
                key: record.get(key)
                for key in fieldnames
                if not key.startswith(("dice_", "iou_"))
            }
            for class_name in LABEL_NAMES.values():
                flat[f"dice_{class_name}"] = record["per_class"][class_name]["dice"]
                flat[f"iou_{class_name}"] = record["per_class"][class_name]["iou"]
            writer.writerow(flat)


def summarise_records(records: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """Aggregate case-level metrics without treating absent classes as perfect."""
    metric_names = (
        "mean_dice",
        "mean_iou",
        "mean_foreground_dice",
        "mean_foreground_iou",
        "prostate_apex_dice",
        "prostate_base_dice",
    )
    means = {
        name: _mean_present([record[name] for record in records])
        for name in metric_names
    }
    class_means: dict[str, dict[str, float | None]] = {}
    for class_name in LABEL_NAMES.values():
        class_means[class_name] = {
            metric: _mean_present(
                [record["per_class"][class_name][metric] for record in records]
            )
            for metric in ("dice", "iou")
        }
    inference_times = [float(record["inference_ms"]) for record in records]
    records_by_patient: dict[str, list[dict[str, Any]]] = {}
    for record in records:
        records_by_patient.setdefault(str(record["patient_id"]), []).append(record)

    patient_metric_means = {
        name: _mean_present(
            [
                _mean_present([record[name] for record in patient_records])
                for patient_records in records_by_patient.values()
            ]
        )
        for name in metric_names
    }
    patient_class_means: dict[str, dict[str, float | None]] = {}
    for class_name in LABEL_NAMES.values():
        patient_class_means[class_name] = {
            metric: _mean_present(
                [
                    _mean_present(
                        [
                            record["per_class"][class_name][metric]
                            for record in patient_records
                        ]
                    )
                    for patient_records in records_by_patient.values()
                ]
            )
            for metric in ("dice", "iou")
        }
    return {
        "num_cases": len(records),
        "num_patients": len(records_by_patient),
        "mean_inference_ms": float(statistics.fmean(inference_times)),
        "median_inference_ms": float(statistics.median(inference_times)),
        "mean_case_metrics": means,
        "mean_case_per_class": class_means,
        "mean_patient_metrics": patient_metric_means,
        "mean_patient_per_class": patient_class_means,
    }


def select_diverse_worst_cases(
    ranked_records: Sequence[dict[str, Any]],
    count: int,
) -> list[dict[str, Any]]:
    """Select low-scoring cases while representing different patients first."""
    selected: list[dict[str, Any]] = []
    selected_indices: set[int] = set()
    represented_patients: set[str] = set()
    for record in ranked_records:
        patient_id = str(record["patient_id"])
        if patient_id in represented_patients:
            continue
        selected.append(record)
        selected_indices.add(int(record["dataset_index"]))
        represented_patients.add(patient_id)
        if len(selected) == count:
            return selected

    for record in ranked_records:
        dataset_index = int(record["dataset_index"])
        if dataset_index in selected_indices:
            continue
        selected.append(record)
        if len(selected) == count:
            break
    return selected


def run_inference(
    checkpoint_path: Path,
    data_root: Path,
    output_dir: Path,
    split_name: str,
    device: torch.device,
    maximum_cases: int | None,
    num_visualisations: int,
    save_nifti: bool,
) -> dict[str, Any]:
    """Run evaluation, save predictions, and create worst-case figures."""
    output_dir.mkdir(parents=True, exist_ok=True)
    model, checkpoint = load_checkpoint_model(checkpoint_path, device)
    config = checkpoint["config"]
    target_shape = tuple(int(value) for value in config["target_shape"])
    datasets = build_datasets(
        data_root,
        target_shape=target_shape,
        seed=int(config["seed"]),
    )
    dataset = datasets[split_name]
    case_count = len(dataset) if maximum_cases is None else min(len(dataset), maximum_cases)
    if case_count <= 0:
        raise ValueError("No cases were selected for inference.")

    amp_enabled = bool(config.get("amp", True)) and device.type == "cuda"
    warm_up_model(model, dataset[0]["image"], device, amp_enabled)
    records: list[dict[str, Any]] = []

    prediction_directory = output_dir / "predictions"
    for index in range(case_count):
        sample = dataset[index]
        prediction, inference_ms = predict_volume(
            model,
            sample["image"],
            device,
            amp_enabled,
        )
        metrics = calculate_case_metrics(prediction, sample["label"])
        record = {
            "case_id": sample["case_id"],
            "patient_id": sample["patient_id"],
            "dataset_index": index,
            "inference_ms": inference_ms,
            **metrics,
        }
        records.append(record)

        if save_nifti:
            save_prediction_nifti(
                prediction,
                sample["image_path"],
                prediction_directory / f"{sample['case_id']}_PREDICTION.nii.gz",
            )
        print(
            f"[{index + 1:02d}/{case_count:02d}] {sample['case_id']} | "
            f"prostate Dice {metrics['per_class']['prostate']['dice']!s} | "
            f"{inference_ms:.1f} ms"
        )

    save_case_csv(records, output_dir / "case_metrics.csv")
    summary = summarise_records(records)
    summary.update(
        {
            "checkpoint": str(checkpoint_path),
            "model": config["model"],
            "input_mode": "2D axial slices" if isinstance(model, UNet2D) else "3D volumes",
            "checkpoint_epoch": int(checkpoint["epoch"]),
            "split": split_name,
            "device": str(device),
            "target_shape": list(target_shape),
        }
    )

    ranked_records = sorted(
        records,
        key=lambda record: (
            record["per_class"]["prostate"]["dice"] is None,
            record["per_class"]["prostate"]["dice"]
            if record["per_class"]["prostate"]["dice"] is not None
            else float("inf"),
        ),
    )
    selected_records = select_diverse_worst_cases(
        ranked_records,
        min(num_visualisations, len(ranked_records)),
    )
    qualitative_directory = output_dir / "qualitative"
    for record in selected_records:
        sample = dataset[int(record["dataset_index"])]
        prediction, _ = predict_volume(
            model,
            sample["image"],
            device,
            amp_enabled,
        )
        prostate_dice = record["per_class"]["prostate"]["dice"]
        score_text = "absent" if prostate_dice is None else f"{prostate_dice:.3f}"
        save_case_figure(
            sample["image"],
            sample["label"],
            prediction,
            sample["case_id"],
            qualitative_directory
            / f"{sample['case_id']}_prostate_dice_{score_text}.png",
        )

    summary["qualitative_cases"] = [
        {
            "case_id": record["case_id"],
            "prostate_dice": record["per_class"]["prostate"]["dice"],
            "prostate_apex_dice": record["prostate_apex_dice"],
            "prostate_base_dice": record["prostate_base_dice"],
        }
        for record in selected_records
    ]
    (output_dir / "inference_summary.json").write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )
    print("\nInference summary:")
    print(json.dumps(summary, indent=2))
    return summary


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Evaluate a trained HipMRI model and create qualitative figures."
    )
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--split", choices=("validation", "test"), default="test")
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda, or cuda:N")
    parser.add_argument("--max-cases", type=int, default=None)
    parser.add_argument("--num-visualisations", type=int, default=5)
    parser.add_argument(
        "--save-nifti",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Save predictions restored to the canonical source-image grid.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    if args.max_cases is not None and args.max_cases <= 0:
        raise ValueError("max-cases must be positive when provided.")
    if args.num_visualisations < 0:
        raise ValueError("num-visualisations cannot be negative.")
    device = select_device(args.device)
    print(f"Device: {device}")
    run_inference(
        checkpoint_path=args.checkpoint,
        data_root=args.data_root,
        output_dir=args.output_dir,
        split_name=args.split,
        device=device,
        maximum_cases=args.max_cases,
        num_visualisations=args.num_visualisations,
        save_nifti=args.save_nifti,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

