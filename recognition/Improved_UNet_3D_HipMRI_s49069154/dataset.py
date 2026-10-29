"""HipMRI NIfTI discovery, auditing, and patient-level splitting utilities.

The split is performed by patient rather than by scan so that repeated weekly
examinations from one person can never leak across dataset partitions.
"""

from __future__ import annotations

import argparse
import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

try:
    from torch.utils.data import Dataset
except ImportError:  # Allows the lightweight audit CLI to run without PyTorch.
    class Dataset:  # type: ignore[no-redef]
        """Fallback base class used only when PyTorch is unavailable."""


IMAGE_DIRECTORY = "semantic_MRs"
LABEL_DIRECTORY = "semantic_labels_only"
IMAGE_SUFFIX = "_LFOV.nii.gz"
LABEL_SUFFIX = "_SEMANTIC.nii.gz"
LABEL_NAMES = {
    0: "background",
    1: "bladder",
    2: "body",
    3: "bone",
    4: "rectum",
    5: "prostate",
}
# Input shape used by the three controlled experiments reported in the README
# and PDF.
DEFAULT_TARGET_SHAPE = (64, 128, 128)  # (depth, height, width)
DEFAULT_DATA_ROOT = Path(
    os.environ.get(
        "HIPMRI_DATA_ROOT",
        "/home/groups/comp3710/HipMRI_Study_open",
    )
)


@dataclass(frozen=True)
class VolumePair:
    """Paths to one MRI volume and its corresponding semantic label volume."""

    case_id: str
    patient_id: str
    image_path: Path
    label_path: Path


@dataclass(frozen=True)
class DatasetSplit:
    """Leakage-free train, validation, and test volume assignments."""

    train: tuple[VolumePair, ...]
    validation: tuple[VolumePair, ...]
    test: tuple[VolumePair, ...]


def _case_id(path: Path, suffix: str) -> str:
    """Remove a known HipMRI suffix without mis-handling ``.nii.gz``."""
    if not path.name.endswith(suffix):
        raise ValueError(f"Unexpected filename '{path.name}'; expected '*{suffix}'.")
    return path.name[: -len(suffix)]


def _patient_id(case_id: str) -> str:
    """Extract the patient identifier from names such as ``B031_Week0``."""
    patient_id, separator, _ = case_id.partition("_Week")
    if not separator or not patient_id:
        raise ValueError(
            f"Case ID '{case_id}' does not follow the expected '<patient>_Week<visit>' format."
        )
    return patient_id


def _index_volumes(directory: Path, suffix: str) -> dict[str, Path]:
    """Index recursively discovered volumes by patient-and-visit case ID."""
    if not directory.is_dir():
        raise FileNotFoundError(f"Required directory was not found: {directory}")

    index: dict[str, Path] = {}
    for path in sorted(directory.rglob(f"*{suffix}")):
        case_id = _case_id(path, suffix)
        if case_id in index:
            raise ValueError(
                f"Duplicate case ID '{case_id}' in {directory}: "
                f"'{index[case_id]}' and '{path}'."
            )
        index[case_id] = path
    return index


def discover_volume_pairs(
    data_root: str | Path,
) -> tuple[list[VolumePair], list[str], list[str]]:
    """Discover MRI/label pairs using the shared patient-and-visit prefix.

    Returns:
        A tuple containing matched pairs, case IDs missing labels, and case IDs
        missing MRI volumes.
    """
    root = Path(data_root).expanduser()
    images = _index_volumes(root / IMAGE_DIRECTORY, IMAGE_SUFFIX)
    labels = _index_volumes(root / LABEL_DIRECTORY, LABEL_SUFFIX)

    paired_ids = sorted(images.keys() & labels.keys())
    missing_labels = sorted(images.keys() - labels.keys())
    missing_images = sorted(labels.keys() - images.keys())
    pairs = [
        VolumePair(
            case_id=case_id,
            patient_id=_patient_id(case_id),
            image_path=images[case_id],
            label_path=labels[case_id],
        )
        for case_id in paired_ids
    ]
    return pairs, missing_labels, missing_images


def _partition_counts(total: int, fractions: Sequence[float]) -> list[int]:
    """Convert fractions into counts while preserving the requested total."""
    raw_counts = [total * fraction for fraction in fractions]
    counts = [int(value) for value in raw_counts]
    unassigned = total - sum(counts)
    remainder_order = sorted(
        range(len(fractions)),
        key=lambda index: (raw_counts[index] - counts[index], -index),
        reverse=True,
    )
    for index in remainder_order[:unassigned]:
        counts[index] += 1
    return counts


def split_by_patient(
    pairs: Sequence[VolumePair],
    fractions: tuple[float, float, float] = (0.70, 0.15, 0.15),
    seed: int = 3710,
) -> DatasetSplit:
    """Create deterministic train/validation/test splits without patient leakage."""
    import random

    if not pairs:
        raise ValueError("Cannot split an empty dataset.")
    if any(fraction <= 0 for fraction in fractions):
        raise ValueError("All split fractions must be greater than zero.")
    if abs(sum(fractions) - 1.0) > 1e-8:
        raise ValueError("Train, validation, and test fractions must sum to 1.0.")

    patient_ids = sorted({pair.patient_id for pair in pairs})
    if len(patient_ids) < 3:
        raise ValueError("At least three patients are required for three dataset splits.")

    random.Random(seed).shuffle(patient_ids)
    train_count, validation_count, test_count = _partition_counts(
        len(patient_ids), fractions
    )
    if min(train_count, validation_count, test_count) == 0:
        raise ValueError("The requested fractions leave one dataset split empty.")

    train_end = train_count
    validation_end = train_end + validation_count
    split_patients = {
        "train": set(patient_ids[:train_end]),
        "validation": set(patient_ids[train_end:validation_end]),
        "test": set(patient_ids[validation_end : validation_end + test_count]),
    }

    overlap = (
        (split_patients["train"] & split_patients["validation"])
        | (split_patients["train"] & split_patients["test"])
        | (split_patients["validation"] & split_patients["test"])
    )
    if overlap:
        raise RuntimeError(f"Patient leakage detected: {sorted(overlap)}")

    def select(patient_set: set[str]) -> tuple[VolumePair, ...]:
        return tuple(pair for pair in pairs if pair.patient_id in patient_set)

    return DatasetSplit(
        train=select(split_patients["train"]),
        validation=select(split_patients["validation"]),
        test=select(split_patients["test"]),
    )


def print_split_summary(split: DatasetSplit) -> None:
    """Print patient and volume counts for a dataset split."""
    print("\nPatient-level split:")
    for name, subset in (
        ("train", split.train),
        ("validation", split.validation),
        ("test", split.test),
    ):
        patient_ids = {pair.patient_id for pair in subset}
        print(f"  {name}: {len(patient_ids)} patients, {len(subset)} volumes")


def _load_nifti_pair(pair: VolumePair) -> tuple[Any, Any, tuple[float, ...]]:
    """Load one aligned NIfTI pair and return arrays in depth-height-width order."""
    try:
        import nibabel as nib
        import numpy as np
    except ImportError as exc:
        raise RuntimeError(
            "Loading HipMRI volumes requires nibabel and numpy. Install them with: "
            "pip install nibabel numpy"
        ) from exc

    image_nifti = nib.as_closest_canonical(nib.load(str(pair.image_path)))
    label_nifti = nib.as_closest_canonical(nib.load(str(pair.label_path)))
    if image_nifti.shape != label_nifti.shape:
        raise ValueError(
            f"Shape mismatch for {pair.case_id}: "
            f"{image_nifti.shape} vs {label_nifti.shape}."
        )
    if not np.allclose(image_nifti.affine, label_nifti.affine, atol=1e-4):
        raise ValueError(f"Affine mismatch for {pair.case_id}.")

    image_xyz = image_nifti.get_fdata(dtype=np.float32)
    label_xyz = np.asanyarray(label_nifti.dataobj)
    rounded_labels = np.rint(label_xyz)
    if not np.allclose(label_xyz, rounded_labels, atol=1e-6):
        raise ValueError(f"Non-integer semantic labels found in {pair.case_id}.")

    label_xyz = rounded_labels.astype(np.int64, copy=False)
    unexpected_labels = sorted(set(np.unique(label_xyz).tolist()) - set(LABEL_NAMES))
    if unexpected_labels:
        raise ValueError(
            f"Unexpected semantic labels in {pair.case_id}: {unexpected_labels}."
        )

    # Nibabel uses (X, Y, Z); Conv3d expects spatial dimensions (D, H, W).
    image_dhw = np.ascontiguousarray(np.transpose(image_xyz, (2, 1, 0)))
    label_dhw = np.ascontiguousarray(np.transpose(label_xyz, (2, 1, 0)))
    spacing_xyz = tuple(float(value) for value in image_nifti.header.get_zooms()[:3])
    spacing_dhw = spacing_xyz[::-1]
    return image_dhw, label_dhw, spacing_dhw


def _normalise_mri(image: Any) -> Any:
    """Apply per-volume z-score normalisation while preserving zero padding."""
    import numpy as np

    image = image.copy()
    finite = np.isfinite(image)
    foreground = finite & (image != 0)
    normalisation_region = foreground if np.any(foreground) else finite
    if not np.any(normalisation_region):
        raise ValueError("MRI volume contains no finite voxels.")

    mean = float(image[normalisation_region].mean())
    standard_deviation = float(image[normalisation_region].std())
    if standard_deviation < 1e-8:
        raise ValueError("MRI intensity standard deviation is too close to zero.")

    image[normalisation_region] = (
        image[normalisation_region] - mean
    ) / standard_deviation
    image[~finite] = 0.0
    image[~foreground] = 0.0
    return image


def _resize_volume(
    image: Any,
    label: Any,
    target_shape: tuple[int, int, int] | None,
) -> tuple[Any, Any]:
    """Convert arrays to tensors and optionally resize image and label safely."""
    try:
        import torch
        import torch.nn.functional as functional
    except ImportError as exc:
        raise RuntimeError(
            "Training data loading requires PyTorch. Install it with: "
            "pip install torch torchvision"
        ) from exc

    image_tensor = torch.from_numpy(image).to(dtype=torch.float32)[None, ...]
    label_tensor = torch.from_numpy(label).to(dtype=torch.long)
    if target_shape is None or tuple(image_tensor.shape[1:]) == target_shape:
        return image_tensor, label_tensor
    if len(target_shape) != 3 or any(dimension <= 0 for dimension in target_shape):
        raise ValueError("target_shape must contain three positive dimensions.")

    image_tensor = functional.interpolate(
        image_tensor[None, ...],
        size=target_shape,
        mode="trilinear",
        align_corners=False,
    )[0]
    label_tensor = functional.interpolate(
        label_tensor[None, None, ...].to(dtype=torch.float32),
        size=target_shape,
        mode="nearest",
    )[0, 0].to(dtype=torch.long)
    return image_tensor, label_tensor


def _augment_training_pair(image: Any, label: Any) -> tuple[Any, Any]:
    """Apply mild anatomy-preserving spatial and MRI intensity augmentation."""
    import torch
    import torch.nn.functional as functional

    # After canonical orientation and conversion to (D, H, W), the final axis
    # is left-right. Other anatomical flips would be implausible.
    if bool(torch.rand(()) < 0.5):
        image = torch.flip(image, dims=(3,))
        label = torch.flip(label, dims=(2,))

    # Small rotations, translations, and isotropic scaling model acquisition
    # and positioning variation without changing the anatomy drastically.
    if bool(torch.rand(()) < 0.35):
        maximum_angle = math.radians(6.0)
        angles = [
            float((2.0 * torch.rand(()) - 1.0) * maximum_angle)
            for _ in range(3)
        ]
        sin_x, sin_y, sin_z = (math.sin(angle) for angle in angles)
        cos_x, cos_y, cos_z = (math.cos(angle) for angle in angles)
        rotation_x = torch.tensor(
            [[1.0, 0.0, 0.0], [0.0, cos_x, -sin_x], [0.0, sin_x, cos_x]],
            dtype=image.dtype,
        )
        rotation_y = torch.tensor(
            [[cos_y, 0.0, sin_y], [0.0, 1.0, 0.0], [-sin_y, 0.0, cos_y]],
            dtype=image.dtype,
        )
        rotation_z = torch.tensor(
            [[cos_z, -sin_z, 0.0], [sin_z, cos_z, 0.0], [0.0, 0.0, 1.0]],
            dtype=image.dtype,
        )
        scale = float(0.95 + 0.10 * torch.rand(()))
        theta = torch.zeros((1, 3, 4), dtype=image.dtype)
        theta[0, :, :3] = (rotation_z @ rotation_y @ rotation_x) * scale
        theta[0, :, 3] = 0.08 * torch.rand(3, dtype=image.dtype) - 0.04
        grid = functional.affine_grid(
            theta,
            size=(1, 1, *image.shape[1:]),
            align_corners=False,
        )
        image = functional.grid_sample(
            image[None, ...],
            grid,
            mode="bilinear",
            padding_mode="zeros",
            align_corners=False,
        )[0]
        label = functional.grid_sample(
            label[None, None, ...].to(dtype=image.dtype),
            grid,
            mode="nearest",
            padding_mode="zeros",
            align_corners=False,
        )[0, 0].to(dtype=torch.long)

    # Scanner-dependent contrast and noise variation. Keep zero padding at zero.
    foreground = image != 0
    intensity_scale = 0.90 + 0.20 * torch.rand(())
    intensity_shift = 0.20 * torch.rand(()) - 0.10
    noise = torch.randn_like(image) * 0.03
    augmented = image * intensity_scale + intensity_shift + noise
    image = torch.where(foreground, augmented, image)
    return image.contiguous(), label.contiguous()


def _augment_basic_pair(image: Any, label: Any) -> tuple[Any, Any]:
    """Apply the left-right flip used by the original 3D comparison runs."""
    import torch

    if bool(torch.rand(()) < 0.5):
        image = torch.flip(image, dims=(3,))
        label = torch.flip(label, dims=(2,))
    return image.contiguous(), label.contiguous()


class HipMRIDataset(Dataset):
    """PyTorch-compatible dataset for aligned 3D HipMRI semantic segmentation."""

    def __init__(
        self,
        pairs: Sequence[VolumePair],
        target_shape: tuple[int, int, int] | None = DEFAULT_TARGET_SHAPE,
        augment: bool = False,
        augmentation_mode: str = "basic",
    ) -> None:
        if not pairs:
            raise ValueError("HipMRIDataset requires at least one volume pair.")
        self.pairs = tuple(pairs)
        self.target_shape = target_shape
        self.augment = augment
        if augmentation_mode not in {"basic", "enhanced"}:
            raise ValueError("augmentation_mode must be 'basic' or 'enhanced'.")
        self.augmentation_mode = augmentation_mode

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, index: int) -> dict[str, Any]:
        import torch

        pair = self.pairs[index]
        image, label, spacing = _load_nifti_pair(pair)
        image = _normalise_mri(image)
        image_tensor, label_tensor = _resize_volume(
            image, label, self.target_shape
        )

        if self.augment:
            if self.augmentation_mode == "basic":
                image_tensor, label_tensor = _augment_basic_pair(
                    image_tensor, label_tensor
                )
            else:
                image_tensor, label_tensor = _augment_training_pair(
                    image_tensor, label_tensor
                )

        return {
            "image": image_tensor,
            "label": label_tensor,
            "case_id": pair.case_id,
            "patient_id": pair.patient_id,
            "spacing_dhw": spacing,
            "image_path": str(pair.image_path),
            "label_path": str(pair.label_path),
        }


def build_datasets(
    data_root: str | Path,
    target_shape: tuple[int, int, int] | None = DEFAULT_TARGET_SHAPE,
    seed: int = 3710,
    augmentation_mode: str = "basic",
) -> dict[str, HipMRIDataset]:
    """Build leakage-free training, validation, and test datasets."""
    pairs, missing_labels, missing_images = discover_volume_pairs(data_root)
    if missing_labels or missing_images:
        raise ValueError(
            "Cannot build datasets while unmatched MRI or label files remain."
        )

    split = split_by_patient(pairs, seed=seed)
    return {
        "train": HipMRIDataset(
            split.train,
            target_shape=target_shape,
            augment=True,
            augmentation_mode=augmentation_mode,
        ),
        "validation": HipMRIDataset(
            split.validation, target_shape=target_shape, augment=False
        ),
        "test": HipMRIDataset(
            split.test, target_shape=target_shape, augment=False
        ),
    }


def print_sample_summary(sample: dict[str, Any]) -> None:
    """Print a compact smoke-test summary for one loaded training sample."""
    import torch

    image = sample["image"]
    label = sample["label"]
    print("\nData-loader smoke test:")
    print(f"  case: {sample['case_id']}")
    print(f"  patient: {sample['patient_id']}")
    print(f"  image tensor: shape={tuple(image.shape)}, dtype={image.dtype}")
    print(f"  label tensor: shape={tuple(label.shape)}, dtype={label.dtype}")
    print(f"  image mean/std: {float(image.mean()):.4f} / {float(image.std()):.4f}")
    print(f"  labels after resize: {torch.unique(label).tolist()}")


def _inspect_pair(pair: VolumePair, include_label_values: bool) -> dict[str, object]:
    """Read NIfTI metadata and optionally inspect the semantic label values."""
    try:
        import nibabel as nib
        import numpy as np
    except ImportError as exc:
        raise RuntimeError(
            "Dataset inspection requires nibabel and numpy. Install them in the "
            "active environment with: pip install nibabel numpy"
        ) from exc

    image = nib.load(str(pair.image_path))
    label = nib.load(str(pair.label_path))
    image_shape = tuple(int(value) for value in image.shape)
    label_shape = tuple(int(value) for value in label.shape)
    image_spacing = tuple(round(float(value), 5) for value in image.header.get_zooms())
    label_spacing = tuple(round(float(value), 5) for value in label.header.get_zooms())

    result: dict[str, object] = {
        "case_id": pair.case_id,
        "image_shape": image_shape,
        "label_shape": label_shape,
        "image_spacing": image_spacing,
        "label_spacing": label_spacing,
        "shape_match": image_shape == label_shape,
        "affine_match": bool(np.allclose(image.affine, label.affine, atol=1e-4)),
    }

    if include_label_values:
        label_data = np.asanyarray(label.dataobj)
        result["label_values"] = np.unique(label_data).tolist()
        result["foreground_voxels"] = int(np.count_nonzero(label_data))

    return result


def audit_dataset(
    data_root: str | Path,
    inspect_limit: int = 5,
    include_label_values: bool = True,
) -> bool:
    """Audit pairing and inspect a representative set of NIfTI volumes.

    Returns ``True`` when all filenames pair and all inspected image/label
    volumes have matching shapes and affine transforms.
    """
    if inspect_limit < 0:
        raise ValueError("inspect_limit must be zero or greater.")

    root = Path(data_root).expanduser()
    pairs, missing_labels, missing_images = discover_volume_pairs(root)
    patient_count = len({pair.patient_id for pair in pairs})

    print(f"Data root: {root}")
    print(f"Matched cases: {len(pairs)}")
    print(f"Unique patients: {patient_count}")
    print(f"MRI volumes without labels: {len(missing_labels)}")
    print(f"Labels without MRI volumes: {len(missing_images)}")
    print("Semantic labels:")
    for label_id, name in LABEL_NAMES.items():
        print(f"  {label_id}: {name}")

    if missing_labels:
        print("Missing labels (first 10):", ", ".join(missing_labels[:10]))
    if missing_images:
        print("Missing MRI volumes (first 10):", ", ".join(missing_images[:10]))

    inspected_ok = True
    pairs_to_inspect = pairs if inspect_limit == 0 else pairs[:inspect_limit]
    print(f"\nInspecting {len(pairs_to_inspect)} paired volume(s):")
    for pair in pairs_to_inspect:
        result = _inspect_pair(pair, include_label_values)
        print(f"\n[{result['case_id']}]")
        print(
            f"  image shape/spacing: {result['image_shape']} / "
            f"{result['image_spacing']}"
        )
        print(
            f"  label shape/spacing: {result['label_shape']} / "
            f"{result['label_spacing']}"
        )
        print(f"  shape match: {result['shape_match']}")
        print(f"  affine match: {result['affine_match']}")
        if include_label_values:
            print(f"  label values: {result['label_values']}")
            print(f"  foreground voxels: {result['foreground_voxels']}")

        inspected_ok &= bool(result["shape_match"] and result["affine_match"])

    pairing_ok = not missing_labels and not missing_images and bool(pairs)
    audit_ok = pairing_ok and inspected_ok
    print(f"\nAudit status: {'PASS' if audit_ok else 'CHECK REQUIRED'}")
    return audit_ok


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Audit HipMRI 3D MRI and semantic-label NIfTI files."
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=DEFAULT_DATA_ROOT,
        help=(
            "HipMRI_Study_open directory. Defaults to HIPMRI_DATA_ROOT or the "
            "COMP3710 Rangpur dataset path."
        ),
    )
    parser.add_argument(
        "--inspect-limit",
        type=int,
        default=5,
        help="Number of matched cases to inspect; use 0 to inspect every case.",
    )
    parser.add_argument(
        "--metadata-only",
        action="store_true",
        help="Skip loading label arrays and only compare NIfTI metadata.",
    )
    parser.add_argument(
        "--show-split",
        action="store_true",
        help="Print the deterministic patient-level train/validation/test split.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=3710,
        help="Random seed used for the patient-level split.",
    )
    parser.add_argument(
        "--smoke-test",
        action="store_true",
        help="Load and preprocess one training case to verify the data pipeline.",
    )
    parser.add_argument(
        "--target-shape",
        type=int,
        nargs=3,
        default=DEFAULT_TARGET_SHAPE,
        metavar=("DEPTH", "HEIGHT", "WIDTH"),
        help=(
            "Downsampled Conv3d input shape "
            f"(default: {' '.join(map(str, DEFAULT_TARGET_SHAPE))})."
        ),
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    success = audit_dataset(
        data_root=args.data_root,
        inspect_limit=args.inspect_limit,
        include_label_values=not args.metadata_only,
    )
    if args.show_split or args.smoke_test:
        pairs, _, _ = discover_volume_pairs(args.data_root)
        split = split_by_patient(pairs, seed=args.seed)
        print_split_summary(split)
        if args.smoke_test:
            dataset = HipMRIDataset(
                split.train,
                target_shape=tuple(args.target_shape),
                augment=False,
            )
            print_sample_summary(dataset[0])
    return 0 if success else 1


if __name__ == "__main__":
    raise SystemExit(main())

