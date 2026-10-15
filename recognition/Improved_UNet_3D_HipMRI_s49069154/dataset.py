"""HipMRI NIfTI discovery and dataset auditing utilities.

The training dataset class will be added after this audit has confirmed the
actual volume shapes, voxel spacing, affine alignment, and semantic label IDs.
"""

from __future__ import annotations

import argparse
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence


IMAGE_DIRECTORY = "semantic_MRs"
LABEL_DIRECTORY = "semantic_labels_only"
IMAGE_SUFFIX = "_LFOV.nii.gz"
LABEL_SUFFIX = "_SEMANTIC.nii.gz"
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
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    success = audit_dataset(
        data_root=args.data_root,
        inspect_limit=args.inspect_limit,
        include_label_values=not args.metadata_only,
    )
    return 0 if success else 1


if __name__ == "__main__":
    raise SystemExit(main())

