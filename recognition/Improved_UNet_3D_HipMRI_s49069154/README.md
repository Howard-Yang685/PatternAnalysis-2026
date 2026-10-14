# 3D Improved U-Net for HipMRI Prostate Segmentation

This project investigates whether an improved three-dimensional U-Net can provide clinically meaningful prostate boundary improvements over a standard 3D U-Net baseline on the HipMRI dataset.

## Project status

The repository structure has been initialised. The next milestone is to audit the NIfTI images and labels, verify image-mask alignment, and construct a reproducible patient-level train/validation/test split.

## Planned comparison

- Baseline: standard 3D U-Net.
- Proposed model: 3D Improved U-Net with residual units and deep supervision.
- Primary metrics: per-class Dice similarity coefficient and Intersection over Union.
- Efficiency measures: parameter count, peak GPU memory, and per-volume inference time.
- Qualitative analysis: 3-5 representative boundary failures, with particular attention to prostate base and apex slices.

## Repository structure

```text
modules.py   Model components and network definitions.
dataset.py   NIfTI loading, preprocessing, and patient-level data splitting.
train.py     Training, validation, testing, and checkpoint management.
predict.py   Saved-model inference and segmentation visualisation.
```

## Reproducibility

Exact dependency versions, data preparation instructions, random seeds, experiment configurations, and execution commands will be documented as the implementation is developed.

