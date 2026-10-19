# 3D Improved U-Net for HipMRI Prostate Segmentation

This project investigates whether an improved three-dimensional U-Net can provide clinically meaningful prostate boundary improvements over a standard 3D U-Net baseline on the HipMRI dataset.

## Enhanced training configuration

The enhanced experiment retains more boundary detail by resizing volumes to
`80 x 128 x 128` rather than `64 x 128 x 128`. Training uses mild
anatomy-preserving affine and MRI intensity augmentation. The combined loss is
60% foreground soft Dice and 40% class-weighted cross-entropy, with larger
weights for the under-represented rectum and prostate. Checkpoints are selected
using validation prostate Dice; the test partition is not used for model
selection. The added depth resolution specifically retains more base/apex
detail. The enhanced run uses 12 base feature channels so that the extra
spatial detail fits reliably on a 12 GB GPU.

The earlier results are preserved by writing the new run to
`results/improved_enhanced` by default.

Example local training command (PowerShell):

```powershell
python train.py --model improved --data-root F:\COMP3710_Data\HipMRI_Study_open --device cuda --amp --num-workers 2 --output-dir results/improved_enhanced
```

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

