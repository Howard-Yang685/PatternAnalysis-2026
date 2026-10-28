# 3D Improved U-Net for HipMRI Prostate Segmentation

This project compares a basic slice-wise 2D U-Net, a standard 3D U-Net, and an
improved 3D U-Net on the HipMRI dataset. The first comparison measures the value
of volumetric context; the second isolates the residual and deep-supervision
changes in the improved architecture.

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

The 2D baseline uses the same resized volumes and leakage-free patient split.
Each volume is converted into an ordered batch of axial slices immediately
before the model forward pass, and its predictions are reassembled into a 3D
label volume for the same case-, patient-, and boundary-level evaluation.

```powershell
python train.py --model 2d --data-root F:\COMP3710_Data\HipMRI_Study_open --device cuda --amp --num-workers 2 --output-dir results/2d_baseline --target-shape 64 128 128 --base-channels 16 --augmentation basic --class-weights 1 1 1 1 1 1 --cross-entropy-weight 0.5 --dice-weight 0.5 --selection-metric mean_foreground_dice
python predict.py --checkpoint results\2d_baseline\best_model.pt --data-root F:\COMP3710_Data\HipMRI_Study_open --output-dir results\2d_baseline\inference --device cuda
```

## Planned comparison

- Basic reference: slice-wise 2D U-Net.
- Volumetric baseline: standard 3D U-Net.
- Proposed model: improved 3D U-Net with residual units and deep supervision.
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

