#!/bin/bash
#SBATCH --job-name=hipmri-smoke
#SBATCH --partition=comp3710
#SBATCH --account=comp3710
#SBATCH --gres=gpu
#SBATCH --time=00:10:00
#SBATCH --output=gpu_smoke_%j.out
#SBATCH --error=gpu_smoke_%j.err

set -euo pipefail

source "$HOME/miniconda3/bin/activate"
conda activate torch

cd "$(dirname "$0")"

python -u train.py \
  --model improved \
  --epochs 1 \
  --base-channels 4 \
  --target-shape 16 32 32 \
  --num-workers 0 \
  --device cuda \
  --max-train-batches 1 \
  --max-validation-batches 1 \
  --max-test-batches 1 \
  --output-dir results/gpu_smoke
