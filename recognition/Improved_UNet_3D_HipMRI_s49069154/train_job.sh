#!/bin/bash
#SBATCH --job-name=hipmri-train
#SBATCH --partition=comp3710
#SBATCH --account=comp3710
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --output=train_%j.out
#SBATCH --error=train_%j.err

set -euo pipefail

MODEL="${1:-}"
if [[ "$MODEL" != "2d" && "$MODEL" != "baseline" && "$MODEL" != "improved" ]]; then
  echo "Usage: sbatch train_job.sh 2d|baseline|improved" >&2
  exit 2
fi

if [[ "$MODEL" == "2d" ]]; then
  OUTPUT_DIR="results/2d_baseline"
  NUM_WORKERS=2
else
  OUTPUT_DIR="results/${MODEL}"
  NUM_WORKERS=0
fi

source "$HOME/miniconda3/bin/activate"
conda activate torch

cd "$(dirname "$0")"

python -u train.py \
  --model "$MODEL" \
  --epochs 100 \
  --batch-size 1 \
  --learning-rate 3e-4 \
  --weight-decay 1e-5 \
  --base-channels 16 \
  --target-shape 64 128 128 \
  --num-workers "$NUM_WORKERS" \
  --seed 3710 \
  --patience 20 \
  --class-weights 1 1 1 1 1 1 \
  --cross-entropy-weight 0.5 \
  --dice-weight 0.5 \
  --device cuda \
  --amp \
  --selection-metric mean_foreground_dice \
  --augmentation basic \
  --output-dir "$OUTPUT_DIR"
