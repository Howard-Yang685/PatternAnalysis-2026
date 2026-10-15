#!/bin/bash
#SBATCH --job-name=hipmri-train
#SBATCH --partition=comp3710
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --output=train_%j.out
#SBATCH --error=train_%j.err

set -euo pipefail

MODEL="${1:-}"
if [[ "$MODEL" != "baseline" && "$MODEL" != "improved" ]]; then
  echo "Usage: sbatch train_job.sh baseline|improved" >&2
  exit 2
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
  --num-workers 4 \
  --seed 3710 \
  --patience 20 \
  --device cuda \
  --amp \
  --output-dir "results/$MODEL"
