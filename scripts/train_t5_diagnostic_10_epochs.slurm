#!/bin/bash --login
# T5-large diagnostic rerun with ten epochs and no early stopping.
# This job preserves epoch-level loss history for later inspection.

#SBATCH --job-name=nlsat-t5-diagnostic
#SBATCH --partition=gpuL
#SBATCH --gpus=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=logs/nlsat-t5-diagnostic-%j.out

set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-$PWD}"
DATA_DIR="${DATA_DIR:-$PROJECT_DIR/data/tagged_splits}"
TRAIN_SCRIPT="${TRAIN_SCRIPT:-$PROJECT_DIR/scripts/train_deberta.py}"
OUTPUT_DIR="$PROJECT_DIR/runs/t5_large_joint_seed42_diagnostic_10ep"

cd "$PROJECT_DIR"
mkdir -p logs "$OUTPUT_DIR"
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1

python "$TRAIN_SCRIPT" \
  --train "$DATA_DIR/train.jsonl" \
  --validation "$DATA_DIR/validation.jsonl" \
  --test "$DATA_DIR/test.jsonl" \
  --output-dir "$OUTPUT_DIR" \
  --model google-t5/t5-large \
  --max-length 256 \
  --batch-size 4 \
  --gradient-accumulation-steps 4 \
  --epochs 10 \
  --early-stopping-patience 0 \
  --learning-rate 1e-5 \
  --weight-decay 0.01 \
  --warmup-ratio 0.1 \
  --seed 42 \
  --num-workers 4
