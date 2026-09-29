#!/bin/bash --login
# Separate T5-large experiments for S, W, and V using the submitted array setup.
# Set PROJECT_DIR and DATA_DIR for the target cluster before submission.

#SBATCH --job-name=nlsat-t5-separate
#SBATCH --partition=gpuL
#SBATCH --gpus=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --array=0-5
#SBATCH --output=logs/nlsat-t5-separate-%A_%a.out

set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-$PWD}"
DATA_ROOT="${DATA_ROOT:-$PROJECT_DIR/data/tagged_fragment_splits}"
TRAIN_SCRIPT="${TRAIN_SCRIPT:-$PROJECT_DIR/train_deberta.py}"
FRAGMENTS=(S W V)
SEEDS=(42 1337 2041)
FRAGMENT_INDEX=$((SLURM_ARRAY_TASK_ID % 3))
SEED_INDEX=$((SLURM_ARRAY_TASK_ID / 3))
FRAGMENT="${FRAGMENTS[$FRAGMENT_INDEX]}"
SEED="${SEEDS[$SEED_INDEX]}"
DATA_DIR="$DATA_ROOT/$FRAGMENT"
OUTPUT_DIR="$PROJECT_DIR/runs/t5_large_${FRAGMENT}_only_seed${SEED}"

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
  --epochs 5 \
  --early-stopping-patience 2 \
  --learning-rate 1e-5 \
  --weight-decay 0.01 \
  --warmup-ratio 0.1 \
  --seed "$SEED" \
  --num-workers 4
