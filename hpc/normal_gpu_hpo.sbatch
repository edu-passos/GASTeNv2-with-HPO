#!/bin/bash
#SBATCH -p normal
#SBATCH --qos=gpu_batch
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=15:50:00
#SBATCH --job-name=gasten-hpo
#SBATCH --output=slurm/%x-%j.out

set -euo pipefail

PROJECT_ROOT="${SLURM_SUBMIT_DIR:-$(pwd)}"
cd "$PROJECT_ROOT"
mkdir -p slurm

if [[ -f .env ]]; then
  set -a
  source .env
  set +a
fi

: "${FILESDIR:?Set FILESDIR in .env or export it before submitting the job.}"

if [[ -n "${ACTIVATE_CMD:-}" ]]; then
  eval "$ACTIVATE_CMD"
fi

export PYTHONUNBUFFERED=1
export WANDB_MODE="${WANDB_MODE:-offline}"

CONFIG="${CONFIG:-experiments/chest-xray.yml}"
STEP1_TRIALS="${STEP1_TRIALS:-5}"
STEP1_WALLTIME="${STEP1_WALLTIME:-18000}"
STEP2_TRIALS="${STEP2_TRIALS:-5}"
STEP2_WALLTIME="${STEP2_WALLTIME:-18000}"

nvidia-smi || true

python3 -m src.optimization \
  --config "$CONFIG" \
  --trials "$STEP1_TRIALS" \
  --walltime "$STEP1_WALLTIME" \
  --step2-trials "$STEP2_TRIALS" \
  --step2-walltime "$STEP2_WALLTIME"
