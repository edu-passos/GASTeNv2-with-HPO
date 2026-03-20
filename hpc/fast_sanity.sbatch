#!/bin/bash
#SBATCH -p fast
#SBATCH --qos=cpu
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:20:00
#SBATCH --job-name=gasten-sanity
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

CONFIG="${CONFIG:-experiments/chest-xray.yml}"
python3 scripts/hpc_sanity_check.py --config "$CONFIG" --device cpu ${SANITY_ARGS:-}
