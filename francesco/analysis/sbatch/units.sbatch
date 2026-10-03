#!/bin/bash
#SBATCH --job-name=probe-units
#SBATCH --partition=gpu_h100
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --time=00:30:00
#SBATCH --output=logs/slurm-%x-%j.out
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
VENV=venv-quant
source scripts/snellius/env.sh
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True CUDA_DEVICE_ORDER=PCI_BUS_ID
BATCH="${BATCH:-80}"
python francesco/analysis/measure_units.py --batch "$BATCH" --reps "${REPS:-20}"
