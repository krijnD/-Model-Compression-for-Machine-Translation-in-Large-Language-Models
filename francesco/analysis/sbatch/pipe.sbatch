#!/bin/bash
#SBATCH --job-name=probe-pipe
#SBATCH --partition=gpu_h100
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --time=00:40:00
#SBATCH --output=logs/slurm-%x-%j.out
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
VENV=venv-quant
source scripts/snellius/env.sh
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True CUDA_DEVICE_ORDER=PCI_BUS_ID
python francesco/analysis/measure_pipeline.py --batches "${BATCHES:-4,16}"
