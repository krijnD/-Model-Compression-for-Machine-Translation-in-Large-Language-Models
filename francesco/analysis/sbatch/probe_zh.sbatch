#!/bin/bash
#SBATCH --job-name=probe-zh
#SBATCH --partition=gpu_h100
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --time=00:45:00
#SBATCH --output=logs/slurm-%x-%j.out
# zh-en runs at BATCH_LONG=1 / source length 512 in the eval jobs - the case where the fixed
# per-forward dequant is amortized over the fewest sequences (5 rows), so the largest expected win.
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
VENV=venv-quant
source scripts/snellius/env.sh
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True CUDA_DEVICE_ORDER=PCI_BUS_ID
python francesco/analysis/run_probe.py \
  --model "${MODEL:-$PROJECT_DIR/models/ALMA-13B-R-gptq-w2g128}" \
  --pair zh-en --batches "${BATCHES:-1,4}" --reps "${REPS:-2}" \
  --max-source-length 512 --max-new-tokens 256 --num-beams 5 --dtype bfloat16 \
  --out "$REPO_DIR/francesco/results/json/probe_zh_w2.json"
