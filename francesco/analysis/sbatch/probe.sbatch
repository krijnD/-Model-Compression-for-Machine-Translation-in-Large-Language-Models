#!/bin/bash
#SBATCH --job-name=probe-batch
#SBATCH --partition=gpu_h100
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --time=00:45:00
#SBATCH --output=logs/slurm-%x-%j.out

# Batch-size probe for the GPTQ 2-bit / 3-bit generation cost. See francesco/README.md.
# Does NOT touch outputs/: it only times generate() calls. Runs beside the four eval_quantized
# jobs on a free H100 (they each hold one node, so this takes a different node).
#   BATCHES="4,16" PAIR=de-en MODEL=...  sbatch --export=ALL,BATCHES=4,16 francesco/analysis/sbatch/probe.sbatch
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
# env.sh activates ${VENV:-venv}; the GPTQ jobs export VENV=venv-quant before sourcing
# (scripts/snellius/generate_quantized.job:21) - gptqmodel only exists in that environment.
VENV=venv-quant
source scripts/snellius/env.sh

MODEL="${MODEL:-$PROJECT_DIR/models/ALMA-13B-R-gptq-w2g128}"
BATCHES="${BATCHES:-4,16}"
PAIR="${PAIR:-de-en}"
REPS="${REPS:-3}"
OUT="${OUT:-$REPO_DIR/francesco/results/json/probe_$(basename "$MODEL").json}"

echo "##### probe $(basename "$MODEL") pair=$PAIR batches=$BATCHES reps=$REPS ($(date))"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
# same env the eval jobs use, so per-line numbers are comparable to the logs
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export CUDA_DEVICE_ORDER=PCI_BUS_ID

python francesco/analysis/run_probe.py \
  --model "$MODEL" --pair "$PAIR" --batches "$BATCHES" --reps "$REPS" \
  --max-source-length 256 --max-new-tokens 256 --num-beams 5 --dtype bfloat16 --out "$OUT"
