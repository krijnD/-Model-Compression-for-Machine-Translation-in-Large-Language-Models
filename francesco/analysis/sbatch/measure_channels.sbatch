#!/bin/bash
#SBATCH --job-name=measure-channels
#SBATCH --partition=gpu_h100
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --time=01:00:00
#SBATCH --output=logs/slurm-%x-%j.out

# RQ5 in reasearch.md: per-channel activation-energy profiles per language.
# Measures the fp16 model only - no quantization and no generation, one forward pass per sequence,
# so this is minutes of GPU, not hours. Writes francesco/results/json/channels_fp16.npz, which
# `--compare` reads back offline.
#   sbatch francesco/analysis/sbatch/measure_channels.sbatch
#   sbatch --export=ALL,SUFFIX=mlp.up_proj francesco/analysis/sbatch/measure_channels.sbatch
#   sbatch --export=ALL,LIMIT=32 francesco/analysis/sbatch/measure_channels.sbatch   # smoke test
set -euo pipefail
# env.sh activates ${VENV:-venv}; use venv (transformers 4.45.2), NOT venv-quant: the fp16 folder
# loads with transformers and nothing here needs GPTQModel.
VENV=venv
source scripts/snellius/env.sh

python francesco/analysis/measure_channels.py \
  --model "${MODEL:-$PROJECT_DIR/models/ALMA-13B-R}" \
  --suffix "${SUFFIX:-mlp.down_proj}" \
  --profiles "${PROFILES:-mixture,calib:is,calib:de,calib:cs,calib:zh,calib:ru,pool:is:1024,pool:de:1024}" \
  --batch-size "${BATCH:-4}" \
  --dtype "${DTYPE:-bfloat16}" \
  ${LIMIT:+--limit "$LIMIT"} \
  --out "${OUT:-$REPO_DIR/francesco/results/json/channels_fp16.npz}"
