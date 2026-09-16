#!/usr/bin/env bash
#SBATCH --job-name=xalma-gen
#SBATCH --partition=gpu_a100
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=06:00:00
#SBATCH --output=logs/gen_%A_%a.out
#SBATCH --error=logs/gen_%A_%a.err
#SBATCH --array=0-17
set -euo pipefail

source "$(conda info --base)/etc/profile.d/conda.sh" 2>/dev/null || true
conda activate almaq 2>/dev/null || source "$(dirname "$0")/../../env/.venv-almaq/bin/activate"

# Matrix position encoded as: RUN_ID MODEL DIRECTION
# Passed via env var MATRIX_FILE (one "run_id model direction" per line) + ARRAY_TASK_ID
MATRIX_FILE="${MATRIX_FILE:?set e.g. MATRIX_FILE=experiments/matrix.txt}"
line=$(sed -n "$((SLURM_ARRAY_TASK_ID + 1))p" "$MATRIX_FILE")
read -r RUN_ID MODEL DIRECTION <<<"$line"
cd "$(dirname "$0")/../.."
echo "[$(date)] gen $RUN_ID $DIRECTION $MODEL"
python src/generate.py \
    --model "$MODEL" --run-id "$RUN_ID" --direction "$DIRECTION" \
    --src-file "data/flores/${DIRECTION}/src.txt" \
    --ref-file "data/flores/${DIRECTION}/ref.txt" 2>&1 | tee "logs/gen_${RUN_ID}_${DIRECTION}.log"
echo "[$(date)] done $RUN_ID $DIRECTION"