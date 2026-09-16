#!/usr/bin/env bash
#SBATCH --job-name=xalma-merge
#SBATCH --partition=gpu_a100
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=logs/merge_%A_%a.out
#SBATCH --error=logs/merge_%A_%a.err
#SBATCH --array=1-8
set -euo pipefail

source "$(conda info --base)/etc/profile.d/conda.sh" 2>/dev/null || true
conda activate almaq 2>/dev/null || source "$(dirname "$0")/../../env/.venv-almaq/bin/activate"

g=$SLURM_ARRAY_TASK_ID
cd "$(dirname "$0")/../.."
echo "[$(date)] merging group $g"
python src/model/merge_xalma.py --groups "$g" --out-dir models/merged 2>&1 | tee "logs/merge_g${g}.log"
echo "[$(date)] done group $g"