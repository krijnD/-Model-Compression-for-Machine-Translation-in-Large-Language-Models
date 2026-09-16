#!/usr/bin/env bash
#SBATCH --job-name=xalma-gptq
#SBATCH --partition=gpu_a100
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=08:00:00
#SBATCH --output=logs/gptq_%A_%a.out
#SBATCH --error=logs/gptq_%A_%a.err
#SBATCH --array=1-8
set -euo pipefail

source "$(conda info --base)/etc/profile.d/conda.sh" 2>/dev/null || true
conda activate almaq 2>/dev/null || source "$(dirname "$0")/../../env/.venv-almaq/bin/activate"

g=$SLURM_ARRAY_TASK_ID
cd "$(dirname "$0")/../.."
for bits in 8 4 3 2; do
  echo "[$(date)] GPTQ group $g bits $bits"
  python src/quantize/quant_gptq.py --model "models/merged/g${g}" \
      --bits "$bits" --out "models/gptq/g${g}/b${bits}" \
      --calib-langs "$(python -c "import json;print(json.load(open('src/common/group_langs.json'))['g$g'])" 2>/dev/null || echo de,is)" \
      2>&1 | tee "logs/gptq_g${g}_b${bits}.log"
done
echo "[$(date)] done group $g GPTQ"