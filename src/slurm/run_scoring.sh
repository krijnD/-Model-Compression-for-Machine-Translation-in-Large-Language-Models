#!/usr/bin/env bash
# Submit scoring jobs on Slurm (one per run_id). Metric deps may live in the
# same env (unbabel-comet, sacrebleu, metricx repo).
#
# Usage: src/slurm/run_scoring.sh <run_id> [directions...]
#   directions default to all FLORES dirs present in outputs/<run_id>/
set -euo pipefail
RUN_ID="${1:?run_id}"
shift || true
cd "$(dirname "$0")/../.."

if [ $# -gt 0 ]; then
  DIRECTIONS=("$@")
else
  DIRECTIONS=($(ls "outputs/$RUN_ID" 2>/dev/null || echo en-de))
fi

for d in "${DIRECTIONS[@]}"; do
  sbatch <<EOF
#!/usr/bin/env bash
#SBATCH --job-name=score-${RUN_ID//[^a-zA-Z0-9]/-}-$d
#SBATCH --partition=gpu_a100
#SBATCH --gres=gpu:1
#SBATCH --time=02:00:00
#SBATCH --cpus-per-task=4
#SBATCH --output=logs/score_%j.out
#SBATCH --error=logs/score_%j.err
set -euo pipefail
source "\$(conda info --base)/etc/profile.d/conda.sh" 2>/dev/null || true
conda activate almaq 2>/dev/null || source env/.venv-almaq/bin/activate
python src/eval/score.py --run-id "$RUN_ID" --direction "$d" \
  --src "data/flores/$d/src.txt" \
  --mt "outputs/$RUN_ID/$d/mt.txt" \
  --ref "data/flores/$d/ref.txt" \
  --out "scores/$RUN_ID/$d.json" 2>&1 | tee "logs/score_${RUN_ID}_$d.log"
EOF
done
echo "submitted $RUN_ID x ${#DIRECTIONS[@]} scoring jobs"