#!/bin/bash
#SBATCH --job-name=comet
#SBATCH --partition=gpu_h100
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --time=03:00:00
#SBATCH --array=0-3
#SBATCH --output=logs/slurm-%x-%A_%a.out

# COMET-family metrics of the ALMA-R paper, exactly as third_party/ALMA/evals/eval_generation.sh.
# One array task per metric: 0 xcomet-xxl, 1 kiwi-xxl, 2 kiwi-22 (no reference), 3 comet-22 (with reference).
#   RUN=paper     : the paper's own ALMA-13B-R outputs (checks the metric setup)
#   RUN=ours      : our translations, paper decoding
#   RUN=ours-beam : our translations, plain beam search
# Usage: sbatch --export=ALL,RUN=ours scripts/snellius/score_comet.slurm
#        sbatch --array=1-3 --export=ALL,RUN=paper scripts/snellius/score_comet.slurm   # only some metrics

set -euo pipefail
source scripts/snellius/env.sh

METRICS=(xcomet-xxl kiwi-xxl kiwi-22 comet-22)
MODELS=(Unbabel/XCOMET-XXL Unbabel/wmt23-cometkiwi-da-xxl Unbabel/wmt22-cometkiwi-da Unbabel/wmt22-comet-da)
BATCH=(8 8 256 256)
i="${SLURM_ARRAY_TASK_ID:-0}"
METRIC=${METRICS[$i]}

RESULTS="$OUTPUT_ROOT/$METRIC/$RUN"
mkdir -p "$RESULTS"
SUMMARY="$RESULTS/summary.tsv"
echo -e "pair\t${METRIC}" > "$SUMMARY"

for pair in ${WMT22_PAIRS//,/ }; do
  src=${pair%-*}; tgt=${pair#*-}
  ref_args=(); [ "$METRIC" = "comet-22" ] && ref_args=(-r "$(ref_path "$pair")")
  echo "=== $METRIC $RUN $pair"
  comet-score -s "$(src_path "$pair")" -t "$(hyp_path "$pair")" "${ref_args[@]}" \
    --model "${MODELS[$i]}" --batch_size "${BATCH[$i]}" --gpus 1 \
    --to_json "$RESULTS/${pair}.json" > "$RESULTS/${pair}.txt"
  score=$(awk '/score:/{s=$NF} END{printf "%.2f", s*100}' "$RESULTS/${pair}.txt")
  echo -e "${pair}\t${score}" | tee -a "$SUMMARY"
done
echo "Compare with the paper: python scripts/summarize.py --run $RUN"
