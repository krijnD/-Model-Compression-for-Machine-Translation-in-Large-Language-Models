#!/bin/bash
#SBATCH --job-name=xcomet-xxl
#SBATCH --partition=gpu_h100
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --time=03:00:00
#SBATCH --output=slurm-%x-%j.out

# Score translations with XCOMET-XXL, reference-free, exactly as
# third_party/ALMA/evals/eval_generation.sh (the ALMA-R paper setup).
#   MODE=paper : the paper's own ALMA-13B-R outputs (checks the metric setup)
#   MODE=ours  : our translations from generate_alma_r.slurm (checks the model)
# Usage: sbatch --export=ALL,MODE=paper scripts/snellius/score_xcomet.slurm

set -euo pipefail
source scripts/snellius/env.sh

MODE="${MODE:-ours}"
TESTSET="$REPO_DIR/third_party/ALMA/outputs/wmt22_outputs/wmt-testset"
PAPER_OUT="$REPO_DIR/third_party/ALMA/outputs/wmt22_outputs/ALMA-13B-R"
OUT="${OUT:-$OUTPUT_ROOT/alma-13b-r/wmt22}"
RESULTS="$OUTPUT_ROOT/xcomet-xxl/$MODE"
mkdir -p "$RESULTS"
SUMMARY="$RESULTS/summary.tsv"
echo -e "pair\txcomet_x100" > "$SUMMARY"

for pair in ${WMT22_PAIRS//,/ }; do
  src=${pair%-*}; tgt=${pair#*-}
  src_path="$TESTSET/${src}${tgt}/test.${src}-${tgt}.${src}"
  if [ "$MODE" = "paper" ]; then
    hyp_path="$PAPER_OUT/${src}${tgt}/test.${src}-${tgt}.${tgt}"
  else
    hyp_path="$OUT/test-${src}-${tgt}"
  fi
  echo "=== $pair"
  comet-score -s "$src_path" -t "$hyp_path" \
    --model Unbabel/XCOMET-XXL --batch_size 8 --gpus 1 \
    --to_json "$RESULTS/${pair}.json" > "$RESULTS/${pair}.txt"
  score=$(awk '/score:/{s=$NF} END{printf "%.2f", s*100}' "$RESULTS/${pair}.txt")
  echo -e "${pair}\t${score}" | tee -a "$SUMMARY"
done

# Averages as reported in the paper (Tables 3 and 4)
awk -F'\t' 'NR>1{ if ($1 ~ /^en-/) {a+=$2; n++} else {b+=$2; m++} }
  END{ printf "avg_en-xx\t%.2f\navg_xx-en\t%.2f\n", a/n, b/m }' "$SUMMARY" | tee -a "$SUMMARY"
echo "Paper (ALMA-13B-R): en-xx 94.05, xx-en 89.11"
