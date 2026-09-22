#!/bin/bash
#SBATCH --job-name=metricx
#SBATCH --partition=gpu_h100
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --time=03:00:00
#SBATCH --output=slurm-%x-%j.out

# MetricX-24 Hybrid XL, reference-based. Lower is better (0-25).
# Usage: sbatch --export=ALL,RUN=ours scripts/snellius/score_metricx.slurm   (RUN = paper | ours | ours-beam)

set -euo pipefail
source scripts/snellius/env.sh

RESULTS="$OUTPUT_ROOT/metricx-24/$RUN"
mkdir -p "$RESULTS"
SUMMARY="$RESULTS/summary.tsv"
echo -e "pair\tmetricx-24" > "$SUMMARY"

cd "$REPO_DIR/third_party/metricx"
for pair in ${WMT22_PAIRS//,/ }; do
  echo "=== metricx-24 $RUN $pair"
  python - "$(src_path "$pair")" "$(hyp_path "$pair")" "$(ref_path "$pair")" "$RESULTS/${pair}.input.jsonl" <<'EOF'
import json, sys
src, hyp, ref, out = sys.argv[1:]
lines = [open(p, encoding="utf-8").read().splitlines() for p in (src, hyp, ref)]
assert len(set(map(len, lines))) == 1, f"line counts differ: {list(map(len, lines))}"
with open(out, "w", encoding="utf-8") as f:
    for s, h, r in zip(*lines):
        f.write(json.dumps({"source": s, "hypothesis": h, "reference": r}, ensure_ascii=False) + "\n")
EOF
  python -m metricx24.predict \
    --tokenizer google/mt5-xl \
    --model_name_or_path google/metricx-24-hybrid-xl-v2p6 \
    --max_input_length 1536 --batch_size 1 \
    --input_file "$RESULTS/${pair}.input.jsonl" --output_file "$RESULTS/${pair}.jsonl"
  rm "$RESULTS/${pair}.input.jsonl"
  score=$(python -c "import json,sys; p=[json.loads(l)['prediction'] for l in open(sys.argv[1])]; print(f'{sum(p)/len(p):.2f}')" "$RESULTS/${pair}.jsonl")
  echo -e "${pair}\t${score}" | tee -a "$SUMMARY"
done
echo "Compare: python scripts/summarize.py --run $RUN"
