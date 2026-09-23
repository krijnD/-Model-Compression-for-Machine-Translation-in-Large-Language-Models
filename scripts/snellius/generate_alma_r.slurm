#!/bin/bash
#SBATCH --job-name=alma-r-gen
#SBATCH --partition=gpu_h100
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=64
#SBATCH --time=08:00:00
#SBATCH --output=slurm-%x-%j.out

# Generate WMT'22 (WMT'21 for is) translations with ALMA-13B-R, using the exact
# settings of third_party/ALMA/evals/alma_13b_r.sh.
#   DECODING=paper : beam 5 + the model's generation_config (do_sample, temp 0.9, top_p 0.6), as in the paper (default)
#   DECODING=beam  : plain beam search, do_sample=False (deterministic)
# Override with: sbatch --export=ALL,DECODING=beam,MODEL=/path/to/weights,OUT=/path scripts/snellius/generate_alma_r.slurm

set -euo pipefail
source scripts/snellius/env.sh

# Local copy of the merged weights WITHOUT the adapter files (see README):
# loading "haoranxu/ALMA-13B-R" by name makes transformers load ALMA-13B-Pretrain + adapter instead.
MODEL="${MODEL:-$PROJECT_DIR/models/ALMA-13B-R}"
[ -f "$MODEL/adapter_config.json" ] && { echo "ERROR: $MODEL contains adapter_config.json; remove adapter_* files"; exit 1; }
DECODING="${DECODING:-paper}"
case "$DECODING" in
  paper) OUT="${OUT:-$OUTPUT_ROOT/alma-13b-r/wmt22}" ;;
  beam)  OUT="${OUT:-$OUTPUT_ROOT/alma-13b-r-beam/wmt22}"
         # run_llmmt.py has no do_sample flag and generate() falls back to the model's
         # generation_config, so point it at a copy of the model folder (symlinks) whose
         # generation_config has sampling switched off.
         BEAM_MODEL="$(mktemp -d)"
         ln -s "$MODEL"/* "$BEAM_MODEL"/
         rm "$BEAM_MODEL/generation_config.json"
         python - "$MODEL/generation_config.json" "$BEAM_MODEL/generation_config.json" <<'EOF'
import json, sys
cfg = json.load(open(sys.argv[1]))
cfg["do_sample"] = False
cfg.pop("temperature", None); cfg.pop("top_p", None)
json.dump(cfg, open(sys.argv[2], "w"), indent=2)
EOF
         MODEL="$BEAM_MODEL" ;;
  *) echo "ERROR: DECODING must be paper or beam"; exit 1 ;;
esac
echo "Decoding: $DECODING | model: $MODEL | output: $OUT"
cat "$MODEL/generation_config.json"

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True   # less fragmentation at long source lengths
NGPUS="${SLURM_GPUS_ON_NODE:-1}"
PORT=$(( RANDOM % 20001 + 30000 ))
MULTI_GPU=""; [ "$NGPUS" -gt 1 ] && MULTI_GPU="--multi_gpu"
mkdir -p "$OUT"

cd "$REPO_DIR/third_party/ALMA"   # run_llmmt.py expects to run from here

generate() {  # $1 = language pairs, $2 = max_source_length, $3 = per-device batch size
  [ -z "$1" ] && return 0
  accelerate launch $MULTI_GPU --num_processes "$NGPUS" --num_machines 1 \
      --mixed_precision bf16 --main_process_port "$PORT" \
    run_llmmt.py \
    --model_name_or_path "$MODEL" \
    --do_predict \
    --low_cpu_mem_usage \
    --language_pairs "$1" \
    --mmt_data_path ./human_written_data/ \
    --per_device_eval_batch_size "$3" \
    --output_dir "$OUT" \
    --predict_with_generate \
    --max_new_tokens 256 \
    --max_source_length "$2" \
    --bf16 \
    --seed 42 \
    --num_beams 5 \
    --overwrite_cache \
    --overwrite_output_dir
}

# The paper also translates zh-en in the first call and then overwrites it with
# source length 512; skipping it here gives the same file and saves ~10% time.
# zh-en uses a smaller batch: at source length 512, batch 4 x 5 beams runs a 13B
# model out of memory on a 94 GB H100. Set PAIRS_SHORT="" to redo only zh-en.
PAIRS_SHORT="${PAIRS_SHORT-${WMT22_PAIRS/zh-en,/}}"
PAIRS_LONG="${PAIRS_LONG-zh-en}"
generate "$PAIRS_SHORT" 256 "${BATCH:-4}"
generate "$PAIRS_LONG" 512 "${BATCH_LONG:-1}"

# Check: one translation per source line, and count empty translations
# (run_llmmt.py writes an empty line when it can't extract a translation).
status=0
printf "%-6s %7s %7s %6s\n" pair sources hyps empty
for pair in ${WMT22_PAIRS//,/ }; do
  hyp="$OUT/test-${pair}"
  n_src=$(awk 'END{print NR}' "$(src_path "$pair")")
  read -r n_hyp n_empty < <( [ -f "$hyp" ] && awk 'NF==0{e++} END{print NR, e+0}' "$hyp" || echo "0 0")
  flag=""; [ "$n_src" -ne "$n_hyp" ] && { flag="  <-- MISMATCH"; status=1; }
  printf "%-6s %7s %7s %6s%s\n" "$pair" "$n_src" "$n_hyp" "$n_empty" "$flag"
done
echo "Translations written to $OUT (files test-<src>-<tgt>)"
exit $status
