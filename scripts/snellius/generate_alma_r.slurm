#!/bin/bash
#SBATCH --job-name=alma-r-gen
#SBATCH --partition=gpu_h100
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=64
#SBATCH --time=04:00:00
#SBATCH --output=slurm-%x-%j.out

# Generate WMT'22 (WMT'21 for is) translations with ALMA-13B-R, using the exact
# settings of third_party/ALMA/evals/alma_13b_r.sh.
# Override with: sbatch --export=ALL,MODEL=/path/to/weights,OUT=/path scripts/snellius/generate_alma_r.slurm

set -euo pipefail
source scripts/snellius/env.sh

# Local copy of the merged weights WITHOUT the adapter files (see README):
# loading "haoranxu/ALMA-13B-R" by name makes transformers load ALMA-13B-Pretrain + adapter instead.
MODEL="${MODEL:-$PROJECT_DIR/models/ALMA-13B-R}"
[ -f "$MODEL/adapter_config.json" ] && { echo "ERROR: $MODEL contains adapter_config.json; remove adapter_* files"; exit 1; }
OUT="${OUT:-$OUTPUT_ROOT/alma-13b-r/wmt22}"
NGPUS="${SLURM_GPUS_ON_NODE:-1}"
PORT=$(( RANDOM % 20001 + 30000 ))
MULTI_GPU=""; [ "$NGPUS" -gt 1 ] && MULTI_GPU="--multi_gpu"
mkdir -p "$OUT"

cd "$REPO_DIR/third_party/ALMA"   # run_llmmt.py expects to run from here

generate() {  # $1 = language pairs, $2 = max_source_length
  accelerate launch $MULTI_GPU --num_processes "$NGPUS" --num_machines 1 \
      --mixed_precision bf16 --main_process_port "$PORT" \
    run_llmmt.py \
    --model_name_or_path "$MODEL" \
    --do_predict \
    --low_cpu_mem_usage \
    --language_pairs "$1" \
    --mmt_data_path ./human_written_data/ \
    --per_device_eval_batch_size 4 \
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

generate "$WMT22_PAIRS" 256
generate "zh-en" 512   # as in the paper: zh-en is regenerated with longer sources

echo "Translations written to $OUT (files test-<src>-<tgt>)"
