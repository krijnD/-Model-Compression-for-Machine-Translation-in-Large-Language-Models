#!/usr/bin/env bash
# GGUF quantization: uniform bit sweep per group. Runs on BOTH platforms.
#
# Usage:
#   src/quant_gguf.sh <group> <f16.gguf> <out-dir>
#   e.g. src/quant_gguf.sh 1 models/gguf/g1/f16.gguf models/gguf/g1
set -euo pipefail
group="${1:?group 1..8}"
src_gguf="${2:?path to f16 gguf}"
out="${3:?out dir}"

QUANTS=(q8_0 q6_k q5_k_m q4_k_m q3_k_m q2_k)
mkdir -p "$out"
for q in "${QUANTS[@]}"; do
  if [ ! -f "$out/$q.gguf" ]; then
    echo "[$(date)] quantize $q"
    llama-quantize "$src_gguf" "$out/$q.gguf" "$q"
  else
    echo "[$(date)] skip $q (exists)"
  fi
done
echo "DONE group=$group -> $out"
for q in "${QUANTS[@]}"; do
  du -sh "$out/$q.gguf" 2>/dev/null || true
done