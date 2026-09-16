#!/usr/bin/env bash
# Layerwise heterogeneous GGUF quantization (ds4-inspired), via llama-quantize
# --tensor-type (regex-matched, repeatable; renamed --override-tensor).
#
# Recipe A het-q8q4: attention/embeddings/lm_head Q8_0; MLP up/gate/down Q4_K.
# Recipe B het-q8q2q3: same Q8_0 shell; MLP up/gate Q2_K, down Q3_K.
# Untouched tensors (incl. norms) stay F32.
#
# Usage:
#   src/quantize/quant_gguf_layerwise.sh <f16.gguf> <out-dir> [A|B]
set -euo pipefail
cd "$(dirname "$0")/../.."
src_gguf="${1:?f16 gguf}"
out="${2:?out dir}"
variant="${3:-A}"
mkdir -p "$out"

case "$variant" in
  A)
    out_file="$out/het_q8q4.gguf"
    args=(
      --token-embedding-type q8_0
      --output-tensor-type q8_0
      --tensor-type 'attn_q=q8_0'
      --tensor-type 'attn_k=q8_0'
      --tensor-type 'attn_v=q8_0'
      --tensor-type 'attn_output=q8_0'
      --tensor-type 'ffn_gate=q4_k'
      --tensor-type 'ffn_up=q4_k'
      --tensor-type 'ffn_down=q4_k'
    )
    ;;
  B)
    out_file="$out/het_q8q2q3.gguf"
    args=(
      --token-embedding-type q8_0
      --output-tensor-type q8_0
      --tensor-type 'attn_q=q8_0'
      --tensor-type 'attn_k=q8_0'
      --tensor-type 'attn_v=q8_0'
      --tensor-type 'attn_output=q8_0'
      --tensor-type 'ffn_gate=q2_k'
      --tensor-type 'ffn_up=q2_k'
      --tensor-type 'ffn_down=q3_k'
    )
    ;;
  *)
    echo "unknown variant $variant (use A or B)"; exit 1;;
esac

echo "[$(date)] layerwise $variant -> $out_file"
llama-quantize "${args[@]}" "$src_gguf" "$out_file" F32
echo "DONE: $out_file"
du -sh "$out_file"