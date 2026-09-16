#!/usr/bin/env bash
# Efficiency for GGUF artifacts via llama-bench (both platforms).
# Protocol: warmup-excluded throughput at batch sizes 1/16/64, prompt 512,
# generation 128. Writes scores/<run_id>/efficiency_gguf.json.
#
# Usage: src/eval/efficiency_gguf.sh <run_id> <model.gguf> [platform]
set -euo pipefail
RUN_ID="${1:?run_id}"
MODEL="${2:?model.gguf}"
cd "$(dirname "$0")/../.."

NGL=999
if [ "$(uname -s)" = "Darwin" ]; then NGL=99; fi
THREADS=$(sysctl -n hw.perflevel0.physicalcpu 2>/dev/null || nproc)

mkdir -p "scores/$RUN_ID"
echo "[$(date)] llama-bench $MODEL (ngl=$NGL threads=$THREADS)"

# -b/--batch-size per run; llama-bench runs each once; repeat with -r 3 (default)
llama-bench -m "$MODEL" -ngl "$NGL" -t "$THREADS" -p 512 -n 128 -r 3 \
  -b 1 -b 16 -b 64 --output json > "scores/$RUN_ID/efficiency_gguf.json" 2>/dev/null

python3 - "$RUN_ID" <<'EOF'
import json, os, sys
run_id = sys.argv[1]
p = f"scores/{run_id}/efficiency_gguf.json"
d = json.load(open(p))
size_gb = os.path.getsize(sys.argv[2] if len(sys.argv) > 2 else "") if False else None
print(f"rows: {len(d)}")
for r in d:
    print(f"  n_batch={r.get('n_batch')} pp={r.get('avg_ts') if r.get('n_prompt') else ''} "
          f"tg={r.get('avg_ts') if r.get('n_gen') else ''} t/s")
EOF
echo "wrote scores/$RUN_ID/efficiency_gguf.json"