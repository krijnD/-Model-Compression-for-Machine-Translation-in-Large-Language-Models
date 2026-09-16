#!/usr/bin/env bash
# Mac (Apple Silicon) environment for the ALMA model-compression project.
# Creates a python 3.11 venv named "almaq" (preferred: python3.11 directly;
# falls back to uv if python3.11 missing), installs the scoring/harness stack,
# clones auxiliary repos, and verifies the MPS + llama.cpp(Metal) gates.
#
# Usage:  env/setup_mac.sh
# Requires: python3.11 (or uv), cmake, git. llama.cpp via Homebrew recommended.
set -euo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ENVDIR="${ENVDIR:-$REPO/env/.venv-almaq}"
VENV_PY="${VENV_PY:-python3.11}"
command -v "$VENV_PY" >/dev/null || command -v uv >/dev/null || {
  echo "Need python3.11 or uv:  brew install python@3.11   or  brew install uv"; exit 1
}

echo "== [1/6] create venv $ENVDIR =="
if [ ! -x "$ENVDIR/bin/python" ]; then
  if command -v "$VENV_PY" >/dev/null; then
    "$VENV_PY" -m venv "$ENVDIR"
  elif command -v uv >/dev/null; then
    uv venv --seed --python 3.11 "$ENVDIR"   # --seed ensures pip is present
  else
    echo "No python3.11 and no uv. Install one: brew install python@3.11 or brew install uv"
    exit 1
  fi
fi
source "$ENVDIR/bin/activate"
python -m pip install -U pip wheel >/dev/null

echo "== [2/6] install python stack (MPS build of torch) =="
cat > "$REPO/env/requirements-mac.txt" <<'EOF'
torch>=2.3
transformers>=4.40
accelerate>=0.31
datasets>=2.19
peft>=0.11
sacrebleu>=2.4
unbabel-comet>=2.2.0
sentencepiece>=0.2
protobuf>=4.25
numpy>=1.26
pandas>=2.2
matplotlib>=3.8
huggingface_hub>=0.25
psutil>=5.9
EOF
pip install -r "$REPO/env/requirements-mac.txt"

echo "== [3/6] auxiliary repos =="
mkdir -p "$REPO/vendor"
[ -d "$REPO/vendor/metricx" ]    || git clone --depth 1 https://github.com/google-research/metricx "$REPO/vendor/metricx"
[ -d "$REPO/vendor/smoothquant" ]|| git clone --depth 1 https://github.com/mit-han-lab/smoothquant "$REPO/vendor/smoothquant"
[ -d "$REPO/vendor/ALMA" ]       || git clone --depth 1 https://github.com/fe1ixxu/ALMA "$REPO/vendor/ALMA"

echo "== [4/6] llama.cpp (expects Homebrew build with Metal) =="
if command -v llama-cli >/dev/null && command -v llama-quantize >/dev/null; then
  echo "llama.cpp binaries found: $(command -v llama-cli)"
  llama-cli --version 2>&1 | head -1 || true
else
  echo "llama.cpp not in PATH — installing via Homebrew (Metal-enabled bottle):"
  brew install llama.cpp
fi

echo "== [5/6] gates =="
MPS_OK=$(python -c "import torch; print(torch.backends.mps.is_available())")
echo "MPS available (must be True): $MPS_OK"
PYVER=$(python --version)
echo "Python: $PYVER"
[ "$MPS_OK" = "True" ] || { echo "WARN: MPS not available on this machine — device.py will fall back to CPU."; }

echo "== [6/6] lock requirements =="
pip freeze > "$REPO/env/requirements-lock-mac.txt"
echo "Lockfile written: env/requirements-lock-mac.txt"
echo "Done. Activate with:  source $ENVDIR/bin/activate"