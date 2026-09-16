#!/usr/bin/env bash
# Slurm/NVIDIA cluster environment for the ALMA model-compression project.
# Creates python 3.11 env "almaq" (conda preferred, else venv), installs the
# CUDA stack incl. GPTQ/bitsandbytes/torchao, clones auxiliary repos, and
# builds llama.cpp with -DGGML_CUDA=ON.
#
# Usage (on a Slurm login node or a machine with an NVIDIA GPU):
#   env/setup_slurm.sh
# Env overrides: CONDA_BASE, TORCH_INDEX (=https://download.pytorch.org/whl/cu118|cu121|cu124),
#                LLAMACPP_REPO, LLAMACPP_COMMIT
set -euo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TORCH_INDEX="${TORCH_INDEX:-https://download.pytorch.org/whl/cu121}"

echo "== [1/7] create env 'almaq' (python 3.11) =="
if command -v conda >/dev/null 2>&1; then
  if ! conda env list | grep -q almaq; then
    conda create -y -n almaq python=3.11
  fi
  source "$(conda info --base)/etc/profile.d/conda.sh"
  conda activate almaq
else
  command -v python3.11 || { echo "no conda and no python3.11 — install one"; exit 1; }
  ENVDIR="${ENVDIR:-$REPO/env/.venv-almaq}"
  [ -x "$ENVDIR/bin/python" ] || python3.11 -m venv "$ENVDIR"
  source "$ENVDIR/bin/activate"
fi
python -m pip install -U pip wheel >/dev/null

echo "== [2/7] install CUDA torch + core stack =="
cat > "$REPO/env/requirements-slurm.txt" <<'EOF'
torch>=2.3
transformers>=4.40
accelerate>=0.31
datasets>=2.19
peft>=0.11
sacrebleu>=2.4
unbabel-comet>=2.2.0
gptqmodel
bitsandbytes
torchao
sentencepiece>=0.2
protobuf>=4.25
numpy>=1.26
pandas>=2.2
matplotlib>=3.8
huggingface_hub>=0.25
psutil>=5.9
EOF
# Torch variant matching driver: cu118/cu121/cu124 -> override via TORCH_INDEX.
pip install torch --index-url "$TORCH_INDEX"
pip install -r "$REPO/env/requirements-slurm.txt"

echo "== [3/7] auxiliary repos =="
mkdir -p "$REPO/vendor"
[ -d "$REPO/vendor/metricx" ]    || git clone --depth 1 https://github.com/google-research/metricx "$REPO/vendor/metricx"
[ -d "$REPO/vendor/smoothquant" ]|| git clone --depth 1 https://github.com/mit-han-lab/smoothquant "$REPO/vendor/smoothquant"
[ -d "$REPO/vendor/ALMA" ]       || git clone --depth 1 https://github.com/fe1ixxu/ALMA "$REPO/vendor/ALMA"

echo "== [4/7] build llama.cpp with CUDA =="
LLAMACPP_REPO="${LLAMACPP_REPO:-https://github.com/ggml-org/llama.cpp}"
LLAMACPP_COMMIT="${LLAMACPP_COMMIT:-master}"
[ -d "$REPO/vendor/llama.cpp" ] || git clone "$LLAMACPP_REPO" "$REPO/vendor/llama.cpp"
cd "$REPO/vendor/llama.cpp"
git fetch origin >/dev/null 2>&1 || true
git checkout "$LLAMACPP_COMMIT" 2>/dev/null || git checkout master
cmake -B build -DGGML_CUDA=ON -DCMAKE_BUILD_TYPE=Release
cmake --build build --config Release -j"$(nproc)" --target llama-cli llama-quantize llama-bench llama-cpp
echo "llama.cpp binaries: $REPO/vendor/llama.cpp/build/bin/"

echo "== [5/7] gates =="
python -c "import torch; print('CUDA:', torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else '')"
nvidia-smi -L 2>/dev/null | head -4 || true

echo "== [6/7] lock requirements =="
pip freeze > "$REPO/env/requirements-lock-slurm.txt"
echo "Lockfile written: env/requirements-lock-slurm.txt"

echo "== [7/7] report =="
echo "Done. Activate with: conda activate almaq   (or venv source $ENVDIR/bin/activate)"
echo "Important: run env/setup_slurm.sh inside the sbatch allocation for GPU jobs;"
echo "          adjust TORCH_INDEX if the driver predates your wheel."