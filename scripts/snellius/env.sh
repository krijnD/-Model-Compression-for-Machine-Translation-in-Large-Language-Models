# Shared setup for the Snellius jobs. Submit jobs from the repo root:
#   sbatch scripts/snellius/<job>.slurm
# Expected layout: <project dir>/{Model-Compression-MT (this repo), venv, hf_cache, outputs}

REPO_DIR="${SLURM_SUBMIT_DIR:-$(pwd)}"
PROJECT_DIR="$(dirname "$REPO_DIR")"

module purge
module load 2024
module load Python/3.12.3-GCCcore-13.3.0
source "$PROJECT_DIR/venv/bin/activate"

export HF_HOME="${HF_HOME:-$PROJECT_DIR/hf_cache}"
# If compute nodes have no internet, pre-download models on the login node and uncomment:
# export HF_HUB_OFFLINE=1

OUTPUT_ROOT="$PROJECT_DIR/outputs"
WMT22_PAIRS="de-en,cs-en,is-en,zh-en,ru-en,en-de,en-cs,en-is,en-zh,en-ru"
