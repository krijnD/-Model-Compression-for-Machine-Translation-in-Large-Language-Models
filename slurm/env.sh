# Shared setup for the Snellius jobs. Submit jobs from the repo root:
#   sbatch scripts/snellius/<job>.job
# Expected layout: <project dir>/{Model-Compression-MT (this repo), venv, hf_cache, outputs}

REPO_DIR="${SLURM_SUBMIT_DIR:-$(pwd)}"
PROJECT_DIR="$(dirname "$REPO_DIR")"

module purge
module load 2024
module load Python/3.12.3-GCCcore-13.3.0
# VENV=venv-quant for the GPTQ jobs (GPTQModel needs a newer transformers than run_llmmt.py, see README)
source "$PROJECT_DIR/${VENV:-venv}/bin/activate"

export HF_HOME="${HF_HOME:-$PROJECT_DIR/hf_cache}"
# If compute nodes have no internet, pre-download models on the login node and uncomment:
# export HF_HUB_OFFLINE=1

OUTPUT_ROOT="$PROJECT_DIR/outputs"
WMT22_PAIRS="de-en,cs-en,is-en,zh-en,ru-en,en-de,en-cs,en-is,en-zh,en-ru"

# Which translations to score (same mapping in scripts/reproduce/score_lexical.py):
#   paper = the paper's ALMA-13B-R outputs, ours = our run with paper decoding, ours-beam = plain beam search,
#   anything else = outputs/<RUN>/wmt22 (e.g. RUN=gptq-w4g128, written by generate_quantized.job)
RUN="${RUN:-ours}"
TESTSET="$REPO_DIR/third_party/ALMA/outputs/wmt22_outputs/wmt-testset"
src_path() { local s=${1%-*} t=${1#*-}; echo "$TESTSET/$s$t/test.$s-$t.$s"; }
ref_path() { local s=${1%-*} t=${1#*-}; echo "$TESTSET/$s$t/test.$s-$t.$t"; }
hyp_path() {
  local s=${1%-*} t=${1#*-}
  case "$RUN" in
    paper)     echo "$REPO_DIR/third_party/ALMA/outputs/wmt22_outputs/ALMA-13B-R/$s$t/test.$s-$t.$t" ;;
    ours)      echo "$OUTPUT_ROOT/alma-13b-r/wmt22/test-$s-$t" ;;
    ours-beam) echo "$OUTPUT_ROOT/alma-13b-r-beam/wmt22/test-$s-$t" ;;
    *)         [ -d "$OUTPUT_ROOT/$RUN/wmt22" ] || { echo "ERROR: unknown RUN=$RUN (no $OUTPUT_ROOT/$RUN/wmt22)" >&2; return 1; }
               echo "$OUTPUT_ROOT/$RUN/wmt22/test-$s-$t" ;;
  esac
}

# Check a generation output folder: one translation per source line, and count empty translations
# (the generation scripts write an empty line when they can't extract a translation). Returns 1 on a mismatch.
check_translations() {
  local out=$1 status=0 pair hyp n_src n_hyp n_empty flag
  printf "%-6s %7s %7s %6s\n" pair sources hyps empty
  for pair in ${WMT22_PAIRS//,/ }; do
    hyp="$out/test-${pair}"
    n_src=$(awk 'END{print NR}' "$(src_path "$pair")")
    read -r n_hyp n_empty < <( [ -f "$hyp" ] && awk 'NF==0{e++} END{print NR, e+0}' "$hyp" || echo "0 0")
    flag=""; [ "$n_src" -ne "$n_hyp" ] && { flag="  <-- MISMATCH"; status=1; }
    printf "%-6s %7s %7s %6s%s\n" "$pair" "$n_src" "$n_hyp" "$n_empty" "$flag"
  done
  echo "Translations written to $out (files test-<src>-<tgt>)"
  return $status
}
