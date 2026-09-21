# Model Compression for Machine Translation in Large Language Models

Everything runs on Snellius. ALMA-R generation and all evaluation metrics share **one** virtual environment.

External code is included as git submodules in `third_party/`:
- `third_party/ALMA`: the [ALMA / ALMA-R repo](https://github.com/fe1ixxu/ALMA), which provides the generation script, the WMT'22 test sets and the paper's own outputs
- `third_party/metricx`: [MetricX](https://github.com/google-research/metricx)

## Installation on Snellius

| Metric | Comes from | Install |
|---|---|---|
| BLEU (SacreBLEU) | [`sacrebleu`](https://github.com/mjpost/sacrebleu) | pip |
| chrF++ | [`sacrebleu`](https://github.com/mjpost/sacrebleu) (`CHRF(word_order=2)`) | pip |
| XCOMET-XXL | [`unbabel-comet`](https://github.com/Unbabel/COMET), model [`Unbabel/XCOMET-XXL`](https://huggingface.co/Unbabel/XCOMET-XXL), as used in the ALMA-R paper | pip, plus the HF license |
| MetricX-24 Hybrid | [`google-research/metricx`](https://github.com/google-research/metricx), model [`google/metricx-24-hybrid-xl-v2p6`](https://huggingface.co/google/metricx-24-hybrid-xl-v2p6) | git submodule `third_party/metricx` (not on PyPI) plus pip deps |
| Hallucination rate | own code (length ratio of candidate to source, in characters, >= 2) | nothing |

Model weights are **not** installed by pip. They download from Hugging Face the first time a model is used.

### Folder layout on Snellius

The venv, data and model weights live **next to** the repo, not inside it:

```
<project dir>/
├── Model-Compression-MT/   # this repo (code only)
├── venv/                   # Python environment
├── data/                   # datasets
├── hf_cache/               # Hugging Face models (HF_HOME)
└── outputs/                # translations and scores written by the jobs
```

### 1. Get the code (including the submodules)

```bash
cd <project dir>/Model-Compression-MT
git pull
git submodule update --init --recursive
```

Check that the submodules are there: `ls third_party/metricx/metricx24/predict.py third_party/ALMA/run_llmmt.py`

### 2. Create the environment

```bash
module purge
module load 2024
module load Python/3.12.3-GCCcore-13.3.0

python -m venv ../venv
source ../venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

(Run `module avail Python` to see which Python modules are available. Use Python 3.10–3.12.)

### 3. Hugging Face setup (once)

1. Log in on huggingface.co and **accept the license** on the [Unbabel/XCOMET-XXL](https://huggingface.co/Unbabel/XCOMET-XXL) page. The model is gated.
2. Point the model cache at `hf_cache/` next to the repo. Otherwise it goes to `~/.cache/huggingface` in your home folder. Use the absolute path, and put this in `~/.bashrc` **and** in every Slurm job script:
   ```bash
   export HF_HOME=<project dir>/hf_cache
   ```
3. Log in with a read token:
   ```bash
   huggingface-cli login
   ```

Compute nodes may not have internet access. If so, download the models once from the login node (they go into `HF_HOME`):

```bash
python -c "from huggingface_hub import snapshot_download as d; d('haoranxu/ALMA-13B-R')"
python -c "from comet import download_model; download_model('Unbabel/XCOMET-XXL')"
python -c "from huggingface_hub import snapshot_download as d; d('google/metricx-24-hybrid-xl-v2p6')"
python -c "from transformers import AutoTokenizer; AutoTokenizer.from_pretrained('google/mt5-xl')"   # tokenizer only
```

### 4. Quick checks

```bash
source ../venv/bin/activate
python -c "import sacrebleu, comet, transformers; print(sacrebleu.__version__, transformers.__version__)"
sacrebleu --help | head -n 3
cd third_party/metricx && python -c "import metricx24.models; print('metricx ok')" && cd -
```

## Reproducing the ALMA-13B-R baseline (XCOMET-XXL)

Goal: reproduce the ALMA-13B-R XCOMET scores from the ALMA-R paper ([Xu et al. 2024](https://arxiv.org/abs/2401.08417), Tables 3 and 4) before quantizing.

| | de | cs | is | zh | ru | **Avg** |
|---|---|---|---|---|---|---|
| en→xx | 97.48 | 93.61 | 91.93 | 92.03 | 95.22 | **94.05** |
| xx→en | 94.20 | 88.03 | 80.49 | 91.65 | 91.18 | **89.11** |

**Paper setup:**
- **Test data:** WMT'22 for de, cs, zh and ru; WMT'21 for is. The files are in `third_party/ALMA/human_written_data/` and `third_party/ALMA/outputs/wmt22_outputs/wmt-testset/`.
- **Generation:** beam 5, bf16, seed 42, max 256 new tokens; source length 256, or 512 for zh→en.
- **Scoring:** `Unbabel/XCOMET-XXL` without a reference, reported × 100.

Submit all jobs **from the repo root**. Results go to `<project dir>/outputs/`.

**Step 1: check the metric.** Score the paper's own ALMA-13B-R translations. This should give about 94.05 and 89.11.
```bash
sbatch --export=ALL,MODE=paper scripts/snellius/score_xcomet.slurm
```

**Step 2: generate.** Translate the test sets with ALMA-13B-R (4× H100, same settings as `third_party/ALMA/evals/alma_13b_r.sh`):
```bash
sbatch scripts/snellius/generate_alma_r.slurm
# if the weights are in a local folder instead of the HF cache:
sbatch --export=ALL,MODEL=/path/to/ALMA-13B-R scripts/snellius/generate_alma_r.slurm
```

**Step 3: score our translations.**
```bash
sbatch --export=ALL,MODE=ours scripts/snellius/score_xcomet.slurm
```
Each run writes `outputs/xcomet-xxl/<mode>/summary.tsv`, with a score per pair and the two averages, plus per-sentence scores and error spans (`<pair>.json`).

Follow a job with `squeue -u $USER` and `tail -f slurm-<job-name>-<id>.out`.

## Notes

- **Reproducibility:** we don't store weights in git. Instead we record exactly which versions were used: `requirements.txt` for packages, the submodule commits for the ALMA and MetricX code, and the Hugging Face model revision (commit hash) for weights. Look up a revision with `ls $HF_HOME/hub/models--google--metricx-24-hybrid-xl-v2p6/snapshots/` and log it with the results.

- **MetricX and transformers versions:** MetricX's own `requirements.txt` pins `transformers==4.30.2`. We use `4.51.1`, the version ALMA pins, so that ALMA, COMET and MetricX share one env. The MT5 internals MetricX uses are unchanged in that version.
- **Running MetricX** (from `third_party/metricx`; the input jsonl has the fields `source`, `hypothesis`, `reference`):
  ```bash
  python -m metricx24.predict \
    --tokenizer google/mt5-xl \
    --model_name_or_path google/metricx-24-hybrid-xl-v2p6 \
    --max_input_length 1536 --batch_size 1 \
    --input_file input.jsonl --output_file output.jsonl
  ```
  Add `--qe` to score without a reference. Lower is better (range 0–25).
- **XCOMET-XXL** has 10.7B parameters (about 43 GB on disk, it doesn't fit on Snellius' 40 GB A100s, so the jobs use the `gpu_h100` partition). The ALMA-R paper uses it without a reference (source + translation only) and reports score × 100. Higher is better (range 0–1).
