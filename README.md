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
| Hallucination rate | own code (`scripts/score_lexical.py`): % of sentences where the candidate is at least 2× as long as the **reference**, in characters. A source-based ratio would flag almost all zh→en sentences. | nothing |

The ALMA-R paper also reports **COMET-22** (`Unbabel/wmt22-comet-da`), **KIWI-22** (`Unbabel/wmt22-cometkiwi-da`) and **KIWI-XXL** (`Unbabel/wmt23-cometkiwi-da-xxl`). We compute them too (same `unbabel-comet` package) so we can compare with the paper on every metric.

Model weights are **not** installed by pip. They download from Hugging Face the first time a model is used.

### Folder layout on Snellius

The venv, data and model weights live **next to** the repo, not inside it:

```
<project dir>/
├── Model-Compression-MT/   # this repo (code only)
├── venv/                   # Python environment
├── data/                   # datasets
├── hf_cache/               # Hugging Face models for the metrics (HF_HOME)
├── models/                 # LLM weights we translate with / quantize (ALMA-13B-R)
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

1. Log in on huggingface.co and **accept the license** on the [Unbabel/XCOMET-XXL](https://huggingface.co/Unbabel/XCOMET-XXL), [Unbabel/wmt23-cometkiwi-da-xxl](https://huggingface.co/Unbabel/wmt23-cometkiwi-da-xxl) and [Unbabel/wmt22-cometkiwi-da](https://huggingface.co/Unbabel/wmt22-cometkiwi-da) pages. These models are gated.
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
python -c "from comet import download_model as d; [d(m) for m in ['Unbabel/XCOMET-XXL', 'Unbabel/wmt23-cometkiwi-da-xxl', 'Unbabel/wmt22-cometkiwi-da', 'Unbabel/wmt22-comet-da']]"
python -c "from huggingface_hub import snapshot_download as d; d('google/metricx-24-hybrid-xl-v2p6')"
python -c "from transformers import AutoTokenizer; AutoTokenizer.from_pretrained('google/mt5-xl')"   # tokenizer only
python -c "import evaluate; evaluate.load('sacrebleu')"   # run_llmmt.py loads this at startup
```

This is a lot of disk space in your home folder: XCOMET-XXL and KIWI-XXL take about 43 GB each, ALMA-13B-R 26 GB. Check with `myquota`.

### 4. Download ALMA-13B-R

Download it into `<project dir>/models/ALMA-13B-R`, **not** into the HF cache, and **without the adapter files**. Run this from the repo root on the login node:

```bash
source scripts/snellius/env.sh
python -c "
from huggingface_hub import snapshot_download
snapshot_download('haoranxu/ALMA-13B-R',
                  revision='831d20301232e54f96c2c9af245ea219f85786d0',
                  local_dir='$PROJECT_DIR/models/ALMA-13B-R',
                  ignore_patterns=['adapter_*'])
"
ls $PROJECT_DIR/models/ALMA-13B-R   # 6 model-*.safetensors files, no adapter_*
```

Why: the `haoranxu/ALMA-13B-R` repo holds two things, the **merged ALMA-13B-R weights** (`model-*.safetensors`, 26 GB, fp16) and a stray `adapter_config.json`. When peft is installed (it is), `from_pretrained("haoranxu/ALMA-13B-R")` sees that adapter file and silently loads **`ALMA-13B-Pretrain` + adapter** instead. `ALMA-13B-Pretrain` isn't a translation model, so you'd get the wrong model. Leaving out `adapter_*` and loading from the local folder avoids this. The revision pins the exact version of the weights.

### 5. Quick checks

```bash
source ../venv/bin/activate
python -c "import sacrebleu, comet, transformers; print(sacrebleu.__version__, transformers.__version__)"
sacrebleu --help | head -n 3
cd third_party/metricx && python -c "import metricx24.models; print('metricx ok')" && cd -
```

## Reproducing the ALMA-13B-R baseline

Our full-precision baseline is a reproduction of the ALMA-R paper ([Xu et al. 2024](https://arxiv.org/abs/2401.08417)), scored with the paper's metrics plus ours. Paper numbers for ALMA-13B-R (Tables 9 and 10), averaged over de, cs, is, zh, ru:

| | BLEU | COMET-22 | KIWI-22 | KIWI-XXL | XCOMET-XXL |
|---|---|---|---|---|---|
| en→xx | 27.03 | 87.74 | 83.34 | 85.74 | 94.05 |
| xx→en | 35.45 | 85.21 | 81.33 | 82.43 | 89.11 |

The per-direction numbers are in `scripts/summarize.py`.

**Paper setup** (`third_party/ALMA/evals/alma_13b_r.sh` and `eval_generation.sh`):
- **Test data:** WMT'22 for de, cs, zh and ru; WMT'21 for is. The files are in `third_party/ALMA/human_written_data/` and `third_party/ALMA/outputs/wmt22_outputs/wmt-testset/`.
- **Generation:** beam 5, bf16, seed 42, max 256 new tokens; source length 256, or 512 for zh→en. The script doesn't set `do_sample`, so the model's `generation_config.json` applies (`do_sample=true`, temperature 0.9, top_p 0.6): the paper used beam **sampling**.
- **Scoring:** BLEU with sacrebleu (tokenizer `zh` for Chinese targets, else `13a`); COMET-22 with a reference; KIWI-22, KIWI-XXL and XCOMET-XXL without one. All × 100.

**Two generation runs:**
- `ours`: paper decoding. This is the reproduction, compared with the paper.
- `ours-beam`: plain beam search (`do_sample=False`), so the output is deterministic. This is the baseline the quantized models are compared against, because with sampling part of any score difference would be sampling noise.

Submit all jobs **from the repo root**. `RUN` selects the translations to score: `paper` (the paper's own outputs), `ours` or `ours-beam`. Scores go to `<project dir>/outputs/<metric>/<RUN>/summary.tsv`.

**Step 1: check the metrics.** Score the paper's own ALMA-13B-R translations. This should reproduce the paper's numbers (XCOMET-XXL is already done: 94.04 / 89.11):
```bash
sbatch --array=1-3 --export=ALL,RUN=paper scripts/snellius/score_comet.slurm   # 0 xcomet-xxl, 1 kiwi-xxl, 2 kiwi-22, 3 comet-22
sbatch --export=ALL,RUN=paper scripts/snellius/score_metricx.slurm
python scripts/score_lexical.py --run paper   # BLEU, chrF++, hallucination rate; seconds, fine on the login node
python scripts/summarize.py --run paper
```

**Step 2: generate** (4× H100, both jobs can run at the same time). At the end, each job checks that there is one translation per source sentence.
```bash
sbatch scripts/snellius/generate_alma_r.slurm                             # -> outputs/alma-13b-r/wmt22
sbatch --export=ALL,DECODING=beam scripts/snellius/generate_alma_r.slurm  # -> outputs/alma-13b-r-beam/wmt22
```
It uses `<project dir>/models/ALMA-13B-R` by default; pass `MODEL=/path` for another folder.

**Step 3: score both runs** (for `RUN=ours` and `RUN=ours-beam`):
```bash
sbatch --export=ALL,RUN=ours scripts/snellius/score_comet.slurm
sbatch --export=ALL,RUN=ours scripts/snellius/score_metricx.slurm
python scripts/score_lexical.py --run ours
```

**Step 4: compare.**
```bash
python scripts/summarize.py --run ours --vs paper   # reproduction: vs the paper's reported numbers and vs the paper's outputs
python scripts/summarize.py --run ours-beam --vs ours
```
It prints all metrics per direction, the difference with the paper, and writes `outputs/baseline/<RUN>.tsv`.

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
