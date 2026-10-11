# Low-Bit Quantization for Multilingual Machine Translation

Code, results and paper source for *Low-Bit Quantization for Multilingual Machine Translation*
(Krijn Dignum, Francesco Massafra, Matthijs Vork, Max Wilde, Reinout Wolting; University of Amsterdam).

We test how much we can shrink [ALMA-13B-R](https://huggingface.co/haoranxu/ALMA-13B-R) while keeping its
translation quality. We use GPTQ to store the model's weights with fewer bits, then measure translation quality
in ten language directions using the ALMA-R paper's test sets (WMT'22, or WMT'21 for Icelandic).
For the 3-bit and 2-bit models, we also train a small add-on, called a LoRA adapter, to learn from the original
model and recover translation quality.

| Model | Disk (GiB) | Peak GPU, batch 1 (GiB) | BLEU | XCOMET-XXL | MetricX-24 ↓ |
|---|---:|---:|---:|---:|---:|
| fp16 | 24.2 | 25.4 | 31.4 | 92.1 | 2.38 |
| w4 | 6.8 | 8.9 | 30.5 | 91.3 | 2.48 |
| w3 | 5.3 | 6.8 | 23.3 | 78.4 | 4.72 |
| w2 | 3.8 | 5.0 | 0.0 | 22.7 | 9.91 |
| **w3 + distilled adapter** (r = 16) | 5.4 | 6.9 | 31.1 | 91.9 | 2.43 |
| **w2 + distilled adapter** (r = 64) | 4.2 | 5.5 | 27.8 | 88.1 | 3.16 |

Averages over the ten language directions. `fp16` is the original 16-bit model; `w4`, `w3` and `w2` store weights
with 4, 3 and 2 bits. Adapter sizes are counted using 16-bit values (bf16); `r` controls the adapter's size.
Higher BLEU and XCOMET-XXL scores are better; lower MetricX-24 scores are better.
Full results: paper Table 1 and Appendix A.

- The 4-bit model uses much less memory with little loss of translation quality. Below that, quality drops
  unevenly across languages, with Icelandic particularly affected.
- Tests using reference translations suggest that the 3-bit model retains knowledge of Icelandic but struggles
  to translate between Icelandic and English.
- An adapter trained using the original model brings the 3-bit model close to the original translation quality,
  and turns the collapsed 2-bit model into
  a usable translator that fits an 8 GB GPU.

To redraw the paper's figures from the saved results, follow the short setup below. To run the translation and
training experiments yourself, use the full reproduction instructions; those require an NVIDIA GPU cluster.

## Contents

- [Repository layout](#repository-layout)
- [Updating an existing checkout](#updating-an-existing-checkout)
- [Rebuild the paper's figures and tables (CPU, one minute)](#rebuild-the-papers-figures-and-tables)
- [Full reproduction](#full-reproduction): [setup](#1-get-the-code), then steps [A](#a-full-precision-baseline-section-21)–[I](#i-rebuild-and-check-the-paper)
- [Notes](#notes), [Licence](#licence), [Citation](#citation)

## Repository layout

```
mtcompress/        shared code: paths.py (all folders), alma_prompt.py (ALMA's prompt), lora.py (the adapter)
scripts/
  reproduce/       score_lexical.py (BLEU, chrF++), summarize.py (one score table per run)
  quantize/        quantize_gptq.py, generate_quantized.py, repack_to_w4.py (lossless 3/2-bit -> 4-bit layout)
  distill/         train adapters using the original model, then translate with them
  analysis/        translation failures, memory, speed, uncertainty and model-weight measurements
  paper/           every figure and table of the paper, built from results/
slurm/             one job per step (Snellius, H100); env.sh sets all paths
results/
  scores/          overall translation scores for each model and language direction
  json/            saved measurements used by the analyses and paper
  distill_runs/    adapter configs and training logs of the three distillation runs
paper/             LaTeX source, figures and generated tables
third_party/       ALMA (prompts, data, test sets, generation code) and MetricX, as git submodules
```

Translations, segment-level scores and model weights are not in the repository.

## Updating an existing checkout

The scripts have moved. Update saved commands and job submissions to use these locations:

| Previous location | Current location |
|---|---|
| `scripts/snellius/*.slurm` | `slurm/*.job` (same job names) |
| `scripts/snellius/env.sh` | `slurm/env.sh` |
| `scripts/score_lexical.py` | `scripts/reproduce/score_lexical.py` |
| `scripts/summarize.py` | `scripts/reproduce/summarize.py` |
| `outputs/baseline/<run>.tsv` | `results/scores/<run>.tsv` for newly written summaries |

Existing translations and metric outputs can stay in the `outputs/` folder next to the repository.
Old summary files are not moved automatically. Running `summarize.py` writes to `results/scores/` and replaces
the saved table for that run, so check the resulting changes before committing them.
Follow steps 2 and 3 below to set the folder locations and install the shared Python code with `pip install -e .`.
Keep the two Python environments separate.

## Rebuild the paper's figures and tables

No GPU and no model files needed. Use Python 3.10–3.12 and `make`:

```bash
git clone https://github.com/krijnD/-Model-Compression-for-Machine-Translation-in-Large-Language-Models.git Model-Compression-MT
cd Model-Compression-MT
python3 -m venv .venv-paper
source .venv-paper/bin/activate
python -m pip install matplotlib==3.9.4 numpy==1.26.4 scipy==1.13.1
make paper                # redraw figures, write the appendix table, check Tables 1 and 2
git diff --exit-code      # check whether the rebuilt files differ from the saved versions
```

If you already have the repository, start with `cd` into its folder and skip cloning it.
This figures-only setup reads the saved results and does not need the external code in `third_party/`.

To build the PDF, install a LaTeX distribution, then run:
`cd paper && pdflatex acl_latex && bibtex acl_latex && pdflatex acl_latex && pdflatex acl_latex`.

| Paper | Built by | From |
|---|---|---|
| Table 1 (memory, time, quality) | `scripts/paper/make_main_table.py` (check) | `results/json/{quant_cost,probe16}_*.json`, `results/scores/` |
| Figure 1 (XCOMET-XXL change per direction) | `scripts/paper/plot_rq1_heatmap.py` | `results/scores/` |
| Figure 2 (memory vs quality, with adapters) | `scripts/paper/plot_rq2_tradeoff.py` | `results/json/probe16_*.json`, `results/scores/` |
| Table 2 (failure rates) | `scripts/paper/make_failures_table.py` (check) | `results/json/failures.json` |
| Table 3 (all metrics, all directions) | `scripts/paper/make_full_table.py` | `results/scores/` |
| Figure 3 (NLL at 3 bits) | `scripts/paper/plot_nll_w3.py` | `results/json/lang_nll_*.json` |
| Figure 4 (recovered 3-bit model) | `scripts/paper/plot_heatmap_adapter.py` | `results/json/adapter_heatmap.json` |

The steps below explain how to produce the main results. The additional inputs needed for the failure-significance
analysis are listed in [step H](#h-cpu-analyses).

## Full reproduction

Everything ran on [Snellius](https://www.surf.nl/en/services/snellius-the-national-supercomputer), one NVIDIA H100
(94 GB) per job, four for the fp16 generation. The jobs are plain Slurm scripts. On another cluster, change the
`module load` lines in `slurm/env.sh` and the `#SBATCH --partition` lines.

| Step | What | GPU time (H100) |
|---|---|---|
| [A](#a-full-precision-baseline-section-21) | fp16 baseline: generate twice, score three runs | ~10 GPU-hours per generation; ~1.5 h scoring per run |
| [B](#b-quantize-and-evaluate-the-grid-sections-22-31-32) | GPTQ at 8/4/3/2 bits, generate, score | 16.5 min per bit width; 10–20 h per model |
| [C](#c-repack-3--and-2-bit-weights-into-the-4-bit-layout) | repack w3/w2, kernel speed and drift | one job, at most 3 h |
| [D](#d-distil-the-adapters-section-23-appendix-b) | three distillation runs | 40–80 min each |
| [E](#e-translate-and-score-with-the-adapters-section-33) | generate with the adapters, score | at most 5 h per generation job; ~1.5 h scoring per run |
| [F](#f-teacher-forced-nll-sections-32-33) | NLL of every model | one job, at most 2 h |
| [G](#g-memory-and-speed-table-1-limitations) | memory and speed probes | nine short jobs, 1–2 h each at most |
| [H](#h-cpu-analyses) | CPU analyses | minutes on a login node, plus one GPU job of at most 1 h |
| [I](#i-rebuild-and-check-the-paper) | figures, tables, checks | seconds, CPU |

Disk: about 75 GB for models (fp16, four GPTQ checkpoints, two repacked copies, adapters, language identifier),
100 GB of metric models in the Hugging Face cache, 0.5 GB of outputs; 22 GB more for the Appendix D weight analysis.

### 1. Get the code

```bash
git clone --recurse-submodules https://github.com/krijnD/-Model-Compression-for-Machine-Translation-in-Large-Language-Models.git Model-Compression-MT
cd Model-Compression-MT
mkdir -p logs   # Slurm writes the job logs here (the folder is in git, but must exist)
```

Check the submodules: `ls third_party/ALMA/run_llmmt.py third_party/metricx/metricx24/predict.py`.

### 2. Folders

Large files live next to the repository. `slurm/env.sh` and `mtcompress/paths.py` read these variables:

| Variable | Default | Holds |
|---|---|---|
| `PROJECT_DIR` | the repository's parent folder | `venv/`, `venv-quant/`, `hf_cache/` |
| `MODELS_DIR` | set explicitly to `$PROJECT_DIR/models` below | fp16 ALMA-13B-R (26 GB), the GPTQ checkpoints (3.8–12.7 GiB each) |
| `ARTIFACTS_DIR` | `$MODELS_DIR` | repacked checkpoints (6.8 GiB each), adapters, the language identifier |
| `OUTPUTS_DIR` | `$PROJECT_DIR/outputs` | translations and segment-level scores |

Run this from the repository root in each new shell before downloading models or submitting jobs.
Change the paths here if your files live elsewhere:

```bash
export PROJECT_DIR="$(dirname "$PWD")"
export MODELS_DIR="$PROJECT_DIR/models"
export ARTIFACTS_DIR="$MODELS_DIR"
export OUTPUTS_DIR="$PROJECT_DIR/outputs"
export HF_HOME="$PROJECT_DIR/hf_cache"
```

**Known setup issue:** `slurm/env.sh` currently fails with `MODELS_DIR: unbound variable` if `MODELS_DIR`
is unset. The explicit setting above avoids this error. Submit every job from the repository root;
`sbatch --export=ALL` passes these settings to the job.

### 3. Environments

Use two separate Python environments: ALMA's translation code needs `transformers==4.45.2`, while the
compression tools need a newer version. The commands below target Snellius: Linux x86-64, Python 3.12 and
an NVIDIA GPU. The GPTQModel download in `requirements-quant.txt` does not work on macOS or other Python versions.

```bash
module purge; module load 2024; module load Python/3.12.3-GCCcore-13.3.0

# venv: fp16 generation, all metrics, CPU analyses, figures
python -m venv ../venv && source ../venv/bin/activate
pip install --upgrade pip && pip install -r requirements.txt && pip install -e .

# venv-quant: quantization, distillation, generation with quantized models, GPU measurements
python -m venv ../venv-quant && source ../venv-quant/bin/activate
pip install --upgrade pip
pip install torch==2.8.0 --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements-quant.txt && pip install -e .
```

The jobs pick the right environment themselves. `requirements.txt` explains the `transformers==4.45.2` pin.

### 4. Models and data

The test sets (WMT'22, WMT'21 for Icelandic) and ALMA's training data come with the ALMA submodule. Download the rest
on the login node, with `venv` active and `HF_HOME` set (`slurm/env.sh` sets it for the jobs):

```bash
# Use the folder settings from step 2 and switch back to the scoring environment.
source "$PROJECT_DIR/venv/bin/activate"
# First accept the licences of the gated models on huggingface.co: Unbabel/XCOMET-XXL,
# Unbabel/wmt23-cometkiwi-da-xxl and Unbabel/wmt22-cometkiwi-da. Then log in with a read token:
huggingface-cli login

# Metrics (XCOMET-XXL and KIWI-XXL are gated and take about 43 GB each)
python -c "from comet import download_model as d; [d(m) for m in ['Unbabel/XCOMET-XXL', 'Unbabel/wmt23-cometkiwi-da-xxl', 'Unbabel/wmt22-cometkiwi-da', 'Unbabel/wmt22-comet-da']]"
python -c "from huggingface_hub import snapshot_download as d; d('google/metricx-24-hybrid-xl-v2p6')"
python -c "from transformers import AutoTokenizer; AutoTokenizer.from_pretrained('google/mt5-xl')"
python -c "import evaluate; evaluate.load('sacrebleu')"   # ALMA's generation script loads it at startup

# ALMA-13B-R, pinned, WITHOUT its adapter files (see below)
python -c "
import os; from huggingface_hub import snapshot_download
snapshot_download('haoranxu/ALMA-13B-R', revision='831d20301232e54f96c2c9af245ea219f85786d0',
                  local_dir=os.environ['MODELS_DIR'] + '/ALMA-13B-R', ignore_patterns=['adapter_*'])"

# NLLB language identifier (failure modes, step H)
python -c "
import os, shutil; from huggingface_hub import hf_hub_download
f = hf_hub_download('facebook/fasttext-language-identification', 'model.bin', revision='3af127d4124fc58b75666f3594bb5143b9757e78')
os.makedirs(os.environ['ARTIFACTS_DIR'] + '/lid', exist_ok=True); shutil.copy(f, os.environ['ARTIFACTS_DIR'] + '/lid/lid218e.bin')"
```

The `haoranxu/ALMA-13B-R` repository holds the merged weights and a stray `adapter_config.json`. With peft installed,
`from_pretrained("haoranxu/ALMA-13B-R")` then silently loads ALMA-13B-Pretrain plus that adapter, which is not a
translation model. Hence the local copy without `adapter_*`.

Only for the weight analysis in Appendix D (step H), 22 GB:

```bash
python -c "
from huggingface_hub import hf_hub_download as d
D = '$PROJECT_DIR/alma_pretrain'
for f in ['pytorch_model.bin.index.json'] + [f'pytorch_model-0000{i}-of-00006.bin' for i in (1, 3, 6)]:
    d('haoranxu/ALMA-13B-Pretrain', f, local_dir=D)
d('haoranxu/ALMA-13B-Pretrain-LoRA', 'adapter_model.bin', local_dir=D)"
```

**Already have our model files?** Put the GPTQ checkpoints in `$MODELS_DIR` and the adapters in `$ARTIFACTS_DIR`,
with their folder names unchanged, and skip `quantize_gptq.job` (step B) and step D.

### A. Full-precision baseline (Section 2.1)

```bash
sbatch slurm/generate_alma_r.job                              # paper decoding (beam sampling)  -> run "ours"
sbatch --export=ALL,DECODING=beam slurm/generate_alma_r.job   # plain beam search               -> run "ours-beam"
# when both are done:
for run in paper ours ours-beam; do
  sbatch --export=ALL,RUN=$run slurm/score_comet.job          # XCOMET-XXL, KIWI-XXL, KIWI-22, COMET-22 (array 0-3)
  sbatch --export=ALL,RUN=$run slurm/score_metricx.job        # MetricX-24
done
# when the scoring is done (CPU, venv active):
for run in paper ours ours-beam; do python scripts/reproduce/score_lexical.py --run $run; done
python scripts/reproduce/summarize.py --run ours --vs paper   # -> results/scores/ours.tsv
python scripts/reproduce/summarize.py --run ours-beam --vs ours
```

`RUN=paper` scores the ALMA-R paper's own outputs (in the ALMA submodule) and checks the metric setup. Our
reproduction is within 0.4 BLEU and 0.5 XCOMET-XXL of the paper on average. All quantized models are compared with
`ours-beam` (deterministic), because sampling would add noise to every difference.

### B. Quantize and evaluate the grid (Sections 2.2, 3.1, 3.2)

```bash
sbatch --array=2,3,4,8 slurm/quantize_gptq.job   # -> $MODELS_DIR/ALMA-13B-R-gptq-w{2,3,4,8}g128, 16.5 min each
# when done:
for q in gptq-w8g128 gptq-w4g128 gptq-w3g128 gptq-w2g128; do
  sbatch --export=ALL,QUANT=$q slurm/eval_quantized.job   # generate + 7 metrics -> results/scores/$q.tsv
done
```

Generation takes 10–20 hours per model (2-bit is slowest, its outputs degenerate). If a job hits its time limit,
submit the same command again: finished directions and metrics are skipped.

### C. Repack 3- and 2-bit weights into the 4-bit layout

GPTQModel has a fused kernel for 4 bits only. `repack_to_w4.py` stores the same 3- and 2-bit values in the 4-bit
layout, losslessly (verified on the written files), so that generation can use the ExLlamaV2 kernel. All adapter
generations in step E run on these copies.

```bash
sbatch slurm/repack.job   # -> $ARTIFACTS_DIR/ALMA-13B-R-gptq-w{3,2}g128-as-w4
                          #    results/json/repack_w{3,2}_written.json (lossless check),
                          #    backend_w{3,2}as4_*.json (speed), repack_quality_*.json (output drift)
```

### D. Distil the adapters (Section 2.3, Appendix B)

```bash
# w3, rank 16, 30K examples (~45 min)
sbatch slurm/distill.job
# w2, rank 64: 30K examples, then 60K more from that adapter with a fresh schedule (submit the second when the first is done)
sbatch --export=ALL,BITS=2,ADAPTER=$ARTIFACTS_DIR/ALMA-13B-R-gptq-w2g128-kd-lora-r64,EXTRA="--r 64 --alpha 128" slurm/distill.job
sbatch --export=ALL,BITS=2,ADAPTER=$ARTIFACTS_DIR/ALMA-13B-R-gptq-w2g128-kd-lora-r64-cont,EXTRA="--r 64 --alpha 128 --n-examples 60000 --init-adapter $ARTIFACTS_DIR/ALMA-13B-R-gptq-w2g128-kd-lora-r64" slurm/distill.job
```

Each run writes `adapter.safetensors`, `adapter_config.json` and `train_log.jsonl`. Our configurations and logs
are in `results/distill_runs/`; the adapter weights are not included. Training needs an H100: it peaks at
36–37 GiB. `SMOKE=1` runs a 30-step test first.
The final 2-bit adapter is `…-r64-cont`; the first-stage `…-r64` is evaluated on two directions in step E for
Appendix D (iv).

### E. Translate and score with the adapters (Section 3.3)

All generations run on the repacked checkpoints of step C. `PAIRS` takes `a:b:c` (Slurm splits commas).

```bash
# w3 + adapter, all ten directions; also its NLL (Section 3.3)
sbatch --export=ALL,NLL=1,TAG=w3kd slurm/gen_lora.job                                      # -> run gptq-w3g128-as4-kd
# w2 + final adapter, all ten directions; also its NLL
sbatch --export=ALL,BITS=2,ADAPTER=$ARTIFACTS_DIR/ALMA-13B-R-gptq-w2g128-kd-lora-r64-cont,KD_RUN=gptq-w2g128-as4-kd-r64-cont,NLL=1,TAG=w2kd-r64-cont slurm/gen_lora.job
# plain repacked w3 on four directions (Figure 4: the same kernel as the w3 adapter run)
sbatch --export=ALL,ADAPTER=none,KD_RUN=gptq-w3g128-as4,PAIRS=is-en:en-is:de-en:en-de slurm/gen_lora.job
# first-stage w2 adapter on two directions (Appendix D (iv), bootstrap)
sbatch --export=ALL,BITS=2,ADAPTER=$ARTIFACTS_DIR/ALMA-13B-R-gptq-w2g128-kd-lora-r64,KD_RUN=gptq-w2g128-as4-kd-r64,PAIRS=is-en:de-en slurm/gen_lora.job

# when the generations are done:
sbatch --export=ALL,RUN=gptq-w3g128-as4-kd slurm/score_full.job            # 7 metrics -> results/scores/<run>.tsv
sbatch --export=ALL,RUN=gptq-w2g128-as4-kd-r64-cont slurm/score_full.job
sbatch --export=ALL,KD_RUN=gptq-w3g128-as4,PAIRS=is-en:en-is:de-en:en-de slurm/score_xcomet.job   # XCOMET-XXL only
sbatch --export=ALL,KD_RUN=gptq-w2g128-as4-kd-r64,PAIRS=is-en:de-en slurm/score_xcomet.job
```

`gen_lora.job` fits a 40 GB A100 too (`sbatch --partition=gpu_a100 ...`); scoring needs an H100 (XCOMET-XXL and
KIWI-XXL need about 43 GB).

### F. Teacher-forced NLL (Sections 3.2, 3.3)

```bash
sbatch slurm/lang_nll.job   # fp16, w8, w4, w3, w2 -> results/json/lang_nll_{fp16,w8,w4,w3,w2}.json
```

The adapter rows (`lang_nll_w3kd.json`, `lang_nll_w2kd-r64-cont.json`) come from `NLL=1` in step E.

### G. Memory and speed (Table 1, Limitations)

Peak memory does not depend on the GPU model; time does, so all times in the paper are from the H100.

```bash
sbatch slurm/quant_cost.job                                           # checkpoint size, load time -> quant_cost_<tag>.json
sbatch slurm/backends.job                                             # every usable kernel per checkpoint -> backend_*.json
sbatch slurm/probe.job                                                # batch 4 and 16 -> probe16_<tag>.json
sbatch --export=ALL,TAGS=fp16:w8:w4:w3:w2,BATCHES=1,SUFFIX=_b1 slurm/probe.job                 # batch 1 -> probe16_<tag>_b1.json
# with the adapters (after steps C and D); peak memory, so an A100 is fine:
sbatch --partition=gpu_a100 slurm/probe_adapter.job                   # -> probe16_{w3kd,w2kd,w3as4kd,w2as4kd}.json
sbatch --partition=gpu_a100 --export=ALL,ONLY=w3kd,BATCHES=1,SUFFIX=_b1 slurm/probe_adapter.job
sbatch --partition=gpu_a100 --export=ALL,ONLY=w2kd,BATCHES=1:2,SUFFIX=_b1 slurm/probe_adapter.job
# time per sentence with the adapters, on the H100:
sbatch --export=ALL,ONLY=w3kd,BATCHES=4,SUFFIX=_h100 slurm/probe_adapter.job
sbatch --export=ALL,ONLY=w2kd,BATCHES=4,SUFFIX=_h100 slurm/probe_adapter.job
```

All outputs go to `results/json/`.

### H. CPU analyses

With `venv` active (`pip install -e .` done, or `export PYTHONPATH=$PWD`), after steps A–E:

| Command | Output | Paper |
|---|---|---|
| `python scripts/analysis/score_failures.py` | `results/json/failures.json` | Table 2, Section 3.4 |
| `python scripts/analysis/failure_significance.py --wmt24pp <WMT24++ test set>` | `results/json/failure_significance.json` | Bold cells of Tables 2, 7 and 9, Tables 7 and 9, Sections 3.5 and D |
| `python scripts/analysis/bootstrap_ci.py` | `results/json/bootstrap_ci.json` | Section 3.3, Appendix A |
| `python scripts/analysis/audit_recovery.py` | `results/json/audit_*.json` | Section 3.4, Appendix D |
| `python scripts/analysis/fragility_by_confidence.py` | `results/json/fragility_by_confidence.json` | Section 3.2 |
| `python scripts/analysis/fragility_by_confidence.py --bits 4 --out results/json/fragility_by_confidence_w4.json` | | Section 3.2 |
| `python scripts/paper/plot_heatmap_adapter.py --from-outputs` | `results/json/adapter_heatmap.json` | Figure 4 |
| `sbatch slurm/measure_channels.job` (GPU, ~1 h), then `python scripts/analysis/measure_channels.py --compare $OUTPUTS_DIR/channels/channels_fp16.npz` | `$OUTPUTS_DIR/channels/` and the printed tables | Section 3.2 (calibration share, outliers) |
| `python scripts/analysis/measure_delta.py --pretrain-dir $PROJECT_DIR/alma_pretrain --out results/json/delta_erasure.json` (venv-quant, ~3 GB RAM per module) | `results/json/delta_erasure.json` | Appendix D |

`score_failures.py` uses the language identifier from step 4; `audit_recovery.py` also writes the worst segments
per direction to `$OUTPUTS_DIR/audit/`.

**Extra inputs for `failure_significance.py`:** steps A–E do not produce everything this script reads.
It also requires translations for `gptq-w2g128-as4-sft-r64-cont`, a model trained directly on reference translations,
in `$OUTPUTS_DIR/gptq-w2g128-as4-sft-r64-cont/wmt22/`. The repository does not currently include instructions
to produce that run. Without those files, the script stops with a missing-file error.
The optional WMT24++ analysis additionally needs its test set and translations for `ood24-fp16`, `ood24-w2`
and `ood24-w2kd`, stored under `$OUTPUTS_DIR/<run>/wmt24pp/`. That part is skipped if the test-set folder is absent.
The saved `results/json/failure_significance.json` is available to inspect without rerunning this analysis.

### I. Rebuild and check the paper

```bash
make PY=python paper   # figures and tables from results/, then the cell-by-cell check of Tables 1 and 2
git diff --stat        # what changed against the published numbers
```

Expect small differences in the time columns on other hardware. Everything else is deterministic (beam search,
seed 42), apart from `ours`, which uses the paper's beam sampling and is not used for any model comparison.

## Notes

- **Evaluation protocol.** As in ALMA-R: beam 5, bf16, seed 42, at most 256 new tokens, source length 256 (512 for
  zh→en, at batch 1); BLEU with sacreBLEU (`zh` tokenizer for Chinese targets, else `13a`); COMET-22 and MetricX-24
  with the reference, XCOMET-XXL and both KIWI models without it. Scores ×100, except MetricX-24 (0–25, lower is
  better).
- **Quantization.** GPTQModel 4.2.5; asymmetric, group size 128, act-order, damping 0.05; 1,024 calibration
  sentences from ALMA's human-written training data, balanced over the ten directions. Embeddings, norms and the
  output head stay fp16. Each checkpoint folder has a `quant_meta.json` with every setting.
- **Two generation paths.** ALMA's `run_llmmt.py` cannot load GPTQ weights, so quantized models translate with
  `scripts/quantize/generate_quantized.py`, which reproduces its prediction path (prompt, left padding to the full
  source length, decoding, translation extraction). Tokenization of all ten test sets is identical in both
  environments.
- **Adapters** are kept in fp32 in every run; the paper counts them in bf16. They use `mtcompress/lora.py`, not peft,
  because the same adapter must wrap two GPTQ kernels (training on Torch, generation on ExLlamaV2).
- **Data.** ALMA's training data (calibration, distillation) does not overlap the WMT'22/'21 test sets; it does
  contain FLORES-200, so FLORES is not a clean test set for these models.
- **Reproducibility.** Packages are pinned in the requirements files, code in the submodule commits, models in their
  Hugging Face revisions. Each result comes from one seed; the paper reports paired bootstrap intervals over
  sentences, not seed variance.
- **MetricX-24** comes from the `third_party/metricx` submodule (not on PyPI) and runs on `transformers==4.45.2`.

## Licence

The code is released under the MIT licence (`LICENSE`). It builds on [ALMA](https://github.com/fe1ixxu/ALMA) (MIT),
[MetricX](https://github.com/google-research/metricx) (Apache 2.0), [GPTQModel](https://github.com/ModelCloud/GPTQModel)
and [COMET](https://github.com/Unbabel/COMET). ALMA-13B-R is released under the MIT licence and is built on LLaMA-2-13B.

## Citation

```bibtex
@misc{dignum2026lowbit,
  title  = {Low-Bit Quantization for Multilingual Machine Translation},
  author = {Dignum, Krijn and Massafra, Francesco and Vork, Matthijs and Wilde, Max and Wolting, Reinout},
  year   = {2026},
  note   = {University of Amsterdam}
}
```
