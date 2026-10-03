# Low-Bit Quantization for Multilingual Machine Translation

Code, results and paper source for *Low-Bit Quantization for Multilingual Machine Translation*
(Krijn Dignum, Francesco Massafra, Matthijs Vork, Max Wilde, Reinout Wolting; University of Amsterdam).

We quantize [ALMA-13B-R](https://huggingface.co/haoranxu/ALMA-13B-R) with GPTQ to 8, 4, 3 and 2 bits and evaluate it
on the ten WMT'22 directions of the ALMA-R paper (WMT'21 for Icelandic) with seven metrics. Keeping the quantizer
fixed, we then distil the full-precision model into a small LoRA adapter on the 3-bit and 2-bit models.

| Model | Disk (GiB) | Peak GPU, batch 1 (GiB) | BLEU | XCOMET-XXL | MetricX-24 ↓ |
|---|---:|---:|---:|---:|---:|
| fp16 | 24.2 | 25.4 | 31.4 | 92.1 | 2.38 |
| w4 | 6.8 | 8.9 | 30.5 | 91.3 | 2.48 |
| w3 | 5.3 | 6.8 | 23.3 | 78.4 | 4.72 |
| w2 | 3.8 | 5.0 | 0.0 | 22.7 | 9.91 |
| **w3 + distilled adapter** (r = 16) | 5.4 | 6.9 | 31.1 | 91.9 | 2.43 |
| **w2 + distilled adapter** (r = 64) | 4.2 | 5.5 | 27.8 | 88.1 | 3.16 |

Averages over the ten directions; adapters counted in bf16. Full results: paper Table 1 and Appendix A.

- Down to 4 bits, quantization is nearly free. Below that, quality collapses unevenly: Icelandic first.
- The 3-bit model keeps its knowledge of Icelandic but loses the Icelandic–English mapping (teacher-forced NLL).
- A distilled adapter restores the 3-bit model to full-precision quality, and turns the collapsed 2-bit model into
  a usable translator that fits an 8 GB GPU.

## Repository layout

```
mtcompress/        shared code: paths.py (all folders), alma_prompt.py (ALMA's prompt), lora.py (the adapter)
scripts/
  reproduce/       score_lexical.py (BLEU, chrF++), summarize.py (one score table per run)
  quantize/        quantize_gptq.py, generate_quantized.py, repack_to_w4.py (lossless 3/2-bit -> 4-bit layout)
  distill/         distill_lora.py (KL distillation into a LoRA), generate_lora.py
  analysis/        NLL, memory/latency probes, failure modes, bootstrap CIs, sanity checks, weight analyses
  paper/           every figure and table of the paper, built from results/
slurm/             one job per step (Snellius, H100), env.sh sets all paths
results/
  scores/          corpus-level scores of every model on every metric and direction
  json/            NLL, probes, kernels, failure rates, bootstrap CIs, audits, weight analyses
  distill_runs/    adapter configs and training logs of the three distillation runs
paper/             LaTeX source, figures and generated tables
third_party/       ALMA (prompts, data, test sets, generation) and MetricX, as git submodules
```

Translations, segment-level scores and model weights are not in the repository.

## Rebuild the paper's figures and tables (CPU, seconds)

```bash
git clone --recurse-submodules <this repository>
cd <repository>
pip install -r requirements.txt   # or only: pip install matplotlib==3.9.4 numpy==1.26.4 scipy==1.13.1
make paper                        # figures, appendix table, and a cell-by-cell check of Tables 1 and 2
git diff --exit-code              # the rebuilt files are byte-identical to the committed ones
```

| Paper | Built by | From |
|---|---|---|
| Table 1 (memory, time, quality) | `scripts/paper/make_main_table.py` (check) | `results/json/{quant_cost,probe16}_*.json`, `results/scores/` |
| Figure 1 (XCOMET-XXL change per direction) | `scripts/paper/plot_rq1_heatmap.py` | `results/scores/` |
| Figure 2 (memory vs quality, with adapters) | `scripts/paper/plot_rq2_tradeoff.py` | `results/json/probe16_*.json`, `results/scores/` |
| Table 2 (failure rates) | `scripts/paper/make_failures_table.py` (check) | `results/json/failures.json` |
| Table 3 (all metrics, all directions) | `scripts/paper/make_full_table.py` | `results/scores/` |
| Figure 3 (NLL at 3 bits) | `scripts/paper/plot_nll_w3.py` | `results/json/lang_nll_*.json` |
| Figure 4 (recovered 3-bit model) | `scripts/paper/plot_heatmap_adapter.py` | `results/json/adapter_heatmap.json` |
| Significance (Section 3.3) | `scripts/analysis/bootstrap_ci.py` | → `results/json/bootstrap_ci.json` |
| Sanity checks, error spans (Section 3.4, App. D) | `scripts/analysis/audit_recovery.py` | → `results/json/audit_*.json` |
| Fine-tuning survives quantization (App. D) | `scripts/analysis/measure_delta.py` | → `results/json/delta_erasure.json` |
| Difficulty-matched damage (Section 3.2) | `scripts/analysis/fragility_by_confidence.py` | → `results/json/fragility_by_confidence*.json` |
| Repack is lossless and faster (Limitations) | `scripts/quantize/repack_to_w4.py`, `slurm/repack.job` | → `results/json/{repack,backend}_*.json` |

Build the PDF with `cd paper && pdflatex acl_latex && bibtex acl_latex && pdflatex acl_latex && pdflatex acl_latex`
(needs the `inconsolata` package).

The scripts marked → need the translations and segment-level scores in `$OUTPUTS_DIR`, which the pipeline below
produces.

## Full pipeline

Everything below ran on [Snellius](https://www.surf.nl/en/services/snellius-the-national-supercomputer) on one
NVIDIA H100 (94 GB) per job (four for the fp16 generation). The jobs are plain Slurm scripts; outside Snellius,
change the `module load` lines in `slurm/env.sh` and the `#SBATCH --partition` lines.

### Folders

Large files live next to the repository. `slurm/env.sh` and `mtcompress/paths.py` read these variables:

| Variable | Default | Holds |
|---|---|---|
| `PROJECT_DIR` | the repository's parent folder | `venv/`, `venv-quant/`, `hf_cache/` |
| `MODELS_DIR` | `$PROJECT_DIR/models` | fp16 ALMA-13B-R (26 GB) and the GPTQ checkpoints (3.8–12.7 GiB each) |
| `ARTIFACTS_DIR` | `$MODELS_DIR` | adapters, repacked checkpoints (6.8 GiB each), the language identifier |
| `OUTPUTS_DIR` | `$PROJECT_DIR/outputs` | translations, segment-level scores (about 0.5 GB) |

Set them in your shell before submitting (commands below expand `$ARTIFACTS_DIR` at submit time), or per job: `sbatch --export=ALL,ARTIFACTS_DIR=/scratch-shared/$USER/models slurm/distill.job`.
Run all jobs from the repository root. Job logs go to `logs/`.

### Environments

Two virtual environments, because ALMA's generation code needs `transformers<=4.45` and GPTQModel needs `>=4.56`:

```bash
module purge; module load 2024; module load Python/3.12.3-GCCcore-13.3.0

# venv: ALMA-R generation, all metrics, analyses, figures
python -m venv ../venv && source ../venv/bin/activate
pip install -r requirements.txt && pip install -e .

# venv-quant: quantization, distillation, generation with quantized models, GPU measurements
python -m venv ../venv-quant && source ../venv-quant/bin/activate
pip install torch==2.8.0 --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements-quant.txt && pip install -e .
```

`requirements.txt` explains the `transformers==4.45.2` pin. The jobs set `PYTHONPATH`, so `pip install -e .` is only
needed to run the scripts outside Slurm.

### Hugging Face models

1. Accept the licences of the gated models
   [Unbabel/XCOMET-XXL](https://huggingface.co/Unbabel/XCOMET-XXL),
   [Unbabel/wmt23-cometkiwi-da-xxl](https://huggingface.co/Unbabel/wmt23-cometkiwi-da-xxl) and
   [Unbabel/wmt22-cometkiwi-da](https://huggingface.co/Unbabel/wmt22-cometkiwi-da), then `huggingface-cli login`.
2. Set `export HF_HOME=$PROJECT_DIR/hf_cache` (`slurm/env.sh` does this for the jobs). XCOMET-XXL and KIWI-XXL take
   about 43 GB each.
3. If compute nodes have no internet, download the metric models on the login node first:
   ```bash
   python -c "from comet import download_model as d; [d(m) for m in ['Unbabel/XCOMET-XXL', 'Unbabel/wmt23-cometkiwi-da-xxl', 'Unbabel/wmt22-cometkiwi-da', 'Unbabel/wmt22-comet-da']]"
   python -c "from huggingface_hub import snapshot_download as d; d('google/metricx-24-hybrid-xl-v2p6')"
   python -c "from transformers import AutoTokenizer; AutoTokenizer.from_pretrained('google/mt5-xl')"
   python -c "import evaluate; evaluate.load('sacrebleu')"
   ```
4. Download ALMA-13B-R **without its adapter files**:
   ```bash
   python -c "
   from huggingface_hub import snapshot_download
   snapshot_download('haoranxu/ALMA-13B-R', revision='831d20301232e54f96c2c9af245ea219f85786d0',
                     local_dir='$MODELS_DIR/ALMA-13B-R', ignore_patterns=['adapter_*'])"
   ```
   The repository holds the merged weights and a stray `adapter_config.json`. With peft installed,
   `from_pretrained("haoranxu/ALMA-13B-R")` then silently loads ALMA-13B-Pretrain plus that adapter, which is not a
   translation model.

### 1. Reproduce ALMA-13B-R (Section 2.1)

```bash
sbatch slurm/generate_alma_r.job                              # paper decoding (beam sampling) -> run "ours"
sbatch --export=ALL,DECODING=beam slurm/generate_alma_r.job   # plain beam search -> run "ours-beam", our fp16 baseline
for run in ours ours-beam; do
  sbatch --export=ALL,RUN=$run slurm/score_comet.job          # XCOMET-XXL, KIWI-XXL, KIWI-22, COMET-22 (array 0-3)
  sbatch --export=ALL,RUN=$run slurm/score_metricx.job
done
python scripts/reproduce/score_lexical.py --run ours          # BLEU, chrF++ (CPU); same for ours-beam
python scripts/reproduce/summarize.py --run ours --vs paper   # -> results/scores/ours.tsv, with the ALMA-R paper's numbers
```

`RUN=paper` scores the ALMA-R paper's own outputs (shipped in the ALMA submodule) to check the metric setup. Our
reproduction is within 0.4 BLEU and 0.5 XCOMET-XXL of the paper on average. We compare all quantized models with
`ours-beam`, because sampling would add noise to every difference.

### 2. Quantize and evaluate (Sections 2.2, 3.1, 3.2)

```bash
sbatch --array=2,3,4,8 slurm/quantize_gptq.job   # -> $MODELS_DIR/ALMA-13B-R-gptq-w{2,3,4,8}g128 (16.5 min each)
for q in gptq-w8g128 gptq-w4g128 gptq-w3g128 gptq-w2g128; do
  sbatch --export=ALL,QUANT=$q slurm/eval_quantized.job   # generate + all metrics -> results/scores/$q.tsv
done
```

Generation takes 10+ hours per model; resubmit the same command after a timeout, finished work is skipped.

### 3. Repack 3- and 2-bit weights into the 4-bit layout

GPTQModel has a fused kernel for 4 bits only. `repack_to_w4.py` stores the same 3- and 2-bit values in the 4-bit
layout, losslessly (verified on the written files), so they run on the ExLlamaV2 kernel. The adapter runs are generated
on these copies.

```bash
sbatch slurm/repack.job   # -> $ARTIFACTS_DIR/ALMA-13B-R-gptq-w{3,2}g128-as-w4, speed and drift records in results/json/
```

### 4. Distil, generate, score (Sections 2.3, 3.3; Appendix B)

The exact commands of the three runs in the paper are at the top of `slurm/distill.job`:

```bash
sbatch slurm/distill.job                                                            # w3, r = 16, 30K examples
sbatch --export=ALL,BITS=2,ADAPTER=$ARTIFACTS_DIR/ALMA-13B-R-gptq-w2g128-kd-lora-r64,EXTRA="--r 64 --alpha 128" slurm/distill.job
sbatch --export=ALL,BITS=2,ADAPTER=$ARTIFACTS_DIR/ALMA-13B-R-gptq-w2g128-kd-lora-r64-cont,EXTRA="--r 64 --alpha 128 --n-examples 60000 --init-adapter $ARTIFACTS_DIR/ALMA-13B-R-gptq-w2g128-kd-lora-r64" slurm/distill.job

sbatch --export=ALL,NLL=1,TAG=w3kd slurm/gen_lora.job                               # NLL + generation, w3 + adapter
BITS=2 ADAPTER=$ARTIFACTS_DIR/ALMA-13B-R-gptq-w2g128-kd-lora-r64-cont KD_RUN=gptq-w2g128-as4-kd-r64-cont NLL=1 TAG=w2kd-r64-cont \
  sbatch --export=ALL slurm/gen_lora.job
sbatch --export=ALL,RUN=gptq-w3g128-as4-kd slurm/score_full.job                     # all metrics -> results/scores/
sbatch --export=ALL,RUN=gptq-w2g128-as4-kd-r64-cont slurm/score_full.job
```

Each training run takes 40–80 minutes on one H100 (peak 36–37 GiB). The adapter weights are not in this
repository; `results/distill_runs/` holds their configs and training logs.

### 5. Analyses

| What | Command | Output |
|---|---|---|
| Teacher-forced NLL per language | `sbatch slurm/lang_nll.job` (adapters: `gen_lora.job` with `NLL=1`) | `results/json/lang_nll_*.json` |
| Peak memory and time, batch 4 and 16 | `sbatch slurm/probe.job` | `results/json/probe16_<tag>.json` |
| … at batch 1 | `TAGS=fp16:w8:w4:w3:w2 BATCHES=1 SUFFIX=_b1 sbatch --export=ALL slurm/probe.job` | `probe16_<tag>_b1.json` |
| … with adapters | `sbatch slurm/probe_adapter.job` (`ONLY=`, `BATCHES=`, `SUFFIX=` as above) | `probe16_<tag>kd*.json` |
| Checkpoint size, load time, kernels | `sbatch slurm/quant_cost.job`, `sbatch slurm/backends.job` | `quant_cost_*.json`, `backend_*.json` |
| Failure modes (Table 2) | `python scripts/analysis/score_failures.py` (needs the [NLLB language identifier](https://dl.fbaipublicfiles.com/nllb/lid/lid218e.bin) in `$ARTIFACTS_DIR/lid/`) | `results/json/failures.json` |
| Paired bootstrap CIs | `python scripts/analysis/bootstrap_ci.py` | `results/json/bootstrap_ci.json` |
| Sanity checks and error analysis of w2 + adapter | `python scripts/analysis/audit_recovery.py` | `results/json/audit_*.json` |
| Damage vs sentence difficulty | `python scripts/analysis/fragility_by_confidence.py [--bits 4 --out …]` | `results/json/fragility_by_confidence*.json` |
| Activation outliers per language | `sbatch slurm/measure_channels.job` | `$OUTPUTS_DIR/channels/channels_fp16.npz` |
| Does quantization erase the fine-tuning? | `python scripts/analysis/measure_delta.py --pretrain-dir … --out …` (see its docstring) | `results/json/delta_erasure.json` |

The CPU analyses take under a minute each on a login node.

## Notes

- **Evaluation protocol.** As in ALMA-R: beam 5, bf16, seed 42, at most 256 new tokens, source length 256 (512 for
  zh→en); BLEU with sacreBLEU (`zh` tokenizer for Chinese targets, else `13a`); COMET-22 and MetricX-24 with the
  reference, XCOMET-XXL and both KIWI models without it. Scores ×100, except MetricX-24 (0–25, lower is better).
- **Quantization.** GPTQModel 4.2.5; asymmetric, group size 128, act-order, damping 0.05; 1,024 calibration
  sentences from ALMA's human-written training data, balanced over the ten directions. Embeddings, norms and the
  output head stay fp16.
- **Data.** ALMA's training data (calibration and distillation) does not overlap the WMT'22/'21 test sets; it does
  contain FLORES-200, so FLORES is not a clean test set for these models.
- **Reproducibility.** Packages are pinned in the requirements files, code in the submodule commits, and the model
  in its Hugging Face revision. Each run is a single seed; the paper reports paired bootstrap intervals over
  sentences, not seed variance.
- **MetricX-24** comes from the `third_party/metricx` submodule (not on PyPI) and runs on `transformers==4.45.2`.
  **XCOMET-XXL** needs about 43 GB of GPU memory, more than a 40 GB A100.

## Citation

```bibtex
@misc{dignum2026lowbit,
  title  = {Low-Bit Quantization for Multilingual Machine Translation},
  author = {Dignum, Krijn and Massafra, Francesco and Vork, Matthijs and Wilde, Max and Wolting, Reinout},
  year   = {2026},
  note   = {University of Amsterdam}
}
```

This work builds on [ALMA](https://github.com/fe1ixxu/ALMA) (MIT), [MetricX](https://github.com/google-research/metricx)
(Apache 2.0), [GPTQModel](https://github.com/ModelCloud/GPTQModel) and [COMET](https://github.com/Unbabel/COMET).
