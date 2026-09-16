# ALMA Model-Compression Project — Agent Brief

## 1. Mission
Quantize the best available ALMA model (**X-ALMA-13B**) and measure quality/efficiency degradation across **4 quantization families** as bit width decreases. Primary outputs: (a) reproduced X-ALMA baseline numbers on their own data, (b) per-method degradation-vs-bit-width curves for every metric, (c) efficiency report. Every agent reports everything (commands, env, artifacts, results) via the mandatory ledger below.

## 2. Fixed contracts (changeable ONLY by the user)
- **Model**: X-ALMA-13B, 8 group modules (`haoranxu/X-ALMA-13B-Pretrain` + `haoranxu/X-ALMA-13B-Group{1..8}`). Each group is merged into a dense fp16 model (paper-supported loading strategy 2; same param count as base).
- **4 quantization kinds → exact toolkits & bit grid**:
  | Kind | Method (toolkit) | Variants / bits |
  |---|---|---|
  | weights | GGUF uniform (llama.cpp) | Q8_0, Q6_K, Q5_K_M, Q4_K_M, Q3_K_M, Q2_K — no calibration, clean bit sweep |
  | activation | SmoothQuant W8A8 (mit-han-lab/smoothquant) | 8-bit, alpha 0.85, calibrated act scales; secondary point: LLM.int8() (transformers `load_in_8bit`) |
  | calibration | GPTQ (GPTQModel) | bits {8,4,3,2}, group_size 128, desc_act, calibration = 128×2048-tok language-matched samples from `haoranxu/X-ALMA-Parallel-Data` |
  | layerwise | ds4-inspired heterogeneous (llama.cpp per-tensor `--override-tensor`) | het-q8q4 (attention/embeddings/lm_head Q8_0; MLP Q4_K) and het-q8q2q3 (same Q8_0 shell; MLP up/gate Q2_K, down Q3_K) |
- **Benchmarks & subset**: FLORES-200 devtest (18 directions), NTREX-128 (18 en-centric directions), WMT'23 test (reproduce X-ALMA numbers on overlapping pairs). Eval language set: **de, is, fr, mg, ru, cs, zh, ja, ar** (tiers: high / mid / low + script diversity; group mapping G1 de,is · G3 ru · G4 fr,mg · G5 cs · G6 zh,ja · G8 ar).
- **Quality metrics**: XCOMET-XL (`Unbabel/XCOMET-XL`, unbabel-comet>=2.2, gated — accept license + HF token), MetricX-24-Hybrid-XL (`google/metricx-24-hybrid-xl-v2p6`, via `google-research/metricx` repo), SacreBLEU spBLEU (tokenize spm/flores), chrF++ (chrf word-order 2).
- **Efficiency metrics**: model size on disk (GB, deployable artifact dir), throughput output-tok/s warmup-excluded, peak GPU VRAM, batch-1 TTFB/prefill-time. Protocol per Gaido et al. 2025 (WMT25 model-compression task) + our instrumentation (VRAM/latency not in the official task).
- **Decode**: ALMA prompt only (no chat template): `Translate this from {src} into {tgt}:\n{src}: {sentence}\n{tgt}:`; num_beams=5, max_new_tokens=256, max_source_length=256, seed 42. Language names per ALMA's `run_llmmt.py` mapping — reuse it.

## 3. Agent collaboration & reporting (MANDATORY)
- Canonical layout (must not diverge):
  ```
  AGENTS.md, project.md
  env/            # env build scripts + locked requirements
  src/common/     # shared: device, ledger, matrix, group_langs.json, sync.sh
  src/model/      # model handling: merge_xalma, generate, generate_gguf
  src/quantize/   # quantization: quant_gptq, quant_gguf, quant_gguf_layerwise
  src/eval/       # scoring/evaluation: score, efficiency, efficiency_gguf, analyze
  src/data/       # data prep: prepare_data
  src/slurm/      # sbatch job scripts (per phase)
  models/         # gitignored: merged/, gptq/, gguf/, w8a8/ artifacts
  data/           # gitignored: downloaded benchmarks
  outputs/<run_id>/<direction>/  src.txt mt.txt ref.txt tokens.json
  scores/<run_id>/<direction>.json
  logs/<run_id>.{env,cmd,out,err}
  experiments/ledger.jsonl
  reports/        # EXPERIMENTS.md, curves/
  ```
- **Ledger**: every run appends one JSON line to `experiments/ledger.jsonl` (atomic `echo >>`):
  `{run_id, kind, method, variant, bits, group, languages, benchmark, num_sentences, model_artifact, calibration_artifact, platform, device, slurm_job_id, commit_sha, command, env_file, started_at, finished_at, scores_path, efficiency_path, status}`
  plus `logs/<run_id>.env` (pip freeze + conda list + nvidia-smi) and `logs/<run_id>.cmd` (exact command).
- **No silent changes**: code + ledger always committed; models/data gitignored. Cross-agent coordination via repo files (`reports/EXPERIMENTS.md` status table), not chat.
- **Handoff report format**: What / Why / How / Results / Next.
- **Milestones** (claim one, update status on completion; M0 → M6):
  - [ ] M0 dual-env probe + setup (Slurm + Mac)   [ ] M1 data/model download + group merge   [ ] M2 harness (generate/score/efficiency) + FP16 baseline generation   [ ] M3 X-ALMA number reproduction on WMT'23   [ ] M4 quantization (4 kinds × grid)   [ ] M5 full matrix generation + scoring + efficiency   [ ] M6 degradation curves + final report

## 4. Environment (dual: Slurm NVIDIA cluster + M2 Pro Mac)
- Build scripts `env/setup_slurm.sh` and `env/setup_mac.sh`; both documented in `env/README.md` (conda preferred, else venv; python 3.11).
- Slurm: probe first (`sinfo`, `scontrol show node`, `module avail`, `nvidia-smi`, `python --version`); CUDA torch; llama.cpp with `-DGGML_CUDA=ON`.
- Mac (Apple Silicon): `pip install torch` (MPS build); llama.cpp with `-DGGML_METAL=ON`; runtime device auto-detected (MPS → CUDA → CPU) by `src/common/device.py`, batch sizes adapted to RAM (`sysctl -n hw.memsize`; < 32 GB ⇒ fp16/GPTQ artifacts must be fetched from the cluster, not built locally).
- Platform split (MUST NOT drift): GGUF uniform + layerwise run on BOTH platforms; SmoothQuant quality eval runs on both (fake-quant), its real-kernel throughput only on cluster; GPTQ quantization + LLM.int8 are cluster-only (GPTQ artifacts rsync'd to Mac for eval); scoring (sacrebleu/chrF/XCOMET-XL/MetricX) runs on both (smaller batch on Mac).
- Artifact sync: `src/common/sync.sh` (rsync of `models/`, `outputs/`, `scores/` keyed by run_id) keeps both environments coherent; every ledger row records `platform` and `job_id`.
- HF login required for gated assets (XCOMET-XL license; FLORES family variants); token at `~/.cache/huggingface/token`.
- Pin versions: commit `env/requirements-lock-slurm.txt` + `env/requirements-lock-mac.txt` (pip freeze after every env change).

## 5. Literature anchors
- Survey/taxonomy: arXiv 2308.07633 (Zhu et al. — quantization taxonomy) · X-ALMA: arXiv 2410.03115 (base=LLaMA-2, LoRA groups, merging strategy, FLORES+WMT'23 eval) · PTQ×MT: arXiv 2508.20893 (55 langs; 4-bit OK for high-resource; low-resource & 2-bit degrade hard; GGUF most consistent) · ds4 (S. Sanfilippo): role-based heterogeneous bits (attention/routing high-bit, MLP aggressive) · WMT25 compression-task protocol: Gaido et al. 2025 (size + tok/s on A100) · XCOMET-XL: Guerreiro et al. 2024 · MetricX-24: Juraska et al. 2024 · ALMA(-R): 2309.11674 / 2401.08417.

## 6. Current status
Nothing executed yet. First agent starts M0.