# ALMA Model-Compression Project — Experiment Log & Status

Cross-agent status table + probe records. Full brief: `AGENTS.md`.
Every run also appends a row to `experiments/ledger.jsonl`.

## Status table (milestones)

| Milestone | Status | Notes |
|---|---|---|
| M0 dual-env probe + setup | **Mac DONE · cluster BLOCKED** | Mac gates passed. Cluster pending authorized access. |
| M1 data/model download + group merge | **data DONE · models DONE (base+G1) · merge NOT NEEDED (see finding)** | All datasets prepared; all X-ALMA repos public. |
| M2 harness + FP16 baseline | harness written; baseline pending | generate/score/efficiency/ledger/quant/analyze written. |
| M3 X-ALMA number reproduction | pending | needs FP16 generations. |
| M4 quantization (4 kinds x grid) | GGUF path started (G1) | GPTQ/LLM.int8 cluster-only. |
| M5 full matrix gen + scoring + efficiency | pending | needs cluster for full matrix. |
| M6 degradation curves + final report | pending | `src/eval/analyze.py` ready. |

## Findings

### F1 — Group repos ship the *merged dense* weights; the merge step is unnecessary (2026-09-16)
Evidence (shard `model-00006-of-00006.safetensors`, layer 38):

| tensor | LoRA target? | base vs `X-ALMA-13B-Group1` |
|---|---|---|
| `model.layers.38.input_layernorm.weight` | no | identical (max_abs_diff 0.0) |
| `model.layers.38.post_attention_layernorm.weight` | no | identical (0.0) |
| `model.layers.38.mlp.down_proj.weight` | yes | differs (max 0.0364, mean 0.0023) |
| `model.layers.38.mlp.up_proj.weight` | yes | differs (max 0.0374, mean 0.0032) |

`haoranxu/X-ALMA-13B-Group{g}` publishes **both** `adapter_config.json` +
`adapter_model.safetensors` (4.0 GB) **and** full 24 GB fp16 shards
(`model-0000{1..6}`, `LlamaForCausalLM`, hidden 5120, 40 layers) whose values
equal base + group LoRA exactly where LoRA applies and are byte-identical
elsewhere. ⇒ the repo already materialises the paper's *loading strategy 2*
(merged dense per group). **Impact:** Phase 2's `merge_and_unload` per group is
redundant; dense fp16 per group is fetched directly. `src/model/merge_xalma.py` is
kept as the verification/fallback path (base + adapter → merge) if a per-group
cross-check is ever wanted on the cluster.

### F2 — llama-cli 10150 requires `-st` for one-shot generation (2026-09-16)
Homebrew llama.cpp build `b10150-dee2a846b`: `--no-cnv`/`-no-cnv` still drops
into an interactive stdin REPL that never exits (prints `>` forever); the
documented `-st, --single-turn` with a predefined `-p` prompt exits cleanly
after one generation. **Impact:** `src/model/generate_gguf.py` uses `-st`; GGUF rows
are greedy (`--temp 0`) because this build exposes no beam search — protocol-
labelled `num_beams=1`, never merged naively with HF beam-5 rows.

### F3 — Dataset layout notes (2026-09-16)
- `haoranxu/FLORES-200`: per-direction configs, split named `test` (= devtest, 1012 rows), rows are `{lang: text}` dicts.
- NTREX-128: `facebook/ntrex_128` and `openlanguagedata/ntrex_128-plus` are 404; used **`mteb/NTREX`** (1997 rows, NLLB-coded parallel columns). Malagasy is `mlg_Latn` here (not FLORES's `plt_Latn`).
- `haoranxu/WMT23-Test`: per-direction configs (`en-de`, `de-en`, …), 11 of our pairs present (fr/ja/ar partial).

## Environment probes

### Mac (M2 Pro) — 2026-09-16
- macOS Darwin 27.0.0 arm64, RAM 32 GB, 460 GiB disk (118 GiB free after model downloads)
- python3.11.14 env `env/.venv-almaq` (torch MPS, transformers 4.x, peft, sacrebleu, unbabel-comet, datasets, psutil)
- llama.cpp Homebrew **b10150 (dee2a846b)**, Metal+BLAS; `llama-bench` on Qwen2.5-0.5B-Q4_K_M: pp32 203.8 t/s, tg16 210.8 t/s
- `llama-quantize`: `--tensor-type` (regex-matched, repeatable — the renamed `--override-tensor`), `--tensor-type-file`, `--output-tensor-type`, `--token-embedding-type` → layerwise recipes supported
- vendored: `vendor/metricx`, `vendor/smoothquant`, `vendor/ALMA` (LANG_TABLE + decode settings mirrored), `vendor/llama.cpp` (83078fe)

### Slurm cluster
- **Blocked pending the user's authorized-access decision.** Earlier read-only probe (before the stop order) saw `snellius.surf.nl` with `gpu_a100`/`gpu_h100` partitions, env modules 2023/2024/2025 + EESSI, system conda broken. Not to be used without explicit authorization.

## Downloads (Mac)

| artifact | size | state |
|---|---|---|
| `models/x-alma-13b-pretrain` (base fp16) | 24 G | ✅ 6/6 shards |
| `models/x-alma-13b-group1` (merged dense + adapter) | 28 G | ✅ 6/6 shards |
| `data/flores` (18 directions) | — | ✅ |
| `data/ntrex` (18 directions) | — | ✅ |
| `data/wmt23` (11 directions) | — | ✅ |

## HF token
Valid OAuth token (user `FireWtap`) at `~/.cache/huggingface/token`. XCOMET-XL is
gated: license acceptance on the HF website is still required (user action).
