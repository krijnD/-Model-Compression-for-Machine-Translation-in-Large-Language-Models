# ALMA Model-Compression Project

Demo/experiment state. See AGENTS.md for the full brief; this file is the
cross-agent status table and probe record.

## Status table (milestones)

| Milestone | Status | Notes |
|---|---|---|
| M0 dual-env probe + setup | **SF-in-progress** | Slurm probe blocked on user's authorized-access decision (2026-09-16). Mac: done (see below). |
| M1 data/model download + group merge | pending | |
| M2 harness + FP16 baseline | pending | |
| M3 X-ALMA number reproduction | pending | |
| M4 quantization (4 kinds x grid) | pending | |
| M5 full matrix gen + scoring + efficiency | pending | |
| M6 degradation curves + final report | pending | |

## Environment probes

### Mac (M2 Pro) — probed 2026-09-16
- OS: macOS (Darwin 27.0.0, arm64), RAM 32 GB (`hw.memsize` = 34359738368)
- Python: Homebrew python3.14 system default; python3.11/3.12/3.13 also installed; `uv` available
- Torch: not yet installed (setup via `env/setup_mac.sh` in progress)
- llama.cpp: Homebrew bottle **10150 (dee2a846b)**, Metal + BLAS backends.
  `llama-bench -m /tmp/gguf-gate/qwen2.5-0.5b-instruct-q4_k_m.gguf -ngl 99`
  → pp32 203.8 tok/s, tg16 210.8 tok/s — **Metal gate PASSED** (git gate verified).
- llama-quantize features: `--tensor-type NAME=TYPE` (repeatable; the renamed
  `--override-tensor`), `--tensor-type-file`, `--output-tensor-type`,
  `--token-embedding-type` — supports the layerwise heterogeneous recipes.
- nvidia-smi: absent (expected on Mac)
- Slurm clients: absent locally

### Slurm cluster
- Blocked pending user authorization. Prior read-only probe (before stop order)
  found: snellius.surf.nl reachable via `RecSysPersonal` alias; partitions
  `gpu_a100` (a100×4/node), `gpu_h100` (h100×4/node), `gpu_mig`, `gpu_vis`;
  environment modules 2023/2024/2025 + EESSI; system conda broken
  (Anaconda3 2025.06-1: `ModuleNotFoundError: No module named 'conda'`),
  python3 3.9.25 on login node. **Not to be used without explicit authorization.**

## HF token
- `~/.cache/huggingface/token` refreshed to a valid OAuth token (user
  `FireWatt`/Massafra Francesco), verified via `/api/whoami-v2` → 200 OK.
- XCOMET-XL is gated: license acceptance still required on the HF website.

## Decisions
- Slurm probe: BLOCKED — do not touch ssh-config hosts for other projects
  without user authorization. Resume only after user names an allowed host.