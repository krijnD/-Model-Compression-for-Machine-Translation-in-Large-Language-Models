# Environment Setup & Access Notes

Two environments: **Slurm NVIDIA cluster** (primary full-matrix runner) and **M2 Pro Mac** (secondary runner). Both share the same repo; heavy artifacts live under `models/`, `data/`, `outputs/`, `scores/` (gitignored) and are kept coherent via `src/common/sync.sh`.

## Building the environments

### Mac (Apple Silicon)
```bash
env/setup_mac.sh          # venv 'almaq' (python 3.11), torch MPS, llama.cpp Metal
source env/.venv-almaq/bin/activate
```
Gate: `python -c "import torch; print(torch.backends.mps.is_available())"` → `True`, and `llama-bench -m <any small gguf>` runs with the Metal backend.

### Slurm cluster (login node)
```bash
env/setup_slurm.sh        # conda env 'almaq' (python 3.11), CUDA torch, llama.cpp -DGGML_CUDA=ON
```
Gate: interactive GPU node runs `nvidia-smi` and `python -c "import torch; print(torch.cuda.get_device_name(0))"`.

## Runtime device selection
`src/common/device.py` picks `mps → cuda → cpu` automatically and caps batch size by RAM:
- `ram >= 32 GB` → `max_batch=32`, `can_fp16=True` (Mac M2 Pro here: 32 GB → fp16 allowed on MPS)
- `ram < 32 GB` → `max_batch=8`, `can_fp16=False` → fp16/GPTQ artifacts must be fetched from the cluster

## Artifact sync between Mac and cluster
`src/common/sync.sh` (rsync over ssh, keyed by run_id):
```bash
src/common/sync.sh pull-run <run_id>   # cluster -> Mac  (models/, outputs/, scores/)
src/common/sync.sh push-run <run_id>   # Mac -> cluster
```
Requires passwordless ssh to the cluster host. Cluster host/user/path are configured at the top of `src/common/sync.sh`
after the authorized access decision.

## HF login
Required for gated assets (XCOMET-XL license, FLORES family variants):
```bash
huggingface-cli login      # token stored at ~/.cache/huggingface/token
```
CLI token was refreshed (2026-09-16) and validated against `api/whoami-v2` — user `FireWtap`.

## Version pins
After any env change, re-freeze: `pip freeze > env/requirements-lock-mac.txt` / `requirements-lock-slurm.txt` and commit.

## Platform split (must not drift)
| Method | Mac | Cluster |
|---|---|---|
| GGUF uniform + layerwise | run | run |
| SmoothQuant W8A8 quality (fake-quant) | run | run |
| SmoothQuant real-kernel throughput | — | run |
| GPTQ quantization | fetch artifacts | run |
| LLM.int8 (bitsandbytes) | — | run |
| scoring (sacrebleu/chrF/XCOMET/MetricX) | run (smaller batch) | run |