# Model Compression for Machine Translation in Large Language Models

## Evaluation metrics: installation on Snellius

All metrics live in **one** virtual environment.

| Metric | Comes from | Install |
|---|---|---|
| BLEU (SacreBLEU) | [`sacrebleu`](https://github.com/mjpost/sacrebleu) | pip |
| chrF++ | [`sacrebleu`](https://github.com/mjpost/sacrebleu) (`CHRF(word_order=2)`) | pip |
| XCOMET-XL | [`unbabel-comet`](https://github.com/Unbabel/COMET), model [`Unbabel/XCOMET-XL`](https://huggingface.co/Unbabel/XCOMET-XL) | pip, plus the HF license |
| MetricX-24 Hybrid | [`google-research/metricx`](https://github.com/google-research/metricx), model [`google/metricx-24-hybrid-xl-v2p6`](https://huggingface.co/google/metricx-24-hybrid-xl-v2p6) | git submodule `third_party/metricx` (not on PyPI) plus pip deps |
| Hallucination rate | own code (length ratio of candidate to source, in characters, >= 2) | nothing |

Model weights are **not** installed by pip. They download from Hugging Face the first time a model is used.

### 1. Get the code (including the MetricX submodule)

```bash
cd /path/to/-Model-Compression-for-Machine-Translation-in-Large-Language-Models
git pull
git submodule update --init --recursive
```

Check that the MetricX code is there: `ls third_party/metricx/metricx24/predict.py`

### 2. Create the environment

```bash
module purge
module load 2024
module load Python/3.12.3-GCCcore-13.3.0

python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

(Run `module avail Python` to see which Python modules are available. Use Python 3.10–3.12.)

### 3. Hugging Face setup (once)

1. Log in on huggingface.co and **accept the license** on the [Unbabel/XCOMET-XL](https://huggingface.co/Unbabel/XCOMET-XL) page. The model is gated.
2. Keep the model cache out of `$HOME`, which has a small quota. Put this in `~/.bashrc` or in your job scripts:
   ```bash
   export HF_HOME=/scratch-shared/$USER/hf_cache   # or your project space
   ```
3. Log in with a read token:
   ```bash
   huggingface-cli login
   ```

Compute nodes may not have internet access. If so, download the models once from the login node (they go into `HF_HOME`):

```bash
python -c "from comet import download_model; download_model('Unbabel/XCOMET-XL')"
python -c "from huggingface_hub import snapshot_download as d; d('google/metricx-24-hybrid-xl-v2p6'); d('google/mt5-xl')"
```

### 4. Quick checks

```bash
source .venv/bin/activate
python -c "import sacrebleu, comet, transformers; print(sacrebleu.__version__, transformers.__version__)"
sacrebleu --help | head -n 3
cd third_party/metricx && python -c "import metricx24.models; print('metricx ok')" && cd -
```

### Notes

- **MetricX and transformers versions:** MetricX's own `requirements.txt` pins `transformers==4.30.2`. We use a newer version so that COMET and MetricX share one env. MetricX uses MT5 internals, so `transformers` must stay `<5`. If MetricX fails with the newer version, try `pip install "transformers[torch]==4.30.2" "datasets==2.13.1"`, and use a separate env only as a last resort.
- **Running MetricX** (from `third_party/metricx`; the input jsonl has the fields `source`, `hypothesis`, `reference`):
  ```bash
  python -m metricx24.predict \
    --tokenizer google/mt5-xl \
    --model_name_or_path google/metricx-24-hybrid-xl-v2p6 \
    --max_input_length 1536 --batch_size 1 \
    --input_file input.jsonl --output_file output.jsonl
  ```
  Add `--qe` to score without a reference. Lower is better (range 0–25).
- **XCOMET-XL** has 3.5B parameters (XXL has 10.7B). Higher is better (range 0–1).
