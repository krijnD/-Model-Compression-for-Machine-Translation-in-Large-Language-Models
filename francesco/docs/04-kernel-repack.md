# Kernel optimisation: 3-bit (and 2-bit) GPTQ on a fused kernel, with no new kernel (2026-09-29)

> *Moved to `docs/` on 2026-09-30, content unchanged except updated doc names. Paths in this file are relative to `francesco/` (`../` = the repo root), as when it was written. Start at `../README.md`.*

This follows `README.md` ("Measurement 5") and `02-cost-and-memory.md` §3.3. Both concluded
that no installed kernel gives w3 or w2 a fused path, so w3 runs `TorchQuantLinear` and w2 runs
`TritonV2QuantLinear`, and both dequantize the whole matrix on every forward. That conclusion holds
for the checkpoints **as stored**. It does not hold for the **numbers inside them**.

## 1. The idea: store the same values in a 4-bit container

A GPTQ asymmetric weight dequantizes as `w = s · (q − z)`, where `q, z ∈ [0, 2^b − 1]`, the scale `s`
is per (group, output column), and `g_idx` gives each input row its group. Every 3-bit or 2-bit
`(q, z)` is also a valid 4-bit `(q, z)`. So we can rewrite `qweight`/`qzeros` as 4-bit fields,
keep `scales`/`g_idx` byte-identical, and set `bits: 4`. The result dequantizes to **exactly the
same matrix**. GPTQModel then loads it as a 4-bit, asymmetric, `desc_act=True` checkpoint. That is
the w4 configuration, and `auto` already gives it the fused `ExllamaV2QuantLinear`, which is the one
kernel in this stack that beats dequant-then-matmul (w4 runs at 0.46 s/line against w3's 1.18).

The cost is memory: the weights go back to w4 size. The quantizer's choices, and so the quality,
do not change.

| checkpoint | weights now | as 4-bit container | Δ |
|---|---|---|---|
| w3 | 5.27 GiB | 6.76 GiB (measured, written) | +1.49 GiB |
| w2 | 3.78 GiB | 6.76 GiB (same tensor shapes as w4) | +2.98 GiB |

## 2. What was verified (CPU, login node) — measured

`analysis/repack_to_w4.py` unpacks the b-bit bitstream (GPTQ's 3-bit `10-1-10-1-10` layout is
simply a little-endian 3-bit stream over 3 int32s) and repacks it 8 values per int32. It also
handles the `checkpoint_format: "gptq"` (v1) zero offset. v1 stores `z − 1` as a packed-constant
subtraction with borrows, and the loader adds the constant back mod 2³². The tool undoes this with
the loader's own addition and redoes it with the 4-bit constant, so `z = 0` wraps and round-trips.

`--verify` runs three checks per module, and each must give **0 mismatches**:
1. GPTQModel's own `TorchQuantLinear` path (buffers ← tensors, `convert_gptq_v1_to_v2_format_module`,
   `dequantize_weight`) on the original b-bit tensors vs. the repacked 4-bit tensors;
2. the integer `q` and `z` fields;
3. the tool's own unpacker vs. GPTQModel's dequant. Without this check, a bug in the unpacker could
   be hidden by a matching bug in the packer.

| checkpoint | modules checked | weights compared | dequant mismatches | q / z mismatches | zero-points = 0 exercised | record |
|---|---|---|---|---|---|---|
| w3 (in memory) | 11 (q/k/v/o/gate/up/down; layers 0, 10, 20, 30, 39) | 511,180,800 | **0** | 0 / 0 | 2 | `results/json/repack_ALMA-13B-R-gptq-w3g128.json` |
| w3 (**written files**, re-read from disk) | 11 | 511,180,800 | **0** | 0 / 0 | 2 | `results/json/repack_w3_written.json` |
| w2 (in memory) | 11 | 511,180,800 | **0** | 0 / 0 | **94** | `results/json/repack_ALMA-13B-R-gptq-w2g128.json` |

**Negative controls, to show the check can fail.** On one repacked `o_proj` I planted three bugs.
Flipping one q bit gave 1 mismatch. Flipping one zero bit gave 128 mismatches (one group). Skipping
the v1 offset gave 26,214,400 mismatches (every weight). So a "0" in the table above is a real result.

**Written:** `/scratch-shared/$USER/models/ALMA-13B-R-gptq-w3g128-as-w4`. It has 41 shards, one per
layer, which keeps the converter at 3.3 GB RSS. Its `quantize_config.json`/`config.json` say
`bits: 4`, and the source is stamped in `meta`. **w2 was not written on the login node.** A first,
unstreamed converter hit the 8 GiB per-user login cap, so I made the converter stream per layer, and
the job converts w2 on the compute node (~6 min). The sampled 11-module verify is what was run for
both checkpoints. A `--verify-all` over all 280 modules was started and stopped for memory reasons
and did not finish.

**Not verified here:** an end-to-end GPU forward pass. That is step (c) of the job below.

## 3. Expected speed — projected, not measured

These come from the existing fixed-workload records: de-en, source 256, `BATCH=4`, beam 5,
`probe16_*.json` / `backend_*.json`. Per decode step = median `generate()` wall ÷ new tokens.

| | per generate() | new tok/line | ≈ ms per step | projected, repacked |
|---|---|---|---|---|
| fp16 | 1.435 s | 30 | 48 | — |
| w4 / ExLlamaV2 | 1.844 s | 31.7 | 58 | — |
| w3 / Torch | 4.752 s | 34 | **140** | **~58 → ~2.4× faster, ≈ 0.46–0.5 s/line** |
| w2 / Torch | 17.08 s | 256 | 67 | ~58 → ~1.15× |
| w2 / TritonV2 (auto) | 18.84 s | 256 | 74 | ~58 → ~1.3× |

- **w3 is where this pays.** At w4's per-step cost, the 13.6 GPU-h w3 grid run would drop to roughly
  5–6 GPU-h (projected; w3 writes slightly longer outputs than w4, e.g. en-is at 9.9 % hallucination).
  It would then be ~1.2–1.3× fp16, the same as w4, not 3.3×. This matters for every experiment
  queued on the w3 checkpoint in `../research.md`: the FLORES screen at 3 bits, RQ3 per-module
  arms, and re-generations.
- **w2 barely gains, and the numbers show why.** w2's per-step cost (67–74 ms) is already close to
  w4's (58 ms). It is slow per *line* because it generates **256 tokens per line instead of ~32**:
  it runs to `max_new_tokens`, which is the collapse itself. The kernel accounts for ≤1.3×. The 8×
  is output length. So "w2 is slow" is a quality fact, not a kernel fact. This corrects how the
  handoff note reads the w2 row.

## 4. Risks

- **fp16 inside the kernel.** ExLlamaV2 casts activations to fp16, computes, and casts back to bf16
  (`exllamav2.py` `forward`). The scored w3 grid ran bf16 matmuls in `TorchQuantLinear`. The weights
  are identical, but rounding and reduction order are not, so beam search can choose different
  outputs. **w4 in the grid already ran this way**, so a repacked w3 would be *more* comparable to
  w4 than the current w3 is. Still, the scored `gptq-w3g128` numbers must not be mixed with
  repacked-w3 generations without the drift check below.
- **Scale for "drift".** On the first 200 lines, the grid's own w3 vs. w4 outputs match exactly on
  10 % (de-en) and 0 % (en-is), with BLEU(w4 | w3) = 59.3 / 20.0 (`compare_outputs.py`, run here).
  Kernel-only drift should be far below that. If it is not, something is wrong.
- **Protocol.** Any re-run should use the same kernel for all bit-widths, following the "uniform
  grid" rule in `README.md`.

## 5. The job (GPU; for you to submit)

`analysis/sbatch/repack_bench.job`, 1 H100, ≤3 h:
(a) converts w2, and w3 if missing, on the compute node, verifying on the written files;
(b) times `w3as4` and `w2as4` with `auto` and `exllama_v2` on the exact backends.sbatch workload
(de-en 256 @4 + zh-en 512 @4) and at `BATCH=16`, writing `results/json/backend_w{3,2}as4_*.json`
next to the existing records;
(c) generates the first 200 lines of de-en and en-is with repacked w3 **and** with the original w3
on Torch (a determinism control), and compares both against `outputs/gptq-w3g128`: exact-match
rate, BLEU/chrF++/hallucination vs. reference, and BLEU vs. the grid. Nothing in `outputs/` is
overwritten. New files go to `outputs/repack-check/`.

```bash
sbatch francesco/analysis/sbatch/repack_bench.job
sbatch --export=ALL,STEPS=time francesco/analysis/sbatch/repack_bench.job      # timing only
```

## 6. Longer-term options, checked against the source (not run)

| option | status | verdict |
|---|---|---|
| **Machete** (GPTQModel 7.5.0, `qlinear/machete.py`) | 4 **and 8** bits, **asym** (`uint4`/`uint8` + zeros), `desc_act=True` (via `input_perm`), sm90 = H100. It JIT-builds against CUTLASS 4.7.1 (downloaded) and needs `transformers>=5.14`, so it requires a **separate venv**. | Best next step for **w8**. It is a fused 8-bit path for our *existing* asymmetric checkpoint, so the handoff's §6.6 "re-quantize with `sym=true` for Marlin" is unnecessary. It would also cover w4 and the repacked w3/w2. The cost is a new env plus one timing job. |
| Marlin (4.2.5 and 7.5.0) | `SUPPORTS_SYM = [True]` in both | Still refuses asymmetric checkpoints. |
| Swordfish (7.5.0) | Blackwell sm100/110 only | Not on H100. |
| **vLLM 0.30.0** | `auto_gptq` TYPE_MAP is only `(4, sym)`/`(8, sym)`, and `desc_act=True` now raises: "GPTQ group activation ordering … is no longer supported" | Unusable for these checkpoints without re-quantizing (`sym=True`, `desc_act=False` or static groups). |
| Custom Triton fused 3-bit (2-bit + 1-bit bit-planes, rows permuted at load, gather `x`) | not prototyped | **Not worth it now.** The only thing it adds over the repack is keeping 5.27 instead of 6.76 GiB. At 20–80 rows per forward, decode is 3.6–7× off the bandwidth roofline (handoff §2), so reading 25 % fewer weight bytes buys little speed. Build it only if 3-bit *memory* becomes a deliverable. |
| Original GPTQ `VecQuant3MatMulKernel` port | scoped in `README.md` | Superseded by the repack. |

## 7. Recommendation

1. Submit `repack_bench.job`. If (b) shows w3as4 ≈ w4 speed and (c) shows drift well inside the
   w3-vs-w4 scale, then **all future 3-bit work should run on the repacked checkpoint**, with the
   kernel named in every record.
2. Do not repack w2 for speed. Its cost is output length, and the kernel is secondary.
3. For w8, try Machete in a separate venv before re-quantizing anything for Marlin.
4. Leave vLLM and a custom Triton kernel off the list unless the protocol or memory target changes.

## Files

| file | role |
|---|---|
| `analysis/repack_to_w4.py` | lossless b-bit → 4-bit container converter, `--verify` / `--verify-all` |
| `analysis/compare_outputs.py` | line-level drift between two output folders (exact match, BLEU/chrF++/halluc); needs the scoring `venv` |
| `analysis/sbatch/repack_bench.job` | convert + time + drift check (GPU) |
| `results/json/repack_*.json` | CPU verification records |

## 8. Measured: job 27351455 (2026-09-29), speed and drift of the repacked w3

Log: `logs/slurm-probe-repack-27351455.out`. Records: `results/json/backend_w3as4_*.json`,
`probe16_w3as4.json`, `repack_quality_w3as4_auto.json`, `repack_quality_w3_torch.json`.

**Speed** (the fixed workloads from §3; `auto` selects ExllamaV2 for the repacked folder):

| workload | w3, TorchQuantLinear | w3-as4, ExllamaV2 | speed-up |
|---|---|---|---|
| de-en 256 @ batch 4 | 1.19 s/line | 0.470 | 2.5× |
| zh-en 512 @ batch 4 | 1.62 | 0.784 | 2.1× |
| de-en @ batch 16 | 0.427 | 0.288 | 1.5× |
| real generation, de-en, first 200 lines | 4.7 min | 1.8 min | 2.6× |
| real generation, en-is, first 200 lines | 17.5 min | 7.6 min | 2.3× |

The §3 projection (~2.4×, ≈0.46–0.5 s/line) holds. w3-as4 is still ~1.3× slower than fp16 per line
(0.359 s at batch 4). Peak memory at batch 4 is 12.3 GiB. w2-as4 gains little (1.46–2.0 s/line), as §3 predicted.

**Drift** (first 200 lines, compared with the scored `outputs/gptq-w3g128`):

| | exact match | BLEU grid → w3-as4 | chrF++ | halluc. | BLEU(w3-as4 \| grid) |
|---|---|---|---|---|---|
| de-en | 80.0 % | 30.36 → 30.19 | 54.04 → 53.99 | 0.0 → 0.5 % | 93.7 |
| en-is | 38.5 % | 11.49 → 11.07 | 38.96 → 38.85 | 8.0 → 10.0 % | 60.7 |
| control: original w3 on Torch, re-run | 100 % / 100 % | identical | identical | identical | 100 |
| scale: grid w4 vs grid w3 | 10 % / 0 % | | | | 59.3 / 20.0 |

- The control is bit-identical, so the drift is the kernel (fp16 compute in ExllamaV2), not nondeterminism.
- The drift is far smaller than the w3→w4 difference, and BLEU moves by ≤0.4. But it isn't zero, and on
  en-is (where w3 is unstable) most lines change.
- **Rule:** repacked-w3 numbers are a separate run (`gptq-w3g128-as4`) and must not replace or be mixed
  with the scored `gptq-w3g128` grid. New 3-bit experiments should compare with w3-as4 on the same kernel.
  The distillation eval (`06-distillation.md`) does this.
- The full 10-direction grid on w3-as4 is `analysis/sbatch/eval_w3as4.job` (optional, ~7 h). It's the
  measured end-to-end speed-up and the full-grid drift.
