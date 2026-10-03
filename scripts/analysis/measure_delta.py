"""Does GPTQ erase the ALMA fine-tuning delta?  CPU-only weight-space measurement.

ALMA-13B-R = ALMA-13B-Pretrain (LLaMA-2-13B + stage-1 monolingual full fine-tune) + a merged
LoRA (stage-2 parallel SFT, then continued by CPO; r=16, alpha=32, target_modules=["down_proj"],
per haoranxu/ALMA-13B-Pretrain-LoRA/adapter_config.json and third_party/ALMA/runs/cpo_ft.sh).
So for every Linear, Delta = W_R - W_pre should be rank <= 16 on down_proj and ~0 elsewhere.

For each (layer, module) this script reports:
  * |Delta|/|W_R|, the effective rank of Delta (energy in the top-16 singular values), and the
    cosine between Delta and the published stage-2 LoRA (2 * B @ A) - how much CPO moved it;
  * for each GPTQ checkpoint b in {8,4,3,2}, with E_b = What_b - W_R:
      snr_b   = |Delta|_F / |E_b|_F           output-SNR proxy for isotropic inputs
      rho_b   = <What_b - W_pre, Delta> / |Delta|^2   1 = delta kept, 0 = model reverted to W_pre
      sub_b   = |U_r^T E_b V_r|_F / |Delta|_F  quantization noise inside the delta's own rank-r
                                                 subspace, relative to the delta (r = 16)
      d/step  = median |Delta_ij| / scale_g(ij)  delta in units of the b-bit grid step
Dequantization is GPTQModel's own TorchQuantLinear.dequantize_weight() after its v1->v2 zero-offset
conversion (checkpoint_format "gptq" = v1), i.e. exactly what the eval runs computed.

Usage (CPU, login node is fine; ~2-3 GB RAM per module):
  python scripts/analysis/measure_delta.py \
      --pretrain-dir <download dir> --layers 0 20 39 \
      --out results/json/delta_erasure.json
The pretrain shards are haoranxu/ALMA-13B-Pretrain pytorch_model-0000{1,3,6}-of-00006.bin
(layers 0-7, 15-22, 38-39); adapter_model.bin from haoranxu/ALMA-13B-Pretrain-LoRA.
"""
import argparse
import json
import os
from pathlib import Path

import torch
from safetensors import safe_open

from mtcompress.paths import MODELS, OUTPUTS

torch.set_grad_enabled(False)
torch.set_num_threads(int(os.environ.get("THREADS", "4")))  # login-node friendly

MODULES = ["self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj", "self_attn.o_proj",
           "mlp.gate_proj", "mlp.up_proj", "mlp.down_proj"]
BITS = [8, 4, 3, 2]
RANK = 16


class Safetensors:
    """Lazy per-tensor reader over a sharded safetensors checkpoint."""

    def __init__(self, d):
        self.d = Path(d)
        self.map = json.load(open(self.d / "model.safetensors.index.json"))["weight_map"]
        self.handles = {}

    def get(self, name):
        f = self.map[name]
        if f not in self.handles:
            self.handles[f] = safe_open(str(self.d / f), framework="pt")
        return self.handles[f].get_tensor(name)


class TorchBin:
    """Lazy reader over a sharded pytorch_model-*.bin checkpoint (mmap, one shard at a time)."""

    def __init__(self, d):
        self.d = Path(d)
        self.map = json.load(open(self.d / "pre_index.json"))["weight_map"]
        self.cur, self.sd = None, None

    def get(self, name):
        f = self.map[name]
        if f != self.cur:
            self.sd = None
            self.sd = torch.load(self.d / f, map_location="cpu", mmap=True, weights_only=True)
            self.cur = f
        return self.sd[name]


def dequant(ckpt, prefix, bits, out_f, in_f):
    from gptqmodel.nn_modules.qlinear.torch import TorchQuantLinear
    from gptqmodel.utils.model import convert_gptq_v1_to_v2_format_module
    m = TorchQuantLinear(bits=bits, group_size=128, sym=False, desc_act=True,
                         in_features=in_f, out_features=out_f, pack_dtype=torch.int32)
    for k in ("qweight", "qzeros", "scales", "g_idx"):
        getattr(m, k).data = ckpt.get(f"{prefix}.{k}").clone()
    m.post_init()
    convert_gptq_v1_to_v2_format_module(m, bits=bits, pack_dtype=torch.int32)
    W = m.dequantize_weight().T.float()                  # [out, in]
    step = m.scales[m.g_idx.long()].T.float()            # per-weight grid step, [out, in]
    return W, step


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", default=str(MODELS))
    ap.add_argument("--pretrain-dir", required=True)
    ap.add_argument("--layers", type=int, nargs="+", default=[0, 20, 39])
    ap.add_argument("--modules", nargs="+", default=MODULES)
    ap.add_argument("--out", required=True)
    ap.add_argument("--channels", default=str(OUTPUTS / "channels/channels_fp16.npz"),
                    help="measure_channels.py per-channel input energy per language (down_proj only); '' to skip")
    ap.add_argument("--profiles", nargs="+",
                    default=["mixture", "calib:is", "calib:de", "calib:cs", "calib:zh", "calib:ru"])
    args = ap.parse_args()
    import numpy as np
    chan = np.load(args.channels) if args.channels and os.path.exists(args.channels) else None

    R = Safetensors(f"{args.models}/ALMA-13B-R")
    P = TorchBin(args.pretrain_dir)
    Q = {b: Safetensors(f"{args.models}/ALMA-13B-R-gptq-w{b}g128") for b in BITS}
    lora_path = Path(args.pretrain_dir) / "adapter_model.bin"
    lora = torch.load(lora_path, map_location="cpu", weights_only=True) if lora_path.exists() else {}

    rows = []
    for L in args.layers:
        for mod in args.modules:
            name = f"model.layers.{L}.{mod}"
            W_R = R.get(f"{name}.weight").float()
            W_p32 = P.get(f"{name}.weight").float()
            # ALMA-13B-R ships fp16; compare against the pretrain weight as fp16 and as bf16->fp16
            # to separate a real delta from a dtype round-trip.
            cand = {"fp16": W_p32.half().float(), "bf16": W_p32.bfloat16().float()}
            dnorm = {k: (W_R - v).norm().item() for k, v in cand.items()}
            ref = min(dnorm, key=dnorm.get)
            W_pre = cand[ref]
            del cand, W_p32
            D = W_R - W_pre
            r = {"layer": L, "module": mod, "pre_dtype_match": ref,
                 "delta_rel": (D.norm() / W_R.norm()).item(),
                 "delta_frac_nonzero": (D != 0).float().mean().item(),
                 "delta_fro": D.norm().item()}
            if r["delta_frac_nonzero"] > 0.01:
                # randomized SVD of the top 3r directions; total energy is exact from |D|_F^2,
                # so the top-r share needs no full decomposition
                torch.manual_seed(0)
                U, S, V = torch.svd_lowrank(D, q=3 * RANK, niter=4)
                tot = D.pow(2).sum()
                r["delta_energy_top16"] = (S[:RANK].pow(2).sum() / tot).item()
                r["delta_energy_top32"] = (S[:2 * RANK].pow(2).sum() / tot).item()
                r["delta_sv"] = S[:24].tolist()
                Ur, Vr = U[:, :RANK], V[:, :RANK]
            else:
                Ur = Vr = None
            key = f"base_model.model.model.layers.{L}.{mod}"
            if f"{key}.lora_A.weight" in lora:
                A, B = lora[f"{key}.lora_A.weight"].float(), lora[f"{key}.lora_B.weight"].float()
                D2 = 2.0 * B @ A                               # alpha / r = 32 / 16
                r["lora_sft_fro"] = D2.norm().item()
                r["cos_delta_vs_sft_lora"] = (torch.sum(D * D2) / (D.norm() * D2.norm())).item()
            out_f, in_f = W_R.shape
            # Per-language output-energy proxies, diagonal approximation E[x x^T] ~ diag(e):
            # E|M x|^2 ~ sum_i |M[:, i]|^2 e_i. Ignores cross-channel correlation [INFERENCE-grade].
            prof = {}
            if chan is not None:
                for pname in args.profiles:
                    k = f"{pname}::{name}"
                    if k in chan.files:
                        e = torch.from_numpy(chan[k]).float()
                        prof[pname] = e / e.sum()
            colW = W_R.pow(2).sum(0)
            colD = D.pow(2).sum(0)
            if prof:
                r["lang"] = {p: {"delta_share": (torch.dot(colD, e) / torch.dot(colW, e)).item()}
                             for p, e in prof.items()}
            for b in BITS:
                What, step = dequant(Q[b], name, b, out_f, in_f)
                E = What - W_R
                q = {"err_rel": (E.norm() / W_R.norm()).item(), "err_fro": E.norm().item()}
                if r["delta_fro"] > 0:
                    q["snr"] = r["delta_fro"] / q["err_fro"]
                    q["rho"] = (torch.sum((What - W_pre) * D) / D.norm().pow(2)).item()
                    q["delta_over_step_median"] = (D.abs() / step).median().item()
                    if Ur is not None:
                        q["sub_noise_over_delta"] = ((Ur.T @ E @ Vr).norm() / D.norm()).item()
                        q["sub_noise_frac_of_err"] = ((Ur.T @ E @ Vr).norm() / E.norm()).item()
                colE = E.pow(2).sum(0)
                colC = ((What - W_pre) * D).sum(0)
                for p, e in prof.items():
                    lq = r["lang"][p]
                    lq[f"w{b}_err"] = (torch.dot(colE, e) / torch.dot(colW, e)).sqrt().item()
                    if r["delta_fro"] > 0:
                        lq[f"w{b}_snr"] = (torch.dot(colD, e) / torch.dot(colE, e)).sqrt().item()
                        lq[f"w{b}_rho"] = (torch.dot(colC, e) / torch.dot(colD, e)).item()
                r[f"w{b}"] = q
                del What, step, E
            rows.append(r)
            brief = {k: v for k, v in r.items() if not k.startswith("w") and k not in ("delta_sv", "lang")}
            print(json.dumps(brief), flush=True)
            for b in BITS:
                print(f"   w{b}", json.dumps(r[f'w{b}']), flush=True)
            for p, lq in r.get("lang", {}).items():
                print(f"   {p:9s}", json.dumps({k: round(v, 4) for k, v in lq.items()}), flush=True)
            # checkpoint after every module so a killed run keeps its finished rows
            os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
            json.dump(rows, open(args.out, "w"), indent=1)
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    json.dump(rows, open(args.out, "w"), indent=1)


if __name__ == "__main__":
    main()
