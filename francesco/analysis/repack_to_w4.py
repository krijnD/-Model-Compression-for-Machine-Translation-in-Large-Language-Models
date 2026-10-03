"""Repack a 3- or 2-bit GPTQ checkpoint into the 4-bit container, losslessly.

Why: GPTQModel 4.2.5 has no fused kernel for 3 or 2 bits, so w3 runs TorchQuantLinear and w2 runs
TritonV2QuantLinear, both of which dequantize the whole matrix to bf16 on every forward and then
matmul (francesco/README.md, "Measurement 5"). The only fused kernel that accepts our asymmetric,
act-ordered (desc_act=True) checkpoints is ExllamaV2QuantLinear, and it is 4-bit only.

A b-bit asymmetric GPTQ weight is w = s * (q - z) with q, z in [0, 2^b - 1]. For b < 4 that value
set is a subset of the 4-bit one with the *same* scale s, zero z and g_idx, so rewriting q and z as
4-bit fields (and leaving scales/g_idx byte-identical) gives exactly the same dequantized matrix -
it only spends 4 bits of storage on each 3- or 2-bit value. Memory goes back up to w4's size; the
numbers the quantizer chose do not change.

Format details this has to get right (gptqmodel/utils/model.py, nn_modules/qlinear/__init__.py):
  * packing: b-bit values are a little-endian bitstream along the packed axis (rows of qweight,
    columns of qzeros); GPTQ's 3-bit "10-1-10-1-10" layout is exactly that stream over 3 int32s.
  * zero points: checkpoint_format "gptq" (v1) stores z - 1, produced by subtracting a packed
    constant from the whole int32 (with borrows) at save; the loader adds the same constant back.
    Both are mod-2^32 inverses, so we recover the true (v2) zeros by the loader's own addition and
    store them for 4 bits by the saver's own subtraction. z = 0 wraps and round-trips correctly.

Every tensor that is not a quantized Linear's qweight/qzeros is copied byte-for-byte.

  # check only, nothing written (CPU, streams one module at a time, minutes):
  python francesco/analysis/repack_to_w4.py --src ../models/ALMA-13B-R-gptq-w3g128 --verify
  # convert (and verify a sample of modules on the written files):
  python francesco/analysis/repack_to_w4.py --src ../models/ALMA-13B-R-gptq-w3g128 \\
      --dst /scratch-shared/$USER/models/ALMA-13B-R-gptq-w3g128-as-w4

--verify compares, per sampled module, three things on CPU: (1) GPTQModel's own TorchQuantLinear
dequantization of the original b-bit tensors against the same code on the repacked 4-bit tensors
(after each goes through GPTQModel's v1->v2 zero conversion, i.e. the real load path), (2) the
integer q and z fields, and (3) this script's unpacker against GPTQModel's dequant, so a bug in the
unpacker cannot hide behind a symmetric bug in the packer. All three must be exactly equal.
"""
import argparse
import json
import re
import shutil
import time
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import save_file

HERE = Path(__file__).resolve().parent
JSON_DIR = HERE.parent / "results" / "json"

# The packed constant GPTQModel adds to v1 qzeros at load (convert_gptq_v1_to_v2_format_module,
# int32 pack dtype). 3 bits cycles through three constants by column position mod 3.
V1_TO_V2 = {
    2: [0b01010101010101010101010101010101],
    3: [0b00100100100100100100100100100100, 0b10010010010010010010010010010010,
        0b01001001001001001001001001001001],
    4: [0b00010001000100010001000100010001],
}
MODULE_RE = re.compile(r"^(.*)\.qweight$")


def wrap32(x: torch.Tensor) -> torch.Tensor:
    """int64 -> int32 with two's-complement wrap-around (what torch int32 += does in place)."""
    x = x & 0xFFFFFFFF
    return torch.where(x >= 2 ** 31, x - 2 ** 32, x).to(torch.int32)


def add_packed_const(packed: torch.Tensor, bits: int, sign: int) -> torch.Tensor:
    """qzeros (int32, packed along dim 1) +/- the v1<->v2 constant, mod 2^32, exactly as the loader."""
    consts = V1_TO_V2[bits]
    x = packed.to(torch.int64)
    c = torch.tensor([consts[j % len(consts)] for j in range(x.shape[1])], dtype=torch.int64)
    return wrap32(x + sign * c)


def unpack(packed: torch.Tensor, bits: int, dim: int) -> torch.Tensor:
    """int32 tensor packed along `dim` -> int64 values in [0, 2^bits), unpacked along `dim`.
    Groups of `bits` words hold 32 values as a little-endian bitstream."""
    x = packed.movedim(dim, 0).to(torch.int64) & 0xFFFFFFFF
    n_words = x.shape[0]
    assert n_words % bits == 0, (n_words, bits)
    x = x.reshape(n_words // bits, bits, *x.shape[1:])
    mask = (1 << bits) - 1
    vals = []
    for i in range(32):
        off = bits * i
        w, s = divmod(off, 32)
        v = x[:, w] >> s
        if s + bits > 32:
            v = v | (x[:, w + 1] << (32 - s))
        vals.append(v & mask)
    out = torch.stack(vals, dim=1).reshape(n_words // bits * 32, *x.shape[2:])
    return out.movedim(0, dim)


def pack4(vals: torch.Tensor, dim: int) -> torch.Tensor:
    """int64 values in [0, 16) -> int32 packed 8 per word along `dim` (GPTQ 4-bit layout)."""
    x = vals.movedim(dim, 0)
    assert x.shape[0] % 8 == 0 and int(x.max()) < 16 and int(x.min()) >= 0
    x = x.reshape(x.shape[0] // 8, 8, *x.shape[1:])
    word = torch.zeros_like(x[:, 0])
    for j in range(8):
        word |= x[:, j] << (4 * j)
    return wrap32(word).movedim(0, dim).contiguous()


def repack_module(t: dict, bits: int) -> tuple[dict, dict]:
    """{qweight, qzeros, scales, g_idx} at `bits` -> same dict at 4 bits, plus integer stats."""
    q = unpack(t["qweight"], bits, dim=0)                                     # [in, out]
    z = unpack(add_packed_const(t["qzeros"], bits, +1), bits, dim=1)         # true (v2) zeros
    out = dict(t)
    out["qweight"] = pack4(q, dim=0)
    out["qzeros"] = add_packed_const(pack4(z, dim=1), 4, -1)                 # store as v1
    stats = {"z_eq_0": int((z == 0).sum()), "z_max": int(z.max()), "q_max": int(q.max()),
             "q_shape": list(q.shape), "z_shape": list(z.shape)}
    return out, stats


def gptqmodel_dequant(t: dict, bits: int, in_f: int, out_f: int) -> torch.Tensor:
    """GPTQModel's own load path on CPU: TorchQuantLinear buffers <- checkpoint tensors, v1->v2
    zero conversion, then PackableQuantLinear.dequantize_weight (uncompiled)."""
    from gptqmodel.nn_modules.qlinear import PackableQuantLinear
    from gptqmodel.nn_modules.qlinear.torch import TorchQuantLinear
    from gptqmodel.utils.model import convert_gptq_v1_to_v2_format_module
    m = TorchQuantLinear(bits=bits, group_size=128, sym=False, desc_act=True, in_features=in_f,
                         out_features=out_f, bias=False, pack_dtype=torch.int32)
    for k in ("qweight", "qzeros", "scales", "g_idx"):
        getattr(m, k).copy_(t[k])
    convert_gptq_v1_to_v2_format_module(m, bits=bits, pack_dtype=torch.int32)
    PackableQuantLinear.post_init(m)            # wf buffers only; skips torch.compile
    with torch.no_grad():
        return PackableQuantLinear.dequantize_weight(m, num_itr=1)


def verify_module(name: str, t: dict, bits: int) -> dict:
    t4, stats = repack_module(t, bits)
    in_f, out_f = t["g_idx"].shape[0], t["scales"].shape[1]
    w_orig = gptqmodel_dequant(t, bits, in_f, out_f)
    w_new = gptqmodel_dequant(t4, 4, in_f, out_f)
    # (3) independent: this script's integers through the same formula
    q = unpack(t["qweight"], bits, 0)
    z = unpack(add_packed_const(t["qzeros"], bits, +1), bits, 1)
    g = t["g_idx"].long()
    w_mine = t["scales"][g] * (q - z[g]).to(torch.float16)
    q4 = unpack(t4["qweight"], 4, 0)
    z4 = unpack(add_packed_const(t4["qzeros"], 4, +1), 4, 1)
    rec = {
        "module": name, "in": in_f, "out": out_f, **stats,
        "dequant_mismatch_vs_gptqmodel": int((w_orig != w_new).sum()),
        "int_q_mismatch": int((q != q4).sum()), "int_z_mismatch": int((z != z4).sum()),
        "unpacker_vs_gptqmodel_mismatch": int((w_mine != w_orig).sum()),
        "scales_g_idx_identical": bool(torch.equal(t["scales"], t4["scales"])
                                       and torch.equal(t["g_idx"], t4["g_idx"])),
        "elements": w_orig.numel(), "max_abs_w": float(w_orig.abs().max()),
    }
    rec["ok"] = (rec["dequant_mismatch_vs_gptqmodel"] == 0 and rec["int_q_mismatch"] == 0
                 and rec["int_z_mismatch"] == 0 and rec["unpacker_vs_gptqmodel_mismatch"] == 0
                 and rec["scales_g_idx_identical"])
    return rec


def load_module(src: Path, weight_map: dict, prefix: str) -> dict:
    out = {}
    for k in ("qweight", "qzeros", "scales", "g_idx"):
        with safe_open(src / weight_map[f"{prefix}.{k}"], framework="pt") as f:
            out[k] = f.get_tensor(f"{prefix}.{k}")
    return out


def sample_modules(prefixes: list[str], n_layers: int) -> list[str]:
    """q_proj and down_proj (the two input distributions that matter most, research.md §6) plus
    one of each other type, spread over first/middle/last layers."""
    want = []
    for layer in sorted({0, n_layers // 2, n_layers - 1}):
        want += [f"model.layers.{layer}.self_attn.q_proj", f"model.layers.{layer}.mlp.down_proj"]
    want += [f"model.layers.{n_layers // 4}.self_attn.{k}_proj" for k in ("k", "v", "o")]
    want += [f"model.layers.{3 * n_layers // 4}.mlp.{k}_proj" for k in ("gate", "up")]
    return [p for p in want if p in prefixes]


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--src", required=True, help="GPTQ checkpoint folder at 2 or 3 bits")
    p.add_argument("--dst", help="output folder; omit with --verify to only check")
    p.add_argument("--verify", action="store_true", help="verify a sample of modules (see docstring)")
    p.add_argument("--verify-all", action="store_true", help="verify every quantized module (slow)")
    p.add_argument("--report", help="JSON verify record (default results/json/repack_<name>.json)")
    args = p.parse_args()

    src = Path(args.src).resolve()
    qcfg = json.loads((src / "quantize_config.json").read_text())
    bits = qcfg["bits"]
    assert bits in (2, 3), f"{src} is {bits}-bit; only 2 and 3 repack into the 4-bit container"
    assert qcfg["pack_dtype"] == "int32" and qcfg["checkpoint_format"] == "gptq", qcfg
    assert qcfg["group_size"] == 128 and not qcfg["sym"], qcfg
    index = json.loads((src / "model.safetensors.index.json").read_text())
    weight_map = index["weight_map"]
    prefixes = sorted({m.group(1) for k in weight_map if (m := MODULE_RE.match(k))})
    n_layers = 1 + max(int(re.search(r"layers\.(\d+)\.", x).group(1)) for x in prefixes)
    print(f"{src.name}: {bits}-bit, {len(prefixes)} quantized modules, {n_layers} layers", flush=True)

    if args.dst:
        dst = Path(args.dst).resolve()
        dst.mkdir(parents=True, exist_ok=True)
        # One output shard per decoder layer (+ one for embeddings/norm/lm_head), so peak memory is
        # one layer's tensors: login nodes cap a user at 8 GiB, and a 4 GB input shard plus its
        # repacked copy does not fit. The index maps every tensor to its new shard.
        t0 = time.time()
        total = 0

        def shard_of(name):
            m = re.match(r"model\.layers\.(\d+)\.", name)
            return int(m.group(1)) if m else n_layers
        n_out = n_layers + 1
        new_map = {}
        for s in range(n_out):
            names = [k for k in weight_map if shard_of(k) == s]
            tensors = {}
            for k in names:
                with safe_open(src / weight_map[k], framework="pt") as f:
                    tensors[k] = f.get_tensor(k)
            for prefix in [x for x in prefixes if f"{x}.qweight" in tensors]:
                t = {k: tensors[f"{prefix}.{k}"] for k in ("qweight", "qzeros", "scales", "g_idx")}
                t4, _ = repack_module(t, bits)
                tensors[f"{prefix}.qweight"], tensors[f"{prefix}.qzeros"] = t4["qweight"], t4["qzeros"]
                del t, t4
            fname = f"model-{s + 1:05d}-of-{n_out:05d}.safetensors"
            total += sum(v.numel() * v.element_size() for v in tensors.values())
            save_file(tensors, dst / fname, metadata={"format": "pt"})
            new_map.update({k: fname for k in names})
            del tensors
            if s % 10 == 0 or s == n_out - 1:
                print(f"  wrote {fname} ({time.time() - t0:.0f}s)", flush=True)
        index = {"metadata": {"total_size": total}, "weight_map": new_map}
        (dst / "model.safetensors.index.json").write_text(json.dumps(index, indent=2))
        stamp = {"repacked_from_bits": bits, "repacked_from": str(src),
                 "tool": "francesco/analysis/repack_to_w4.py (lossless b-bit -> 4-bit container)"}
        qcfg4 = dict(qcfg, bits=4)
        qcfg4["meta"] = dict(qcfg.get("meta", {}), **stamp)
        (dst / "quantize_config.json").write_text(json.dumps(qcfg4, indent=2))
        cfg = json.loads((src / "config.json").read_text())
        cfg["quantization_config"] = dict(cfg["quantization_config"], bits=4)
        cfg["quantization_config"]["meta"] = dict(cfg["quantization_config"].get("meta", {}), **stamp)
        (dst / "config.json").write_text(json.dumps(cfg, indent=2))
        if (src / "quant_meta.json").exists():
            qm = json.loads((src / "quant_meta.json").read_text())
            (dst / "quant_meta.json").write_text(json.dumps(dict(qm, **stamp, container_bits=4), indent=2))
        for f in src.iterdir():
            if f.name.startswith(("tokenizer", "special_tokens", "generation_config", "quant_log")):
                shutil.copy2(f, dst / f.name)
        print(f"converted -> {dst}, {total / 2 ** 30:.2f} GiB of tensors", flush=True)

    if args.verify or args.verify_all or args.dst:
        # verify against the written files when converting, else in memory
        chosen = prefixes if args.verify_all else sample_modules(prefixes, n_layers)
        records = []
        for prefix in chosen:
            t = load_module(src, weight_map, prefix)                    # original index
            rec = verify_module(prefix, t, bits)
            if args.dst:
                dst_map = json.loads((Path(args.dst) / "model.safetensors.index.json").read_text())["weight_map"]
                t4_disk = load_module(Path(args.dst).resolve(), dst_map, prefix)
                t4_mem, _ = repack_module(t, bits)
                rec["disk_equals_memory"] = all(torch.equal(t4_disk[k], t4_mem[k]) for k in t4_mem)
                rec["ok"] = rec["ok"] and rec["disk_equals_memory"]
            records.append(rec)
            print(f"  {'OK ' if rec['ok'] else 'BAD'} {prefix:40s} {rec['elements']:>10,d} weights  "
                  f"dequant mismatches {rec['dequant_mismatch_vs_gptqmodel']}  "
                  f"q/z mismatches {rec['int_q_mismatch']}/{rec['int_z_mismatch']}  "
                  f"unpacker {rec['unpacker_vs_gptqmodel_mismatch']}  z==0: {rec['z_eq_0']}", flush=True)
        summary = {"src": str(src), "bits": bits, "modules_checked": len(records),
                   "weights_checked": sum(r["elements"] for r in records),
                   "all_ok": all(r["ok"] for r in records), "records": records}
        out = Path(args.report) if args.report else JSON_DIR / f"repack_{src.name}.json"
        out.write_text(json.dumps(summary, indent=2))
        print(f"verify: {summary['modules_checked']} modules, {summary['weights_checked']:,d} weights, "
              f"all_ok={summary['all_ok']} -> {out}", flush=True)
        if not summary["all_ok"]:
            raise SystemExit(1)


if __name__ == "__main__":
    main()
