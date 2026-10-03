"""Per-channel activation-energy profiles per language: the RQ5 measurement in reasearch.md.

Why this measures anything. Weight-only GPTQ quantizes W and never x, so a Linear's output error is
dW @ x -- weight error *weighted by the activation magnitude of each input channel*. GPTQ minimises
||Wx - Wx||^2 over its calibration set, and with desc_act=true (all four of our checkpoints) it sorts
columns by decreasing Hessian diagonal and quantizes the largest first, i.e. it is most careful with
whatever is loudest *on the 10-direction calibration mixture*. RQ5 asks whether a given language's
loud channels are the same channels as the mixture's -- if they are, giving that language more of the
calibration budget cannot help, and no re-quantize needs to be run to find out.

What it does. Hooks one Linear type (default the 40 mlp.down_proj), runs a fixed prompt set per
profile, and accumulates sum(x^2) per input channel per layer -- the diagonal of GPTQ's own Hessian.
No quantization and no generation: one forward pass per sequence.

Profiles:
  mixture        the quantizer's 1024 calibration examples over all 10 directions (what GPTQ saw)
  calib:<lang>   that language's two directions inside those 1024 examples (~205 each)
  pool:<lang>:N  N examples per direction from the same train pool, a larger sample used to check
                 that the top-k channel set is stable and not an artifact of ~100 examples

Every prompt is ALMA's fine-tuning format (scripts/alma_prompt.py), so the only difference between
profiles is the language, not the template. The mixture profile is cross-checked against
alma_prompt.calibration_examples and the script refuses to continue if the two disagree.

The raw per-channel vectors go to an .npz, so all ranking comparisons are offline and re-runnable.

Usage (venv, one GPU), from the repo root:
  python francesco/analysis/measure_channels.py --out francesco/results/json/channels.npz
  python francesco/analysis/measure_channels.py --out ... --suffix mlp.down_proj,self_attn.o_proj
  python francesco/analysis/measure_channels.py --compare francesco/results/json/channels.npz
"""
import argparse
import json
import random
import re
import sys
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

HERE = Path(__file__).resolve().parent        # francesco/analysis
REPO = HERE.parents[1]                        # the repository root
PROJECT_DIR = REPO.parent                     # holds models/, venv/, hf_cache/, outputs/
sys.path.insert(0, str(REPO / "scripts"))     # alma_prompt.py lives with the other quant scripts
from alma_prompt import WMT22_PAIRS, get_prompt, load_train_pairs  # noqa: E402

KS = (1, 8, 32)


# ------------------------------------------------------------------------------- profiles

def _encode(tokenizer, src, tgt, source, target, max_length):
    """ALMA fine-tuning format, exactly as scripts/alma_prompt.py builds calibration examples."""
    ids = tokenizer(get_prompt(src, tgt, source) + target,
                    max_length=max_length - 1, truncation=True).input_ids
    ids.append(tokenizer.eos_token_id)
    return ids


def labelled_calibration(tokenizer, n_samples, pairs, seed, max_length):
    """[(pair, input_ids)] for the quantizer's calibration set, keeping the direction label.

    Replicates scripts/alma_prompt.calibration_examples draw-for-draw (same per-pair counts, same
    Random(seed) call order); the final shuffle there is a permutation, which does not change any
    per-channel sum. Verified against that function by the caller.
    """
    rng = random.Random(seed)
    per_pair = [n_samples // len(pairs) + (i < n_samples % len(pairs)) for i in range(len(pairs))]
    out = []
    for pair, n in zip(pairs, per_pair):
        src, tgt = pair.split("-")
        for s, t in rng.sample(load_train_pairs(src, tgt), n):
            out.append((pair, _encode(tokenizer, src, tgt, s, t, max_length)))
    return out


def pool_profile(tokenizer, lang, n_per_direction, max_length, seed):
    """[(pair, input_ids)] drawn from the same train pool, a different sample than the calibration."""
    rng = random.Random(seed)
    out = []
    for pair in (f"{lang}-en", f"en-{lang}"):
        if pair not in WMT22_PAIRS:
            sys.exit(f"ERROR: {pair} is not one of the ALMA directions {WMT22_PAIRS}")
        src, tgt = pair.split("-")
        for s, t in rng.sample(load_train_pairs(src, tgt), n_per_direction):
            out.append((pair, _encode(tokenizer, src, tgt, s, t, max_length)))
    return out


def build_profiles(tokenizer, specs, languages, pool_n, max_length, seed, limit):
    """{profile name: [(pair, input_ids)]}. `mix` holds the labelled calibration set."""
    mix = None
    profiles = {}
    for spec in specs:
        if spec == "mixture":
            mix = labelled_calibration(tokenizer, 1024, WMT22_PAIRS, seed, max_length)
            profiles["mixture"] = mix
        elif spec.startswith("calib:"):
            if mix is None:
                mix = labelled_calibration(tokenizer, 1024, WMT22_PAIRS, seed, max_length)
            lang = spec.split(":", 1)[1]
            profiles[spec] = [(p, i) for p, i in mix if lang in p.split("-")]
        elif spec.startswith("pool:"):
            _, lang, n = spec.split(":")
            profiles[spec] = pool_profile(tokenizer, lang, int(n), max_length, seed=seed + 1000)
        else:
            sys.exit(f"ERROR: unknown profile spec {spec!r} (mixture | calib:<lang> | pool:<lang>:<n>)")
    if limit:
        for k in profiles:
            profiles[k] = profiles[k][:limit]
    # languages=... is shorthand for the per-language calib profiles
    if languages:
        if mix is None:
            mix = labelled_calibration(tokenizer, 1024, WMT22_PAIRS, seed, max_length)
        for lang in languages:
            profiles.setdefault(f"calib:{lang}", [(p, i) for p, i in mix if lang in p.split("-")])
    return profiles, mix


# ------------------------------------------------------------------------------- measurement

class ChannelProfiler:
    """sum(x^2) per input channel, per hooked module, over every real token of every batch."""

    def __init__(self, names):
        self.names = names
        self.energy = {}
        self.mask = None

    def _hook(self, name):
        def fn(_module, inputs, _output):
            x = inputs[0].detach().to(torch.float32)
            if self.mask is not None:
                # the mask lives on the host; match the activations' device and dtype
                x = x * self.mask.unsqueeze(-1).to(device=x.device, dtype=x.dtype)
            e = (x * x).sum(dim=(0, 1)).to(torch.float64).cpu()
            if name not in self.energy:
                self.energy[name] = torch.zeros_like(e)
            self.energy[name] += e
        return fn

    def attach(self, model):
        modules = dict(model.named_modules())
        handles = []
        for name in self.names:
            handles.append(modules[name].register_forward_hook(self._hook(name)))
        return handles


def profile_one(model, profiler, tokenizer, examples, batch_size, device):
    total = 0
    for i in range(0, len(examples), batch_size):
        chunk = examples[i:i + batch_size]
        width = max(len(ids) for _, ids in chunk)
        pad = tokenizer.pad_token_id
        input_ids = torch.full((len(chunk), width), pad, dtype=torch.long)
        mask = torch.zeros((len(chunk), width), dtype=torch.long)
        for j, (_, ids) in enumerate(chunk):
            input_ids[j, :len(ids)] = torch.tensor(ids)
            mask[j, :len(ids)] = 1
        profiler.mask = mask
        with torch.inference_mode():
            model(input_ids=input_ids.to(device), attention_mask=mask.to(device))
        total += int(mask.sum())
    profiler.mask = None
    return total


def describe(energy):
    """Concentration statistics of one layer's per-channel energy vector."""
    e = np.asarray(energy, dtype=np.float64)
    total = e.sum()
    order = np.argsort(e)[::-1]
    out = {"total": float(total), "n_channels": int(len(e))}
    for k in KS:
        out[f"tau{k}"] = float(e[order[:k]].sum() / total) if total > 0 else float("nan")
    amp = np.sqrt(e)
    # scale-free concentration: ratio of the loudest channel to the median channel
    out["max_over_median"] = float(amp.max() / np.median(amp)) if np.median(amp) > 0 else float("nan")
    return out


def rankdata(a):
    order = np.argsort(a, kind="stable")
    ranks = np.empty(len(a), dtype=np.float64)
    ranks[order] = np.arange(len(a), dtype=np.float64)
    return ranks


def spearman(a, b):
    ra, rb = rankdata(a), rankdata(b)
    ra -= ra.mean()
    rb -= rb.mean()
    return float((ra * rb).sum() / np.sqrt((ra * ra).sum() * (rb * rb).sum()))


def jaccard(a, b, k):
    ta = set(np.argsort(a)[::-1][:k].tolist())
    tb = set(np.argsort(b)[::-1][:k].tolist())
    return len(ta & tb) / len(ta | tb)


# ------------------------------------------------------------------------------------- compare

def coverage_table(z, profiles, layers, ref, ks=(8, 32, 128)):
    """The operationally meaningful statistic, next to the rank agreement above.

    GPTQ with desc_act=true quantizes the columns of largest *calibration* Hessian diagonal first,
    so the channels in `ref`'s top-k are the ones it is told to be most careful about. Coverage is
    the share of a profile's own energy sitting inside exactly those channels; `self` is the ceiling
    it would get from its own ordering, so `gain` is the most any re-ordering could buy it.
    """
    C = len(z[f"{ref}::{layers[0]}"])
    print(f"\ncoverage: share of a profile's activation energy inside {ref!r}'s top-k channels")
    print("(what a desc_act ordering derived from that profile protects). 'self' is the ceiling")
    print("from the profile's own ordering, so 'gain' bounds what re-ordering could buy it.")
    for k in ks:
        print(f"\n  top-k = {k} of {C} channels   uniform = {100 * k / C:.2f} %")
        print(f"  {'profile':16s} {'coverage':>9s} {'self':>7s} {'gain':>6s}")
        for p in profiles:
            cov, sel = [], []
            for L in layers:
                er = np.asarray(z[f"{ref}::{L}"], dtype=np.float64)
                ep = np.asarray(z[f"{p}::{L}"], dtype=np.float64)
                cov.append(ep[np.argsort(er)[::-1][:k]].sum() / ep.sum())
                sel.append(ep[np.argsort(ep)[::-1][:k]].sum() / ep.sum())
            print(f"  {p:16s} {100 * np.mean(cov):8.1f}% {100 * np.mean(sel):6.1f}% "
                  f"{100 * (np.mean(sel) - np.mean(cov)):5.1f}")


def compare(path, k=32):
    z = np.load(path, allow_pickle=False)
    meta = json.loads(str(z["_meta"]))
    profiles = meta["profiles"]
    layers = meta["layers"]
    n_channels = len(z[f"{profiles[0]}::{layers[0]}"])
    n_modules = len(layers)
    print(f"{path}: {n_modules} modules x {n_channels} input channels, {len(profiles)} profiles")
    print(f"modules: {meta['suffixes']}, dtype {meta['dtype']}, max_length {meta['max_length']}\n")
    print(f"{'profile':16s} {'tau1':>6s} {'tau8':>6s} {'tau32':>6s} {'max/med':>8s}")
    for p in profiles:
        stats = [describe(z[f"{p}::{L}"]) for L in layers]
        row = [np.median([s[key] for s in stats]) for key in ("tau1", "tau8", "tau32")]
        mm = np.median([s["max_over_median"] for s in stats])
        print(f"{p:16s} " + " ".join(f"{v:6.3f}" for v in row) + f" {mm:8.1f}")
    ref = "mixture" if "mixture" in profiles else profiles[0]
    print(f"\nper-layer rank agreement with {ref!r} (median over {len(layers)} modules, worst module):")
    print(f"{'profile':16s} {'spearman':>9s} {'worst':>7s} {'jaccard@32':>11s} {'worst':>7s}")
    for p in profiles:
        if p == ref:
            continue
        sp = [spearman(z[f"{p}::{L}"], z[f"{ref}::{L}"]) for L in layers]
        jc = [jaccard(z[f"{p}::{L}"], z[f"{ref}::{L}"], k) for L in layers]
        print(f"{p:16s} {np.median(sp):9.3f} {min(sp):7.3f} {np.median(jc):11.3f} {min(jc):7.3f}")
    coverage_table(z, profiles, layers, ref)


# ---------------------------------------------------------------------------------------- main

def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--compare", type=Path, help="read an existing .npz and print the comparison, then exit")
    p.add_argument("--model", default=str(PROJECT_DIR / "models/ALMA-13B-R"), help="fp16 ALMA-13B-R")
    p.add_argument("--tokenizer", default=None, help="defaults to --model (quantized folders ship their own)")
    p.add_argument("--suffix", default="mlp.down_proj", help="comma-separated module name suffixes")
    p.add_argument("--profiles", default="mixture,calib:is,calib:de,calib:cs,calib:zh,calib:ru,pool:is:1024,pool:de:1024")
    p.add_argument("--languages", default="", help="extra per-language calib profiles, e.g. is,de,cs,zh,ru")
    p.add_argument("--pool-n", type=int, default=1024, help="examples per direction for pool:<lang>")
    p.add_argument("--max-length", type=int, default=512)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float16", "float32"])
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--limit", type=int, help="truncate every profile to N examples (smoke tests)")
    p.add_argument("--out", type=Path, default=Path("channels.npz"))
    args = p.parse_args()

    if args.compare:
        compare(args.compare)
        return

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer or args.model, add_eos_token=False)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    profiles, mix = build_profiles(tokenizer, [s for s in args.profiles.split(",") if s],
                                  [s for s in args.languages.split(",") if s],
                                  args.pool_n, args.max_length, args.seed, args.limit)

    if mix is not None:
        # same set as the shipped checkpoints' calibration (alma_prompt shuffles, which is a no-op here)
        from alma_prompt import calibration_examples
        ref = sorted(tuple(e["input_ids"]) for e in
                     calibration_examples(tokenizer, 1024, WMT22_PAIRS, seed=args.seed,
                                          max_length=args.max_length))
        mine = sorted(tuple(i) for _, i in mix)
        if ref != mine:
            sys.exit(f"ERROR: calibration replication differs from alma_prompt.calibration_examples "
                     f"({len(ref)} vs {len(mine)} examples); refusing to compare against the mixture")
        print(f"calibration replication verified against alma_prompt: {len(mine)} examples, identical")

    # commas fall foul of sbatch --export, so accept whitespace too
    suffixes = [s for s in re.split(r"[,\s]+", args.suffix) if s]
    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=getattr(torch, args.dtype))
    model.eval().to(args.device)
    names = [n for n, m in model.named_modules()
             if any(n.endswith(s) for s in suffixes) and isinstance(m, torch.nn.Linear)]
    if not names:
        sys.exit(f"ERROR: no Linear matches {suffixes}")
    print(f"hooked {len(names)} modules matching {suffixes} (e.g. {names[0]})")

    profiler = ChannelProfiler(names)
    handles = profiler.attach(model)
    z, meta_profiles = {}, {}
    for name, examples in profiles.items():
        profiler.energy = {}
        tokens = profile_one(model, profiler, tokenizer, examples,
                             args.batch_size, args.device)
        for n in names:
            out = profiler.energy.get(n)
            if out is None:
                sys.exit(f"ERROR: hook for {n} never fired on profile {name}")
            z[f"{name}::{n}"] = out.numpy()
        meta_profiles[name] = {"examples": len(examples), "tokens": tokens}
        rows = [describe(z[f"{name}::{n}"]) for n in names]
        print(f"{name:16s} {len(examples):5d} examples {tokens:7d} tokens | "
              + " ".join(f"tau{k}={np.median([r[f'tau{k}'] for r in rows]):.3f}" for k in KS)
              + f" max/med={np.median([r['max_over_median'] for r in rows]):.1f}")
    for h in handles:
        h.remove()

    z["_meta"] = np.array(json.dumps({"model": args.model, "dtype": args.dtype, "suffixes": suffixes,
                                      "layers": names, "ks": list(KS), "profiles": list(profiles),
                                      "per_profile": meta_profiles, "max_length": args.max_length,
                                      "seed": args.seed, "limit": args.limit}))
    np.savez_compressed(args.out, **z)
    print(f"wrote {args.out}")
    compare(args.out)


if __name__ == "__main__":
    main()
