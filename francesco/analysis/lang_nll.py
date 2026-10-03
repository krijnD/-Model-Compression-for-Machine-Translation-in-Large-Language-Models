"""Does 3-bit GPTQ erase the *language* or the *translation skill*?  Teacher-forced, no generation.

For each checkpoint and each language, two per-token NLLs on WMT'22 test text (clean: not in
ALMA's training data, unlike FLORES):
  mono  - the sentence alone, after BOS: plain language modelling, the stage-1 (OSCAR) skill;
  trans - the reference translation given ALMA's prompt, loss on the target only: the stage-2 skill.
Report dNLL = NLL(quantized) - NLL(fp16) per language. If `is` collapses because the model lost
Icelandic itself, `mono` dNLL for `is` dwarfs `de`'s; if it lost the mapping, `mono` is flat
and only `trans` diverges.

  python francesco/analysis/lang_nll.py --model ../models/ALMA-13B-R [--quantized] \
      --tag fp16 --out francesco/results/json/lang_nll_fp16.json [--limit 500]
  python francesco/analysis/lang_nll.py --compare francesco/results/json/lang_nll_*.json
  --adapter <dir>: add a distill_lora.py adapter on top of --model (the w3 + KD run, docs/06-distillation.md)
"""
import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "scripts"))
from alma_prompt import get_prompt, pair_dir  # noqa: E402

LANGS = ["is", "de", "cs", "zh", "ru"]


def test_pairs(src, tgt):
    d, _ = pair_dir(src, tgt)
    with open(d / f"test.{src}-{tgt}.json", encoding="utf-8") as f:
        return [(ex["translation"][src], ex["translation"][tgt]) for ex in json.load(f)]


def build_items(limit):
    """{(kind, lang): [(prefix, text)]}. kind mono: prefix ''. kind trans: xx->en and en->xx refs."""
    items = {}
    for L in LANGS:
        xe, ex = test_pairs(L, "en")[:limit], test_pairs("en", L)[:limit]
        items[("mono", L)] = [("", s) for s, _ in xe]                        # native-side sources
        items[("mono", "en")] = items.get(("mono", "en"), []) + [("", s) for s, _ in ex[: limit // len(LANGS)]]
        items[("trans", f"{L}-en")] = [(get_prompt(L, "en", s), " " + t) for s, t in xe]
        items[("trans", f"en-{L}")] = [(get_prompt("en", L, s), " " + t) for s, t in ex]
    return items


def score(model, tok, pairs, device, batch):
    import torch
    tot_nll, tot_tok = 0.0, 0
    for i in range(0, len(pairs), batch):
        ids, labels = [], []
        for prefix, text in pairs[i:i + batch]:
            p = tok(prefix, add_special_tokens=True).input_ids if prefix else [tok.bos_token_id]
            t = tok(text, add_special_tokens=False).input_ids[:256]
            ids.append(p + t)
            labels.append([-100] * len(p) + t)
        n = max(map(len, ids))
        pad = tok.pad_token_id if tok.pad_token_id is not None else 0
        x = torch.tensor([a + [pad] * (n - len(a)) for a in ids], device=device)
        y = torch.tensor([b + [-100] * (n - len(b)) for b in labels], device=device)
        m = torch.tensor([[1] * len(a) + [0] * (n - len(a)) for a in ids], device=device)
        with torch.inference_mode():
            logits = model(input_ids=x, attention_mask=m).logits.float()
        lp = torch.nn.functional.cross_entropy(logits[:, :-1].transpose(1, 2), y[:, 1:],
                                               ignore_index=-100, reduction="sum")
        tot_nll += lp.item()
        tot_tok += (y[:, 1:] != -100).sum().item()
    return tot_nll / tot_tok, tot_tok


def compare(paths):
    recs = {json.load(open(p))["tag"]: json.load(open(p))["nll"] for p in paths}
    base = recs.pop("fp16")
    keys = list(base)
    print("dNLL vs fp16 (nats/token)\n" + f"{'item':18s}" + "".join(f"{t:>10s}" for t in recs))
    for k in keys:
        print(f"{k:18s}" + "".join(f"{r[k] - base[k]:+10.3f}" for r in recs.values()) + f"   fp16 {base[k]:.3f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model")
    ap.add_argument("--quantized", action="store_true")
    ap.add_argument("--tag")
    ap.add_argument("--out")
    ap.add_argument("--limit", type=int, default=500)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--compare", nargs="+")
    ap.add_argument("--adapter", help="LoRA folder from distill_lora.py, applied on top of --model")
    a = ap.parse_args()
    if a.compare:
        return compare(a.compare)
    import torch
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(a.model, add_eos_token=False)
    if a.quantized:
        from gptqmodel import GPTQModel
        model = GPTQModel.load(a.model, device=a.device, torch_dtype=torch.bfloat16).model
    else:
        from transformers import AutoModelForCausalLM
        model = AutoModelForCausalLM.from_pretrained(a.model, torch_dtype=torch.bfloat16).to(a.device)
    if a.adapter:
        sys.path.insert(0, str(Path(__file__).resolve().parent))
        import lora
        lora.load(model, a.adapter)
    model.eval()
    res = {"tag": a.tag, "model": a.model, "adapter": a.adapter, "limit": a.limit, "nll": {}, "tokens": {}}
    for (kind, L), pairs in build_items(a.limit).items():
        nll, n = score(model, tok, pairs, a.device, a.batch)
        res["nll"][f"{kind}:{L}"], res["tokens"][f"{kind}:{L}"] = nll, n
        print(f"{a.tag} {kind}:{L} nll={nll:.4f} tokens={n}", flush=True)
    json.dump(res, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
