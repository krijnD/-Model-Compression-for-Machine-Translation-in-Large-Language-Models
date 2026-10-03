"""Distil fp16 ALMA-13B-R into a LoRA on top of the frozen 3-bit GPTQ checkpoint (paper Section 2.3, Appendix B).

Teacher: fp16 ALMA-13B-R (bf16 compute). Student: ALMA-13B-R-gptq-w3g128 on TorchQuantLinear, all
weights frozen, plus a rank-r LoRA on every decoder Linear (mtcompress/lora.py). Loss: token-level forward KL
KL(p_teacher || p_student) over the full vocabulary, on the target tokens only (the prompt is context),
i.e. the student learns to put its probability mass where fp16 puts it, not to imitate the references.
--ce-weight adds the usual cross-entropy on the reference tokens (the "SFT on references" variant is
--kd-weight 0 --ce-weight 1, which also skips loading the teacher).

Data: ALMA's human-written parallel train files (mtcompress/alma_prompt.py load_train_pairs), in ALMA's
fine-tuning format: BOS + prompt + target + EOS, loss on target + EOS, exactly the calibration format.
No overlap with the WMT'22 (WMT'21 for is) test sets. --heldout rows per language pair are held out of
training in *both* directions (the two directions of a pair share the same sentence pairs) and give the
held-out KL per direction, logged at step 0 (= the damage of plain w3) and every --eval-every steps.
Sampling: every remaining Icelandic row, repeated --is-repeat times, the rest of --n-examples split
evenly over the other 8 directions.

  python scripts/distill/distill_lora.py --teacher $MODELS_DIR/ALMA-13B-R \
      --student $MODELS_DIR/ALMA-13B-R-gptq-w3g128 --out $ARTIFACTS_DIR/w3-kd-lora
  (--init-adapter <dir> continues a finished run with a fresh schedule; --max-steps 30 for a smoke test; a student folder without quantize_config.json is loaded as a
   plain HF model, which is how the fp16 + LoRA control and the CPU test with a tiny model run)
Output: <out>/adapter.safetensors, adapter_config.json (all settings + data counts), train_log.jsonl.
"""
import argparse
import json
import math
import random
import time
from pathlib import Path

import torch
import torch.nn.functional as F

from mtcompress import lora
from mtcompress.alma_prompt import WMT22_PAIRS, get_prompt, load_train_pairs


def build_data(tok, a):
    """-> (train, heldout, counts). Each example: (pair, input_ids, n_prompt_tokens)."""
    rng = random.Random(a.seed)
    langs = sorted({s if s != "en" else t for s, t in (p.split("-") for p in WMT22_PAIRS)})
    held_idx, rows = {}, {}
    for L in langs:
        n = len(load_train_pairs(L, "en"))
        idx = list(range(n))
        rng.shuffle(idx)
        held_idx[L] = set(idx[: a.heldout])
    for pair in WMT22_PAIRS:
        src, tgt = pair.split("-")
        L = src if src != "en" else tgt
        allrows = load_train_pairs(src, tgt)  # same row order for both directions of a pair
        rows[pair] = ([r for i, r in enumerate(allrows) if i not in held_idx[L]],
                      [r for i, r in enumerate(allrows) if i in held_idx[L]])

    is_pairs = [p for p in WMT22_PAIRS if "is" in p.split("-")]
    other = [p for p in WMT22_PAIRS if p not in is_pairs]
    n_is = sum(len(rows[p][0]) for p in is_pairs) * a.is_repeat
    per_other = max(0, (a.n_examples - n_is) // len(other))
    chosen = []
    for p in is_pairs:
        chosen += [(p, r) for r in rows[p][0]] * a.is_repeat
    for p in other:
        pool = rows[p][0]
        chosen += [(p, r) for r in rng.sample(pool, min(per_other, len(pool)))]
    rng.shuffle(chosen)

    def encode(p, r):
        src, tgt = p.split("-")
        prompt = get_prompt(src, tgt, r[0])
        ids = tok(prompt + r[1], max_length=a.max_len - 1, truncation=True).input_ids + [tok.eos_token_id]
        n_prompt = len(tok(prompt).input_ids)  # as ALMA's ignore_prompt_token_for_loss
        return (p, ids, n_prompt) if n_prompt < len(ids) else None

    train = [e for e in (encode(p, r) for p, r in chosen) if e]
    held = [e for p in WMT22_PAIRS for e in (encode(p, r) for r in rows[p][1]) if e]
    counts = {p: sum(e[0] == p for e in train) for p in WMT22_PAIRS}
    return train, held, counts


def collate(batch, pad_id, device):
    n = max(len(ids) for _, ids, _ in batch)
    x = torch.full((len(batch), n), pad_id, dtype=torch.long)
    m = torch.zeros((len(batch), n), dtype=torch.long)
    y = torch.full((len(batch), n), -100, dtype=torch.long)
    for i, (_, ids, n_prompt) in enumerate(batch):
        x[i, : len(ids)] = torch.tensor(ids)
        m[i, : len(ids)] = 1
        y[i, n_prompt: len(ids)] = torch.tensor(ids[n_prompt:])
    return x.to(device), m.to(device), y.to(device)


def losses(student, teacher, x, m, y, a):
    """Sum over target tokens of KD (forward KL) and CE, plus the token count."""
    tgt = y[:, 1:]
    sel = tgt != -100
    s = student(input_ids=x, attention_mask=m).logits[:, :-1][sel].float()
    kd = ce = torch.zeros((), device=x.device)
    if teacher is not None:
        with torch.no_grad():
            t = teacher(input_ids=x, attention_mask=m).logits[:, :-1][sel].float()
        kd = F.kl_div(F.log_softmax(s, -1), F.log_softmax(t, -1), log_target=True, reduction="sum")
    if a.ce_weight or teacher is None:
        ce = F.cross_entropy(s, tgt[sel], reduction="sum")
    return kd, ce, sel.sum()


@torch.no_grad()
def evaluate(student, teacher, held, a, pad_id):
    """Per-direction held-out KL (if a teacher) and CE on the references, per target token."""
    was = student.training
    student.eval()
    acc = {}
    for i in range(0, len(held), a.micro_batch):
        batch = held[i:i + a.micro_batch]
        for p in {e[0] for e in batch}:  # per direction: one sub-batch each
            sub = [e for e in batch if e[0] == p]
            kd, ce, n = losses(student, teacher, *collate(sub, pad_id, a.device), argparse.Namespace(ce_weight=1))
            k, c, t = acc.get(p, (0.0, 0.0, 0))
            acc[p] = (k + kd.item(), c + ce.item(), t + n.item())
    student.train(was)
    return {p: {"kl": k / t, "ce": c / t, "tokens": t} for p, (k, c, t) in acc.items()}


def load_student(path, device):
    if (Path(path) / "quantize_config.json").exists():
        from gptqmodel import BACKEND, GPTQModel
        # TORCH: the only kernel here with a backward pass (ExllamaV2 has SUPPORTS_TRAINING = False)
        m = GPTQModel.load(path, device=device, backend=BACKEND.TORCH, torch_dtype=torch.bfloat16).model
    else:
        from transformers import AutoModelForCausalLM
        m = AutoModelForCausalLM.from_pretrained(path, torch_dtype=torch.bfloat16).to(device)
    return m


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--teacher", help="fp16 model folder (not needed with --kd-weight 0)")
    p.add_argument("--student", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--n-examples", type=int, default=30000)
    p.add_argument("--is-repeat", type=int, default=2)
    p.add_argument("--heldout", type=int, default=32, help="rows per language pair, held out of training")
    p.add_argument("--max-len", type=int, default=256)
    p.add_argument("--r", type=int, default=16)
    p.add_argument("--alpha", type=float, default=32)
    p.add_argument("--targets", default=",".join(lora.TARGETS))
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--warmup", type=float, default=0.03)
    p.add_argument("--micro-batch", type=int, default=16)
    p.add_argument("--grad-accum", type=int, default=2)
    p.add_argument("--epochs", type=int, default=1)
    p.add_argument("--max-steps", type=int, help="stop after this many optimizer steps (smoke test)")
    p.add_argument("--eval-every", type=int, default=200)
    p.add_argument("--kd-weight", type=float, default=1.0)
    p.add_argument("--ce-weight", type=float, default=0.0)
    p.add_argument("--no-grad-ckpt", action="store_true")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--init-adapter", help="start from this distill_lora.py adapter (same r/alpha/targets) instead of "
                   "B = 0, i.e. continue training with a fresh lr schedule. Keep --seed: the held-out rows depend on it")
    p.add_argument("--device", default="cuda:0")
    a = p.parse_args()
    torch.manual_seed(a.seed)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    log = open(out / "train_log.jsonl", "w")

    def emit(rec):
        print(json.dumps(rec), flush=True)
        log.write(json.dumps(rec) + "\n")
        log.flush()

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(a.student, add_eos_token=False)
    pad_id = tok.pad_token_id if tok.pad_token_id is not None else 0
    train, held, counts = build_data(tok, a)
    n_tok = sum(len(ids) - n for _, ids, n in train)
    emit({"data": counts, "train_examples": len(train), "train_target_tokens": n_tok,
          "train_total_tokens": sum(len(ids) for _, ids, _ in train), "heldout_examples": len(held)})

    t0 = time.time()
    teacher = None
    if a.kd_weight:
        from transformers import AutoModelForCausalLM
        teacher = AutoModelForCausalLM.from_pretrained(a.teacher, torch_dtype=torch.bfloat16).to(a.device).eval()
        for q in teacher.parameters():
            q.requires_grad_(False)
    student = load_student(a.student, a.device)
    kernels = sorted({type(m).__name__ for m in student.modules() if "QuantLinear" in type(m).__name__})
    wrapped = lora.inject(student, a.r, a.alpha, a.targets.split(","))
    if a.init_adapter:
        from safetensors.torch import load_file
        cfg = json.loads((Path(a.init_adapter) / "adapter_config.json").read_text())
        if (cfg["r"], cfg["alpha"], cfg["targets"]) != (a.r, a.alpha, a.targets.split(",")):
            raise ValueError(f"--init-adapter has r={cfg['r']} alpha={cfg['alpha']}, this run r={a.r} alpha={a.alpha}")
        sd = load_file(str(Path(a.init_adapter) / "adapter.safetensors"))
        with torch.no_grad():
            for k, v in sd.items():
                student.get_parameter(k).copy_(v)
        print(f"initialised {len(sd)} LoRA tensors from {a.init_adapter} ({cfg['steps']} steps)", flush=True)
    student.config.use_cache = False
    if not a.no_grad_ckpt:
        student.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    student.train()
    params = [q for q in student.parameters() if q.requires_grad]
    emit({"loaded_s": round(time.time() - t0, 1), "student_kernels": kernels, "lora_modules": len(wrapped),
          "lora_params": sum(q.numel() for q in params),
          "peak_mem_gib": round(torch.cuda.max_memory_allocated() / 2**30, 2) if torch.cuda.is_available() else None})

    steps_per_epoch = len(train) // (a.micro_batch * a.grad_accum)
    total = min(a.max_steps or 10**9, steps_per_epoch * a.epochs)
    opt = torch.optim.AdamW(params, lr=a.lr, weight_decay=0.0)
    warm = max(1, int(a.warmup * total))
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: (s + 1) / warm if s < warm else 0.5 * (1 + math.cos(math.pi * (s - warm) / max(1, total - warm))))

    emit({"step": 0, "heldout": evaluate(student, teacher, held, a, pad_id)})
    step, mb, seen_tok, t_train = 0, 0, 0, time.time()
    order = [i for _ in range(a.epochs) for i in range(len(train))]
    while step < total:
        acc = {"kd": 0.0, "ce": 0.0, "tok": 0}
        for _ in range(a.grad_accum):
            batch = [train[order[(mb * a.micro_batch + j) % len(order)]] for j in range(a.micro_batch)]
            mb += 1
            kd, ce, n = losses(student, teacher, *collate(batch, pad_id, a.device), a)
            ((a.kd_weight * kd + a.ce_weight * ce) / n / a.grad_accum).backward()
            acc["kd"] += kd.item(); acc["ce"] += ce.item(); acc["tok"] += n.item()
            seen_tok += sum(len(ids) for _, ids, _ in batch)
        if step == 0:  # the check the smoke test is for: gradients reach every adapter through the kernel
            got = sum(q.grad is not None and q.grad.abs().sum().item() > 0 for q in params)
            emit({"grad_check": f"{got}/{len(params)} LoRA tensors got a nonzero gradient "
                                "(lora_A gets none on step 1: B starts at 0)"})
        gnorm = torch.nn.utils.clip_grad_norm_(params, 1.0).item()
        opt.step(); sched.step(); opt.zero_grad(set_to_none=True)
        step += 1
        el = time.time() - t_train
        if step % 10 == 0 or step == 1 or step == total:
            emit({"step": step, "of": total, "kd": acc["kd"] / acc["tok"], "ce": acc["ce"] / acc["tok"],
                  "grad_norm": round(gnorm, 4), "lr": sched.get_last_lr()[0], "tok_per_s": round(seen_tok / el),
                  "s_per_step": round(el / step, 2), "eta_min": round(el / step * (total - step) / 60, 1),
                  "full_epoch_eta_min": round(el / step * steps_per_epoch / 60, 1),
                  "peak_mem_gib": round(torch.cuda.max_memory_allocated() / 2**30, 2) if torch.cuda.is_available() else None})
        if not math.isfinite(gnorm):
            raise RuntimeError(f"non-finite gradient norm at step {step}")
        if step % a.eval_every == 0 and step < total:
            emit({"step": step, "heldout": evaluate(student, teacher, held, a, pad_id)})
    emit({"step": step, "heldout": evaluate(student, teacher, held, a, pad_id), "train_min": round((time.time() - t_train) / 60, 1)})
    lora.save(student, out, {"r": a.r, "alpha": a.alpha, "targets": a.targets.split(","), "student": a.student,
                             "teacher": a.teacher, "steps": step, "args": vars(a), "data": counts,
                             "student_kernels": kernels})
    print(f"Saved adapter to {out}")


if __name__ == "__main__":
    main()
