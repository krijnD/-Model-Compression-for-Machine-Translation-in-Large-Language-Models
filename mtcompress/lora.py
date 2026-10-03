"""Minimal LoRA that wraps *any* Linear-like module, including GPTQModel's packed QuantLinears.

Why not peft: peft decides per kernel class whether it can wrap a quantized layer, and we need the same
adapter on two kernels. Training runs on TorchQuantLinear (the only GPTQ kernel here with a backward
pass; ExllamaV2QuantLinear has SUPPORTS_TRAINING = False), and generation runs on the repacked w3 on
ExllamaV2QuantLinear (docs/04-kernel-repack.md, 2.5x faster). Both dequantize to the same weights, so a
wrapper that only calls base(x) and adds the low-rank term works on either.

    y = base(x) + (alpha / r) * x A^T B^T      A: r x in (Kaiming), B: out x r (zeros) -> starts as a no-op

The adapter is kept in fp32 and added to the kernel's bf16/fp16 output. Saved as safetensors with keys
"<module name>.lora_A" / ".lora_B" plus adapter_config.json.
"""
import json
import math
from pathlib import Path

import torch
from torch import nn

TARGETS = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]


class LoRA(nn.Module):
    def __init__(self, base, in_features, out_features, r, alpha):
        super().__init__()
        self.base = base
        self.scale = alpha / r
        dev = next(iter([*base.parameters(), *base.buffers()])).device  # QuantLinears only have buffers
        self.lora_A = nn.Parameter(torch.empty(r, in_features, device=dev, dtype=torch.float32))
        self.lora_B = nn.Parameter(torch.zeros(out_features, r, device=dev, dtype=torch.float32))
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))  # as peft

    def forward(self, x):
        out = self.base(x)
        return out + (self.scale * (x.float() @ self.lora_A.T @ self.lora_B.T)).to(out.dtype)


def inject(model, r=16, alpha=32, targets=TARGETS):
    """Freeze the whole model and wrap every decoder Linear whose name ends in one of `targets`."""
    for p in model.parameters():
        p.requires_grad_(False)
    names = [n for n, _ in model.named_modules() if n.split(".")[-1] in targets and ".layers." in n]
    for n in names:
        parent_name, child = n.rsplit(".", 1)
        parent = model.get_submodule(parent_name)
        base = getattr(parent, child)
        # nn.Linear and GPTQModel's QuantLinears both expose in_features / out_features
        setattr(parent, child, LoRA(base, base.in_features, base.out_features, r, alpha))
    return names


def lora_state(model):
    return {k: v.detach().cpu().contiguous() for k, v in model.state_dict().items()
            if k.endswith(".lora_A") or k.endswith(".lora_B")}


def save(model, out_dir, config):
    from safetensors.torch import save_file
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    save_file(lora_state(model), str(out / "adapter.safetensors"))
    (out / "adapter_config.json").write_text(json.dumps(config, indent=1))


def load(model, adapter_dir):
    """Wrap `model` as the adapter was trained and load its weights. Returns the adapter config."""
    from safetensors.torch import load_file
    d = Path(adapter_dir)
    cfg = json.loads((d / "adapter_config.json").read_text())
    names = inject(model, cfg["r"], cfg["alpha"], cfg["targets"])
    sd = load_file(str(d / "adapter.safetensors"))
    missing = {f"{n}.lora_{x}" for n in names for x in "AB"} ^ set(sd)
    if missing:
        raise ValueError(f"adapter/model mismatch on {len(missing)} keys, e.g. {sorted(missing)[:3]}")
    for k, v in sd.items():
        model.get_parameter(k).data.copy_(v)
    return cfg
