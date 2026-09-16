"""Runtime device/batch resolution for the ALMA compression harness (dual-platform).

Order: MPS -> CUDA -> CPU. Batch capped by RAM: >=32 GB -> 32, else 8.
fp16 allowed on CUDA always; on MPS only if RAM >= 32 GB.
"""
from __future__ import annotations

import os
import platform
import subprocess


def _ram_bytes() -> int:
    if platform.system() == "Darwin":
        out = subprocess.check_output(["sysctl", "-n", "hw.memsize"]).strip()
        return int(out)
    if platform.system() == "Linux":
        for line in open("/proc/meminfo"):
            if line.startswith("MemTotal:"):
                return int(line.split()[1]) * 1024
    return 8 << 30  # conservative default (8 GB)


def ram_gb() -> float:
    return _ram_bytes() / (1024 ** 3)


def device_name() -> str:
    """mps | cuda | cpu (imports torch lazily; safe to call without GPU)."""
    try:
        import torch
    except ImportError:
        return "cpu"
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def device():
    import torch
    return torch.device(device_name())


def max_batch() -> int:
    return 32 if ram_gb() >= 32 else 8


def can_fp16() -> bool:
    """True if fp16 weights/activations are safe on this machine."""
    d = device_name()
    if d == "cuda":
        return True
    if d == "mps":
        return ram_gb() >= 32
    return False


def platform_tag() -> str:
    """'slurm' | 'mac' | 'linux'. Ledger field."""
    if os.environ.get("SLURM_JOB_ID") or os.environ.get("SLURM_JOB_UID"):
        return "slurm"
    if platform.system() == "Darwin":
        return "mac"
    if platform.system() == "Linux":
        return "slurm" if os.path.exists("/usr/bin/sbatch") else "linux"
    return platform.system().lower()


if __name__ == "__main__":
    print(f"device={device_name()} ram_gb={ram_gb():.1f} max_batch={max_batch()} "
          f"can_fp16={can_fp16()} platform={platform_tag()}")