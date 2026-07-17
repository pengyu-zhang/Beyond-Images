"""Seeding, device resolution, and run banners shared by every stage."""

from __future__ import annotations

import os
import random

import numpy as np


def set_all_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import torch

        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def resolve_device(preference: str = "auto", tf32: bool = True) -> str:
    """Resolve the compute device and print a run banner."""
    try:
        import torch
    except ImportError:
        print("[device] torch not installed; running on cpu")
        return "cpu"

    if preference == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = preference

    if device.startswith("cuda") and torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = tf32
        torch.backends.cudnn.allow_tf32 = tf32
        name = torch.cuda.get_device_name(0)
        vram = torch.cuda.get_device_properties(0).total_memory / 1024**3
        print(f"[device] cuda: {name} ({vram:.1f} GB), tf32={tf32}")
    else:
        device = "cpu"
        print("[device] cpu")
    return device
