"""Mixed-precision helpers for GAN training.

Two modes:
- 'bf16': bfloat16 autocast. Stable, no GradScaler needed. Requires Ampere+.
- 'fp16': float16 autocast. Faster on older GPUs but needs a GradScaler.

Off by default. Enable via cfg.train.amp = 'bf16' (recommended on modern GPUs)
or 'fp16'.

R1 / WGP gradient penalties must be computed in fp32 because second-order
gradients through autocast are unstable. Each penalty implementation wraps
its `autograd.grad` call in `autocast(enabled=False)`.
"""
from __future__ import annotations

from contextlib import nullcontext
from typing import Optional, Tuple

import torch


def parse_amp(mode) -> Tuple[Optional[torch.dtype], Optional[object]]:
    """Return (dtype, scaler). Both None when AMP is disabled."""
    if mode is None:
        return None, None
    s = str(mode).strip().lower()
    if s in ("", "off", "false", "no", "none", "disabled"):
        return None, None
    if s in ("bf16", "bfloat16"):
        return torch.bfloat16, None
    if s in ("fp16", "float16", "half"):
        return torch.float16, torch.amp.GradScaler("cuda")
    raise ValueError(f"unknown amp mode {mode!r}; expected one of off|bf16|fp16")


def autocast_ctx(amp_dtype: Optional[torch.dtype], device: torch.device):
    """Return an autocast context for the active AMP dtype, or a no-op."""
    if amp_dtype is None:
        return nullcontext()
    return torch.autocast(device_type=device.type, dtype=amp_dtype)


def fp32_ctx(device: torch.device):
    """Disable autocast for an fp32 island (used inside R1/WGP penalties)."""
    return torch.autocast(device_type=device.type, enabled=False)
