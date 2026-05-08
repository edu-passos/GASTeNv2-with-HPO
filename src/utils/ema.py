"""Exponential moving average of model parameters.

Used on the GAN generator to smooth out epoch-to-epoch noise in samples and
metrics, and to give step-2 a less-noisy starting point. Routinely worth a
few FID points on adversarial training.
"""
from __future__ import annotations

from contextlib import contextmanager
from typing import Iterable

import torch
import torch.nn as nn


class EMA:
    """Maintain a parameter-shadow that follows ``model`` with decay ``decay``.

    Update rule per call to ``update(model)``::

        ema_p = decay * ema_p + (1 - decay) * model_p

    Buffers (e.g. BatchNorm running stats) are mirrored exactly — there's no
    moving average for those; we just keep the latest copy.
    """

    def __init__(self, model: nn.Module, decay: float = 0.999):
        if not (0.0 < decay < 1.0):
            raise ValueError(f"EMA decay must be in (0, 1); got {decay}")
        self.decay = float(decay)
        self.shadow_params: dict[str, torch.Tensor] = {}
        self.shadow_buffers: dict[str, torch.Tensor] = {}
        for n, p in model.named_parameters():
            if p.requires_grad:
                self.shadow_params[n] = p.detach().clone()
        for n, b in model.named_buffers():
            self.shadow_buffers[n] = b.detach().clone()

    @torch.no_grad()
    def update(self, model: nn.Module) -> None:
        d = self.decay
        for n, p in model.named_parameters():
            if not p.requires_grad:
                continue
            s = self.shadow_params.get(n)
            if s is None:
                self.shadow_params[n] = p.detach().clone()
                continue
            # in-place: s = d*s + (1-d)*p
            s.mul_(d).add_(p.detach(), alpha=1.0 - d)
        for n, b in model.named_buffers():
            self.shadow_buffers[n] = b.detach().clone()

    @contextmanager
    def swapped(self, model: nn.Module):
        """Temporarily load EMA weights into ``model``; restore on exit."""
        backup_params: dict[str, torch.Tensor] = {}
        backup_buffers: dict[str, torch.Tensor] = {}
        try:
            for n, p in model.named_parameters():
                if n in self.shadow_params:
                    backup_params[n] = p.detach().clone()
                    p.data.copy_(self.shadow_params[n].to(p.device, dtype=p.dtype))
            for n, b in model.named_buffers():
                if n in self.shadow_buffers:
                    backup_buffers[n] = b.detach().clone()
                    b.data.copy_(self.shadow_buffers[n].to(b.device, dtype=b.dtype))
            yield model
        finally:
            for n, p in model.named_parameters():
                if n in backup_params:
                    p.data.copy_(backup_params[n])
            for n, b in model.named_buffers():
                if n in backup_buffers:
                    b.data.copy_(backup_buffers[n])

    def state_dict(self) -> dict:
        return {
            "decay": self.decay,
            "params": {n: t.detach().cpu() for n, t in self.shadow_params.items()},
            "buffers": {n: t.detach().cpu() for n, t in self.shadow_buffers.items()},
        }

    def load_state_dict(self, sd: dict, device: torch.device | None = None) -> None:
        self.decay = float(sd.get("decay", self.decay))
        params = sd.get("params", {})
        buffers = sd.get("buffers", {})
        if device is None:
            self.shadow_params = {n: t.detach().clone() for n, t in params.items()}
            self.shadow_buffers = {n: t.detach().clone() for n, t in buffers.items()}
        else:
            self.shadow_params = {n: t.detach().to(device) for n, t in params.items()}
            self.shadow_buffers = {n: t.detach().to(device) for n, t in buffers.items()}

    @classmethod
    def from_model(cls, model: nn.Module, decay: float = 0.999) -> "EMA":
        return cls(model, decay=decay)


def maybe_make_ema(cfg_section: dict | None, model: nn.Module) -> EMA | None:
    """Build an EMA from a config sub-section like ``{"ema": {"decay": 0.999}}``.

    Returns None if the config disables EMA. Accepts either ``{"ema": ...}`` or
    a bare ``{"decay": ...}`` mapping.
    """
    if not cfg_section:
        return None
    if "ema" in cfg_section:
        cfg_section = cfg_section["ema"]
    if cfg_section is None or cfg_section is False:
        return None
    if isinstance(cfg_section, bool):
        return EMA(model) if cfg_section else None
    decay = float(cfg_section.get("decay", 0.999))
    return EMA(model, decay=decay)
