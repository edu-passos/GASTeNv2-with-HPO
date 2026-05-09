"""Differentiable augmentation for data-efficient GAN training.

Reference: Zhao et al., "Differentiable Augmentation for Data-Efficient GAN
Training," NeurIPS 2020. https://arxiv.org/abs/2006.10738

The same augmentation policy is applied identically to real and fake images
before they reach the discriminator. Because the operations are differentiable,
gradients flow through to the generator. This is *the* fix for collapse on
small datasets like chest-xray (≈4 000 images).

Usage::

    aug = DiffAugmenter("color,translation,cutout")
    real_aug = aug(real)
    fake_aug = aug(fake)
    d_real = D(real_aug)
    d_fake = D(fake_aug)

Policies (mix and match, comma separated):
- ``color``        — brightness, saturation, contrast jitter
- ``translation``  — random shift up to 1/8 image (zero pad)
- ``cutout``       — zero out a 1/2 × 1/2 random patch
"""
from __future__ import annotations

from typing import Iterable

import torch


# ---------------------------------------------------------------------------
# Color
# ---------------------------------------------------------------------------
def _rand_brightness(x: torch.Tensor) -> torch.Tensor:
    return x + (torch.rand(x.size(0), 1, 1, 1, dtype=x.dtype, device=x.device) - 0.5)


def _rand_saturation(x: torch.Tensor) -> torch.Tensor:
    mean = x.mean(dim=1, keepdim=True)
    return (x - mean) * (torch.rand(x.size(0), 1, 1, 1, dtype=x.dtype, device=x.device) * 2.0) + mean


def _rand_contrast(x: torch.Tensor) -> torch.Tensor:
    mean = x.mean(dim=[1, 2, 3], keepdim=True)
    return (x - mean) * (torch.rand(x.size(0), 1, 1, 1, dtype=x.dtype, device=x.device) + 0.5) + mean


# ---------------------------------------------------------------------------
# Translation
# ---------------------------------------------------------------------------
def _rand_translation(x: torch.Tensor, ratio: float = 0.125) -> torch.Tensor:
    n, _, h, w = x.size()
    sh = int(h * ratio + 0.5)
    sw = int(w * ratio + 0.5)
    if sh == 0 and sw == 0:
        return x
    th = torch.randint(-sh, sh + 1, size=(n, 1, 1), device=x.device)
    tw = torch.randint(-sw, sw + 1, size=(n, 1, 1), device=x.device)

    grid_y, grid_x = torch.meshgrid(
        torch.arange(h, device=x.device),
        torch.arange(w, device=x.device),
        indexing="ij",
    )
    grid_y = grid_y.unsqueeze(0).expand(n, -1, -1) + th
    grid_x = grid_x.unsqueeze(0).expand(n, -1, -1) + tw
    grid_y = grid_y.clamp(0, h - 1)
    grid_x = grid_x.clamp(0, w - 1)

    # gather; pad by replicate at borders is fine — we clamped indices.
    out = x.permute(0, 2, 3, 1)  # n h w c
    batch_idx = torch.arange(n, device=x.device).view(n, 1, 1).expand(-1, h, w)
    out = out[batch_idx, grid_y, grid_x]
    return out.permute(0, 3, 1, 2)


# ---------------------------------------------------------------------------
# Cutout
# ---------------------------------------------------------------------------
def _rand_cutout(x: torch.Tensor, ratio: float = 0.5) -> torch.Tensor:
    n, _, h, w = x.size()
    cw = int(w * ratio + 0.5)
    ch = int(h * ratio + 0.5)
    if cw == 0 or ch == 0:
        return x
    cy = torch.randint(0, h, size=(n,), device=x.device)
    cx = torch.randint(0, w, size=(n,), device=x.device)

    mask = torch.ones(n, 1, h, w, dtype=x.dtype, device=x.device)
    grid_y, grid_x = torch.meshgrid(
        torch.arange(h, device=x.device),
        torch.arange(w, device=x.device),
        indexing="ij",
    )
    grid_y = grid_y.unsqueeze(0).expand(n, -1, -1)
    grid_x = grid_x.unsqueeze(0).expand(n, -1, -1)

    in_y = (grid_y >= (cy.view(n, 1, 1) - ch // 2)) & (grid_y < (cy.view(n, 1, 1) + ch // 2))
    in_x = (grid_x >= (cx.view(n, 1, 1) - cw // 2)) & (grid_x < (cx.view(n, 1, 1) + cw // 2))
    mask[(in_y & in_x).unsqueeze(1).expand(-1, 1, -1, -1)] = 0
    return x * mask


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
_OPS = {
    "color": (_rand_brightness, _rand_saturation, _rand_contrast),
    "translation": (_rand_translation,),
    "cutout": (_rand_cutout,),
}


class DiffAugmenter:
    """Apply a sequence of differentiable augmentations.

    ``policy`` is a comma-separated string (e.g. ``"color,translation,cutout"``)
    or an iterable of policy names. ``DiffAugmenter("")`` / passing ``None``
    yields a no-op augmenter (handy for callers that always pass an instance).
    """
    def __init__(self, policy: str | Iterable[str] | None = "color,translation,cutout"):
        if policy is None:
            self.ops: list = []
            return
        if isinstance(policy, str):
            tokens = [t.strip() for t in policy.split(",") if t.strip()]
        else:
            tokens = [str(t).strip() for t in policy if str(t).strip()]

        ops: list = []
        for t in tokens:
            if t not in _OPS:
                raise ValueError(f"Unknown DiffAugment policy {t!r}; valid: {list(_OPS)}")
            ops.extend(_OPS[t])
        self.ops = ops

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        for op in self.ops:
            x = op(x)
        return x

    def __bool__(self) -> bool:
        return bool(self.ops)


def make_augmenter(cfg) -> DiffAugmenter | None:
    """Build a :class:`DiffAugmenter` from a config slice (str or dict).

    Accepted shapes::

        diffaug: "color,translation,cutout"
        diffaug: { policy: "color,translation,cutout" }
        diffaug: false  # disabled
        diffaug: null   # disabled

    Returns None if disabled, otherwise a DiffAugmenter (which may itself be
    falsy if the policy string is empty).
    """
    if cfg is None or cfg is False:
        return None
    if isinstance(cfg, str):
        return DiffAugmenter(cfg)
    if isinstance(cfg, dict):
        return DiffAugmenter(cfg.get("policy", "color,translation,cutout"))
    raise TypeError(f"unsupported diffaug config: {type(cfg).__name__}")
