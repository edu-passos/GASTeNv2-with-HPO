from __future__ import annotations

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import spectral_norm


# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------
def pixel_norm(x: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    return x * torch.rsqrt(torch.mean(x ** 2, dim=1, keepdim=True) + eps)


def he_init(m: nn.Module) -> None:
    """
    Kaiming init for regular Conv2d/Linear modules.

    IMPORTANT: we intentionally skip the ModConv style affine (marked with
    _skip_he_init) because for stability it should start at weight=0, bias=1.
    """
    if isinstance(m, nn.Linear) and getattr(m, "_skip_he_init", False):
        return

    if isinstance(m, (nn.Conv2d, nn.Linear)):
        nn.init.kaiming_normal_(m.weight, a=0.2)
        if m.bias is not None:
            nn.init.zeros_(m.bias)


# ---------------------------------------------------------------------
# Attention
# ---------------------------------------------------------------------
class SelfAttention(nn.Module):
    def __init__(self, ch: int, qk_dim: int | None = None):
        super().__init__()
        qk_dim = qk_dim or max(1, ch // 8)
        self.to_q   = spectral_norm(nn.Conv2d(ch,     qk_dim, 1))
        self.to_k   = spectral_norm(nn.Conv2d(ch,     qk_dim, 1))
        self.to_v   = spectral_norm(nn.Conv2d(ch,   ch // 2, 1))
        self.to_out = spectral_norm(nn.Conv2d(ch // 2, ch,   1))
        self.gamma  = nn.Parameter(torch.zeros(1))

    def forward(self, x: torch.Tensor, w: torch.Tensor | None = None) -> torch.Tensor:
        b, c, h, w_ = x.shape
        q = self.to_q(x).reshape(b, -1, h * w_).permute(0, 2, 1)   # (B, HW, Cq)
        k = self.to_k(x).reshape(b, -1, h * w_).permute(0, 2, 1)   # (B, HW, Cq)
        v = self.to_v(x).reshape(b, -1, h * w_).permute(0, 2, 1)   # (B, HW, Cv)

        attn = F.scaled_dot_product_attention(q, k, v)            # (B, HW, Cv)
        attn = attn.permute(0, 2, 1).reshape(b, c // 2, h, w_)     # (B, Cv, H, W)
        return x + self.gamma * self.to_out(attn)


# ---------------------------------------------------------------------
# Mapping network
# ---------------------------------------------------------------------
class Mapping(nn.Sequential):
    def __init__(self, z_dim: int = 128, w_dim: int = 512, n_layers: int = 8):
        layers: list[nn.Module] = []
        for i in range(n_layers):
            inp = z_dim if i == 0 else w_dim
            layers += [nn.Linear(inp, w_dim), nn.LeakyReLU(0.2)]
        super().__init__(*layers)
        self.apply(he_init)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return pixel_norm(super().forward(z))


# ---------------------------------------------------------------------
# Modulated Conv (stability-fixed)
# ---------------------------------------------------------------------
class ModConv(nn.Module):
    """
    StyleGAN-like modulated conv.

    Fixes:
      - weight scaling (equalized-ish) to prevent early blow-up and tanh saturation
      - style affine init: weight=0, bias=1 -> styles start near 1
    """
    def __init__(self, in_ch: int, out_ch: int, k: int = 3, demod: bool = True, w_dim: int = 512):
        super().__init__()
        self.k = k
        self.pad = k // 2
        self.demod = demod

        # weight scaling: N(0,1)/sqrt(fan_in)
        fan_in = in_ch * k * k
        w = torch.randn(out_ch, in_ch, k, k) / math.sqrt(fan_in)
        self.weight = nn.Parameter(w)

        self.affine = nn.Linear(w_dim, in_ch)
        # mark so global he_init() won't override
        self.affine._skip_he_init = True

        # style affine init: weight=0, bias=1 (critical)
        nn.init.zeros_(self.affine.weight)
        nn.init.ones_(self.affine.bias)

    def forward(self, x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
        b, c, h, w_ = x.shape

        style = self.affine(w).view(b, 1, c, 1, 1)  # starts ~1
        wgt = self.weight[None] * style             # (B, out, in, k, k)

        if self.demod:
            d = torch.rsqrt((wgt ** 2).sum((2, 3, 4)) + 1e-8)       # (B, out)
            wgt = wgt * d.view(b, -1, 1, 1, 1)

        wgt = wgt.view(-1, c, self.k, self.k)      # (B*out, in, k, k)
        x   = x.view(1, -1, h, w_)                  # (1, B*in, H, W)
        x   = F.conv2d(x, wgt, padding=self.pad, groups=b)
        return x.view(b, -1, h, w_)


# ---------------------------------------------------------------------
# Blocks
# ---------------------------------------------------------------------
class GBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, w_dim: int = 512):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode="nearest")
        self.c1 = ModConv(in_ch,  out_ch, k=3, demod=True,  w_dim=w_dim)
        self.c2 = ModConv(out_ch, out_ch, k=3, demod=True,  w_dim=w_dim)

        # noise strengths start at 0 (fine)
        self.n1 = nn.Parameter(torch.zeros(1, out_ch, 1, 1))
        self.n2 = nn.Parameter(torch.zeros(1, out_ch, 1, 1))
        self.act = nn.LeakyReLU(0.2)

    def forward(self, x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
        x = self.up(x)
        out1 = self.c1(x, w)
        x = self.act(out1 + self.n1 * torch.randn_like(out1))

        out2 = self.c2(x, w)
        x = self.act(out2 + self.n2 * torch.randn_like(out2))
        return x


class DBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.c1   = spectral_norm(nn.Conv2d(in_ch, out_ch, 3, 1, 1))
        self.c2   = spectral_norm(nn.Conv2d(out_ch, out_ch, 3, 1, 1))
        self.skip = spectral_norm(nn.Conv2d(in_ch, out_ch, 1))
        self.pool = nn.AvgPool2d(2)
        self.act  = nn.LeakyReLU(0.2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.pool(self.act(self.c2(self.act(self.c1(x)))))
        return y + self.pool(self.skip(x))


# ---------------------------------------------------------------------
# Generator / Discriminator
# ---------------------------------------------------------------------
class Generator(nn.Module):
    """StyleGAN2-style chest-xray generator, optionally class-conditional.

    When ``num_classes > 1`` a class embedding is added to the mapping output;
    pass ``y`` (LongTensor) to ``forward``. Float ``y`` is also accepted: it's
    used as a one-hot-like soft mix between the class embeddings, which lets
    callers (e.g. step-2) interpolate between classes for boundary search.
    """
    def __init__(self, img_size=(1, 128, 128), z_dim=128, fmap=64, w_dim=512,
                 num_classes: int = 0, clamp_tanh: bool = True, **_):
        super().__init__()
        self.z_dim = z_dim
        self.num_classes = int(num_classes)
        self.clamp_tanh = clamp_tanh

        c, h, _ = img_size
        log_res = int(math.log2(h))

        self.mapping = Mapping(z_dim=z_dim, w_dim=w_dim)

        if self.num_classes > 1:
            # init small so conditioning starts as a perturbation
            self.class_emb = nn.Embedding(self.num_classes, w_dim)
            nn.init.normal_(self.class_emb.weight, mean=0.0, std=0.02)

        # small constant init (critical)
        self.const = nn.Parameter(torch.randn(1, fmap * 16, 4, 4) * 0.02)

        self.upsample = nn.Upsample(scale_factor=2, mode="nearest")

        in_ch = fmap * 16
        self.blocks = nn.ModuleList()
        self.torgb  = nn.ModuleList()

        for i in range(log_res - 2):  # 4->8->...->128
            out_ch = max(fmap, in_ch // 2)

            self.blocks.append(GBlock(in_ch, out_ch, w_dim=w_dim))

            # attention at 32x32
            if 4 * (2 ** i) == 32:
                self.blocks.append(SelfAttention(out_ch))

            # 1x1 modconv to RGB/gray; keep weight scaling via ModConv
            self.torgb.append(ModConv(out_ch, c, k=1, demod=False, w_dim=w_dim))
            in_ch = out_ch

        self.tanh = nn.Tanh()

        # init "regular" conv/linear; ModConv.affine is protected by _skip_he_init
        self.apply(he_init)

    def _class_offset(self, y: torch.Tensor | None, n: int, device, dtype) -> torch.Tensor | None:
        if self.num_classes <= 1:
            return None
        if y is None:
            # uniform random class per sample
            y = torch.randint(0, self.num_classes, (n,), device=device)
        if y.dtype.is_floating_point:
            # treat as continuous mix over class embeddings (n,) ∈ [0,1] for binary,
            # or (n, num_classes) for general soft labels.
            if y.dim() == 1:
                # binary case: y is a P(class=1) score
                e0 = self.class_emb.weight[0]
                e1 = self.class_emb.weight[1] if self.num_classes > 1 else e0
                return ((1.0 - y).unsqueeze(-1) * e0 + y.unsqueeze(-1) * e1).to(dtype)
            return (y.to(dtype) @ self.class_emb.weight).to(dtype)
        return self.class_emb(y.to(torch.long)).to(dtype)

    def forward(self, z: torch.Tensor, y: torch.Tensor | None = None) -> torch.Tensor:
        w = self.mapping(z)
        offset = self._class_offset(y, z.size(0), z.device, w.dtype)
        if offset is not None:
            w = w + offset
        x = self.const.expand(z.size(0), -1, -1, -1)

        rgb = None
        ti = 0
        for blk in self.blocks:
            if isinstance(blk, GBlock):
                x = blk(x, w)
                new_rgb = self.torgb[ti](x, w)

                if rgb is None:
                    rgb = new_rgb
                else:
                    # prevent amplitude blow-up (critical)
                    rgb = (self.upsample(rgb) + new_rgb) * (1.0 / math.sqrt(2.0))
                ti += 1
            else:
                x = blk(x)

        if self.clamp_tanh:
            rgb = rgb.clamp(-10.0, 10.0)
        return self.tanh(rgb)


class Discriminator(nn.Module):
    """Projection-style class-conditional discriminator (Miyato 2018).

    When ``num_classes > 1``, output is the standard logit plus a projection
    term ``<features, class_emb(y)>``. When unconditional or ``y is None`` the
    projection term is zero — the unconditional path is unchanged.
    """
    def __init__(self, img_size=(1, 128, 128), fmap=64, num_classes: int = 0,
                 is_critic=False, **_):
        super().__init__()
        self.is_critic = bool(is_critic)
        self.num_classes = int(num_classes)

        c, h, _ = img_size
        log_res = int(math.log2(h))
        fmap16  = fmap * 16

        layers: list[nn.Module] = [spectral_norm(nn.Conv2d(c, fmap, 3, 1, 1))]
        in_ch = fmap
        res = h

        for _ in range(log_res - 2):
            out_ch = min(fmap16, in_ch * 2)
            layers.append(DBlock(in_ch, out_ch))
            if res == 32:
                layers.append(SelfAttention(out_ch))
            in_ch = out_ch
            res //= 2

        self.features  = nn.Sequential(*layers)
        self.feat_dim  = in_ch * res * res
        self.fc        = spectral_norm(nn.Linear(self.feat_dim, 1))
        self.act_out   = nn.Identity()

        if self.num_classes > 1:
            self.proj = spectral_norm(nn.Embedding(self.num_classes, self.feat_dim))
            nn.init.normal_(self.proj.weight, mean=0.0, std=0.02)

        self.apply(he_init)

    def forward(self, x: torch.Tensor, y: torch.Tensor | None = None) -> torch.Tensor:
        feats = self.features(x).flatten(1)
        out = self.fc(feats).view(-1)
        if self.num_classes > 1 and y is not None:
            if y.dtype.is_floating_point:
                # continuous mix over class embeddings (matches Generator's path)
                if y.dim() == 1:
                    e0 = self.proj.weight[0]
                    e1 = self.proj.weight[1] if self.num_classes > 1 else e0
                    emb = (1.0 - y).unsqueeze(-1) * e0 + y.unsqueeze(-1) * e1
                else:
                    emb = y.to(feats.dtype) @ self.proj.weight
            else:
                emb = self.proj(y.to(torch.long))
            out = out + (feats * emb).sum(dim=-1)
        return self.act_out(out)


def build_cxr_g(z_dim=128, base_ch=64):
    # base_ch=64 is a safer default for 128x128
    return Generator((1, 128, 128), z_dim=z_dim, fmap=base_ch)


def build_cxr_d(base_ch=64):
    return Discriminator((1, 128, 128), fmap=base_ch)
