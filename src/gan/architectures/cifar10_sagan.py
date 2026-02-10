from __future__ import annotations
import math, torch, torch.nn as nn, torch.nn.functional as F
from torch.nn.utils import spectral_norm

# ───────────────────────── helpers ─────────────────────────
def pixel_norm(x, eps=1e-8):
    return x * torch.rsqrt(torch.mean(x**2, dim=1, keepdim=True) + eps)

def he(m: nn.Module):
    if isinstance(m, (nn.Conv2d, nn.Linear)):
        nn.init.kaiming_normal_(m.weight, a=.2)
        if m.bias is not None:
            nn.init.zeros_(m.bias)

# ───────────────────────── mapping ─────────────────────────
class Mapping(nn.Sequential):
    def __init__(self, z_dim=128, w_dim=512, n_layers=8):
        super().__init__(*sum([[nn.Linear(z_dim if i == 0 else w_dim, w_dim),
                                nn.LeakyReLU(.2)] for i in range(n_layers)], []))
        self.apply(he)
    def forward(self, z): return pixel_norm(super().forward(z))

# ───────────────────── modulated conv ─────────────────────
class ModConv(nn.Module):
    def __init__(self, in_ch, out_ch, k=3, demod=True):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(out_ch, in_ch, k, k))
        self.affine = nn.Linear(512, in_ch)
        self.k, self.pad, self.demod = k, k // 2, demod
    def forward(self, x, w):
        b, c, h, w_ = x.shape
        s = self.affine(w).view(b, 1, c, 1, 1) + 1
        wgt = self.weight[None] * s
        if self.demod:
            d = torch.rsqrt((wgt**2).sum([2, 3, 4]) + 1e-8)
            wgt = wgt * d.view(b, -1, 1, 1, 1)
        wgt = wgt.view(-1, c, self.k, self.k)
        x = x.view(1, -1, h, w_)
        x = F.conv2d(x, wgt, padding=self.pad, groups=b)
        return x.view(b, -1, h, w_)

# ───────────────────── generator block ────────────────────
class GBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.up  = nn.Upsample(scale_factor=2, mode="nearest")
        self.c1, self.c2 = ModConv(in_ch, out_ch), ModConv(out_ch, out_ch)
        self.n1 = nn.Parameter(torch.zeros(1, out_ch, 1, 1))
        self.n2 = nn.Parameter(torch.zeros(1, out_ch, 1, 1))
        self.act = nn.LeakyReLU(.2)
    def forward(self, x, w):
        x = self.up(x)

        x = self.c1(x, w)
        x = x + self.n1 * torch.randn_like(x)
        x = self.act(x)

        x = self.c2(x, w)
        x = x + self.n2 * torch.randn_like(x)
        x = self.act(x)
        return x

# ─────────────────── discriminator block ──────────────────
class DBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.c1  = spectral_norm(nn.Conv2d(in_ch, out_ch, 3, 1, 1))
        self.c2  = spectral_norm(nn.Conv2d(out_ch, out_ch, 3, 1, 1))
        self.skip = spectral_norm(nn.Conv2d(in_ch, out_ch, 1))
        self.pool = nn.AvgPool2d(2)
        self.act  = nn.LeakyReLU(.2)
    def forward(self, x):
        y = self.pool(self.act(self.c2(self.act(self.c1(x)))))
        return y + self.pool(self.skip(x))

# ───────────── minibatch-stddev (4×4 resolution) ──────────
class MinibatchStd(nn.Module):
    def forward(self, x):
        b, _, h, w = x.size()
        std = torch.std(x, dim=0, keepdim=True) + 1e-8
        mean_std = std.mean().view(1, 1, 1, 1).expand(b, 1, h, w)
        return torch.cat([x, mean_std], 1)

# ───────────────────────── Generator ──────────────────────
class Generator(nn.Module):
    def __init__(self,
                 img_size: tuple[int, int, int] = (3, 32, 32),
                 z_dim: int = 128,
                 fmap: int = 128,
                 **_):
        super().__init__()
        self.z_dim = z_dim
        c, h, _ = img_size
        assert h == 32, "This architecture expects 32×32 resolution"
        self.mapping = Mapping(z_dim)
        self.const   = nn.Parameter(torch.randn(1, fmap * 16, 4, 4))

        self.blocks = nn.ModuleList([
            GBlock(fmap*16, fmap*8),   # 8×8
            GBlock(fmap*8,  fmap*4),   # 16×16
            GBlock(fmap*4,  fmap*2),   # 32×32
        ])
        self.torgb = nn.ModuleList([
            ModConv(fmap*8, c, 1, demod=False),
            ModConv(fmap*4, c, 1, demod=False),
            ModConv(fmap*2, c, 1, demod=False),
        ])
        self.tanh = nn.Tanh()
        self.apply(he)

    def forward(self, z):
        w, x = self.mapping(z), self.const.expand(z.size(0), -1, -1, -1)
        rgb = None
        for blk, tor in zip(self.blocks, self.torgb):
            x   = blk(x, w)
            rgb = tor(x, w) if rgb is None else \
                  F.interpolate(rgb, scale_factor=2, mode="nearest") + tor(x, w)
        return self.tanh(rgb)

# ──────────────────────── Discriminator ───────────────────
class Discriminator(nn.Module):
    def __init__(self,
                 img_size: tuple[int, int, int] = (3, 32, 32),
                 fmap: int = 128,
                 is_critic: bool = False,
                 **_):
        super().__init__()
        self.is_critic = bool(is_critic)
        c, h, _ = img_size
        assert h == 32
        blocks = [spectral_norm(nn.Conv2d(c, fmap, 3, 1, 1))]
        blocks += [
            DBlock(fmap,    fmap*2),   # 16×16
            DBlock(fmap*2,  fmap*4),   # 8×8
            DBlock(fmap*4,  fmap*8),   # 4×4
            MinibatchStd()
        ]
        self.blocks = nn.Sequential(*blocks)
        self.final_conv  = spectral_norm(nn.Conv2d(fmap*8 + 1, fmap*8, 3, 1, 1))
        self.final_dense = spectral_norm(nn.Linear(fmap*8 * 4 * 4, 1))
        self.act = nn.LeakyReLU(.2)
        self.out_act = nn.Identity() if self.is_critic else nn.Sigmoid()
        self.apply(he)

    def forward(self, x):
        x = self.blocks(x)
        x = self.act(self.final_conv(x))
        return self.out_act(self.final_dense(x.flatten(1))).view(-1)

# ────────────────────────── builders ──────────────────────
def build_cifar10_g(z_dim: int = 128, base_ch: int = 128):
    return Generator((3, 32, 32), z_dim=z_dim, fmap=base_ch)

def build_cifar10_d(base_ch: int = 128):
    return Discriminator((3, 32, 32), fmap=base_ch)
