import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import spectral_norm

# ─── Conditional BatchNorm ─────────────────────────────────────────────────────
class ConditionalBatchNorm2d(nn.Module):
    def __init__(self, num_features, num_classes):
        super().__init__()
        self.bn = nn.BatchNorm2d(num_features, affine=False)
        self.embed = nn.Embedding(num_classes, num_features * 2)
        self.embed.weight.data[:, :num_features].fill_(1)
        self.embed.weight.data[:, num_features:].zero_()

    def forward(self, x, y):
        gamma, beta = self.embed(y).chunk(2, dim=1)
        gamma = gamma.view(-1, x.size(1), 1, 1)
        beta  = beta .view(-1, x.size(1), 1, 1)
        return self.bn(x) * gamma + beta

# ─── Self-Attention ─────────────────────────────────────────────────────────────
class SelfAttention(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.f = spectral_norm(nn.Conv2d(channels, channels//8, 1))
        self.g = spectral_norm(nn.Conv2d(channels, channels//8, 1))
        self.h = spectral_norm(nn.Conv2d(channels, channels//2, 1))
        self.v = spectral_norm(nn.Conv2d(channels//2, channels, 1))
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        b,c,h,w = x.size()
        f = self.f(x).view(b, -1, h*w)
        g = self.g(x).view(b, -1, h*w)
        beta = F.softmax(torch.bmm(f.permute(0,2,1), g), dim=-1)
        h_   = self.h(x).view(b, -1, h*w)
        o    = torch.bmm(h_, beta).view(b, c//2, h,w)
        o    = self.v(o)
        return x + self.gamma * o

# ─── Weight Init ────────────────────────────────────────────────────────────────
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
        if getattr(m, 'bias', None) is not None:
            nn.init.constant_(m.bias, 0)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias, 0)

# ─── Generator with CBN + Self-Attention ────────────────────────────────────────
class Generator(nn.Module):
    def __init__(self, image_size, num_classes, z_dim=100, base_ch=64):
        super().__init__()
        _, H, _ = image_size
        init_spatial = H // 16
        dims = [base_ch * (2**i) for i in range(5)]
        gen_dims = list(reversed(dims))

        self.project = nn.Sequential(
            nn.ConvTranspose2d(z_dim, gen_dims[0], init_spatial, 1, 0, bias=False),
            ConditionalBatchNorm2d(gen_dims[0], num_classes),
            nn.ReLU(True),
        )

        blocks = []
        curr_h = init_spatial
        for i in range(len(gen_dims)-1):
            in_c, out_c = gen_dims[i], gen_dims[i+1]
            seq = []
            seq += [nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)]
            if curr_h == H//4:
                seq += [SelfAttention(in_c)]
            seq += [
                ConditionalBatchNorm2d(in_c, num_classes),
                nn.ReLU(True),
                nn.Conv2d(in_c, out_c, 3,1,1, bias=False),
            ]
            blocks.append(nn.Sequential(*seq))
            curr_h *= 2

        self.gen_blocks = nn.Sequential(*blocks)
        self.to_rgb = nn.Sequential(nn.Conv2d(gen_dims[-1], 3, 3,1,1), nn.Tanh())

        self.apply(weights_init)

    def forward(self, z, labels):
        x = z.view(-1, z.size(1), 1, 1)
        x = self.project[0](x)
        x = self.project[1](x, labels)
        x = self.project[2](x)
        for blk in self.gen_blocks:
            if isinstance(blk[0], SelfAttention):
                x = blk(x)
            else:
                for layer in blk:
                    if isinstance(layer, ConditionalBatchNorm2d):
                        x = layer(x, labels)
                    else:
                        x = layer(x)
        return self.to_rgb(x)

# ─── Discriminator with spectral‐norm, Self‐Attention & Projection Head ───────
class Discriminator(nn.Module):
    def __init__(self, image_size, num_classes, base_ch=64, use_bn=False, is_critic=True):
        super().__init__()
        self.is_critic = bool(is_critic)
        _, H, W = image_size
        dims = [base_ch * (2**i) for i in range(5)]
        blocks = []
        curr_h, curr_w = H, W
        in_c = 3
        for i, out_c in enumerate(dims):
            seq = []
            seq.append(spectral_norm(nn.Conv2d(in_c, out_c, 4,2,1, bias=(i==0))))
            if use_bn and i>0:
                seq.append(nn.BatchNorm2d(out_c))
            seq.append(nn.LeakyReLU(0.2, inplace=True))
            if curr_h == H//4:
                seq.append(SelfAttention(out_c))
            blocks.append(nn.Sequential(*seq))
            in_c = out_c
            curr_h //= 2; curr_w //= 2

        self.conv_blocks = nn.Sequential(*blocks)
        self.final_conv = spectral_norm(nn.Conv2d(dims[-1], dims[-1], (curr_h, curr_w), 1,0, bias=False))
        self.flatten    = nn.Flatten()

        self.embed = nn.Embedding(num_classes, dims[-1])
        self.out_act = nn.Identity() if self.is_critic else nn.Sigmoid()

        self.apply(weights_init)

    def forward(self, x, labels):
        h = x
        for blk in self.conv_blocks:
            h = blk(h)
        feats = self.final_conv(h)
        flat  = self.flatten(feats)
        logit_uncond = (flat).sum(dim=1, keepdim=True)
        emb = self.embed(labels)
        proj = torch.sum(flat * emb, dim=1, keepdim=True)
        out  = logit_uncond + proj
        return self.out_act(out).view(-1)
