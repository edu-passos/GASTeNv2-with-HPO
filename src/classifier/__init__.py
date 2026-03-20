from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
import timm


# ────────────────────────────────────────────────────────────────────
# helpers
# ────────────────────────────────────────────────────────────────────
class _ResizeForward(nn.Module):
    def __init__(self, backbone: nn.Module, tgt_sz: int):
        super().__init__()
        self.backbone = backbone
        self.tgt_sz = tgt_sz
        self.resize = nn.Upsample(size=tgt_sz, mode="bilinear", align_corners=False)

    def forward(self, x: torch.Tensor):
        if x.shape[-1] != self.tgt_sz:
            x = self.resize(x)
        return self.backbone(x)


# ────────────────────────────────────────────────────────────────────
# simple CNN (for README-compatible `--classifier-type cnn`)
# ────────────────────────────────────────────────────────────────────
class SimpleCNN(nn.Module):
    """
    Small, robust CNN for binary/multiclass classification.

    - Works for any HxW >= 32 (uses adaptive pooling)
    - If output_feature_maps=True: returns (logits, feature_map, pooled_feature)
      to support your gaussianV2 / hubris / hist pipelines.
    """
    def __init__(self, in_ch: int, n_classes: int, nf: int = 32):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(in_ch, nf, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            nn.Conv2d(nf, nf * 2, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            nn.Conv2d(nf * 2, nf * 4, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            nn.Conv2d(nf * 4, nf * 4, 3, padding=1),
            nn.ReLU(inplace=True),
        )
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Linear(nf * 4, n_classes)

    def forward(self, x: torch.Tensor, output_feature_maps: bool = False):
        fmap = self.features(x)              # [B, C, h, w]
        pooled = self.pool(fmap).flatten(1)  # [B, C]
        logits = self.classifier(pooled)     # [B, n_classes]
        if output_feature_maps:
            # match your downstream expectations:
            #   output[0] = logits
            #   output[-2] used as "feature_map" in some code
            return (logits, fmap, pooled)
        return logits


# ────────────────────────────────────────────────────────────────────
# pooled MLP
# ────────────────────────────────────────────────────────────────────
class PooledMLP(nn.Module):
    def __init__(self, in_ch: int, n_classes: int, pool_sz: int = 32, hidden_dim: int = 512):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d((pool_sz, pool_sz))
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_ch * pool_sz * pool_sz, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim // 2, n_classes),
        )

    def forward(self, x: torch.Tensor):
        x = self.pool(x)
        return self.net(x)


# ────────────────────────────────────────────────────────────────────
# MiniViT (your existing one)
# ────────────────────────────────────────────────────────────────────
class MiniViT(nn.Module):
    """Minimal ViT: square images, divisible by patch_size."""
    def __init__(
        self,
        img_size: tuple[int, int, int],
        patch_size: int,
        dim: int,
        depth: int,
        heads: int,
        mlp_dim: int,
        num_classes: int
    ):
        super().__init__()
        C, H, W = img_size
        assert H == W and H % patch_size == 0, "Non-square or not divisible → fallback"
        n_patches = (H // patch_size) ** 2

        self.to_patches = nn.Conv2d(C, dim, patch_size, patch_size)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, dim))
        self.pos_emb = nn.Parameter(torch.randn(1, n_patches + 1, dim))

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=dim, nhead=heads, dim_feedforward=mlp_dim,
            activation="gelu", batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=depth)
        self.to_logits = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, num_classes))

    def forward(self, x: torch.Tensor):
        b = x.size(0)
        x = self.to_patches(x)
        x = x.flatten(2).transpose(1, 2)
        cls = self.cls_token.expand(b, -1, -1)
        x = torch.cat((cls, x), dim=1) + self.pos_emb
        x = self.transformer(x)
        return self.to_logits(x[:, 0])


# ────────────────────────────────────────────────────────────────────
# factory
# ────────────────────────────────────────────────────────────────────
def construct_classifier(params: dict, device: str | torch.device | None = None) -> nn.Module:
    mtype = params["type"].lower()
    n_cls = int(params["n_classes"])
    C, H, W = params.get("img_size", (3, 28, 28))
    hidden_dim = params.get("hidden_dim", params.get("nf", 512))

    # CNN branch
    if mtype in {"cnn", "simple_cnn"}:
        nf = int(params.get("nf", 32))
        model = SimpleCNN(in_ch=C, n_classes=n_cls, nf=nf)
        return model.to(device) if device else model

    # MLP branch
    if mtype in {"mlp", "simple_mlp", "pooledmlp"}:
        model = PooledMLP(C, n_cls, pool_sz=int(params.get("pool_sz", 32)), hidden_dim=int(hidden_dim))
        return model.to(device) if device else model

    # MiniViT branch
    if mtype.startswith("minivit"):
        patch_size = int(params.get("patch_size", 8))
        dim = int(params.get("vit_dim", 128))
        depth = int(params.get("vit_depth", 4))
        heads = int(params.get("vit_heads", 4))
        mlp_dim = int(params.get("vit_mlp_dim", dim * 2))
        try:
            model = MiniViT((C, H, W), patch_size, dim, depth, heads, mlp_dim, n_cls)
        except AssertionError:
            model = PooledMLP(C, n_cls, pool_sz=int(params.get("pool_sz", 32)), hidden_dim=int(hidden_dim))
        return model.to(device) if device else model

    # TIMM branch
    freeze = mtype.startswith("frozen_")
    if freeze:
        mtype = mtype[len("frozen_"):]

    if mtype not in timm.list_models():
        raise ValueError(f"Unknown TIMM model '{mtype}'")

    is_patch_model = any(tok in mtype for tok in ("vit", "deit", "beit", "swin", "mixer"))
    timm_kwargs = dict(pretrained=True, in_chans=C, num_classes=n_cls)

    # if patch model + non-224 size: avoid pretrained positional embedding mismatch
    if is_patch_model and H != 224 and H % 16 == 0:
        timm_kwargs.update(img_size=H, pretrained=False)

    backbone = timm.create_model(mtype, **timm_kwargs)

    if freeze:
        for p in backbone.parameters():
            p.requires_grad = False
        # timm exposes classifier head differently across models; this is common enough:
        try:
            head = backbone.get_classifier()
            for p in head.parameters():
                p.requires_grad = True
        except Exception:
            pass

    tgt = 224 if is_patch_model else H
    model = _ResizeForward(backbone, tgt) if H != tgt else backbone
    return model.to(device) if device else model
