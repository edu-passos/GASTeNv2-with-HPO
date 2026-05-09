"""GPU smoke test for the Phase 1 + 2 + 3 changes.

Bypasses SMAC (not installed locally) and exercises the actual training
primitives: construct_gan, train_disc, train_gen with bf16 AMP, EMA on the
raw generator, torch.compile on G, and checkpoint round-trip — including a
hinge-r1 path to make sure the fp32 R1 island works under autocast.

Run::

    python scripts/gpu_smoke.py
"""
from __future__ import annotations

import os
import sys
import shutil
import tempfile

import torch
import torch.nn as nn
from torch.optim import Adam

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
if REPO not in sys.path:
    sys.path.insert(0, REPO)

from src.gan import construct_gan, construct_loss
from src.gan.architectures.chest_xray import Generator as CXR_G, Discriminator as CXR_D
from src.gan.train import train_disc, train_gen
from src.gan.update_g import UpdateGeneratorGAN
from src.utils.ema import EMA, maybe_make_ema
from src.utils.amp import parse_amp
from src.utils.perf import apply_perf_settings, maybe_compile, unwrap_compiled
from src.utils.metrics_logger import MetricsLogger
from src.utils.checkpoint import checkpoint_gan, construct_gan_from_checkpoint
from src.utils.diffaug import DiffAugmenter


def _model_cfg(loss_name: str) -> dict:
    return {
        "z_dim": 64,
        "architecture": {
            "name": "dcgan",
            "g_filter_dim": 32,
            "d_filter_dim": 32,
            "g_num_blocks": 3,
            "d_num_blocks": 3,
        },
        "loss": {"name": loss_name, "args": {"lambda": 10.0}},
    }


def _make_logger(updater, d_crit) -> tuple[MetricsLogger, MetricsLogger]:
    tr = MetricsLogger("train")
    ev = MetricsLogger("eval")
    tr.add("G_loss", True)
    tr.add("D_loss", True)
    for t in updater.get_loss_terms() + d_crit.get_loss_terms():
        tr.add(t, True)
    return tr, ev


def _step(loss_name: str, *, amp_mode: str, compile_g: bool, device: torch.device) -> None:
    print(f"\n=== {loss_name}  amp={amp_mode}  compile_g={compile_g} ===")
    img_size = (3, 32, 32)
    cfg = _model_cfg(loss_name)
    G, D = construct_gan(cfg, img_size, device)

    # Build EMA from the raw module first, then compile.
    g_ema = EMA(G, decay=0.9)  # exaggerated decay so a few steps move it visibly
    G = maybe_compile(G, "default" if compile_g else False)
    g_raw = unwrap_compiled(G)

    g_crit, d_crit = construct_loss(cfg["loss"], D)
    g_up = UpdateGeneratorGAN(g_crit)
    g_opt = Adam(G.parameters(), lr=2e-4, betas=(0.5, 0.999))
    d_opt = Adam(D.parameters(), lr=2e-4, betas=(0.5, 0.999))

    amp_dtype, scaler = parse_amp(amp_mode)
    tr_log, _ = _make_logger(g_up, d_crit)

    # Snapshot one EMA shadow param so we can verify update() actually moved it.
    sample_name = next(iter(g_ema.shadow_params.keys()))
    before = g_ema.shadow_params[sample_name].clone()

    bs = 4
    fake_real = torch.randn(bs, *img_size, device=device)

    losses = []
    for step in range(3):
        train_disc(G, D, d_opt, d_crit, fake_real, bs, tr_log, device,
                   amp_dtype=amp_dtype, scaler=scaler)
        g_loss, _ = train_gen(g_up, G, D, g_opt, bs, tr_log, device,
                              amp_dtype=amp_dtype, scaler=scaler)
        g_ema.update(g_raw)
        losses.append(float(g_loss))

    after = g_ema.shadow_params[sample_name]
    assert torch.isfinite(after).all(), "EMA shadow has non-finite values"
    assert not torch.allclose(before, after), "EMA shadow did not change after update()"
    for L in losses:
        assert L == L, f"NaN G loss at step (loss={L})"

    # EMA swap actually swaps params, then restores.
    cur = {n: p.detach().clone() for n, p in g_raw.named_parameters()}
    with g_ema.swapped(g_raw):
        for n, p in g_raw.named_parameters():
            if n in g_ema.shadow_params:
                assert torch.allclose(p, g_ema.shadow_params[n].to(p.device, dtype=p.dtype)), n
    for n, p in g_raw.named_parameters():
        assert torch.allclose(p, cur[n]), f"swap did not restore {n}"

    print(f"  losses: {[round(L, 4) for L in losses]}  ok")


def _checkpoint_roundtrip(device: torch.device) -> None:
    print("\n=== checkpoint roundtrip with EMA + compile ===")
    img_size = (3, 32, 32)
    cfg_model = _model_cfg("ns")
    G, D = construct_gan(cfg_model, img_size, device)
    g_ema = EMA(G, decay=0.9)
    G = maybe_compile(G, "default")
    g_raw = unwrap_compiled(G)

    g_crit, d_crit = construct_loss(cfg_model["loss"], D)
    g_up = UpdateGeneratorGAN(g_crit)
    g_opt = Adam(G.parameters(), lr=2e-4, betas=(0.5, 0.999))
    d_opt = Adam(D.parameters(), lr=2e-4, betas=(0.5, 0.999))

    tr_log, _ = _make_logger(g_up, d_crit)
    bs = 4
    real = torch.randn(bs, *img_size, device=device)
    for _ in range(2):
        train_disc(G, D, d_opt, d_crit, real, bs, tr_log, device, amp_dtype=torch.bfloat16)
        train_gen(g_up, G, D, g_opt, bs, tr_log, device, amp_dtype=torch.bfloat16)
        g_ema.update(g_raw)

    # checkpoint_gan must save *raw* state dicts (no _orig_mod prefix).
    # Pretend this is a step-1 dump including EMA shadow.
    full_cfg = {
        "model": {**cfg_model, "image-size": list(img_size)},
        "optimizer": {"lr": 2e-4, "beta1": 0.5, "beta2": 0.999},
    }
    tmp = tempfile.mkdtemp(prefix="gpu_smoke_ckpt_")
    try:
        ckpt_path = checkpoint_gan(
            G, D, g_opt, d_opt,
            state={"epoch": 2, "best_epoch": 2, "best_fid": 1.23, "seed": 0},
            stats={},
            config=full_cfg,
            output_dir=tmp,
            epoch=2,
            g_ema=g_ema,
        )

        # Saved keys must be the *raw* names.
        gen_blob = torch.load(os.path.join(ckpt_path, "generator.pth"), map_location="cpu", weights_only=False)
        bad = [k for k in gen_blob["state"] if k.startswith("_orig_mod.")]
        assert not bad, f"saved generator keys leaked _orig_mod prefix: {bad[:5]}"

        # construct_gan_from_checkpoint should prefer EMA weights.
        G2, D2, _, _ = construct_gan_from_checkpoint(ckpt_path, device=device, prefer_ema=True)

        # Compare loaded G2 params against EMA shadow — they should match.
        for n, p in G2.named_parameters():
            if n in g_ema.shadow_params:
                want = g_ema.shadow_params[n].to(p.device, dtype=p.dtype)
                assert torch.allclose(p, want, atol=1e-5), f"EMA-loaded param mismatch: {n}"
        print("  EMA-prefer load matches saved shadow  ok")

        # And without EMA preference, raw G state should load.
        G3, _, _, _ = construct_gan_from_checkpoint(ckpt_path, device=device, prefer_ema=False)
        raw_state = gen_blob["state"]
        for n, p in G3.named_parameters():
            assert torch.allclose(p, raw_state[n].to(p.device, dtype=p.dtype), atol=1e-5), n
        print("  raw load matches generator.pth  ok")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def main() -> None:
    if not torch.cuda.is_available():
        print("CUDA not available; skipping GPU smoke.")
        return
    device = torch.device("cuda")
    print(f"device: {torch.cuda.get_device_name(0)}  torch={torch.__version__}")

    # MetricsLogger calls wandb.define_metric on add(); silence wandb so it
    # doesn't try to talk to the network.
    import wandb
    wandb.init(mode="disabled")

    apply_perf_settings({
        "cudnn_benchmark": True,
        "cudnn_deterministic": False,
        "tf32_matmul": True,
    })

    # ns: simplest path; verify AMP off and on, compile off and on.
    _step("ns", amp_mode="off",  compile_g=False, device=device)
    _step("ns", amp_mode="bf16", compile_g=False, device=device)
    _step("ns", amp_mode="bf16", compile_g=True,  device=device)

    # hinge-r1: stresses the fp32 R1 island under bf16 autocast.
    _step("hinge-r1", amp_mode="bf16", compile_g=False, device=device)
    _step("hinge-r1", amp_mode="bf16", compile_g=True,  device=device)

    _checkpoint_roundtrip(device)
    _conditional_diffaug_smoke(device)
    print("\nALL SMOKE CHECKS PASSED")


def _conditional_diffaug_smoke(device: torch.device) -> None:
    print("\n=== conditional G+D + DiffAugment + bf16 + EMA + hinge-r1 ===")
    img_size = (1, 128, 128)
    G = CXR_G(img_size=img_size, z_dim=64, fmap=32, num_classes=2).to(device)
    D = CXR_D(img_size=img_size, fmap=32, num_classes=2).to(device)
    assert G.num_classes == 2 and D.num_classes == 2

    g_ema = EMA(G, decay=0.9)
    aug = DiffAugmenter("color,translation,cutout")
    cfg_loss = {"name": "hinge-r1", "args": {"lambda": 5.0}}
    g_crit, d_crit = construct_loss(cfg_loss, D)
    g_up = UpdateGeneratorGAN(g_crit)
    g_opt = Adam(G.parameters(), lr=2e-4, betas=(0.0, 0.999))
    d_opt = Adam(D.parameters(), lr=2e-4, betas=(0.0, 0.999))

    tr_log, _ = _make_logger(g_up, d_crit)

    bs = 4
    real = torch.randn(bs, *img_size, device=device)
    real_y = torch.randint(0, 2, (bs,), device=device)

    losses = []
    for _ in range(3):
        train_disc(G, D, d_opt, d_crit, real, bs, tr_log, device,
                   amp_dtype=torch.bfloat16, real_label=real_y, augmenter=aug)
        gl, _ = train_gen(g_up, G, D, g_opt, bs, tr_log, device,
                          amp_dtype=torch.bfloat16, augmenter=aug)
        g_ema.update(G)
        losses.append(float(gl))

    for L in losses:
        assert L == L, f"NaN G loss in cond+diffaug path: {L}"
    print(f"  losses: {[round(L, 4) for L in losses]}  ok")

    # boundary-mix sample (continuous y in [0,1] for binary)
    z = torch.randn(4, 64, device=device)
    y_soft = torch.linspace(0.0, 1.0, 4, device=device)
    with torch.no_grad():
        x_soft = G(z, y_soft)
    assert x_soft.shape == (4, 1, 128, 128) and torch.isfinite(x_soft).all()
    print(f"  continuous-y interpolation OK  shape={tuple(x_soft.shape)}")

    # checkpoint round-trip preserves num_classes via the saved config
    full_cfg = {
        "model": {
            "num_classes": 2,
            "z_dim": 64,
            "image-size": list(img_size),
            "architecture": {
                "name": "chest-xray",
                "g_filter_dim": 32,
                "d_filter_dim": 32,
                "g_num_blocks": 3,
                "d_num_blocks": 3,
            },
            "loss": cfg_loss,
        },
        "optimizer": {"lr": 2e-4, "beta1": 0.0, "beta2": 0.999},
    }
    tmp = tempfile.mkdtemp(prefix="gpu_smoke_cond_")
    try:
        ckpt_path = checkpoint_gan(
            G, D, g_opt, d_opt,
            state={"epoch": 1, "best_epoch": 1, "best_fid": 1.0, "seed": 0},
            stats={},
            config=full_cfg,
            output_dir=tmp,
            epoch=1,
            g_ema=g_ema,
        )
        G2, D2, _, _ = construct_gan_from_checkpoint(ckpt_path, device=device, prefer_ema=True)
        assert G2.num_classes == 2 and D2.num_classes == 2, (
            f"num_classes lost in roundtrip: G2={G2.num_classes} D2={D2.num_classes}"
        )
        print(f"  checkpoint roundtrip preserves num_classes  G2.nc={G2.num_classes}")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    main()
