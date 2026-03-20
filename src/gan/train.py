from __future__ import annotations
import math
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt

from src.utils import (
    MetricsLogger,
    seed_worker,
    group_images,
)
from src.utils.checkpoint import checkpoint_gan, checkpoint_image


# ────────────────────────────────────────────────────────────────
# helpers
# ────────────────────────────────────────────────────────────────
def loss_terms_to_str(loss_items: dict[str, float]) -> str:
    return " ".join(f"{k}: {v:.4f}" for k, v in loss_items.items())


# ────────────────────────────────────────────────────────────────
# evaluation on a fixed Z-set
# ────────────────────────────────────────────────────────────────
def evaluate(
    G,
    fid_metrics: dict[str, object],
    stats_logger: MetricsLogger,
    batch_size: int,
    test_noise: torch.Tensor,
    device: torch.device,
    c_out_hist=None,
    *,
    rgb_repeat: bool = False,
) -> None:
    was_training = G.training
    G.eval()

    n_batches = math.ceil(test_noise.size(0) / batch_size)
    start = 0

    for _ in tqdm(range(n_batches), desc="Evaluating", leave=False):
        real_sz = min(batch_size, test_noise.size(0) - start)
        z_batch = test_noise[start : start + real_sz]

        with torch.no_grad():
            gen_batch = G(z_batch.to(device))
        # quick collapse diagnostics
        print("gen_batch std:", gen_batch.float().std().item(), "min/max:", gen_batch.min().item(), gen_batch.max().item())


        if rgb_repeat and gen_batch.shape[1] == 1:
            gen_batch = gen_batch.repeat_interleave(3, dim=1)

        for metric in fid_metrics.values():
            metric.update(gen_batch, (start, real_sz))

        if c_out_hist is not None:
            c_out_hist.update(gen_batch, (start, real_sz))

        start += real_sz

    for name, metric in fid_metrics.items():
        value = metric.finalize()
        stats_logger.update_epoch_metric(name, value, prnt=True)
        metric.reset()

    if c_out_hist is not None:
        c_out_hist.plot()
        c_out_hist.reset()
        plt.clf()

    if was_training:
        G.train()


# ────────────────────────────────────────────────────────────────
# 1-step of discriminator update
# ────────────────────────────────────────────────────────────────
def train_disc(
    G,
    D,
    d_opt,
    d_crit,
    real_data: torch.Tensor,
    batch_size: int,
    train_metrics: MetricsLogger,
    device: torch.device,
):
    D.zero_grad()

    # real pass
    real_data = real_data.to(device)
    d_real = D(real_data)

    # fake pass
    noise = torch.randn(batch_size, G.z_dim, device=device)
    with torch.no_grad():
        fake_data = G(noise)
    d_fake = D(fake_data.detach())

    # loss + update
    d_loss, d_terms = d_crit(real_data, fake_data, d_real, d_fake, device)
    d_loss.backward()
    d_opt.step()

    for k, v in d_terms.items():
        train_metrics.update_it_metric(k, v)
    train_metrics.update_it_metric("D_loss", d_loss.item())

    return d_loss, d_terms


# ────────────────────────────────────────────────────────────────
# 1-step of generator update
# ────────────────────────────────────────────────────────────────
def train_gen(
    update_fn,
    G,
    D,
    g_opt,
    batch_size: int,
    train_metrics: MetricsLogger,
    device: torch.device,
):
    noise = torch.randn(batch_size, G.z_dim, device=device)
    g_loss, g_terms = update_fn(G, D, g_opt, noise, device)

    for k, v in g_terms.items():
        train_metrics.update_it_metric(k, v)
    train_metrics.update_it_metric("G_loss", g_loss.item())

    return g_loss, g_terms


# ────────────────────────────────────────────────────────────────
# full training loop (step-1 / vanilla GAN)
# ────────────────────────────────────────────────────────────────
def train(
    config: dict,
    dataset,
    device: torch.device,
    n_epochs: int,
    batch_size: int,
    G,
    g_opt,
    g_updater,
    D,
    d_opt,
    d_crit,
    test_noise: torch.Tensor,
    fid_metrics: dict[str, object],
    n_disc_iters: int,
    *,
    early_stop: tuple[str, int] | None = None,
    start_early_stop_when: tuple[str, int] | None = None,
    checkpoint_dir: str | None = None,
    checkpoint_every: int = 10,
    fixed_noise: torch.Tensor | None = None,
    c_out_hist=None,
    classifier=None,
):
    # fixed Z for visuals
    fixed_noise = (
        torch.randn(64, G.z_dim, device=device)
        if fixed_noise is None
        else fixed_noise
    )

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=config["num-workers"],
        worker_init_fn=seed_worker,
    )

    tr_log, ev_log = MetricsLogger("train"), MetricsLogger("eval")
    tr_log.add("G_loss", True)
    tr_log.add("D_loss", True)
    for t in g_updater.get_loss_terms() + d_crit.get_loss_terms():
        tr_log.add(t, True)
    for m in fid_metrics:
        ev_log.add(m)
    ev_log.add_media_metric("samples")
    ev_log.add_media_metric("histogram")

    # -------- initial snapshot ------------------------------------
    with torch.no_grad():
        G.eval()
        first_fake = G(fixed_noise).cpu()
        G.train()
    img0 = group_images(first_fake, classifier, device)
    if checkpoint_dir is not None:
        checkpoint_image(img0, 0, checkpoint_dir)
        checkpoint_gan(
            G, D, g_opt, d_opt,
            state={},
            stats={},
            config=config,
            output_dir=checkpoint_dir,
            epoch=0,
        )

    # -------- training loop ---------------------------------------
    iters_per_epoch = (len(loader) // n_disc_iters) * n_disc_iters
    log_every_g = 50

    for epoch in range(1, n_epochs + 1):
        it_loader = iter(loader)
        g_it = 0

        for i in range(1, iters_per_epoch + 1):
            real, _ = next(it_loader)
            train_disc(G, D, d_opt, d_crit, real, batch_size, tr_log, device)

            if i % n_disc_iters == 0:
                g_it += 1
                train_gen(g_updater, G, D, g_opt, batch_size, tr_log, device)

                if g_it % log_every_g == 0 or g_it == iters_per_epoch // n_disc_iters:
                    try:
                        g_last = tr_log.last('G_loss')
                        d_last = tr_log.last('D_loss')
                        print(
                            f"[{epoch}/{n_epochs}] g_it {g_it}/{iters_per_epoch//n_disc_iters} "
                            f"G {g_last:.3f} | D {d_last:.3f}"
                        )
                    except RuntimeError:
                        pass

        # ---------- epoch end: images & metrics -------------------
        with torch.no_grad():
            G.eval()
            fake = G(fixed_noise).cpu()
            G.train()
        img = group_images(fake, classifier, device)
        ev_log.log_image("samples", img)
        if checkpoint_dir is not None:
            checkpoint_image(img, epoch, checkpoint_dir)

        tr_log.finalize_epoch()
        evaluate(G, fid_metrics, ev_log, batch_size, test_noise, device, c_out_hist)
        ev_log.finalize_epoch()

        if checkpoint_dir and (epoch % checkpoint_every == 0 or epoch == n_epochs):
            checkpoint_gan(
                G, D, g_opt, d_opt,
                state=None,
                stats={"eval": ev_log.stats, "train": tr_log.stats},
                config=config,
                output_dir=checkpoint_dir,
                epoch=epoch,
            )


    metrics = {"train": tr_log.stats, "eval": ev_log.stats}
    optim_state = {"g_opt": g_opt.state_dict(), "d_opt": d_opt.state_dict()}
    return G, D, optim_state, metrics


# ======================================================================
#  Back-compatibility re-exports
# ======================================================================

__all__ = ["train_disc", "train_gen", "evaluate"]

globals().update({name: globals()[name] for name in __all__})
