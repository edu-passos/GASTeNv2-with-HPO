from __future__ import annotations

import argparse
import os
import json
import hashlib
from contextlib import contextmanager

import torch
from dotenv import load_dotenv
import wandb
from torch.optim import Adam

from ConfigSpace import ConfigurationSpace, Float, Integer
from smac import Scenario, HyperparameterOptimizationFacade

from src.utils.config import read_config
from src.data_loaders import load_dataset
from src.metrics import fid
from src.utils import MetricsLogger, group_images, load_z, seed_worker, setup_reprod, create_checkpoint_path
from src.utils.ema import maybe_make_ema
from src.utils.amp import parse_amp
from src.utils.perf import apply_perf_settings, maybe_compile, unwrap_compiled
from src.gan import construct_gan, construct_loss
from src.gan.update_g import UpdateGeneratorGAN
from src.gan.train import train_disc, train_gen, evaluate
from src.utils.checkpoint import checkpoint_gan


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True, help="YAML experiment config")
    p.add_argument("--trials", type=int, default=40, help="SMAC trials")
    p.add_argument("--walltime", type=int, default=20_000, help="SMAC wall-time limit (s)")
    return p.parse_args()


def _hash_cfg(d: dict) -> str:
    blob = json.dumps(d, sort_keys=True).encode("utf-8")
    return hashlib.md5(blob).hexdigest()[:10]


@contextmanager
def _nullctx(x):
    yield x


def _load_fixed_noise(cfg: dict, device: torch.device) -> torch.Tensor:
    fixed = cfg["fixed-noise"]
    z_dim = cfg["model"]["z_dim"]

    if isinstance(fixed, str):
        np = __import__("numpy")
        arr = np.load(fixed)
        t = torch.tensor(arr, dtype=torch.float32, device=device)
        if t.ndim != 2 or t.size(1) != z_dim:
            raise ValueError(f"fixed-noise {fixed} has shape {tuple(t.shape)}, expected (N, {z_dim})")
        return t

    n = int(fixed)
    return torch.randn(n, z_dim, device=device)


def main() -> None:
    load_dotenv()
    args = parse_args()
    cfg = read_config(args.config)

    out_dir = cfg.get("out-dir", "out")
    os.makedirs(out_dir, exist_ok=True)

    device = torch.device(cfg["device"])
    ds_cfg = cfg["dataset"]
    dataset_name = ds_cfg["name"]
    pos_cls, neg_cls = ds_cfg["binary"]["pos"], ds_cfg["binary"]["neg"]

    dataset, _, img_size = load_dataset(dataset_name, cfg["data-dir"], pos_cls, neg_cls)
    cfg["model"]["image-size"] = list(img_size)

    batch_size = cfg["train"]["step-1"]["batch-size"]
    n_disc_iters = cfg["train"]["step-1"]["disc-iters"]
    epochs = int(cfg["train"]["step-1"].get("epochs", 30))

    # Patience-based early stop on FID plateau within a trial.
    # Set to 0 / None / negative to disable.
    es_cfg = cfg["train"]["step-1"].get("early-stop", {}) or {}
    es_patience = int(es_cfg.get("patience", 0) or 0)
    es_min_epochs = int(es_cfg.get("min-epochs", 5) or 5)

    # EMA on the generator (off by default; configured per cfg.train.step-1.ema)
    ema_cfg = cfg["train"]["step-1"].get("ema", None)

    # AMP for G/D forward; off by default. cfg.train.step-1.amp = 'bf16' | 'fp16'
    amp_dtype, scaler = parse_amp(cfg["train"]["step-1"].get("amp", None))

    # Top-level perf knobs (cudnn.benchmark / determinism / TF32). Per-step
    # compile mode is handled inside the trial loop.
    apply_perf_settings(cfg.get("train", {}).get("perf", None))
    compile_g_mode = cfg["train"]["step-1"].get("compile", False)
    # Compiling D is opt-in: hinge-r1 and wgan-gp do double backward through
    # D for the gradient penalty, which can recompile/fall back under dynamo.
    compile_d_mode = cfg["train"]["step-1"].get("compile_d", False)

    # FID
    fm, dims = fid.get_inception_feature_map_fn(device)
    mu, sigma = fid.load_statistics_from_path(cfg["fid-stats-path"])
    test_noise, _ = load_z(cfg["test-noise"])
    fid_metric = fid.FID(fm, dims, test_noise.size(0), mu, sigma, device=device)
    fid_metrics = {"fid": fid_metric}

    fixed_noise = _load_fixed_noise(cfg, device=device)

    nw = int(cfg["num-workers"])
    dl = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=nw,
        worker_init_fn=seed_worker,
        pin_memory=(device.type == "cuda"),
        persistent_workers=nw > 0,
    )

    run_id = wandb.util.generate_id()
    cp_root = create_checkpoint_path(cfg, run_id)  # your existing helper
    best_score = float("inf")
    best_ckpt_dir = None

    wandb.init(
        project=cfg["project"],
        name=f"{run_id}-step1",
        group=cfg["name"],
        entity=os.environ.get("ENTITY", None),
        job_type="step-1",
        config={"id": run_id, "gan": cfg["model"], "train": cfg["train"]["step-1"]},
    )

    def objective(params, seed: int) -> float:
        nonlocal best_score, best_ckpt_dir

        setup_reprod(seed)

        arch = cfg["model"]["architecture"]
        arch["g_num_blocks"] = arch["d_num_blocks"] = int(params["n_blocks"])

        G, D = construct_gan(cfg["model"], img_size, device)
        # EMA must reference the *raw* module so parameter names are stable
        # across torch.compile (compile wraps params under _orig_mod.).
        g_ema = maybe_make_ema(ema_cfg, G)

        # Now optionally compile for speed. d_crit closes over D — pass the
        # compiled instance so its forwards take the optimised path.
        G = maybe_compile(G, compile_g_mode)
        D = maybe_compile(D, compile_d_mode)
        g_crit, d_crit = construct_loss(cfg["model"]["loss"], D)

        g_opt = Adam(G.parameters(), lr=float(params["g_lr"]), betas=(float(params["g_beta1"]), float(params["g_beta2"])))
        d_opt = Adam(D.parameters(), lr=float(params["d_lr"]), betas=(float(params["d_beta1"]), float(params["d_beta2"])))

        g_up = UpdateGeneratorGAN(g_crit)
        g_raw = unwrap_compiled(G)

        tr_log, ev_log = MetricsLogger("train"), MetricsLogger("eval")
        tr_log.add("G_loss", True)
        tr_log.add("D_loss", True)
        for t in g_up.get_loss_terms() + d_crit.get_loss_terms():
            tr_log.add(t, True)
        for m in fid_metrics:
            ev_log.add(m)
        ev_log.add_media_metric("samples")

        iters_per_epoch = (len(dl) // n_disc_iters) * n_disc_iters

        # Per-trial output dirs
        p_dict = dict(params)
        key = _hash_cfg(p_dict)
        cfg_dir = os.path.join(cp_root, f"cfg_{key}")
        os.makedirs(cfg_dir, exist_ok=True)
        with open(os.path.join(cfg_dir, "params.json"), "w") as f:
            json.dump(p_dict, f, indent=2, sort_keys=True)
        seed_dir = os.path.join(cfg_dir, f"seed_{seed}")
        os.makedirs(seed_dir, exist_ok=True)

        trial_best_fid = float("inf")
        trial_best_epoch = -1
        trial_best_path = None
        epochs_since_improve = 0

        for ep in range(1, epochs + 1):
            it_dl = iter(dl)

            for i in range(1, iters_per_epoch + 1):
                real, _ = next(it_dl)
                train_disc(G, D, d_opt, d_crit, real, batch_size, tr_log, device,
                           amp_dtype=amp_dtype, scaler=scaler)
                if i % n_disc_iters == 0:
                    train_gen(g_up, G, D, g_opt, batch_size, tr_log, device,
                              amp_dtype=amp_dtype, scaler=scaler)
                    if g_ema is not None:
                        g_ema.update(g_raw)

            # Evaluate on EMA weights when available — gives a much smoother
            # FID signal across epochs. Fall through to raw G otherwise.
            ema_ctx = g_ema.swapped(g_raw) if g_ema is not None else _nullctx(G)
            with ema_ctx:
                with torch.no_grad():
                    G.eval()
                    fake = G(fixed_noise).cpu()
                    G.train()
                ev_log.log_image("samples", group_images(fake, None, device))

                tr_log.finalize_epoch()

                # If your dataset is grayscale but FID inception expects RGB, set rgb_repeat=True.
                # For chest-xray here (RGB), rgb_repeat=False is correct.
                evaluate(G, fid_metrics, ev_log, batch_size, test_noise, device, c_out_hist=None, rgb_repeat=False)
                ev_log.finalize_epoch()

            ep_fid = float(ev_log.stats["fid"][-1])

            if ep_fid < trial_best_fid:
                trial_best_fid = ep_fid
                trial_best_epoch = ep
                epochs_since_improve = 0

                train_state = {
                    "epoch": ep,
                    "best_epoch": ep,
                    "best_fid": trial_best_fid,
                    "seed": seed,
                }
                stats = {"eval": ev_log.stats, "train": tr_log.stats}
                trial_best_path = checkpoint_gan(
                    G, D, g_opt, d_opt,
                    state=train_state,
                    stats=stats,
                    config=cfg,
                    output_dir=seed_dir,
                    epoch=ep,
                    g_ema=g_ema,
                )
            else:
                epochs_since_improve += 1
                if (
                    es_patience > 0
                    and ep >= es_min_epochs
                    and epochs_since_improve >= es_patience
                ):
                    print(
                        f"[step1] early-stop trial after epoch {ep}: "
                        f"no FID improvement for {epochs_since_improve} evals "
                        f"(best={trial_best_fid:.4f} @ epoch {trial_best_epoch})"
                    )
                    break

        # Fallback: if every epoch produced inf/NaN FID, save current weights so something is loadable.
        if trial_best_path is None:
            train_state = {
                "epoch": epochs,
                "best_epoch": -1,
                "best_fid": trial_best_fid,
                "seed": seed,
            }
            stats = {"eval": ev_log.stats, "train": tr_log.stats}
            trial_best_path = checkpoint_gan(
                G, D, g_opt, d_opt,
                state=train_state,
                stats=stats,
                config=cfg,
                output_dir=seed_dir,
                epoch=epochs,
                g_ema=g_ema,
            )

        if trial_best_fid < best_score:
            best_score = trial_best_fid
            best_ckpt_dir = trial_best_path

        return trial_best_fid

    hp_space = ConfigurationSpace()
    hp_space.add_hyperparameters(
        [
            Float("g_lr", (1e-5, 1e-3), default=2e-4, log=True),
            Float("d_lr", (1e-5, 1e-3), default=2e-4, log=True),
            Float("g_beta1", (0.0, 0.9), default=0.5),
            Float("d_beta1", (0.0, 0.9), default=0.5),
            Float("g_beta2", (0.9, 0.9999), default=0.999),
            Float("d_beta2", (0.9, 0.9999), default=0.999),
            Integer("n_blocks", (4, 5), default=4),
        ]
    )

    scenario = Scenario(hp_space, deterministic=True, n_trials=args.trials, walltime_limit=args.walltime)
    smac = HyperparameterOptimizationFacade(scenario, objective, overwrite=True)
    incumbent = smac.optimize()

    pointer_path = os.path.join(out_dir, f"step1-best-{dataset_name}-{pos_cls}v{neg_cls}.txt")
    with open(pointer_path, "w") as f:
        f.write(best_ckpt_dir if best_ckpt_dir is not None else "")

    print("Best SMAC incumbent:", dict(incumbent))
    print("Saved best checkpoint dir →", best_ckpt_dir)
    print("Saved pointer file      →", pointer_path)

    wandb.finish()


if __name__ == "__main__":
    main()
