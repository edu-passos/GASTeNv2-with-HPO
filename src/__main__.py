import os
import argparse
import json
import numpy as np
from dotenv import load_dotenv
import pandas as pd

import torch
import wandb

from src.metrics import fid, LossSecondTerm, Hubris
from src.data_loaders import load_dataset
from src.gan.train import train
from src.gan.update_g import (
    UpdateGeneratorGAN,
    UpdateGeneratorGASTEN_gaussian,
    UpdateGeneratorGASTEN_gaussianV2,
)
from src.metrics.c_output_hist import OutputsHistogram
from src.utils import (
    load_z,
    set_seed,
    setup_reprod,
    create_checkpoint_path,
    gen_seed,
    seed_worker,
)
from src.utils.plot import plot_metrics
from src.utils.config import read_config
from src.utils.checkpoint import (
    construct_gan_from_checkpoint,
    construct_classifier_from_checkpoint,
    get_gan_path_at_epoch,
)
from src.gan import construct_gan, construct_loss


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", dest="config_path", required=True, help="Path to config file")
    parser.add_argument("--no-plots", action="store_true", help="Disable creation of images for plots")
    return parser.parse_args()


def construct_optimizers(config, G, D):
    g_optim = torch.optim.Adam(G.parameters(), lr=config["lr"], betas=(config["beta1"], config["beta2"]))
    d_optim = torch.optim.Adam(D.parameters(), lr=config["lr"], betas=(config["beta1"], config["beta2"]))
    return g_optim, d_optim


def compute_dataset_fid_stats(dset, get_feature_map_fn, dims, batch_size=64, device="cpu", num_workers=0):
    dataloader = torch.utils.data.DataLoader(
        dset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        worker_init_fn=seed_worker,
    )
    m, s = fid.calculate_activation_statistics_dataloader(
        dataloader, get_feature_map_fn, dims=dims, device=device
    )
    return m, s


def load_gan_stats(output_dir: str) -> dict:
    """
    Load GAN stats saved by checkpoint_gan().
    Expected file: <output_dir>/stats.json
    """
    path = os.path.join(str(output_dir), "stats.json")
    if not os.path.exists(path):
        return {}
    try:
        with open(path, "r") as f:
            data = json.load(f)
        return data if isinstance(data, dict) else {}
    except (OSError, json.JSONDecodeError):
        return {}


def list_saved_step1_epochs(step1_dir: str) -> list[int]:
    """
    Return sorted epoch numbers for checkpoints available in <step1_dir>/<epoch>.
    """
    if not os.path.isdir(step1_dir):
        return []

    epochs = []
    for name in os.listdir(step1_dir):
        try:
            epoch = int(name)
        except ValueError:
            continue

        epoch_dir = os.path.join(step1_dir, name)
        if os.path.isdir(epoch_dir) and os.path.exists(os.path.join(epoch_dir, "generator.pth")):
            epochs.append(epoch)

    return sorted(set(epochs))


def resolve_step1_epoch(step1_epoch_label, step1_metrics: dict, step1_dir: str) -> int:
    """
    Resolve requested step-1 epoch label ('best'/'last'/int) to a saved checkpoint epoch.
    """
    saved_epochs = list_saved_step1_epochs(step1_dir)
    if not saved_epochs:
        raise FileNotFoundError(f"No step-1 checkpoints found under: {step1_dir}")

    eval_stats = step1_metrics.get("eval", {}) if isinstance(step1_metrics, dict) else {}
    fid_hist = eval_stats.get("fid", []) if isinstance(eval_stats, dict) else []
    fid_hist = fid_hist if isinstance(fid_hist, list) else []

    best_eval_epoch = None
    last_eval_epoch = None
    if len(fid_hist) > 0:
        best_eval_epoch = int(np.argmin(np.asarray(fid_hist))) + 1
        last_eval_epoch = len(fid_hist)

    def nearest_saved_at_or_before(target_epoch: int, reason: str) -> int:
        if target_epoch in saved_epochs:
            return target_epoch
        lower = [e for e in saved_epochs if e <= target_epoch]
        if lower:
            chosen = max(lower)
            print(
                f"Requested {reason} epoch {target_epoch} is not checkpointed; "
                f"using nearest saved epoch {chosen}."
            )
            return chosen
        chosen = saved_epochs[0]
        print(
            f"Requested {reason} epoch {target_epoch} is before first checkpoint; "
            f"using epoch {chosen}."
        )
        return chosen

    if step1_epoch_label == "best":
        if best_eval_epoch is not None:
            return nearest_saved_at_or_before(best_eval_epoch, "best")
        print("Step-1 metrics missing; using latest saved checkpoint for 'best'.")
        return saved_epochs[-1]

    if step1_epoch_label == "last":
        if last_eval_epoch is not None:
            return nearest_saved_at_or_before(last_eval_epoch, "last")
        return saved_epochs[-1]

    requested = int(step1_epoch_label)
    if requested not in saved_epochs:
        raise FileNotFoundError(
            f"Requested step-1 epoch {requested} not found in {step1_dir}. "
            f"Available epochs: {saved_epochs}"
        )
    return requested


def train_modified_gan(
    config,
    dataset,
    cp_dir,
    gan_checkpoint_path,
    test_noise,
    fid_metrics,
    c_out_hist,
    C,
    C_name,
    C_params,
    C_stats,
    C_args,
    weight,
    fixed_noise,
    device,
    seed,
    run_id,
    *,
    step1_epoch_label: str,
    step1_epoch_int: int,
):
    print(f"Running experiment with classifier {C_name} and weight {weight} ...")

    # Ensure weight is a dictionary for Gaussian loss.
    if not (isinstance(weight, dict) and ("gaussian" in weight or "gaussian-v2" in weight)):
        raise ValueError(
            "For GASTeN v2 we require weight to be specified as a dictionary with "
            "'gaussian' or 'gaussian-v2'."
        )

    if "gaussian" in weight:
        weight_txt = "gauss_" + "_".join([f"{k}_{v}" for k, v in weight["gaussian"].items()])
    else:
        weight_txt = "gauss_v2_" + "_".join([f"{k}_{v}" for k, v in weight["gaussian-v2"].items()])

    run_name = f"{C_name}_{weight_txt}_{step1_epoch_label}"
    gan_cp_dir = os.path.join(cp_dir, run_name)

    batch_size = config["train"]["step-2"]["batch-size"]
    n_epochs = config["train"]["step-2"]["epochs"]
    n_disc_iters = config["train"]["step-2"]["disc-iters"]
    checkpoint_every = config["train"]["step-2"]["checkpoint-every"]

    # Load GAN checkpoint
    G, D, _, _ = construct_gan_from_checkpoint(gan_checkpoint_path, device=device)

    # Construct losses/optimizers
    g_crit, d_crit = construct_loss(config["model"]["loss"], D)
    g_optim, d_optim = construct_optimizers(config["optimizer"], G, D)

    # Select generator updater
    if "gaussian" in weight:
        alpha = weight["gaussian"]["alpha"]
        var = weight["gaussian"]["var"]
        g_updater = UpdateGeneratorGASTEN_gaussian(g_crit, C, alpha=alpha, var=var)
    else:
        alpha = weight["gaussian-v2"]["alpha"]
        var = weight["gaussian-v2"]["var"]
        g_updater = UpdateGeneratorGASTEN_gaussianV2(g_crit, C, alpha=alpha, var=var)

    # Early stopping config
    early_stop_key = "conf_dist"
    early_stop_crit = config["train"]["step-2"].get("early-stop", {}).get("criteria", None)
    early_stop = (early_stop_key, early_stop_crit) if early_stop_crit is not None else (early_stop_key, None)

    step1_cfg = config["train"]["step-1"]
    if isinstance(step1_cfg, dict):
        early_stop_crit_step_1 = step1_cfg.get("early-stop", {}).get("criteria", None)
    else:
        early_stop_crit_step_1 = None
    start_early_stop_when = None
    if early_stop_crit_step_1 is not None:
        start_early_stop_when = ("fid", early_stop_crit_step_1)

    set_seed(seed)

    wandb.init(
        project=config["project"],
        group=config["name"],
        entity=os.environ["ENTITY"],
        job_type="step-2",
        name=f"{run_id}-{run_name}",
        config={
            "id": run_id,
            "seed": seed,
            "weight": weight_txt,
            "train": config["train"]["step-2"],
            "classifier_loss": C_stats.get("test_loss", 0.0),
            "classifier": C_name,
            "classifier_args": C_args,
            "classifier_params": C_params,
            "step1_epoch": step1_epoch_label,
            "step1_epoch_int": step1_epoch_int,
        },
    )

    _, _, _, eval_metrics = train(
        config,
        dataset,
        device,
        n_epochs,
        batch_size,
        G,
        g_optim,
        g_updater,
        D,
        d_optim,
        d_crit,
        test_noise,
        fid_metrics,
        n_disc_iters,
        early_stop=early_stop,
        start_early_stop_when=start_early_stop_when,
        checkpoint_dir=gan_cp_dir,
        fixed_noise=fixed_noise,
        c_out_hist=c_out_hist,
        checkpoint_every=checkpoint_every,
        classifier=C,
    )

    wandb.finish()
    return eval_metrics


def main():
    load_dotenv()
    args = parse_args()
    config = read_config(args.config_path)
    print(f"Loaded experiment configuration from {args.config_path}")

    run_seeds = config.get("step-1-seeds", [gen_seed() for _ in range(config["num-runs"])])
    step_2_seeds = config.get("step-2-seeds", [gen_seed() for _ in range(config["num-runs"])])

    device = torch.device(config["device"])
    print(f"Using device {device}")

    # Load dataset
    pos_class = config["dataset"].get("binary", {}).get("pos", None)
    neg_class = config["dataset"].get("binary", {}).get("neg", None)
    dataset, num_classes, img_size = load_dataset(
        config["dataset"]["name"], config["data-dir"], pos_class, neg_class
    )
    num_workers = config["num-workers"]
    print(" > Num workers", num_workers)

    # Fixed noise
    if isinstance(config["fixed-noise"], str):
        arr = np.load(config["fixed-noise"])
        fixed_noise = torch.tensor(arr, device=device, dtype=torch.float32)
    else:
        fixed_noise = torch.randn(config["fixed-noise"], config["model"]["z_dim"], device=device)

    # Test noise
    test_noise, test_noise_conf = load_z(config["test-noise"])
    print("Loaded test noise from", config["test-noise"])
    print("\t", test_noise_conf)

    # Inception FID reference
    mu_ref, sigma_ref = fid.load_statistics_from_path(config["fid-stats-path"])
    fm_fn, dims_inception = fid.get_inception_feature_map_fn(device)
    original_fid = fid.FID(fm_fn, dims_inception, test_noise.size(0), mu_ref, sigma_ref, device=device)

    # ---- Step 1: Baseline GAN Training ----
    num_runs = config["num-runs"]
    for i in range(num_runs):
        print("##\n# Starting run", i, "\n##")
        run_id = wandb.util.generate_id()
        cp_dir = create_checkpoint_path(config, run_id)

        with open(os.path.join(cp_dir, "fixed_noise.npy"), "wb") as f:
            np.save(f, fixed_noise.detach().cpu().numpy())

        seed = run_seeds[i]
        setup_reprod(seed)
        config["model"]["image-size"] = img_size

        # Train baseline GAN (Step 1)
        G, D = construct_gan(config["model"], img_size, device)
        g_optim, d_optim = construct_optimizers(config["optimizer"], G, D)
        g_crit, d_crit = construct_loss(config["model"]["loss"], D)
        g_updater = UpdateGeneratorGAN(g_crit)

        print(f"Storing generated artifacts in {cp_dir}")
        original_gan_cp_dir = os.path.join(cp_dir, "step-1")

        if not isinstance(config["train"]["step-1"], str):
            batch_size = config["train"]["step-1"]["batch-size"]
            n_epochs = config["train"]["step-1"]["epochs"]
            n_disc_iters = config["train"]["step-1"]["disc-iters"]
            checkpoint_every = config["train"]["step-1"]["checkpoint-every"]

            fid_metrics = {"fid": original_fid}

            early_stop_key = "fid"
            early_stop_crit = config["train"]["step-1"].get("early-stop", {}).get("criteria", None)
            early_stop = (early_stop_key, early_stop_crit) if early_stop_crit is not None else (early_stop_key, None)

            wandb.init(
                project=config["project"],
                group=config["name"],
                entity=os.environ["ENTITY"],
                job_type="step-1",
                name=f"{run_id}-step-1",
                config={
                    "id": run_id,
                    "seed": seed,
                    "gan": config["model"],
                    "optim": config["optimizer"],
                    "train": config["train"]["step-1"],
                    "dataset": config["dataset"],
                    "num-workers": config["num-workers"],
                    "test-noise": test_noise_conf,
                },
            )

            _, _, _, step_1_metrics = train(
                config,
                dataset,
                device,
                n_epochs,
                batch_size,
                G,
                g_optim,
                g_updater,
                D,
                d_optim,
                d_crit,
                test_noise,
                fid_metrics,
                n_disc_iters,
                early_stop=early_stop,
                checkpoint_dir=original_gan_cp_dir,
                fixed_noise=fixed_noise,
                checkpoint_every=checkpoint_every,
            )
            wandb.finish()
        else:
            original_gan_cp_dir = config["train"]["step-1"]
            step_1_metrics = load_gan_stats(original_gan_cp_dir)

        # ---- Step 2: Modified GAN Training (GASTeN v2) ----
        print(" > Start step 2 (GAN with modified loss)")

        # Choose step-1 epoch
        step_1_epochs = config["train"]["step-2"].get("step-1-epochs", ["best"])
        step1_epoch_label = step_1_epochs[0]

        epoch_int = resolve_step1_epoch(step1_epoch_label, step_1_metrics, original_gan_cp_dir)

        gan_checkpoint_path = get_gan_path_at_epoch(original_gan_cp_dir, epoch=epoch_int)

        # Single classifier: pick the first path
        classifier_path = config["train"]["step-2"]["classifier"][0]
        C_name = os.path.splitext(os.path.basename(classifier_path))[0]
        C, C_params, C_stats, C_args = construct_classifier_from_checkpoint(classifier_path, device=device)
        C.to(device)
        C.eval()

        # Feature-map extraction for classifier-based FID ("focd")
        def get_feature_map_fn(images, batch_idx, batch_size):
            output = C(images, output_feature_maps=True)
            feature_map = output[-2]  # expected shape: (B, dims, h, w)
            pooled = torch.nn.functional.adaptive_avg_pool2d(feature_map, (1, 1))
            return pooled.view(pooled.size(0), -1)

        # Robust dims inference (works even if dataset has no .data attribute)
        x0, _ = dataset[0]
        x0 = x0.unsqueeze(0).to(device)
        dims = get_feature_map_fn(x0, 0, 1).size(1)

        print(" > Computing statistics using original dataset")
        mu, sigma = compute_dataset_fid_stats(
            dataset, get_feature_map_fn, dims, device=device, num_workers=num_workers
        )
        print("   ... done")

        our_class_fid = fid.FID(get_feature_map_fn, dims, test_noise.size(0), mu, sigma, device=device)

        # Additional metrics
        conf_dist = LossSecondTerm(C)
        fid_metrics = {
            "fid": original_fid,
            "focd": our_class_fid,
            "conf_dist": conf_dist,
            "hubris": Hubris(C, test_noise.size(0)),
        }

        c_out_hist = OutputsHistogram(C, test_noise.size(0))
        if args.no_plots:
            c_out_hist = None

        weight = config["train"]["step-2"]["weight"][0]

        eval_metrics = train_modified_gan(
            config,
            dataset,
            cp_dir,
            gan_checkpoint_path,
            test_noise,
            fid_metrics,
            c_out_hist,
            C,
            C_name,
            C_params,
            C_stats,
            C_args,
            weight,
            fixed_noise,
            device,
            step_2_seeds[i],
            run_id,
            step1_epoch_label=step1_epoch_label,
            step1_epoch_int=epoch_int,
        )

        # Plot metrics
        if not args.no_plots:
            eval_stats = eval_metrics.get("eval", {}) if isinstance(eval_metrics, dict) else {}
            fid_hist = eval_stats.get("fid", [])
            conf_hist = eval_stats.get("conf_dist", [])
            hubris_hist = eval_stats.get("hubris", [])

            if not fid_hist:
                print("Step-2 plotting skipped: no eval/fid history found.")
                continue

            n = len(fid_hist)

            def _fit_len(vals):
                if len(vals) == n:
                    return vals
                if len(vals) == 0:
                    return [float("nan")] * n
                if len(vals) > n:
                    return vals[:n]
                return vals + [vals[-1]] * (n - len(vals))

            step2_metrics = pd.DataFrame(
                {
                    "fid": _fit_len(fid_hist),
                    "conf_dist": _fit_len(conf_hist),
                    "hubris": _fit_len(hubris_hist),
                    "s1_epochs": [epoch_int] * n,
                    "weight": [str(weight)] * n,
                    "classifier": [classifier_path] * n,
                    "epoch": [j + 1 for j in range(n)],
                }
            )
            plot_metrics(step2_metrics, cp_dir, f"{C_name}-{run_id}")


if __name__ == "__main__":
    main()
