import os
import json
import argparse
import numpy as np
from dotenv import load_dotenv

import torch
import torch.nn as nn
import wandb

from smac import HyperparameterOptimizationFacade, Scenario
from ConfigSpace import ConfigurationSpace, Float

from src.metrics import fid, LossSecondTerm, Hubris
from src.data_loaders import load_dataset
from src.gan.train import train
from src.gan.update_g import UpdateGeneratorGASTEN_gaussian, UpdateGeneratorGASTEN_gaussianV2
from src.metrics.c_output_hist import OutputsHistogram
from src.utils import load_z, set_seed, gen_seed, create_checkpoint_path
from src.utils.config import read_config
from src.utils.checkpoint import construct_gan_from_checkpoint, construct_classifier_from_checkpoint
from src.gan import construct_loss


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True, help="Path to YAML config")
    p.add_argument("--no-plots", action="store_true", help="Skip plots/histograms")
    p.add_argument("--trials", type=int, default=None, help="Override step-2 SMAC trials")
    p.add_argument("--walltime", type=int, default=None, help="Override step-2 SMAC wall-time limit (s)")
    return p.parse_args()


def construct_optimizers(opt_cfg, G, D):
    g_optim = torch.optim.Adam(G.parameters(), lr=opt_cfg["lr"], betas=(opt_cfg["beta1"], opt_cfg["beta2"]))
    d_optim = torch.optim.Adam(D.parameters(), lr=opt_cfg["lr"], betas=(opt_cfg["beta1"], opt_cfg["beta2"]))
    return g_optim, d_optim


def train_modified_gan(
    config, dataset, cp_dir, gan_ckpt_dir, test_noise,
    fid_metrics, c_out_hist,
    C, C_name, C_params, C_stats, C_args,
    weight, fixed_noise, num_classes, device, seed, run_id
):
    if "gaussian" in weight:
        a, v = weight["gaussian"]["alpha"], weight["gaussian"]["var"]
        updater_cls = UpdateGeneratorGASTEN_gaussian
        tag = f"gauss_α{a:.3f}_σ{v:.4f}"
    elif "gaussian-v2" in weight:
        a, v = weight["gaussian-v2"]["alpha"], weight["gaussian-v2"]["var"]
        updater_cls = UpdateGeneratorGASTEN_gaussianV2
        tag = f"gaussV2_α{a:.3f}_σ{v:.4f}"
    else:
        raise ValueError("weight must contain 'gaussian' or 'gaussian-v2'")

    run_name = f"{C_name}_{tag}"
    out_dir = os.path.join(cp_dir, run_name)
    os.makedirs(out_dir, exist_ok=True)

    # load step-1 GAN
    G, D, _, _ = construct_gan_from_checkpoint(gan_ckpt_dir, device=torch.device("cpu"))

    # multi-GPU
    if torch.cuda.device_count() > 1:
        G = nn.DataParallel(G)
        D = nn.DataParallel(D)
        if hasattr(G.module, "z_dim"):
            G.z_dim = G.module.z_dim

    G, D = G.to(device), D.to(device)

    g_crit, d_crit = construct_loss(config["model"]["loss"], D)
    g_opt, d_opt = construct_optimizers(config["optimizer"], G, D)
    g_updater = updater_cls(g_crit, C, alpha=a, var=v)

    tcfg = config["train"]["step-2"]
    bs = tcfg["batch-size"]
    nepochs = tcfg["epochs"]
    nd = tcfg["disc-iters"]
    chk_every = tcfg["checkpoint-every"]

    early_crit = tcfg.get("early-stop", {}).get("criteria", None)
    early_stop = ("conf_dist", early_crit) if early_crit is not None else ("conf_dist", None)

    set_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    wandb.init(
        project=config["project"],
        group=config["name"],
        entity=os.environ.get("ENTITY", None),
        job_type="step-2",
        name=f"{run_id}-{run_name}",
        config={
            "seed": seed,
            "weight": weight,
            "train": tcfg,
            "classifier": C_name,
            "classifier_loss": C_stats.get("test_loss", 0.0),
        },
    )

    _, _, _, metrics = train(
        config, dataset, device, nepochs, bs,
        G, g_opt, g_updater,
        D, d_opt, d_crit,
        test_noise, fid_metrics,
        nd,
        early_stop=early_stop,
        checkpoint_dir=out_dir,
        fixed_noise=fixed_noise,
        c_out_hist=c_out_hist,
        checkpoint_every=chk_every,
        classifier=C,
    )

    wandb.finish()
    return metrics


def main():
    load_dotenv()
    args = parse_args()
    config = read_config(args.config)

    out_dir = config.get("out-dir", "out")
    os.makedirs(out_dir, exist_ok=True)

    ds = config["dataset"]
    dataset_name = ds["name"]
    pos = ds["binary"]["pos"]
    neg = ds["binary"]["neg"]

    pointer = os.path.join(out_dir, f"step1-best-{dataset_name}-{pos}v{neg}.txt")
    if not os.path.exists(pointer):
        raise FileNotFoundError(f"Missing step-1 pointer file: {pointer}")

    gan_ckpt_dir = open(pointer).read().strip()
    if not gan_ckpt_dir or not os.path.exists(gan_ckpt_dir):
        raise FileNotFoundError(f"Invalid step-1 checkpoint dir: '{gan_ckpt_dir}'")

    device = torch.device(config["device"])

    dataset, num_classes, _ = load_dataset(dataset_name, config["data-dir"], pos, neg)

    if isinstance(config["fixed-noise"], str):
        fixed_noise = torch.tensor(np.load(config["fixed-noise"]), dtype=torch.float32, device=device)
    else:
        fixed_noise = torch.randn(int(config["fixed-noise"]), config["model"]["z_dim"], device=device)

    test_noise, _ = load_z(config["test-noise"])

    # Original inception FID
    mu, sigma = fid.load_statistics_from_path(config["fid-stats-path"])
    fm_fn, dims = fid.get_inception_feature_map_fn(device)
    original_fid = fid.FID(fm_fn, dims, test_noise.size(0), mu, sigma, device=device)

    clf_path = config["train"]["step-2"]["classifier"][0]
    C, C_p, C_s, C_a = construct_classifier_from_checkpoint(clf_path, device=device)
    C.eval()

    fid_metrics = {
        "fid": original_fid,
        "conf_dist": LossSecondTerm(C),
        "hubris": Hubris(C, test_noise.size(0)),
    }
    c_out_hist = None if args.no_plots else OutputsHistogram(C, test_noise.size(0))

    run_id = wandb.util.generate_id()
    cp_dir = create_checkpoint_path(config, run_id)

    def step2_obj(cfg_hp, seed: int) -> float:
        weight = {"gaussian": {"alpha": float(cfg_hp["alpha"]), "var": float(cfg_hp["var"])}}

        metrics = train_modified_gan(
            config, dataset, cp_dir, gan_ckpt_dir, test_noise,
            fid_metrics, c_out_hist,
            C, os.path.basename(clf_path), C_p, C_s, C_a,
            weight, fixed_noise, num_classes, device, seed,
            wandb.util.generate_id(),
        )

        ev = metrics.get("eval", {})
        fid_list = ev.get("fid", [])
        cd_list = ev.get("conf_dist", [])
        if not fid_list or not cd_list:
            return float("inf")

        f = float(fid_list[-1])
        c = float(cd_list[-1])
        return 1.0 * f + 0.001 * c

    cs = ConfigurationSpace()
    cs.add_hyperparameters(
        [
            Float("alpha", (0.0, 5.0), default=1.0),
            Float("var", (1e-4, 1.0), default=0.01, log=True),
        ]
    )

    scenario = Scenario(
        cs,
        deterministic=True,
        n_trials=args.trials if args.trials is not None else config["train"]["step-2"].get("hpo-trials", 50),
        walltime_limit=(
            args.walltime
            if args.walltime is not None
            else config["train"]["step-2"].get("hpo-walltime", 10_000)
        ),
    )

    smac = HyperparameterOptimizationFacade(scenario, step2_obj, overwrite=True)
    incumbent = smac.optimize()
    best_cfg = incumbent.get_dictionary()

    out_json = os.path.join(cp_dir, f"step2-best-gauss-{pos}v{neg}.json")
    with open(out_json, "w") as f:
        json.dump(best_cfg, f, indent=2)

    # retrain once with best cfg
    final_metrics = train_modified_gan(
        config, dataset, cp_dir, gan_ckpt_dir, test_noise,
        fid_metrics, c_out_hist,
        C, os.path.basename(clf_path), C_p, C_s, C_a,
        {"gaussian": {"alpha": float(best_cfg["alpha"]), "var": float(best_cfg["var"])}},
        fixed_noise, num_classes, device,
        gen_seed(), wandb.util.generate_id(),
    )

    ev = final_metrics.get("eval", {})
    fid_list = ev.get("fid", [])
    cd_list = ev.get("conf_dist", [])
    last_fid = float(fid_list[-1]) if fid_list else float("inf")
    last_cd = float(cd_list[-1]) if cd_list else float("inf")
    best_epoch = (int(min(range(len(fid_list)), key=lambda i: fid_list[i])) + 1) if fid_list else -1

    summary_txt = os.path.join(cp_dir, f"step2-best-gauss-{pos}v{neg}-summary.txt")
    with open(summary_txt, "w") as f:
        f.write("Best Gaussian parameters:\n")
        f.write(json.dumps(best_cfg, indent=2) + "\n\n")
        f.write("Performance metrics:\n")
        f.write(f"  FID       = {last_fid:.4f}\n")
        f.write(f"  Conf_dist = {last_cd:.4f}\n")
        f.write(f"  Best Epoch= {best_epoch}\n")
        f.write("\n")
        f.write(f"Loaded step-1 checkpoint: {gan_ckpt_dir}\n")

    print("→ Step 2 complete.")
    print(f"   * Best hyperparams JSON: {out_json}")
    print(f"   * Summary TXT:            {summary_txt}")


if __name__ == "__main__":
    main()
