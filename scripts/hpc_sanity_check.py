#!/usr/bin/env python3
import argparse
import os
from pathlib import Path

import numpy as np
import torch
from dotenv import load_dotenv

from src.data_loaders import load_dataset
from src.gan import construct_gan
from src.metrics import fid
from src.utils import load_z
from src.utils.checkpoint import construct_classifier_from_checkpoint
from src.utils.config import read_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Lightweight HPC smoke test for config resolution, assets, and model wiring."
    )
    parser.add_argument("--config", required=True, help="Experiment YAML")
    parser.add_argument("--device", default=None, help="Override config device for the check")
    parser.add_argument("--batch-size", type=int, default=4, help="Mini-batch size for sample checks")
    parser.add_argument(
        "--skip-fid-model",
        action="store_true",
        help="Skip loading the FID Inception model (useful if you only want path checks).",
    )
    return parser.parse_args()


def _resolve_device(device_name: str) -> torch.device:
    requested = (device_name or "cpu").lower()
    if requested.startswith("cuda") and not torch.cuda.is_available():
        print(f"Requested device '{device_name}' is unavailable on this node; using cpu instead.")
        return torch.device("cpu")
    return torch.device(requested)


def _describe_path(label: str, path_str: str) -> None:
    path = Path(path_str)
    print(f"{label}: {path}")
    if not path.exists():
        raise FileNotFoundError(f"Missing required path for {label}: {path}")


def main() -> None:
    load_dotenv()
    args = parse_args()

    if "FILESDIR" not in os.environ:
        raise EnvironmentError("FILESDIR is not set. Define it in the environment or in .env before running.")

    cfg = read_config(args.config)
    if args.device is not None:
        cfg["device"] = args.device

    device = _resolve_device(cfg["device"])
    print(f"Using device: {device}")
    print(f"FILESDIR: {os.environ['FILESDIR']}")
    print(f"Config: {Path(args.config).resolve()}")

    _describe_path("out-dir", cfg["out-dir"])
    _describe_path("data-dir", cfg["data-dir"])
    _describe_path("fid-stats-path", cfg["fid-stats-path"])
    _describe_path("test-noise", cfg["test-noise"])

    classifier_path = cfg["train"]["step-2"]["classifier"][0]
    _describe_path("classifier", classifier_path)

    dataset_cfg = cfg["dataset"]
    pos = dataset_cfg.get("binary", {}).get("pos")
    neg = dataset_cfg.get("binary", {}).get("neg")

    dataset, num_classes, img_size = load_dataset(dataset_cfg["name"], cfg["data-dir"], pos, neg)
    print(f"Dataset size: {len(dataset)}")
    print(f"Dataset image size: {img_size}")
    print(f"Dataset classes: {num_classes}")

    loader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)
    batch, labels = next(iter(loader))
    batch = batch.to(device)
    print(f"First batch tensor: {tuple(batch.shape)}")
    print(f"First batch labels: {labels[:min(4, labels.size(0))].tolist()}")

    test_noise, test_noise_meta = load_z(cfg["test-noise"])
    print(f"Test noise tensor: {tuple(test_noise.shape)}")
    print(f"Test noise metadata: {test_noise_meta}")

    fixed_noise_cfg = cfg["fixed-noise"]
    if isinstance(fixed_noise_cfg, str):
        fixed_noise = np.load(fixed_noise_cfg)
        print(f"Fixed noise file: {fixed_noise_cfg}")
        print(f"Fixed noise shape: {tuple(fixed_noise.shape)}")
    else:
        print(f"Fixed noise count: {fixed_noise_cfg}")

    mu, sigma = fid.load_statistics_from_path(cfg["fid-stats-path"])
    print(f"FID stats shapes: mu={tuple(mu.shape)}, sigma={tuple(sigma.shape)}")

    cfg["model"]["image-size"] = list(img_size)
    G, D = construct_gan(cfg["model"], img_size, device=device)
    G.eval()
    D.eval()

    z = torch.randn(min(args.batch_size, 4), cfg["model"]["z_dim"], device=device)
    with torch.no_grad():
        fake = G(z)
        disc_out = D(fake)
    print(f"Generator output shape: {tuple(fake.shape)}")
    print(f"Discriminator output shape: {tuple(disc_out.shape)}")

    classifier, clf_params, clf_stats, _ = construct_classifier_from_checkpoint(classifier_path, device=device)
    classifier = classifier.to(device)
    classifier.eval()
    with torch.no_grad():
        clf_out = classifier(batch)
    print(f"Classifier output shape: {tuple(clf_out.shape)}")
    print(f"Classifier params: {clf_params}")
    print(f"Classifier best epoch: {clf_stats.get('best_epoch', 'n/a')}")

    if not args.skip_fid_model:
        fm_fn, dims = fid.get_inception_feature_map_fn(device)
        with torch.no_grad():
            feats = fm_fn(batch[: min(batch.size(0), 2)], 0, min(batch.size(0), 2))
        print(f"FID feature shape: {tuple(feats.shape)}")
        print(f"FID dims: {dims}")

    canary = Path(cfg["out-dir"]) / ".hpc_sanity_check"
    canary.write_text("ok\n", encoding="ascii")
    print(f"Wrote canary file: {canary}")
    print("HPC sanity check completed successfully.")


if __name__ == "__main__":
    main()
