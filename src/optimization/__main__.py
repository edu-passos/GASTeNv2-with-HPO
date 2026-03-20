import os
import subprocess
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
from dotenv import load_dotenv

from src.utils import create_and_store_z, gen_seed, set_seed
from src.utils.config import read_config


def parse_args():
    p = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
    p.add_argument("--config", dest="config_path", required=True, help="YAML experiment config")
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--trials", type=int, default=None)
    p.add_argument("--walltime", type=int, default=None)
    p.add_argument("--step2-trials", type=int, default=None)
    p.add_argument("--step2-walltime", type=int, default=None)

    # used only if we need to create z-pool
    p.add_argument("--nz", type=int, default=2048)
    p.add_argument("--z-dim", dest="z_dim", type=int, default=64)
    return p.parse_args()


def _ensure_test_noise(cfg: dict, seed: int, nz: int, z_dim: int) -> None:
    """
    Ensures cfg['test-noise'] exists. If it does not, create it in-place.
    """
    z_path = cfg.get("test-noise")
    if not isinstance(z_path, str):
        raise ValueError("Config must define 'test-noise' as a path string")

    # We treat 'test-noise' as a directory produced by create_and_store_z
    if os.path.exists(z_path):
        return

    parent = os.path.dirname(z_path)
    os.makedirs(parent, exist_ok=True)

    # create_and_store_z takes output directory and will create z_{nz}_{z_dim} under it,
    # so we pass parent, then require that it matches our desired leaf name.
    leaf = os.path.basename(z_path)
    expected_leaf = f"z_{nz}_{z_dim}"
    if leaf != expected_leaf:
        raise ValueError(
            f"'test-noise' is '{z_path}', but if we auto-create we expect leaf '{expected_leaf}'. "
            "Either pre-create the noise pool, or set test-noise accordingly."
        )

    print(f"Creating test noise pool at {z_path} (seed={seed}, nz={nz}, z_dim={z_dim})")
    create_and_store_z(parent, nz, z_dim, config={"seed": seed, "n_z": nz, "z_dim": z_dim})
    if not os.path.exists(z_path):
        raise RuntimeError(f"Failed to create test-noise directory at {z_path}")


def main():
    load_dotenv()
    args = parse_args()

    seed = gen_seed() if args.seed is None else args.seed
    set_seed(seed)

    cfg = read_config(args.config_path)
    _ensure_test_noise(cfg, seed=seed, nz=args.nz, z_dim=args.z_dim)

    cmd1 = ["python3", "-m", "src.optimization.gasten_bayesian_optimization_step1", "--config", args.config_path]
    if args.trials is not None:
        cmd1 += ["--trials", str(args.trials)]
    if args.walltime is not None:
        cmd1 += ["--walltime", str(args.walltime)]

    print("\nRunning:", " ".join(cmd1))
    subprocess.run(cmd1, check=True)

    cmd2 = ["python3", "-m", "src.optimization.gasten_bayesian_optimization_step2", "--config", args.config_path]
    if args.step2_trials is not None:
        cmd2 += ["--trials", str(args.step2_trials)]
    if args.step2_walltime is not None:
        cmd2 += ["--walltime", str(args.step2_walltime)]
    print("\nRunning:", " ".join(cmd2))
    subprocess.run(cmd2, check=True)


if __name__ == "__main__":
    main()
