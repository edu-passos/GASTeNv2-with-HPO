# HPC Usage

This repo now includes two SLURM entrypoints for the MSc. AI cluster:

- `hpc/fast_sanity.sbatch`: a short CPU smoke test for config resolution, dataset access, checkpoint loading, GAN construction, and optional FID model loading.
- `hpc/normal_gpu_hpo.sbatch`: the real batch job for the two-stage HPO pipeline on the `normal` partition with `gpu_batch`.

## 1. Prepare the repo on the cluster

Clone or copy the repo to your HPC workspace, and make sure the project assets are available under a single storage root, for example:

```text
/path/to/GASTeNv2-with-HPO
/path/to/GASTeNv2-with-HPO/large_files
```

Create a Python environment and install dependencies:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

## 2. Create `.env`

From the repo root, create a `.env` file that points to your storage root:

```bash
cat > .env <<'EOF'
FILESDIR=/absolute/path/to/GASTeNv2-with-HPO/large_files
WANDB_MODE=offline
EOF
```

`FILESDIR` is required because the experiment YAML uses relative paths like `data/...`, `models/...`, and `out/...`, and `read_config()` resolves them under that root.

For the current chest-xray config, these assets must exist under `FILESDIR`:

- `data/fid-stats/stats.inception.chest-xray.1v0.npz`
- `data/z/z_512_64`
- `models/chest-xray/1v0/cnn-24-24-4441`

## 3. Fast smoke test on CPU

Submit from the repo root:

```bash
sbatch --export=ALL,ACTIVATE_CMD='source .venv/bin/activate' hpc/fast_sanity.sbatch
```

If you want to skip the FID Inception model download/load during the smoke test:

```bash
sbatch --export=ALL,ACTIVATE_CMD='source .venv/bin/activate',SANITY_ARGS='--skip-fid-model' hpc/fast_sanity.sbatch
```

This job runs on:

- partition: `fast`
- qos: `cpu`
- limits in script: `2` CPUs, `8G` RAM, `20` minutes

## 4. Full chest-xray HPO run on GPU

Submit from the repo root:

```bash
sbatch --export=ALL,ACTIVATE_CMD='source .venv/bin/activate' hpc/normal_gpu_hpo.sbatch
```

The script defaults to a shorter trial budget that is more realistic for a shared HPC queue:

- step 1 trials: `5`
- step 1 walltime: `18000` seconds
- step 2 trials: `5`
- step 2 walltime: `18000` seconds

Override them at submission time if needed:

```bash
sbatch --export=ALL,\
ACTIVATE_CMD='source .venv/bin/activate',\
STEP1_TRIALS=8,\
STEP1_WALLTIME=25000,\
STEP2_TRIALS=8,\
STEP2_WALLTIME=25000,\
CONFIG=experiments/chest-xray.yml \
hpc/normal_gpu_hpo.sbatch
```

This job runs on:

- partition: `normal`
- qos: `gpu_batch`
- GPU request: `1`
- limits in script: `4` CPUs, `16G` RAM, `15h50m`

## 5. Notes

- SLURM logs are written to `slurm/<job-name>-<job-id>.out` under the repo root.
- Training outputs are written under `FILESDIR/out/...`.
- The first chest-xray run may need to download the Hugging Face dataset and the FID Inception weights. If compute nodes do not have outbound network access, pre-populate those caches before submitting the GPU job.
