# ConvNet-4 lambda sweep on PARAM Rudra (SLURM)

Same experiment as `README.md` (4 methods × {L1, L2, SAM, GSAM, GAM} × SVHN/CIFAR-10, GRID of 29 values);
this file only covers running it on **PARAM Rudra** (SLURM, 2 × A100-80GB per GPU node, partition `gpu`).
Facts below come from the *PARAM Rudra User Manual (IIT Patna)* and the *2FA User Guide*.

## 0. What you do vs. what is automatic

| step | who | where |
|---|---|---|
| log in (CAPTCHA → 6-digit code → password) | **you** (2FA needs your phone) | your terminal |
| upload + unzip | you | `scp` below |
| `setup_env.sh`, `gpu_check.sbatch`, `submit_rudra.sh` | you (copy-paste) | login node |

## 1. Log in (2FA)

```bash
ssh <username>@<rudra-login-host>        # host/IP from your account mail; manual example: paramrudra.iitp.ac.in
```
First login: CAPTCHA → cluster password → a QR code appears → scan with *Google Authenticator* → enter the 6-digit code.
Later logins: CAPTCHA → **verification code** → password. Lost phone / reset: run `google-authenticator` on the cluster
(answer `y` to the four prompts) and keep the emergency codes. Do not run jobs on the login node (they are killed).

## 2. Upload and unzip

```bash
# from your laptop
scp -r -P 22 learn-regg_convnet4_sweep_rudra_with_data.zip <username>@<host>:/home/<username>/
# on Rudra
cd ~ && unzip -q learn-regg_convnet4_sweep_rudra_with_data.zip && cd learn-regg_convnet4_sweep_rudra
```
Data (`datasets/cifar-10-batches-py`, `datasets/{train,test}_32x32.mat`, ~420 MB) is inside the "with_data" zip, so the
compute nodes need no internet. `/home` quota is limited; `/scratch` has no quota issue but files unused for 3 months are
deleted — you can unzip there instead (`cd /scratch/<username>`); everything is relative to the project root.

## 3. One-time environment check (login node, seconds)

```bash
bash experiments/convnet4_lambda_sweep_svhn_cifar10/slurm/setup_env.sh
```
Loads `mldl/Pytorch` (PyTorch 2.2.1, py3.10, CUDA 11.8 — one of the ready-made envs in the manual), prints versions and
installs `scipy` (needed for the SVHN `.mat` files) with `pip install --user` only if missing.
If that module name differs on your system: `module avail 2>&1 | grep -i -E "torch|conda"` and re-run with
`MODULE_ENV=<name> bash .../setup_env.sh` (the same variable is read by every `.sbatch`).

## 4. GPU check (≈ 5 min of A100)

```bash
mkdir -p logs
sbatch experiments/convnet4_lambda_sweep_svhn_cifar10/slurm/gpu_check.sbatch
squeue --me                                   # PD = pending, R = running
cat logs/gpucheck.<jobid>.out                 # expect SMOKE_OK and GPUCHECK_OK
```
Then (not before) the real sweep. Either submit both at once if the queue is long — the sweep does not depend on the check.

## 5. Submit the sweep

```bash
bash experiments/convnet4_lambda_sweep_svhn_cifar10/slurm/submit_rudra.sh  full  1  4  24:00:00
#                                                                       PRESET COPIES LANES WALLTIME
squeue --me
```
* **LANES = 4**: one SLURM array of 4 jobs, each `--gres=gpu:1` on partition `gpu` with 5 worker processes sharing the
  80 GB A100 (the code caps worker processes by GPU memory: a GAM cell peaks at ≈ 13 GB). Manual: ≤ 10 simultaneous jobs.
* **COPIES**: chain length (`--dependency=afterany`). A lane stops cleanly 30 min before the walltime
  (`WALLTIME_EXIT`), the next copy resumes and skips finished cells; surplus copies exit in seconds.
* **WALLTIME**: manual says max 4 days (`96:00:00`) on the default QoS; default walltime is 2 h, so it is always set.
  Shorter requests backfill sooner.
* Presets: `smoke` (minutes, wiring only), `fast` (width 32), `full` (width 64, the real run).
* A final `report` job (partition `cpu`) writes `results/<preset>/summary.md` and `best_lambdas.json`.

Work estimate (from V100 timings; A100 is the same or faster): ≈ 430 LO cells (2–4 min each) + ≈ 280 AdamW runs
→ roughly 10–15 GPU-hours in total, i.e. **a few hours wall-clock on 4 lanes** once the jobs start.

## 6. Useful SLURM commands (from the manual)

| | |
|---|---|
| `squeue --me` | my jobs (ST: PD pending, R running) |
| `scancel <jobid>` / `scancel -u $USER` | cancel one / all my jobs |
| `sinfo -p gpu` | state of the GPU partition |
| `scontrol show job <jobid>` | why pending (`Reason=`), node, limits |
| `sacct -j <jobid> --format=JobID,State,Elapsed,MaxRSS` | accounting after the job |
| `salloc --nodes=1 --time=1:00:00 --partition=gpu --gres=gpu:1` | interactive GPU node (then `ssh <node>`) |
| `module avail` / `module list` | modules |

## 7. Files

```
slurm/setup_env.sh      login-node env check
slurm/gpu_check.sbatch  smoke + timing on one A100 (uses slurm/gpucheck.py)
slurm/sweep.sbatch      one lane (array task) of the sweep   [env: PRESET LANES MAX_HOURS PROCS MODULE_ENV CONDA_ENV]
slurm/report.sbatch     aggregation
slurm/submit_rudra.sh   array + chain + report submission
```
Own conda env instead of `mldl/Pytorch`: `module load mldl/Miniconda; conda create -n myenv python=3.10 ...`, then
`MODULE_ENV=mldl/Miniconda CONDA_ENV=myenv bash slurm/submit_rudra.sh full 1 4 24:00:00`.

## 8. Caveats

* Written from the manual and tested on CPU only (and earlier on a V100/PBS cluster); the SLURM scripts themselves have
  **not been run on Rudra**. If `sbatch` rejects a directive (`--cpus-per-task`, `--gres`, partition name), the fix is a
  one-line edit in the `#SBATCH` header of the file named in the error.
* Conventions (radius SAM 0.001 / GAM 0.04 / GSAM 0.005, GSAM = true surrogate gap, L1/L2 as sums) are as in `README.md`.
