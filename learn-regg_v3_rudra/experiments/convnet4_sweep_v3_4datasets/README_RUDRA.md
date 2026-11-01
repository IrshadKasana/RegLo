# Five-method, four-dataset sweep on PARAM Rudra (SLURM)

Same experiment as `README.md` (M1 AdamW, M2 AdamW+λR, M3 learned optimizer, M4a LO+smooth, M5 LO+smooth+R; R ∈ L1, L2, SAM, GSAM,
GAM; SVHN, CIFAR-10, MNIST, FashionMNIST; 70/20/10 split; ID test + OOD between all pairs). This file only covers running it on
**PARAM Rudra** (SLURM; GPU nodes with A100-80GB; partition `gpu`; max 4 days per job; ≤ 10 simultaneous jobs).

## 0. What is where
| | |
|---|---|
| `learn-regg_v3_rudra_code.zip` (≈ 75 KB) | code, SLURM scripts, the 6 reused files of `src/`, checksum list |
| `learn-regg_v3_rudra_newdata.zip` (≈ 41 MB) | MNIST + FashionMNIST raw files (the only datasets Rudra does not have yet) |
| `learn-regg_v3_rudra_with_data.zip` (≈ 440 MB, optional) | everything incl. CIFAR-10 + SVHN, if the old Rudra folder is gone |

CIFAR-10 and SVHN are already in `~/Rego/convnet4_rudra/learn-regg_convnet4_sweep_rudra/datasets/`; `ASSEMBLE.sh` copies them.
**Every job loads all four datasets** (each one is an OOD target of the others), so all four must be on disk even if this
cluster computes only some of them as *sources*.

## 1. Upload (from a SEPARATE terminal on your PC, not the logged-in one)
```bash
cd ~/Downloads/Reglo    # where the zips are
scp -P 4422 learn-regg_v3_rudra_code.zip learn-regg_v3_rudra_newdata.zip aroofa_ext@paramrudra.iitp.ac.in:~/Rego/convnet4_rudra/
```

## 2. Assemble (on Rudra) — a NEW root, the old project is not touched
```bash
ssh -o ConnectTimeout=30 aroofa_ext@paramrudra.iitp.ac.in -p 4422
cd ~/Rego/convnet4_rudra
unzip -q -n learn-regg_v3_rudra_code.zip learn-regg_v3_rudra/ASSEMBLE.sh      # only to get the script
bash learn-regg_v3_rudra/ASSEMBLE.sh            # unzips both zips, copies cifar10/svhn from the old root, checks md5
cd learn-regg_v3_rudra
bash experiments/convnet4_sweep_v3_4datasets/slurm/setup_env.sh      # env + data check (login node, seconds)
```
Expect `ALL DATASET FILES OK` and `OK torch/torchvision/numpy/scipy`. If no python with torch is found (module names differ on Rudra): `module avail 2>&1 | grep -i -E "torch|conda|python"` or `conda env list`,
then re-run with `MODULE_ENV="<module(s)>"` and/or `CONDA_ENV=<env>` (or `VENV=/path`, or `KEEP_ENV=1`) in front of
`bash .../setup_env.sh`, `sbatch ...gpu_check.sbatch` and `bash ...submit_rudra.sh` (the variables are passed on to the jobs).

## 3. GPU check (≈ 10 min of one A100) — do this first
```bash
mkdir -p logs
sbatch experiments/convnet4_sweep_v3_4datasets/slurm/gpu_check.sbatch
squeue --me ; cat logs/gpucheck.<jobid>.out      # expect SMOKE_OK, GPUCHECK_OK and the line  "PROCS <= N worker processes per GPU"
```
`gpucheck.py` times one full-width M5-GAM trial (the most expensive kind) and prints its seconds and peak GPU memory. Use that
to decide `PROCS` (default 4) and to estimate the wall time: total work ≈ (number of trials) × (seconds per trial) / (lanes × procs).

## 4. Submit the production sweep
```bash
# all four datasets here:
bash experiments/convnet4_sweep_v3_4datasets/slurm/submit_rudra.sh full 2 4 96:00:00
#                                                              PRESET COPIES LANES WALLTIME
# only some datasets (two clusters, no duplicated work):
DATASETS="mnist fmnist" bash experiments/convnet4_sweep_v3_4datasets/slurm/submit_rudra.sh full 2 4 96:00:00
# optional knobs (environment variables):  PROCS=4  REGS="l1 l2 sam gsam gam"  M5_MODE=full|block  MODULE_ENV=...  WAIT_HOURS=24
squeue --me
```
* **LANES**: one SLURM array task per A100. Lanes wait for each other's result files at every phase boundary, so they should
  be able to **start together** — choose LANES = the number of GPUs you can get at once (barrier timeout `WAIT_HOURS`, default 24 h).
* **COPIES**: chain length (`--dependency=afterany`). A lane stops cleanly 30 min before its walltime (`WALLTIME_EXIT`); the next
  copy resumes and skips finished trials; surplus copies exit in seconds.
* A final `report` job (CPU partition) writes `results/full/{summary.md, best_config.json, all_cells.csv, plots/, trial_table/}`.
* Resume after any interruption: run the same `submit_rudra.sh` command again.

## 5. Splitting the work with Agastya, then merging
Give the two clusters **disjoint `DATASETS`** (trials of different source datasets are independent files). Afterwards, on one machine:
```bash
bash slurm/merge_results.sh  /path/to/other/results/full   /path/to/this/results/full            # dry run
bash slurm/merge_results.sh  /path/to/other/results/full   /path/to/this/results/full  --apply
cd experiments/convnet4_sweep_v3_4datasets
python run_sweep.py report --preset full ; python plot_sweep_results.py --preset full ; python make_trial_table.py --preset full
```
(Transfer a results folder with `scp -r -P 4422 ...` or `rsync -av -e "ssh -p 4422" ...`.) Existing files are never overwritten.

## 6. Conservative merge of code / data from another copy of the project
```bash
bash slurm/merge_conservative.sh  SRC_DIR  DST_DIR            # dry run: new files / differing files
bash slurm/merge_conservative.sh  SRC_DIR  DST_DIR  --apply   # copies only NEW files; differing ones are saved as <name>.incoming
```
Status when this bundle was made: `project_paths.py` and the 6 reused `src/` files are byte-identical on the PC, on Agastya and in
the old Rudra project, and the MNIST / FashionMNIST / SVHN files are identical on the PC and Agastya — so no update of the old
files is needed. The only code difference to the Agastya copy of `run_sweep.py` is
`MAX_PROCS_PER_GPU = int(os.environ.get("MAX_PROCS_PER_GPU", "2"))` (default unchanged; `sweep.sbatch` sets it to `PROCS`).

## 7. SLURM cheat sheet
`squeue --me` · `scancel <jobid>` / `scancel -u $USER` · `sinfo -p gpu` · `scontrol show job <jobid>` (why pending: `Reason=`) ·
`sacct -j <jobid> --format=JobID,State,Elapsed,MaxRSS` · `salloc --nodes=1 --time=1:00:00 --partition=gpu --gres=gpu:1`

## 8. Caveats
* The SLURM scripts follow the Rudra manual and the earlier Rudra bundle, but the v3 scripts have **not been run on Rudra**. If
  `sbatch` rejects a directive (`--cpus-per-task`, `--gres`, `-p gpu`), edit that one `#SBATCH` line in the file named in the error.
* The code was tested on torch 1.9 (Agastya V100, CPU smoke). Rudra's `MLDL/Pytorch-gpu` has a newer torch (version printed by setup_env.sh): `gpu_check.sbatch` runs the
  smoke trials on it first — read `logs/gpucheck.*.err` if it fails.
* An A100 holds more worker processes than a 32 GB V100; if a trial dies with CUDA OOM, lower `PROCS` (a failed trial is retried
  automatically up to 6 times after a cool-down, then reported as FAILED in `trial_table/trial_status.md`).
