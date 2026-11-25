# Full pipeline on PARAM Rudra: three cases

All 6 methods × R ∈ {L1, L2, SAM, GAM, GSAM} × B ∈ {Adam, SGD}, i.e. 24 run ids per case (`run_pipeline.py --list`),
phases 1 → 2 → 3 → 3D tables. All hyperparameters come from the existing YAMLs (`configs/dmo/...`) and
`configs/pipeline/pipeline.yaml`. Nothing is shortened: base meta-training budget and curriculum (100 → 200 → 500 → 1000),
1000-step optimizee trainings × 5 seeds, 20-epoch linear probes.

| case | meta-training (= seen) task | phase 2 optimizees (on the seen dataset) | phase 3: frozen optimizee on |
|---|---|---|---|
| `mlp_mnist` | MLP + MNIST | MLP (seen), ConvNet-4, ResNet-18 | MNIST (seen); SVHN, FMNIST, CIFAR-10, CIFAR-100, ImageNet-32 (linear probe) |
| `convnet4_cifar10` | ConvNet-4 + CIFAR-10 | ConvNet-4 (seen), MLP, ResNet-18 | CIFAR-10 (seen); SVHN, MNIST, FMNIST, CIFAR-100, ImageNet-32 |
| `resnet18_imagenet` | ResNet-18 + ImageNet-32 | ResNet-18 (seen), ConvNet-4, MLP | ImageNet-32 (seen); SVHN, MNIST, FMNIST, CIFAR-10, CIFAR-100 |

**ResNet-18 case:** a full-width ResNet-18 (11.7 M weights) cannot be meta-trained: the unrolled LSTM graph needs
about 315 GB at the base truncation of 20 steps. The learned optimizer is therefore meta-trained on a **width-16
ResNet-18 proxy** (0.83 M weights, about 24 GB; about 46 GB with R_smooth). It is coordinate-wise, so it then trains
the **full width-64 ResNet-18** in phases 2 and 3 unchanged. To change the proxy width, set `R18_META_WIDTH=8`
(cheaper) or `R18_META_WIDTH=32` (does not fit at truncation 20) when you submit.

## Jobs

`rudra_jobs.py` runs each case as a queue of 24 units, one per run id. A unit runs phases 1–3 for its run id in
one process. A **lane** (`lane.sbatch`) is one A100 job that runs several units at once, as many as fit in its
GPU-memory budget, longest first. Several lanes can work on one case. Default layout: **5 GPU jobs**, the
gpu-QOS limit of this account (5 running jobs, 48 h walltime):

| case | lanes | processes per lane | why |
|---|---|---|---|
| mlp_mnist | 1 | 6 | small nets, CPU/data-loading bound (≤10 GB each) |
| convnet4_cifar10 | 1 | 6 | same |
| resnet18_imagenet | 3 | ≤3 (memory-packed) | ResNet proxy meta-training: ~25 GB (M3, M5), ~46 GB (M4, M6) |

**Walltime:** the slowest units (M6 with GAM: R_smooth costs 10 extra PGA steps per inner step, and GAM evaluates
the truncation about 4×) need more than 48 h. So:
* Meta-training writes a resume file (`checkpoints/lo/<run>.resume.pt`) after **every meta-trained optimizee**.
  It holds the LO, the outer-optimizer and scheduler state, curriculum counters, RNGs and FLOP counters.
  Optimizee trainings are cached and finished phases are skipped.
* A lane stops its processes 30 min before the walltime and **resubmits itself** (same case, same settings,
  at most `MAX_GEN=8` times). The next lane continues from the resume files. No job waits in the queue on a
  dependency (held jobs annoy the admins).
* If a lane's GPU is not empty when it starts (stale processes were seen on ragpu004/ragpu006), the lane
  requeues itself on another node. `ragpu004` is excluded by default (`EXCLUDE=`).
* A unit that fails twice is not retried. Its log is in `rudra_queue/logs/<run>.log`.

## Steps (on Rudra; Claude cannot SSH, so you run these)

```bash
# 0. copy the code (and the small datasets) to Rudra, e.g. into ~/Rego/learn-regularizers-main
#    scp learn-regularizers-rudra_code.tgz learn-regularizers-rudra_small_datasets.tgz <user>@<rudra>:~/Rego/
cd ~/Rego && mkdir -p learn-regularizers-main && tar -xzf learn-regularizers-rudra_code.tgz -C learn-regularizers-main
cd learn-regularizers-main && tar -xzf ../learn-regularizers-rudra_small_datasets.tgz       # -> datasets/{MNIST,FashionMNIST,cifar-*,SVHN}
ln -s ~/Rego/learn-regg_v5_consolidated_1/datasets/imagenet32 datasets/imagenet32         # ImageNet-32 already converted on Rudra

# 1. environment + data check (login node; light)
bash experiments/pipeline/rudra/check_setup.sh          # env.sh uses ~/rudra_env.sh / MLDL/Pytorch-gpu like before
#    (login node hangs on module search? -> load your env by hand and use KEEP_ENV=1 bash ...)

# 2. GPU check (one A100, <= 4 h): smoke of all 24 units x 3 cases on real data + speed/memory of the heavy units
mkdir -p logs/rudra && sbatch experiments/pipeline/rudra/gpu_check.sbatch
#    -> logs/rudra/gpucheck_<id>.out must contain SMOKE_OK_mlp_mnist, SMOKE_OK_convnet4_cifar10,
#       SMOKE_OK_resnet18_imagenet and GPUCHECK_OK. Read the SUMMARY table: measured peak GB vs the
#       estimate used for packing (if flagged, submit with MEM_SCALE=1.3), and projected meta-training hours.

# 3. full run (5 GPU jobs; resubmit themselves until done)
bash experiments/pipeline/rudra/submit_rudra.sh

# progress / control
squeue --me
python experiments/pipeline/rudra/rudra_jobs.py status            # per case: done / running / waiting / failed
tail -f logs/rudra/rgR18_<jobid>.out                                # lane log
tail -f artifacts/pipeline/rudra_resnet18_imagenet/rudra_queue/logs/M6_LO_gam.log   # one unit
bash experiments/pipeline/rudra/submit_rudra.sh resnet18_imagenet  # add / restart lanes of one case (LANES=, PROCS=)
python experiments/pipeline/rudra/rudra_jobs.py reset-failed <case>  # after fixing a failure, then submit again
```

Results of each case (written by the lane that finishes the case's last unit):
`artifacts/pipeline/rudra_<case>/tables/`: `summary.md`, `breakdown.md`, `results_long.csv`, and
`<metric>__B-<adam|sgd>.{npy,csv,png}` for test accuracy, generalization gap and FLOPs, each
Regularizer × Method × {seen optimizee, unseen optimizee, seen dataset, unseen dataset}. To rebuild a report by
hand: `python experiments/pipeline/rudra/rudra_jobs.py report <case>` (or in a short job).

## Expected cost (measure it with step 2)

Measured on a laptop RTX 4060 (single process, ms per meta-training inner step; A100 numbers come from step 2):

| unit | MLP+MNIST | ConvNet-4+CIFAR-10 | ResNet-18 w8 proxy + ImageNet-32 |
|---|---|---|---|
| M3 (LO) | 68 | 82 | 178 |
| M4 (LO + R_smooth) | 268 | 285 | 375 |
| M5-GAM | 161 | 260 | 673 |
| M6-GAM (proposed) | 974 | 1072 | 1517 |

A unit runs about 230 k inner steps if the curriculum stays about 10 outer iterations at unroll 1000. The stopping
rule decides this, so it is not known in advance. At that length M6-GAM is about 60–100 h, which is 2–3 lane
generations. The other 23 units of a case run in parallel next to it. Rough wall time: MLP and ConvNet-4 cases
about 3–4 days, ResNet-18 case about 4–6 days at width 16. Phases 2–3 add a few hours per case: 1000 steps × 5
seeds per optimizee, and phase 2 trains the full ResNet-18 in every case.

Disk: ~5 GB of optimizee checkpoints per case (full ResNet-18 state dicts) under `artifacts/pipeline/`. The home
quota is 50 GB (`quota -s`). If it is tight, run from `/scratch/$USER`, which is purged after 3 months without
access, and copy `tables/` back.

## Files

```
experiments/pipeline/rudra/
    README_RUDRA.md     this file
    rudra_jobs.py       case definitions, work queue / lane pool, status, report
    lane.sbatch         one lane (GPU guard, pool, self-resubmission)
    submit_rudra.sh     submits the lanes of one or all cases
    gpu_check.sbatch    smoke (real data, --quick) + gpu_check.py (speed / memory / projected hours)
    check_setup.sh      login-node check of python packages and the six datasets
    env.sh              finds the torch python (as in the earlier Rudra sweeps) + GPU fencing
```
Code changes for HPC use (an uninterrupted run computes exactly what it did before; a resumed meta-training run continues with a freshly shuffled data stream): meta-training resume (`pipeline/meta_train.py`); a cross-process
lock and atomic writes in the optimizee-run cache (`pipeline/optimizee_runs.py`); SIGTERM → clean exit
(`pipeline/cli.py`).
