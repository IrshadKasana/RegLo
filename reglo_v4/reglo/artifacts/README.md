# artifacts/ — every result lands here, one folder per profile

Git-ignored except for this README and the empty folder markers. Move it with
`REGLO_ARTIFACTS=/path/to/artifacts` (e.g. to scratch storage on the HPC). Smoke and
production never mix: delete `artifacts/smoke/` to rerun the smoke test from scratch.

```
artifacts/
├── production/                            the real study (configs/profiles/production.yaml)
├── smoke/                                 the smoke test on the real datasets (smoke.yaml)
├── smoke_offline/                         the smoke test on stand-in images (smoke_offline.yaml)
│
└── <profile>/
    ├── meta_training/<round>/                 STAGE 1 — per round (configs/rounds.yaml: A, B), one folder
    │   │                                      per LSTM optimizer, meta-trained on the round's source pair
    │   ├── m3_lstm/                           φ for M3 (L_meta only)
    │   └── m4_lstm_reg-<r>/                   φ*_r for M4, r ∈ {l1, l2, sam, gam, gsam}
    │       ├── config.yaml                    fully resolved configuration
    │       ├── phi_best.pt                    selected by meta-validation — loaded by the M3 / M4 runs
    │       ├── phi_last.pt                    latest state, for resuming after a walltime / max_hours stop
    │       ├── meta_log.jsonl                 L_meta, L_smooth, L_reg, ‖∇φ‖ per meta-step; validations
    │       └── summary.json                   written when meta-training finishes
    │
    ├── runs/<study>/<optimizee>-<dataset>/<method>[-<r>][@<round>]/seed<k>/
    │   │                                      one optimizee run — M2 (Stage 1, on the round's source pair,
    │   │                                      with r), M1 and M3/M4 (Stages 2–3; M3/M4 once per round, @A,
    │   │                                      @B: whose φ they used). One pool per study: each run trained
    │   │                                      once, read by every setting and round that uses it
    │   ├── config.yaml
    │   ├── environment.json                   GPU model, versions, PBS / SLURM job id
    │   ├── metrics.jsonl                      per-epoch accuracy, losses, timing (the training curves)
    │   ├── final.json                         headline numbers: accuracy, gaps, FLOPs, norms, sparsity
    │   ├── weights.json                       norms, sparsity, per-layer norms, log10|w| histogram
    │   ├── model_final.pt                     trained optimizee ("at completion")
    │   ├── model_converged.pt                 "at convergence" snapshot (the paper's PoC tests), if reached
    │   └── .lock                              held while a process trains this run (one at a time)
    │
    ├── diagnostics/                           STAGE 4 supplements
    │   ├── <study>/…/seed<k>/diagnostics.json λ_max, Tr(H), ε-ball tests, norms per snapshot
    │   └── compute/<optimizee>-<dataset>.json all 12 variants timed on one GPU (+ hours per run)
    │
    ├── tables/<study>/                        STAGE 4 — start with summary.md; round_<R>/cube_accuracy, cube_gen_gap
    │                                          and cube_flops are the 3D tables of round R (method × regularizer ×
    │                                          setting), round_<R>/benchmark the Stage-1 M1-vs-M2(r) comparison,
    │                                          transfer_* the cross-evaluation of the rounds
    ├── figures/<study>/round_<R>/             cube figures, training curves, Pareto, PoC, weights (PNG + PDF)
    ├── tables/meta_training/, figures/meta_training/      Stage 1
    ├── manifests/                             run lists (one line per run); submitted/ = frozen copies the jobs read
    └── logs/                                  output of each run started by run_study.py with --gpus / --jobs
```

Example: `production/runs/main/resnet18-svhn/m4_lstm_reg-sam@A/seed42/` is M4 with round A's
SAM-trained LSTM optimizer training ResNet-18 on SVHN (for round A: Stage 2, unseen optimizee),
seed 42; `…/m4_lstm_reg-sam@B/seed42/` the same with round B's optimizer (for round B: unseen
optimizee and dataset). `production/runs/main/convnet4-svhn/m2_adamw_reg-sam/seed42/` is M2 —
AdamW + SAM — in Stage 1 on round A's source pair.
