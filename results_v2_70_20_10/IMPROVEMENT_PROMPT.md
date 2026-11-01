# Prompt: make the sweep results complete, auditable and publication-ready

## Context
The five-method sweep (M1 AdamW, M2 AdamW+λR, M3 learned optimizer (LO), M4a LO+λ_smooth, M5 LO+λ_smooth+λ_reg·R;
R ∈ {L1, L2, SAM, GSAM, GAM}; SVHN and CIFAR-10; 70/20/10 split; ID test + cross-dataset OOD) has finished on Agastya in
`~/learn-regg_v5_consolidated_1/experiments/convnet4_sweep_v2_70_20_10/`. The local copy is in
`/home/hp/Downloads/Reglo/results_v2_70_20_10/`. Keep every existing result: do not re-run anything that has already
passed, and reuse the code in `run_sweep.py`, `common.py`, `plot_sweep_results.py` and `make_trial_table.py`.

I want two kinds of output:
1. **Best-pick summary images**, like `01_summary_svhn_to_cifar10.png` and `01_summary_cifar10_to_svhn.png`.
2. **A full evidence record** proving the sweep ran at full scale: every planned grid combination, every trial, its
   status, hyperparameters, steps, losses, accuracies, checkpoint and run time, plus how the best pick was chosen.

Fix the problems below.

## A. Gaps in the evidence record (`trial_table/`)
1. **Coverage is not stated against the full grid.** `trials_full.csv` lists the 3,032 *planned* trials, but the plan
   is staged, not the full Cartesian product:
   - M2 uses only the AdamW lr chosen by M1.
   - M4a and M5 use only the meta_lr chosen by M3 (1e-4).
   - M5 runs a λ_reg axis plus a 5×5 block: 40–48 of the 784 (λ_smooth × λ_reg) cells per regularizer.
   - ρ is swept in M2 only; M5 uses fixed ρ.

   Add a `coverage.csv` and a section in the status report with one row per (dataset, method, regularizer). Columns:
   size of the full product, planned, ran, passed, failed, missing, and coverage %. Add a plain sentence on why each
   staged part is not exhaustive. In the M5 heatmaps, mark each cell as "not planned" or "planned but missing", not
   just blank.
2. **Per-init detail is incomplete.** Only `val_acc` and `train_loss` are stored per init. Add a long-format
   `trials_per_init.csv`, one row per trial × init. Columns:
   - init seed value
   - train, val and test loss and accuracy
   - OOD loss, accuracy and AUROC
   - diverged flag
   - seconds
3. **Curves are thin.** `trials_curves.csv` has 10 points per init (every 100 steps): train loss, val loss and val acc.
   - Add train accuracy and step 0, the value before training.
   - Record the final step explicitly.
   - Keep the 100-step interval unless you need denser curves; if so, add a `CURVE_EVERY` option for future runs.
4. **Learned-optimizer meta-training is only summarised.** `trials_meta.csv` has 4 stage checkpoints per LO trial.
   Add the meta-training loss per stage or per block if the code has it; if not, say it was not logged. Also record
   which stage's checkpoint was finally used.
5. **No checkpoints for 2,530 of 3,090 trials.** M1 and M2 never saved weights. Say this explicitly in the table and
   README. For future runs, add an option to save the final optimizee weights of the selected cells only, which is
   cheap. Checkpoint rows should also record file size and a hash, so they can be verified.
6. **No provenance per trial.** Add the following where it can be recovered from the logs and PBS output; otherwise
   mark it unknown:
   - start and end timestamp
   - PBS job id
   - lane, worker and node
   - GPU name
   - attempt number
   - code version (file hash of `common.py`, `run_sweep.py`, `sweep_config.py`)
   - a snapshot or hash of the grid configuration used
7. **Two overlapping tables disagree.** `all_cells.csv` has 3,034 rows; `trials_full.csv` has 3,090. The extra 58 are
   the old-plan results marked `EXTRA_NOT_IN_CURRENT_GRID`. Keep one master table, or document clearly how they
   differ and point both at the same trial id. Use one trial-id format everywhere: M2 ids have an `m2_` prefix in one
   table and not the other, and M1 trial ids differ from their log job names.
8. **Wide OOD columns are half empty.** `ood_svhn_*` and `ood_cifar10_*` are each empty for half the rows. Use
   `ood_target` plus `ood_acc`, `ood_loss` and `ood_auroc` columns, or split per direction.
9. **Units are inconsistent.** The CSVs store accuracy as fractions (0–1) and the images use %. Pick one, state it in
   the header or a data dictionary, and write a `DATA_DICTIONARY.md` listing every column, its meaning and units.
10. **The selection itself is not evidenced.** For every method × regularizer, write `selection_trace.csv`. Columns:
    - candidate count
    - selected trial and its val acc
    - runner-up trial and its val acc
    - margin to the runner-up, in val SE units
    - rank of the selected trial by test acc, to show the winner's-curse gap

    Also add a column confirming that selection read validation only.
11. **The folder is untidy.** The top level has duplicate copies of `trials_curves.csv`, `trials_full.html` and the
    two `01_summary_*.png` images next to `trial_table/` and `plots/`. Keep a single copy of each, add a `README.md`
    index of every file, and add an `INDEX.html` linking the images and tables.

## B. Improvements to the summary images (`01_summary_*.png`)
1. **Show the search size.** Add a "cells tried / planned" column (5, 28, 392, 48, ...), so the reader sees each best
   pick was chosen from a full search.
2. **Highlight on validation, not test.** The bold row is the highest ID *test* accuracy, which looks like selection
   on test. Highlight on validation instead, or remove the highlight and add a footnote. Mark statistical ties, where
   differences are below about 1 SE.
3. **Add missing uncertainty.** OOD AUROC has no ±. Add its standard error (inits for M1/M2, meta-seeds for LO).
   `val acc` is a single seed-0 number; say so in the header.
4. **Shorten the title.** The current title is one long line. Move the protocol text to a footnote under the table.
5. **Put the config in columns.** Show λ, ρ, λ_smooth, λ_reg, lr and meta_lr in separate columns, with "–" where not
   applicable, instead of free text.
6. **Add a compact companion figure per dataset.** It should cover: grid coverage per method, pass, retry and fail
   counts, total GPU hours, and the number of seeds.

## C. Checks to run and report
- Every planned (dataset, method, regularizer, hyperparameter) combination appears exactly once with status
  PASSED, PASSED_AFTER_RETRY, FAILED or MISSING. Report totals that add up.
- Every LO trial has an existing checkpoint file. Every trial has `n_inits` = 3 with distinct init seeds.
- The best picks in the images equal the max-val rows in `selection_trace.csv` and the `best_config.json` entries.
- The numbers in the images equal the numbers in the CSVs, after unit conversion.

## Deliverables
- Updated `make_trial_table.py` and `plot_sweep_results.py`, still reading only the existing result files.
- New files: `coverage.csv`, `trials_per_init.csv`, `selection_trace.csv`, `DATA_DICTIONARY.md`, `README.md` and
  `INDEX.html`.
- Regenerated `trial_status.md`, the HTML table and the summary images.
- A short report of the checks in section C, with any gaps that cannot be filled from the existing data (for example,
  missing M1/M2 weights or unlogged meta-loss) listed explicitly.
