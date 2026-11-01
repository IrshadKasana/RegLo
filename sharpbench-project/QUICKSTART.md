# QUICKSTART — running the experiment in VS Code

Assumes your Python environment and PyTorch are already installed. Every
command is run from the project root, in the VS Code integrated terminal
(``Ctrl+` ``), with the correct interpreter selected.

**Total time: about 5 hours wall clock on a 6 GB GPU with 4 shards.**
Steps 0–3 take fifteen minutes and tell you whether the remaining hours will
be wasted, so do not skip them.

---

## Step 0 — Open the folder, install deps

```powershell
cd path\to\sharpbench-project
pip install -r requirements.txt
```

**File → Open Folder** on `sharpbench-project` (the folder, not a file).
The `.vscode/` configs and every relative path assume it is the workspace root.

---

## Step 1 — Verify the environment

```powershell
python -m sharpbench check
```

Three things to confirm in the output:

| Line | Want to see |
|---|---|
| `cuda available` | `True` — if `False` you have the CPU-only wheel |
| `model params` | `15,910  OK` |
| `concurrent shards this card can hold` | 4 or more |

The bottom of the output is a **time estimate for your specific machine**.
If it says more than you want to spend, set `epochs: 50` in
`configs/sweep_grid.yaml` before continuing.

---

## Step 2 — Download MNIST

```powershell
python -m sharpbench fetch
```

~11 MB, one time. Everything after this runs offline.

---

## Step 3 — Smoke test each arm

```powershell
python -m sharpbench train --regularizer none --epochs 3
python -m sharpbench train --regularizer sam  --rho 0.05 --epochs 3
python -m sharpbench train --regularizer gsam --rho 0.05 --alpha 0.1 --epochs 3
python -m sharpbench train --regularizer gam  --epochs 3
```

**The one check that matters most:** `fwd_bwd_passes ÷ steps` must be

| arm | ratio |
|---|---|
| `none` | 1× |
| `sam`, `gsam` | 2× |
| `gam` | 4× |

If SAM reports 1×, the regularizer is not being applied and every downstream
number is meaningless. This catches the failure in three seconds instead of
after a four-hour sweep.

Also sanity-check `best_val_acc` is around 0.90–0.93 even after 3 epochs.

---

## Step 4 — Validate the whole pipeline in 5 minutes

Before committing hours, run the tiny config end to end. This exercises every
stage — sweep, analyze, winner selection, stage 2, plots — on a grid small
enough to finish while you watch.

```powershell
python -m sharpbench sweep   --config configs/quick.yaml --out results/quick.jsonl
python -m sharpbench analyze --results "results/quick.jsonl" --emit-winners configs/winners_quick.yaml
python -m sharpbench sweep   --config configs/quick.yaml --stage2 configs/winners_quick.yaml --out results/quick_s2.jsonl
python -m sharpbench plot    --results "results/quick.jsonl" --outdir figures/quick
```

Open `figures/quick/01_rho_sensitivity.png`. The *numbers* are meaningless at
4 epochs — you are only confirming that every stage runs and produces output.

> **Quote the wildcards.** PowerShell does not expand `*` for external
> commands. The tool expands globs internally, but the quotes stop PowerShell
> mangling the string first.

---

## Step 5 — Preview the real grid

```powershell
python -m sharpbench sweep --config configs/sweep_grid.yaml --stage1 --dry-run --out results/stage1_0.jsonl
```

Prints the run count (~297) and the first 25 config keys. Read them. Confirm
`adaptive=False` for the `sam` arm and `adaptive=True` for `asam`, and that the
rho values match what you intended.

---

## Step 6 — Stage 1: hyperparameter search

Four parallel shards, each in its own window:

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\run_all.ps1 -Shards 4 -Stage 1
```

Or a single process if you prefer:

```powershell
python -m sharpbench sweep --config configs/sweep_grid.yaml --stage1 --out results/stage1_0.jsonl
```

Progress looks like:

```
[42/75] sam    lr=0.001  seed=1  -> val 93.12 test 93.04 (118s, eta 65m)
```

**It is resumable.** Close a window, lose power, kill it to use the GPU — just
re-run the same command and it skips everything already written. You lose at
most the single run that was in flight.

If shards fight over memory, cap them: `--memory-fraction 0.2`.

---

## Step 7 — Select the winners

```powershell
python -m sharpbench analyze --results "results/stage1_*.jsonl" --emit-winners configs/winners.yaml --top-k 5
```

Read three things before moving on:

1. **Boundary warnings.** If you see
   `best rho=0.5 is the LARGEST value in the grid`, the optimum is outside your
   search range. Widen the grid in `configs/sweep_grid.yaml` and re-run step 6 —
   it will only run the newly added points.
2. **The top-5 list per arm.** If the top few rho values are within one
   standard deviation of each other, rho is not well identified. That is worth
   knowing, and worth saying in the write-up.
3. **ASAM's winner** should land in `0.5–2.0`. If it picked `0.1`, the adaptive
   flag may not be reaching the optimizer.

---

## Step 8 — Stage 2: winners on all 10 seeds

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\run_all.ps1 -Shards 4 -Stage 2
```

Only ~50 runs. This is what produces the reported table — stage 1 was
selection, stage 2 is measurement.

---

## Step 9 — Final table and figures

```powershell
python -m sharpbench analyze --results "results/stage2_*.jsonl" `
       --latex tables/main.tex --csv tables/main.csv

python -m sharpbench plot --results "results/stage1_*.jsonl" "results/stage2_*.jsonl" `
       --outdir figures
```

Note the plots take **both** stages: stage 1 has the full rho grid the
sensitivity curves need; stage 2 has the 10-seed statistics.

Outputs:

```
tables/main.tex     paste into the paper
tables/main.csv     for your own analysis
figures/01..07      PNG (300 dpi) + PDF (vector)
```

---

## Step 10 — The companion experiments

Only if the default result is too small to resolve (see README §4). Same
sequence, different config:

```powershell
# overparameterized: hidden=200
python -m sharpbench sweep   --config configs/wide.yaml --stage1 --out results/wide_0.jsonl
python -m sharpbench analyze --results "results/wide_*.jsonl" --emit-winners configs/winners_wide.yaml

# 20% label noise -- the strongest stress test for flat-minima claims
python -m sharpbench sweep   --config configs/label_noise.yaml --stage1 --out results/noise_0.jsonl
python -m sharpbench analyze --results "results/noise_*.jsonl" --emit-winners configs/winners_noise.yaml
```

---

## Debugging in VS Code

`.vscode/launch.json` has ready configs. **Run and Debug** (`Ctrl+Shift+D`),
pick one, `F5`:

- *check environment*
- *train: SAM / GAM / GSAM (short)*
- *sweep: dry run*
- *sweep: quick config*
- *analyze / plot*

Worth doing once: set a breakpoint inside `SAM.step` in
`sharpbench/optimizers.py` and step through the two-pass ascent-descent. If
someone asks you to defend the implementation, you will have actually watched
it run.

---

## Troubleshooting

| Symptom | Cause | Fix |
|---|---|---|
| `ModuleNotFoundError: torch` in terminal, but pip says installed | VS Code terminal on the wrong interpreter | `Ctrl+Shift+P` → Python: Select Interpreter, then open a **new** terminal |
| `cuda available: False` | CPU-only wheel | `pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121` |
| `.ps1 cannot be loaded` | Execution policy | `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned` |
| `no result files matched` | Wrong directory, or unquoted wildcard | `cd` to project root; quote the pattern |
| `ModuleNotFoundError: sharpbench` | Not in project root | `cd` to the folder containing `sharpbench/` |
| SAM's `fwd_bwd_passes` == `steps` | Regularizer not applied | Check `--regularizer` spelling; step 3 |
| All arms give identical accuracy | Same as above | Same fix |
| `CUDA out of memory` | Too many shards | Lower `-Shards`, or add `--memory-fraction 0.2` |
| Sweep much slower than `check` predicted | Another GPU process, or antivirus scanning `results/` | `nvidia-smi`; exclude the folder from Defender |
| Plots skip `03_training_curves` | History not saved | Set `keep_history: true` in the config |
| Plots skip `06_gsam_heatmap` | GSAM grid is 1-D | Needs both rho and alpha varied |

---

## Command reference

```
python -m sharpbench check    [--device auto|cpu|cuda:0] [--hidden 20]
python -m sharpbench fetch    [--data-root ./data]
python -m sharpbench train    --regularizer {none,sam,gam,gsam} [--rho R] [--alpha A]
                              [--adaptive] [--epochs N] [--seed S] [--lr LR]
                              [--label-noise F] [--m-sharpness M] [--measure-hessian]
python -m sharpbench sweep    --config C --out O [--stage1] [--stage2 W]
                              [--shard i/n] [--arms sam gam] [--dry-run]
                              [--memory-fraction F] [--device D]
python -m sharpbench analyze  --results P [P ...] [--control adamw] [--top-k K]
                              [--emit-winners W] [--latex T] [--csv T]
python -m sharpbench plot     --results P [P ...] [--outdir D] [--control adamw]
```
