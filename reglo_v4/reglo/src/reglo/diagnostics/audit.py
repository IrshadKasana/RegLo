"""Stage 4 (supplementary): audit one finished optimizee run.

Reads a run folder (config.yaml, final.json, model_final.pt, model_converged.pt) and writes
``artifacts/diagnostics/<study>/<optimizee>-<dataset>/<method>[-<r>]/seed<k>/diagnostics.json``:

    hessian     λ_max (power iteration) and Tr(H) (Hutchinson)              Exp 4.1
    sharpness   max loss / surrogate gap / max ‖∇L‖ in ℓ∞ balls              paper's PoC tests
    weights     ‖θ‖₁, ‖θ‖₂, sparsity                                         Exp 4.2

each for the snapshots the experiment grid asks for ("final" = at completion, "converged" =
at convergence). Settings come from configs/experiments/stage4_analysis.yaml.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Union

import torch

from reglo.config import load_saved_config
from reglo.data.loaders import build_optimizee_data, eval_stream
from reglo.diagnostics.hessian import hessian_trace, loss_terms, top_eigenvalue
from reglo.diagnostics.norms import weight_statistics
from reglo.diagnostics.sharpness import max_grad_norm_in_ball, max_loss_in_ball
from reglo.models.optimizees import build_optimizee
from reglo.utils.logging import read_json, write_json
from reglo.utils.paths import diagnostics_dir_for
from reglo.utils.runtime import resolve_device

__all__ = ["run_audit"]

SNAPSHOTS = {"final": "model_final.pt", "converged": "model_converged.pt"}
_COMPACT = ("num_params", "num_penalised", "l1", "l2", "l1_all", "l2_all", "sparsity")


def _first(stream, count: int):
    batches = []
    for batch in stream:
        batches.append(batch)
        if len(batches) >= count:
            break
    return batches


def run_audit(run_path: Union[str, Path], spec: Mapping[str, Any], device: Optional[torch.device] = None) -> Dict[str, Any]:
    """Audit one run; ``spec`` is the ``diagnostics`` block of the Stage-4 experiment file."""
    run_path = Path(run_path).resolve()
    out_path = diagnostics_dir_for(run_path) / "diagnostics.json"
    final = read_json(run_path / "final.json")
    identity = {k: final.get(k) for k in ("study", "optimizee", "dataset", "method", "label", "regularizer", "seed")}
    if final.get("status") != "completed":
        result = {"status": "skipped", "reason": f"run status is {final.get('status')}", "run": run_path.as_posix(), **identity}
        write_json(out_path, result)
        return result

    cfg = load_saved_config(run_path / "config.yaml")
    cfg.paths.data = os.environ.get("REGLO_DATA", cfg.paths.data)  # the data may live elsewhere now
    device = torch.device(device) if device is not None else resolve_device(str(cfg.device))
    data = build_optimizee_data(cfg, device)
    batch = int(cfg.training.batch_size)
    hcfg, scfg, ncfg = spec["hessian"], spec["sharpness"], spec["norms"]
    curvature_batches = _first(eval_stream(cfg, data.train.data, batch, device), int(hcfg["num_batches"]))
    held_out_batches = _first(eval_stream(cfg, data.test.data, batch, device), int(scfg["num_batches"]))

    model = build_optimizee(cfg.optimizee).to(device)
    params = [p for p in model.parameters() if p.requires_grad]
    exclude = list(cfg.regularizer.get("exclude", ["bias", "norm"]))
    result: Dict[str, Any] = {"status": "completed", "run": run_path.as_posix(), **identity,
                              "hessian": {}, "sharpness": {}, "weights": {}}

    for name, filename in SNAPSHOTS.items():
        path = run_path / filename
        wanted = name in hcfg["snapshots"] or name in scfg["snapshots"]
        if not wanted or not path.is_file():
            continue
        model.load_state_dict(torch.load(path, map_location=device, weights_only=True))
        model.eval()
        if name in hcfg["snapshots"]:
            terms = loss_terms(model, curvature_batches)
            result["hessian"][name] = {
                **top_eigenvalue(terms, params, int(hcfg["power_iterations"])),
                **hessian_trace(terms, params, int(hcfg["hutchinson_samples"])),
            }
        if name in scfg["snapshots"]:
            terms = loss_terms(model, held_out_batches)
            options = dict(steps=int(scfg["pga_steps"]), step_ratio=float(scfg["step_ratio"]), relative=bool(scfg["relative"]))
            per_radius = {}
            for radius in scfg["radii"]:
                entry = max_loss_in_ball(terms, params, float(radius), **options)
                if scfg.get("grad_norm", False):
                    entry.update(max_grad_norm_in_ball(terms, params, float(radius), **options))
                per_radius[str(radius)] = entry
            result["sharpness"][name] = per_radius
        stats = weight_statistics(model, float(ncfg["sparsity_threshold"]), exclude)
        result["weights"][name] = {k: stats[k] for k in _COMPACT}

    write_json(out_path, result)
    return result
