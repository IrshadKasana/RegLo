#################################################################

#           Train the optimizer model     

#################################################################
import numpy as np
import tqdm
import argparse
import os
import random
import numpy as np
import sys
from pathlib import Path
import torch
import torch.nn.functional as F
import torch.optim as optim
import copy
from torch.autograd import Variable
from torchvision import datasets, transforms
from tqdm import tqdm

from utils import *  # includes consolidated_reg_train_opt_func
from meta_optimizer import *
from model import *
from dataloaders import *
from regularizers import *
from artifact_manager import get_checkpoint_path, get_checkpoints_dir

sys.path.append(str(Path(__file__).resolve().parents[3]))

from project_paths import ensure_dir, remap_workspace_path
from experiment_config import EXPERIMENT_DIR, load_train_args, print_arguments


parser = argparse.ArgumentParser()
parser.add_argument("--regularizer", type=str, default='no_reg', help="regularization type (no_reg, sam, sam_opzee, gam, gsam)")
parser.add_argument("--profile", type=str, default=None,
                    help="scale profile from configs/dmo/profiles/<name>_train.yaml "
                         "(production | smoke): keeps epochs/steps/train_period/"
                         "curriculum in YAML instead of on the command line.")
parser.add_argument("--set", dest="overrides", nargs='*', default=[], metavar="KEY=VALUE",
                    help="Override any YAML key, e.g. --set dataset=cifar100 "
                         "optimizee_model_name=resnet8 smooth_lamb=0.5 rho=0.1 "
                         "model_name=SAM_r8_c100_LO if_smooth=True")
reg_info = parser.parse_args()


print('****----------extracting yaml files------------****')

from experiment_config import parse_overrides
overrides = parse_overrides(reg_info.overrides)
args = load_train_args(reg_info.regularizer, overrides=overrides,
                       profile=reg_info.profile)
print_arguments(args)

# print(f'printing to check {args.batch_size}')
################################################################################################################################


# Device setup: CUDA > Apple Silicon (MPS) > CPU
# NOTE: torch.backends.mps only exists from PyTorch 1.12; Agastya runs
# 1.9.0+cu111 where the bare call raises AttributeError. Guard required.
HAS_MPS = hasattr(torch.backends, "mps") and torch.backends.mps.is_available()

if torch.cuda.is_available():
    device = torch.device("cuda:0")
elif HAS_MPS:
    device = torch.device("mps")
else:
    device = torch.device("cpu")

assert args.optimizee_steps % args.truncated_bptt_step == 0

random.seed(args.seed)
torch.manual_seed(args.seed)
np.random.seed(args.seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(args.seed)

# refuse to run silently on CPU unless explicitly allowed (Change.MD item 25)
if device.type == 'cpu' and not bool(getattr(args, 'allow_cpu', False)):
    raise RuntimeError(
        "CUDA/MPS not available and allow_cpu is not set. The cluster env "
        "previously fell back to CPU silently (train_sam_bigcnn ran a full "
        "walltime on CPU). Set --set allow_cpu=True for local smoke runs.")

# Use professional artifact manager for checkpoints
save_root = get_checkpoints_dir(experiment="dmo")
print(f"Checkpoints will be saved to: {save_root}")


#dataloaders
train_data_loader, heldout_data_loader = data_loader(
    args.dataset, batch_size=args.batch_size, random_seed=args.seed,
    split_seed=int(getattr(args, 'split_seed', 42)))
# heldout_data_loader = meta-validation stream (Change.MD item 28)

torch.cuda.empty_cache()

_stop_state = {"reason": "max_epochs"}   # module-level: __main__ reads it


def train_optimizer():

    loss_min = 99999.9999
    n = 1

    if args.if_cl == True:
        # Curriculum ladder now comes from YAML (profiles) instead of being
        # hardcoded. THIS was why --profile smoke still ran big: with
        # if_cl=True the unroll length was taken from this list and ignored
        # optimizee_steps/profile entirely (stage 2 = 500 steps).
        # Falls back to the original ladder if the key is absent.
        N_train_steps = [int(v) for v in (getattr(args, 'curriculum_steps', None)
                                          or [100, 200, 500, 1000])]
        N_eval_steps = N_train_steps[1:]
        curriculum_idx = 0
        num_unroll_train = N_train_steps[curriculum_idx]
        num_unroll_eval = N_eval_steps[curriculum_idx]
    
    else:
        num_unroll_train = args.optimizee_steps
        num_unroll_eval = args.optimizee_steps

    # # Start recording memory snapshot history
    # start_record_memory_history()


    meta_model = initialize_model(args.dataset, args.optimizee_model_name)
    meta_model = meta_model.to(device)


    meta_optimizer = MetaOptimizerLSTM(MetaModel(meta_model), args.num_layers, args.hidden_size,
                                       preproc_factor=10.0, preproc=True,
                                       param_chunk_size=int(getattr(args, 'param_chunk_size', 0)) or None)
    meta_optimizer = meta_optimizer.to(device)

    # finetuning pretrained model
    if args.if_pretrained:
        PT_model_dir = get_checkpoint_path(args.pretrained_model_name, experiment="dmo")
        print(f"Loading pretrained model from: {PT_model_dir}")
        meta_optimizer.load_state_dict(torch.load(PT_model_dir))
        meta_optimizer.reset_lstm(keep_states=False, model=meta_model)
    
    optimizer_path = get_checkpoint_path(args.model_name, experiment="dmo")
    print(f"Current model checkpoint will be saved to: {optimizer_path}")



    
    #############_________________________

    smooth_reg_object = smooth_regularize(args.smooth_lamb, args.smooth_eps, args.smooth_num_pga_steps, meta_optimizer)
    

    improved = False
    best_meta_optimizer = None     # guarded (2026-07-31): resume can restore
                                   # improved=True without has_best; the
                                   # stage-transition eval then NameError'd.
    stale = 0                      # consecutive non-improving windows (item 107)

    # ------------------------------------------------------------------
    # 24h-walltime survival (HPC): a rolling resume checkpoint is written
    # after EVERY outer epoch-block; when the same command is re-run (the
    # default resume=True only activates if the file exists), training
    # continues exactly where the previous job stopped. Set
    # --set max_hours=23.0 so the job exits cleanly BEFORE the scheduler
    # kills it; resubmitting the identical job continues the run.
    # ------------------------------------------------------------------
    import time as _t
    resume_path = str(optimizer_path).replace('.pt', '_resume.pt')
    _wall_t0 = _t.time()
    max_hours = float(getattr(args, 'max_hours', 0) or 0)
    _wall_state = {}                 # last-block timing for the adaptive exit
    # BUGFIX (2026-07-31): _stop_state was a LOCAL here but is read in the
    # module-level __main__ traininfo block -> every run died with
    # NameError AFTER training finished, so <model>_traininfo.json was
    # never written (meta_train_flops lost, stop_reason lost, PBS rc=1).
    global _stop_state
    _stop_state["reason"] = "max_epochs"     # overwritten by earlier stops
    start_eph = 0
    if bool(getattr(args, 'resume', True)) and Path(resume_path).exists():
        rs = torch.load(resume_path, map_location=device)
        meta_optimizer.load_state_dict(rs['model_state'])
        import flops_utils as _fu_res
        _fu_res.reset_counters(rs.get('meta_updates', 0),
                               rs.get('eval_updates', 0))
        start_eph = int(rs['eph']) + 1
        n = int(rs['n']); loss_min = float(rs['loss_min'])
        improved = bool(rs['improved'])
        stale = int(rs.get('stale', 0))
        if args.if_cl and rs.get('curriculum_idx', -1) >= 0:
            curriculum_idx = int(rs['curriculum_idx'])
            num_unroll_train = int(rs['num_unroll_train'])
            num_unroll_eval = int(rs['num_unroll_eval'])
            if rs.get('has_best') and Path(optimizer_path).exists():
                best_meta_optimizer = copy.deepcopy(meta_optimizer)
                best_meta_optimizer.load_state_dict(
                    torch.load(optimizer_path, map_location=device))
        print(f"[resume] '{args.model_name}': continuing from epoch-block "
              f"{start_eph}/{args.max_epochs//args.train_period} "
              f"(loss_min={loss_min:.4f}, curriculum stage "
              f"{rs.get('curriculum_idx', '-')})")

    def _resume_tick(_eph):
        """Save the rolling resume checkpoint; True => walltime reached,
        caller must return. Called from BOTH the CL and non-CL paths."""
        import flops_utils as _fu_save
        torch.save({'model_state': meta_optimizer.state_dict(),
                    'eph': _eph, 'n': n, 'loss_min': loss_min,
                    'improved': improved,
                    'stale': stale,
                    'meta_updates': _fu_save.COUNTERS['meta_updates'],
                    'eval_updates': _fu_save.COUNTERS['eval_updates'],
                    'curriculum_idx': curriculum_idx if args.if_cl else -1,
                    'num_unroll_train': num_unroll_train,
                    'num_unroll_eval': num_unroll_eval,
                    'has_best': Path(optimizer_path).exists()}, resume_path)
        # ---- ADAPTIVE deadline (fix: two production jobs were KILLED by
        # PBS at 86400s despite max_hours=23). The old check ran only
        # BETWEEN blocks, but a curriculum-stage-3 block (train_period=20
        # meta-epochs x 1000-step unrolls) can take >1h: the 22.9h check
        # passed, the block ran past 24h, PBS killed it mid-block and the
        # block's work was lost. Now we exit early unless there is room for
        # 1.25x the LAST block's duration.
        _now = _t.time()
        _last_blk = _now - _wall_state.get("prev", _wall_t0)
        _wall_state["prev"] = _now
        _elapsed_h = (_now - _wall_t0) / 3600.0
        _needed_h = _elapsed_h + 1.25 * (_last_blk / 3600.0)
        if max_hours > 0 and _needed_h >= max_hours:
            _stop_state["reason"] = "walltime_exit"
            print(f"WALLTIME_EXIT: {_elapsed_h:.2f}h elapsed; next block "
                  f"(~{_last_blk/60:.0f} min x 1.25 margin) would overrun the "
                  f"{max_hours:.2f}h budget. Resume state -> {resume_path}. "
                  f"Resubmit the SAME command/job to continue.")
            return True
        return False

    # ---- checkpoint-overwrite guard (final design): a FRESH run must not
    # silently clobber an existing checkpoint. Resume (default) continues a
    # walltime-split run; otherwise pick a new model_name or pass
    # --set overwrite=True deliberately.
    if start_eph == 0 and Path(optimizer_path).exists() \
            and not bool(getattr(args, 'overwrite', False)):
        raise RuntimeError(
            f"Checkpoint already exists: {optimizer_path}. Refusing to start a "
            f"FRESH run over it. Either (a) resume a walltime-split run "
            f"(needs {resume_path}), (b) choose a new model_name, or "
            f"(c) pass --set overwrite=True.")

    # ---- guarantee a checkpoint exists ------------------------------
    # Bug this fixes: the weights were only ever saved inside
    # `if loss_eval < loss_min`, so a run that early-stopped BEFORE its
    # first improvement wrote traininfo.json but NO .pt — and every
    # downstream eval died with
    #   FileNotFoundError: .../checkpoints/SAMLO_cifar10_resnet8.pt
    # Must sit AFTER the overwrite guard above (otherwise the guard would
    # see the file we just wrote and refuse to start) and AFTER start_eph
    # is defined by the resume block.
    if start_eph == 0 and not Path(optimizer_path).exists():
        torch.save(meta_optimizer.state_dict(), optimizer_path)
        print(f"[init] baseline checkpoint written -> {optimizer_path} "
              f"(overwritten by the first real improvement)")

    for eph in range(start_eph, args.max_epochs//args.train_period):
        
        # Dispatch on args.regularizer (the YAML value), NOT the raw CLI
        # string: '--regularizer sam_bigcnn' previously matched nothing and
        # silently skipped training (Change.MD item 20). reg_mode='penalty'
        # routes sam/gsam/gam to the consolidated Eq. 7 objective (item 70).
        reg_name = str(getattr(args, 'regularizer', reg_info.regularizer))
        reg_mode = str(getattr(args, 'reg_mode', 'ascent'))
        if args.if_avreg:
            if reg_name in ('sam', 'gsam', 'gam') and reg_mode == 'penalty':
                consolidated_reg_train_opt_func(
                                    num_unroll_train,
                                    train_data_loader,
                                    device,
                                    meta_optimizer,
                                    smooth_reg_object,
                                    args,
                                    val_loader=heldout_data_loader)
            elif reg_name == 'sam':
                SAMoptimizer_reg_train_opt_func(
                                    num_unroll_train,
                                    train_data_loader,
                                    device,
                                    meta_optimizer,
                                    smooth_reg_object,
                                    args) #all other arguments are included in args.
            elif reg_name == 'sam_opzee':
                SAMoptimizee_reg_train_opt_func(
                                    num_unroll_train,
                                    train_data_loader,
                                    device,
                                    meta_optimizer,
                                    smooth_reg_object,
                                    args)
                    
            elif reg_name == 'gam':
                GAM_reg_train_opt_func(
                                    num_unroll_train,
                                    train_data_loader,
                                    device,
                                    meta_optimizer,
                                    smooth_reg_object,
                                    args)
            elif reg_name == 'gsam':
                GSAM_reg_train_opt_func(
                                    num_unroll_train,
                                    train_data_loader,
                                    device,
                                    meta_optimizer,
                                    smooth_reg_object,
                                    args)
            # (l2 regularizer RETIRED per final design — config in extras/deprecated)
            else:
                raise ValueError(
                    f"if_avreg=True but regularizer '{reg_name}' matches no "
                    f"training routine (valid: sam, sam_opzee, gam, gsam)."
                    f" This used to be a SILENT no-op (Change.MD item 20).")

        else:    
            train_opt_func(num_unroll_train,
                        train_data_loader,
                        device,
                        meta_optimizer,
                        args)
        
        torch.cuda.empty_cache()


        #evaluation of the optimizer
        n = n + 1

        
        loss_eval = eval_opt_func(num_unroll_eval,
                                train_data_loader,
                                device,
                                meta_optimizer,
                                args,
                                val_loader=heldout_data_loader)
        torch.cuda.empty_cache()
        
        if not args.if_cl:
            if loss_eval < loss_min:
                loss_min = loss_eval
                torch.save(meta_optimizer.state_dict(), optimizer_path)
                print('Model Saved in : ', optimizer_path)

            if _resume_tick(eph):
                return
            continue
        


        # Curriculum Learning

        if loss_eval < loss_min:
            
            loss_min = loss_eval
            improved = True

            # Save model (artifact manager ensures directory exists)
            best_meta_optimizer = copy.deepcopy(meta_optimizer)
            torch.save(meta_optimizer.state_dict(), optimizer_path)
            print(f"Best model saved to: {optimizer_path}")
            print('Model Saved in : ', optimizer_path)

        if args.if_cl == True:
            if n >= args.n_period and improved:

                print(f"\n\nCurriculum stage {curriculum_idx} completed !!\n\n\n")
                n = 1
                stale = 0          # stage cleared -> patience refunded
                improved = False   # each stage must improve on its own (item 29b)
                curriculum_idx += 1
                if curriculum_idx >= len(N_train_steps):
                    # clamp at the final stage instead of wrapping to -1 -> 0,
                    # which silently RESTARTED the curriculum (item 29a)
                    curriculum_idx = len(N_train_steps) - 1
                num_unroll_train = N_train_steps[curriculum_idx]
                # Chen-2020 protocol: evaluate on the NEXT (longer) stage
                # when one exists (item 29c)
                num_unroll_eval = N_train_steps[min(curriculum_idx + 1,
                                                    len(N_train_steps) - 1)]


                if best_meta_optimizer is None:
                    # resume edge: improved carried over but the best ckpt
                    # copy was not restorable — fall back to current weights
                    # (identical selection behaviour, no crash).
                    best_meta_optimizer = copy.deepcopy(meta_optimizer)
                #initial evaluation for next curriculum
                loss_min =  eval_opt_func(num_unroll_eval,
                                        train_data_loader,
                                        device,
                                        best_meta_optimizer,
                                        args,
                                        val_loader=heldout_data_loader)
                torch.cuda.empty_cache()


            elif n >= args.n_period and not improved:
                # ---- early-stop PATIENCE (item 107) --------------------
                # Was: a single non-improving window (n_period-1 = 2 evals)
                # killed the whole run. eval_opt_func is stochastic and
                # loss_min is re-baselined at the NEXT, LONGER unroll, so
                # the LO had to beat its own best on a HARDER task within 2
                # noisy evals — close to a coin flip. That is why the smoke
                # runs died at stages 0/1/3 and SAMLO_cifar10_resnet8 never
                # even saved a checkpoint.
                # Now: a non-improving window costs one unit of patience and
                # the stage is retried. cl_patience=1 reproduces the OLD
                # behaviour exactly, so this is opt-in via YAML.
                stale += 1
                _pat = int(getattr(args, 'cl_patience', 1) or 1)
                if stale >= _pat:
                    _stop_state["reason"] = "curriculum_early_stop"
                    print(f"No improvement with curriculum {curriculum_idx} "
                          f"after {stale}/{_pat} window(s) : STOPPING Here!")
                    break
                n = 1              # grant this stage another window
                print(f"[curriculum {curriculum_idx}] no improvement in "
                      f"window {stale}/{_pat} — retrying the stage "
                      f"(loss_min={loss_min:.4f})")

        # ---- rolling resume checkpoint + clean walltime exit (CL path) ----
        if _resume_tick(eph):
            return


    # natural completion: the resume file is no longer needed
    try:
        Path(resume_path).unlink()
        print(f"[resume] training complete — removed {resume_path}")
    except FileNotFoundError:
        pass


if __name__ == "__main__":
    import json as _json
    import time as _time

    _t0 = _time.time()
    train_optimizer()
    _elapsed = _time.time() - _t0

    # Record meta-training cost + config next to the checkpoint. This is the
    # "one-time meta-training investment" number for the efficiency section
    # of the paper, and generate_paper_plots.py reads it.
    _meta_info = {
        "model_name": args.model_name,
        "regularizer": reg_info.regularizer,
        "dataset": args.dataset,
        "optimizee_model_name": args.optimizee_model_name,
        "meta_train_wallclock_sec": _elapsed,
        # why this run ended: curriculum_early_stop | walltime_exit |
        # max_epochs — lets the report verify all arms stopped by CRITERION,
        # not by the scheduler (fact-check: 2 of 4 production arms were
        # walltime-killed, making budgets unequal).
        "stop_reason": _stop_state["reason"],
        # one-time meta-training COMPUTE (paper metric: meta-train FLOPs).
        # measured meta_update count x analytic per-update cost; survives
        # walltime resumes because the counter is checkpointed.
        "meta_train_flops": (lambda: (lambda _fu: _fu.meta_train_flops(
            _fu.COUNTERS["meta_updates"],
            initialize_model(args.dataset, args.optimizee_model_name),
            _fu.dummy_batch(args.dataset, args.batch_size),
            eval_updates=_fu.COUNTERS["eval_updates"]))(
                __import__("flops_utils")))(),
        "overrides": overrides,
        "args": {k: v for k, v in vars(args).items()
                 if isinstance(v, (int, float, str, bool, type(None)))},
    }
    _info_path = get_checkpoints_dir(experiment="dmo") / f"{args.model_name}_traininfo.json"
    with open(_info_path, "w") as fh:
        _json.dump(_meta_info, fh, indent=2)
    print(f"Meta-training took {_elapsed/3600:.2f} h. Info saved to: {_info_path}")