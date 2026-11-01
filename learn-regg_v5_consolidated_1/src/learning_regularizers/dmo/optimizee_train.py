import tqdm
import argparse
import os
import random
import sys
from pathlib import Path
import torch
import torch.nn.functional as F
import torch.optim as optim
import copy
from torch.autograd import Variable
from torchvision import datasets, transforms
from tqdm import tqdm
import json
import time
from utils import *
from meta_optimizer import *
from model import *
from dataloaders import *
from analysis import (
    accuracy,
    compute_hessian_spectrum,
    copy_gradients,
    max_grad_norm_PGD,
    max_loss_PGD,
)
from artifact_manager import get_checkpoint_path, get_result_file_path, get_timestamped_result_file, get_results_dir

from datetime import datetime

sys.path.append(str(Path(__file__).resolve().parents[3]))

from project_paths import ensure_dir, remap_workspace_path
from experiment_config import EXPERIMENT_DIR, load_optimizee_args, print_arguments

# @ #
ENABLE_MAX_GRAD_NORM = False


parser = argparse.ArgumentParser()
parser.add_argument("--regularizer", type=str, default='no_reg', help="regularization type (sam, gsam, gam)")
parser.add_argument("--profile", type=str, default=None,
                    help="scale profile from configs/dmo/profiles/<name>_eval.yaml "
                         "(production | smoke): keeps epochs/steps/n_training/"
                         "patience/hessian in YAML instead of on the command line.")
parser.add_argument('--base_optimizer', type=str, default='sgd', help="use sgd or adam for normal optimization")
parser.add_argument('--if_normal_optim', action= 'store_true', help = "if need to train using a normal optimizer")
parser.add_argument('--saved_model', type=str, default='SAMLO_mnist_mlp', help='name of the saved LO checkpoint (without .pt); default matches the design-v2 naming {METHOD}LO_{dataset}_{arch}')
parser.add_argument('--dataset', type = str, default= 'mnist', help = 'dataset for the optimizee training (mnist/fmnist/cifar10/cifar100)')
parser.add_argument('--optimizee_model', type = str, default= 'mlp', help='optimizee architecture: mlp | resnet8 | resnet18 (design v2 active set)')
parser.add_argument('--lo_train_dataset', type=str, default='',
                    help='dataset the LOADED LO was meta-trained on (for the cross-dataset '
                         'generalization matrix; e.g. mnist). Stored in the result JSON.')
parser.add_argument('--lo_train_arch', type=str, default='',
                    help="architecture the LO was meta-trained on (bookkeeping "
                         "for the cross-architecture Experiments 7/8)")
parser.add_argument('--eval_tag', type=str, default='',
                    help='free-form tag stored in the JSON + filename (e.g. sweep id).')
parser.add_argument('--param_chunk_size', type=int, default=0,
                    help='If >0, apply the coordinate-wise LSTM in chunks of this many '
                         'parameters. Use e.g. 262144 for ResNet-18-scale (cuDNN-safe; exact)')
parser.add_argument('--augment', action='store_true',
                    help='CIFAR standard augmentation (RandomCrop+Flip) for big-model runs.')
parser.add_argument("--set", dest="overrides", nargs='*', default=[], metavar="KEY=VALUE",
                    help="Override any YAML key, e.g. --set optimizee_epochs=3000 "
                         "batch_size=128 n_training=10 calc_hessian=True")

reg_info = parser.parse_args()


print('****----------extracting yaml files------------****')

from experiment_config import parse_overrides
overrides = parse_overrides(reg_info.overrides)
args = load_optimizee_args(reg_info, overrides=overrides)
print_arguments(args)

# Device setup: CUDA > Apple Silicon (MPS) > CPU
# NOTE: torch.backends.mps only exists from PyTorch 1.12. Agastya runs
# 1.9.0+cu111, where calling torch.backends.mps.is_available() directly
# raises AttributeError — hence the hasattr guard. (User's fix, kept.)
HAS_MPS = hasattr(torch.backends, "mps") and torch.backends.mps.is_available()

if torch.cuda.is_available():
    device = torch.device("cuda:0")
elif HAS_MPS:
    device = torch.device("mps")
else:
    device = torch.device("cpu")
print(f"CUDA available: {torch.cuda.is_available()}, "
      f"MPS available: {HAS_MPS}, Device: {device}")

# Use professional artifact manager for results
save_root = get_result_file_path(".", experiment="dmo", result_type="json").parent
print(f"Results will be saved to: {save_root}")

random.seed(args.seed)
torch.manual_seed(args.seed)



#defining optimizee
dataset = args.dataset
model_name = args.optimizee_model

meta_model = initialize_model(dataset, model_name)
meta_model = meta_model.to(device)


# optimizer checkpoint read
if not args.if_normal_optim:
    path = get_checkpoint_path(args.saved_model, experiment="dmo")
    print(f"Loading learned optimizer checkpoint from: {path}")
    # load the optimizer
    num_layers = args.num_layers
    hidden_size = args.hidden_size
    meta_optimizer = MetaOptimizerLSTM(MetaModel(meta_model), num_layers, hidden_size, preproc_factor = 10.0, preproc =True)
    # map_location added 2026-07-31: checkpoints are saved on cuda; a bare
    # torch.load crashes on CPU-only debug nodes and pins GPU0 on multi-GPU.
    meta_optimizer.load_state_dict(torch.load(path, map_location=device))
    meta_optimizer.reset_lstm(keep_states=False, model=meta_model)
    meta_optimizer = meta_optimizer.to(device)

    # Chunked coordinate processing for big optimizees (resnet18/vgg16)
    if getattr(args, 'param_chunk_size', 0):
        meta_optimizer.param_chunk_size = int(args.param_chunk_size)
        print(f"LSTM will process parameters in chunks of {meta_optimizer.param_chunk_size}")
    



def evaluate_model(model, data_loader):
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for x, y in data_loader:
            x, y = x.to(device), y.to(device)
            outputs = model(x)
            total_loss += F.nll_loss(outputs, y).item()
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == y).sum().item()
            total += y.size(0)

    model.train()
    avg_loss = total_loss / max(len(data_loader), 1)
    avg_acc = correct / total if total else 0.0
    return avg_loss, avg_acc

#eigen values calculation at the convergence points

def _eps_list(args):
    """PGD radius set for hyper_eps sweeps. Configurable via the eps_list
    key (optimizee_eval.yaml / --set); default = user-specified radii."""
    return [float(e) for e in getattr(args, 'eps_list', [0.001, 0.002, 0.005, 0.01])]


def train_optimizee(epochs, train_data_loader, test_data_loader,model,meta_optimizer, val_loader=None):
    loss_arr = []
    loss_val_arr = []
    acc_val_arr = []
    acc_tr_arr = []
    loss_val_track, acc_val_track = [], []
    w_loss, w_acc = float('nan'), float('nan')      # per-step train (mini-batch) accuracy -> generalization gap
    step_times = []      # per-step wall-clock (pure optimization cost, excl. eval)

    train_iter = iter(train_data_loader)
    test_iter = iter(test_data_loader)

    diff = 1e-3
    max_loss_dict = {}
    max_grad_norm_dict = {}
    stage_properties = {}   # {'at_convergence': {...}, 'at_completion': {...}}
    patience = args.patience
    min_tr_loss = 999999
    early_stop_Flag = False
    eigen_values = {}
    best_model = copy.deepcopy(model)
    best_iter = 0
    tolerance = 0

    for j in tqdm(range(epochs), "epoch: "):
        try:
            x, y = next(train_iter)
        except StopIteration:
            train_iter = iter(train_data_loader)
            x, y = next(train_iter)
        x, y = x.to(device), y.to(device)

        # ---- timed optimization step (fwd + bwd + LO update; excludes eval) ----
        if device.type == 'cuda':
            torch.cuda.synchronize()
        step_t0 = time.time()

        # First we need to compute the gradients wrt model parameters at a point
        f_x = model(x)
        loss = F.nll_loss(f_x, y)

        model.zero_grad()
        loss.backward()

        train_loss_iter = loss.item()

        loss_arr.append(loss.item())

        #Update the values of the parameters using the meta optimizer, meta_model is the new model params
        with torch.no_grad():
            meta_optimizer.meta_update(model)

        if device.type == 'cuda':
            torch.cuda.synchronize()
        step_times.append(time.time() - step_t0)

        # train mini-batch accuracy (cheap; enables acc-based generalization gap)
        with torch.no_grad():
            acc_tr_arr.append((f_x.argmax(dim=1) == y).float().mean().item())
        loss_val_track.append(float(w_loss))
        acc_val_track.append(float(w_acc))

        #####validating......
        _ev = int(getattr(args, 'eval_every', 1))
        if j % _ev == 0 or j == epochs - 1:
            v_loss, v_acc = evaluate_model(model, test_data_loader)
            if val_loader is not None:
                w_loss, w_acc = evaluate_model(model, val_loader)
        # else: reuse previous v_loss / v_acc (recorded every step for
        # curve continuity; full evaluation only every eval_every steps —
        # fixes the evaluation-dominated wallclock, Change.MD item 51)
        loss_val_arr.append(v_loss)
        acc_val_arr.append(v_acc)

        # ---- per-epoch sharpness tracker (Experiment 12 / fig 19):
        # sharpness_every=k (0=off) measures PGD max-loss at each radius in
        # eps_list every k epochs — Accuracy/Sharpness/Generalization
        # together over training. Cost respects PGD_NUM_BATCHES.
        _sv = int(getattr(args, 'sharpness_every', 0) or 0)
        if _sv > 0 and (j % _sv == 0 or j == epochs - 1):
            try:
                _sharp_track
            except NameError:
                _sharp_track = {"every": _sv, "steps": [], "eps": {
                    str(_e): [] for _e in
                    (list(getattr(args, 'eps_list', None) or [args.eps]))}}
            _sharp_track["steps"].append(int(j))
            for _e_s, _vals in _sharp_track["eps"].items():
                _r = max_loss_PGD(model, float(_e_s),
                                  train_data_loader, device)
                # returns (perturbed_loss_profile, actual_loss); track the max
                _vals.append(float(max(_r[0]) if isinstance(_r[0], (list, tuple))
                                   else _r[0]))

        print("training loss:", train_loss_iter, "validation loss:", v_loss)


        #early stopping
        check = min_tr_loss - (train_loss_iter)
        
        
        if (check > diff):
            min_tr_loss = train_loss_iter
            best_model = copy.deepcopy(model)
            
            #required for GAM Property
            copy_gradients(model, best_model)
            
            best_iter = j
            if not early_stop_Flag:
                tolerance = 0

        else:
            if not early_stop_Flag:
                tolerance += 1
                
                if(tolerance <= patience):
                    continue
                
                #True if tolerance has gone beyond patience

                if args.calc_hessian:
                    eigen_vals, eigen_vecs = compute_hessian_spectrum(args.num_ev_hess, best_model, train_data_loader)
                    eigen_values[best_iter] = eigen_vals
                    print(eigen_vals)

                if args.hyper_eps:
                    early_stop_Flag = True
                    max_loss_nghb_dict = {}
                    max_grad_norm_nghb_dict = {}

                    for e in _eps_list(args):
                        print(f"Computing max loss and max gradient in {e}-nghb of the converged point")
                        pgd_loss_profile = max_loss_PGD(best_model, e, train_data_loader, device)
                        # pgd_grad_norm_profile = max_grad_norm_PGD(best_model, e, train_data_loader, device)
                        # Disabled due to CUDA OOM on large models
                        pgd_grad_norm_profile = None
                        max_loss_nghb_dict[e] = pgd_loss_profile
                        max_grad_norm_nghb_dict[e] = pgd_grad_norm_profile

                    max_loss_dict[best_iter] = max_loss_nghb_dict
                    max_grad_norm_dict[best_iter] = max_grad_norm_nghb_dict
                    stage_properties['at_convergence'] = {
                        'iter': best_iter,
                        'max_loss': max_loss_nghb_dict,
                        'max_grad_norm': max_grad_norm_nghb_dict}
                else:
                    early_stop_Flag = True
                    # max_loss_nghb_list = max_loss_PGD(best_model, args.eps, train_data_loader, device)
                    # Disabled to avoid CUDA OOM during evaluation
                    max_grad_norm_nghb_list = None
                    max_grad_norm_nghb_list = (max_grad_norm_PGD(best_model, args.eps, train_data_loader, device) if getattr(args, 'calc_grad_norm', True) else None)
                    
                    max_loss_dict[best_iter] = max_loss_nghb_list
                    max_grad_norm_dict[best_iter] = max_grad_norm_nghb_list
                    stage_properties['at_convergence'] = {
                        'iter': best_iter,
                        'max_loss': max_loss_nghb_list,
                        'max_grad_norm': max_grad_norm_nghb_list}
            

            # do not break the loop--- donot want the trainign to stop and keep training untill the last step
            # but want to keep track of when the model first converged to
            # will have two outputs if the model converged in between and at the very end
            # only have one output if it never converged in between
            
            

    # ---- 'at completion' stage (Alg. 2 fixed-iteration): measured at the
    # FINAL model by default (completion_at='final'), or at best_model for
    # the legacy behaviour (completion_at='best'). Previously both stages
    # wrote max_loss_dict[best_iter] and the completion write OVERWROTE the
    # convergence one whenever best_iter had not moved (Change.MD items
    # 37/50) — stage_properties keeps them separate.
    completion_model = model if str(getattr(args, 'completion_at', 'final')) == 'final' else best_model
    completion_iter = epochs - 1 if str(getattr(args, 'completion_at', 'final')) == 'final' else best_iter
    if args.calc_hessian:            
        eigen_vals, eigen_vecs = compute_hessian_spectrum(args.num_ev_hess, best_model, train_data_loader)
        eigen_values[best_iter] = eigen_vals
        print(eigen_vals)
        
    if args.hyper_eps:
        max_loss_nghb_dict = {}
        max_grad_norm_nghb_dict = {}

        for e in _eps_list(args):
            print(f"Computing max loss and max gradient in {e}-nghb (completion stage)")
            pgd_loss_profile = max_loss_PGD(completion_model, e, train_data_loader, device)
            # Disabled to avoid CUDA OOM during evaluation
            pgd_grad_norm_profile = None
            #pgd_grad_norm_profile = max_grad_norm_PGD(completion_model, e, train_data_loader, device)

            max_loss_nghb_dict[e] = pgd_loss_profile
            max_grad_norm_nghb_dict[e] = pgd_grad_norm_profile

        max_loss_dict[completion_iter] = max_loss_nghb_dict
        max_grad_norm_dict[completion_iter] = max_grad_norm_nghb_dict
        stage_properties['at_completion'] = {
            'iter': completion_iter,
            'max_loss': max_loss_nghb_dict,
            'max_grad_norm': max_grad_norm_nghb_dict}

    else:
        max_loss_nghb_list = max_loss_PGD(completion_model, args.eps, train_data_loader, device)
        # Disabled to avoid CUDA OOM during evaluation
        max_grad_norm_nghb_list = None
        #max_grad_norm_nghb_list = (max_grad_norm_PGD(completion_model, args.eps, train_data_loader, device) if getattr(args, 'calc_grad_norm', True) else None)
        
        max_loss_dict[completion_iter] = max_loss_nghb_list
        max_grad_norm_dict[completion_iter] = max_grad_norm_nghb_list
        stage_properties['at_completion'] = {
            'iter': completion_iter,
            'max_loss': max_loss_nghb_list,
            'max_grad_norm': max_grad_norm_nghb_list}


    try:
        _st = _sharp_track
    except NameError:
        _st = None
    extras = {
        "acc_train": acc_tr_arr,
        "loss_val": loss_val_track,
        "acc_val": acc_val_track,
        "step_times": step_times,
        "stage_properties": stage_properties,   # two-stage keys (items 37/50)
        "eigen_values": eigen_values,   # was computed but never saved before
        "sharpness_track": _st,
    }
    return loss_arr,loss_val_arr,acc_val_arr, max_loss_dict, max_grad_norm_dict, extras






#########################################################################################################
##############################  For NORMAL Optimizers #############################################
#########################################################################################################



#########################################################################################################
########### Some required auxillary functions for GAM and GSAM ###################################
#########################################################################################################


def OPZEE_get_optim_and_schedulers(model, args): # optzee_model, data_laoder, 
    if str(args.base_optimizer).lower() == 'sgd':
        base_optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum,
                                  weight_decay=args.weight_decay)
    elif args.base_optimizer == 'adam':
        base_optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    elif args.base_optimizer == 'AdamW':
        base_optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    else:
        raise NotImplementedError

    

    # optional grad_rho_scheduler and grad_norm_rho_scheduler
    if args.regularizer=='gam':
        lr_scheduler = optim.lr_scheduler.CosineAnnealingLR(base_optimizer, T_max=args.optimizee_epochs)
        grad_rho_scheduler = ProportionScheduler(pytorch_lr_scheduler=lr_scheduler, max_lr=args.lr, min_lr=0.0,
                                                max_value=args.grad_rho_max, min_value=args.grad_rho_min)

        grad_norm_rho_scheduler = ProportionScheduler(pytorch_lr_scheduler=lr_scheduler, max_lr=args.lr, min_lr=0.0,
                                                  max_value=args.grad_norm_rho_max, min_value=args.grad_norm_rho_min)

    
        optimizer = GAM(params=model.parameters(), base_optimizer=base_optimizer, model=model,
                        grad_rho_scheduler=grad_rho_scheduler, grad_norm_rho_scheduler=grad_norm_rho_scheduler,
                        adaptive=args.adaptive, args=args)
        
    
    elif args.regularizer=='gsam':
        scheduler = CosineScheduler(T_max=args.optimizee_epochs, max_value=args.lr, min_value=0.0, optimizer=base_optimizer)
        rho_scheduler = ProportionScheduler(pytorch_lr_scheduler=scheduler, max_lr=args.lr, min_lr=0.0, 
                                            max_value=args.rho_max, min_value=args.rho_min)

        optimizer = GSAM(params=model.parameters(), base_optimizer=base_optimizer, model=model, gsam_alpha=args.alpha, 
                         rho_scheduler=rho_scheduler, adaptive=args.adaptive, args=args)


    return optimizer #, base_optimizer, lr_scheduler, grad_rho_scheduler, grad_norm_rho_scheduler

#########################################################################################################
#########################################################################################################







def fit_w_normal_optimizer(epochs, train_data_loader, test_data_loader,model, val_loader=None):
    # ---- regularizer FAMILY normalisation (correctness fix) -------------
    # At meta-test, args.regularizer is the CONFIG NAME the CLI was given
    # (e.g. 'adamw_gam_baseline') because the CLI wins the YAML merge — NOT
    # the YAML's `regularizer: gam`. The dispatch below compared that raw
    # string against 'gam'/'gsam', so every non-SAM benchmark fell through
    # to the plain-optimizer `else` branch: AdamW+GAM and AdamW+GSAM would
    # have silently run as BARE AdamW while being labelled as regularized.
    # get_optim_and_schedulers() reads args.regularizer too, so normalise
    # the attribute itself (family), keeping the config name for the record.
    _raw = str(getattr(args, 'regularizer', '') or '').lower()
    if 'gsam' in _raw:
        _fam = 'gsam'
    elif 'gam' in _raw:
        _fam = 'gam'
    elif 'sam' in _raw:
        _fam = 'sam'
    else:
        _fam = 'no_reg'
    if _fam != args.regularizer:
        print(f"[baseline] regularizer config '{args.regularizer}' "
              f"-> family '{_fam}'")
        args.regularizer_config = args.regularizer
        args.regularizer = _fam

    
    
    
    #if normal optimization
    if args.if_normal_optim:
        if str(args.base_optimizer).lower() == 'sgd':
            optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum,
                                  weight_decay=args.weight_decay)
        elif args.base_optimizer == 'adam':
            optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

        elif args.base_optimizer == 'AdamW':
            optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

        else:
            raise NotImplementedError
        ##################################################################################################
        if args.regularizer == 'sam':
            sam_optimizer = SAM(model.parameters(),args = args)
            scheduler = StepLR(sam_optimizer, args.lr, args.optimizee_epochs)

        elif args.regularizer=='gam':
            gam_optimizer = OPZEE_get_optim_and_schedulers(model, args = args)

        elif args.regularizer=='gsam':
            gsam_optimizer = OPZEE_get_optim_and_schedulers(model, args= args)


    
    
    loss_arr = []
    loss_val_arr = []
    acc_val_arr = []
    acc_tr_arr = []
    loss_val_track, acc_val_track = [], []
    w_loss, w_acc = float('nan'), float('nan')      # per-step train (mini-batch) accuracy -> generalization gap
    step_times = []      # per-step wall-clock (pure optimization cost, excl. eval)

    train_iter = iter(train_data_loader)
    test_iter = iter(test_data_loader)

    diff = 1e-3
    max_loss_dict= {}
    stage_properties = {}   # {'at_convergence': {...}, 'at_completion': {...}}
    max_grad_norm_dict = {}
    patience = args.patience
    min_tr_loss = 999999
    early_stop_Flag = False
    eigen_values = {}
    best_model = copy.deepcopy(model)
    best_iter = 0
    tolerance = 0

    for j in tqdm(range(epochs), "epoch: "):
        try:
            x, y = next(train_iter)
        except StopIteration:
            train_iter = iter(train_data_loader)
            x, y = next(train_iter)
        x, y = x.to(device), y.to(device)

        # ---- timed optimization step (includes SAM/GAM/GSAM extra passes) ----
        if device.type == 'cuda':
            torch.cuda.synchronize()
        step_t0 = time.time()

        # First we need to compute the gradients wrt model parameters at a point
        f_x = model(x)
        loss_fn = F.nll_loss
        loss = loss_fn(f_x, y)

        model.zero_grad()
        loss.backward()

        train_loss_iter = loss.item()

        loss_arr.append(loss.item())

        #Update the values of the parameters using an optimizer
        
        # Optimization step for different Regularization
        if args.regularizer == 'sam':
            sam_optimizer.first_step(zero_grad=True)
            loss_fn(model(x), y).backward()  # make sure to do a full forward pass
            sam_optimizer.second_step()
            scheduler(j)

        elif args.regularizer=='gam':
            gam_optimizer.set_closure(loss_fn, x, y)
            gam_optimizer.update_rho_t()
            gam_optimizer.step()
        
        elif args.regularizer=='gsam':
            gsam_optimizer.set_closure(loss_fn, x, y)
            gsam_optimizer.update_rho_t()
            gsam_optimizer.step()
        
        else:
            optimizer.step()

        if device.type == 'cuda':
            torch.cuda.synchronize()
        step_times.append(time.time() - step_t0)

        # train mini-batch accuracy (cheap; enables acc-based generalization gap)
        with torch.no_grad():
            acc_tr_arr.append((f_x.argmax(dim=1) == y).float().mean().item())
        loss_val_track.append(float(w_loss))
        acc_val_track.append(float(w_acc))

        #####validating......
        _ev = int(getattr(args, 'eval_every', 1))
        if j % _ev == 0 or j == epochs - 1:
            v_loss, v_acc = evaluate_model(model, test_data_loader)
            if val_loader is not None:
                w_loss, w_acc = evaluate_model(model, val_loader)
        # else: reuse previous v_loss / v_acc (recorded every step for
        # curve continuity; full evaluation only every eval_every steps —
        # fixes the evaluation-dominated wallclock, Change.MD item 51)
        loss_val_arr.append(v_loss)
        acc_val_arr.append(v_acc)

        # ---- per-epoch sharpness tracker (Experiment 12 / fig 19):
        # sharpness_every=k (0=off) measures PGD max-loss at each radius in
        # eps_list every k epochs — Accuracy/Sharpness/Generalization
        # together over training. Cost respects PGD_NUM_BATCHES.
        _sv = int(getattr(args, 'sharpness_every', 0) or 0)
        if _sv > 0 and (j % _sv == 0 or j == epochs - 1):
            try:
                _sharp_track
            except NameError:
                _sharp_track = {"every": _sv, "steps": [], "eps": {
                    str(_e): [] for _e in
                    (list(getattr(args, 'eps_list', None) or [args.eps]))}}
            _sharp_track["steps"].append(int(j))
            for _e_s, _vals in _sharp_track["eps"].items():
                _r = max_loss_PGD(model, float(_e_s),
                                  train_data_loader, device)
                # returns (perturbed_loss_profile, actual_loss); track the max
                _vals.append(float(max(_r[0]) if isinstance(_r[0], (list, tuple))
                                   else _r[0]))

        #early stopping
        check = min_tr_loss - (train_loss_iter)

        print(train_loss_iter)


        if (check > diff):
            min_tr_loss = train_loss_iter
            best_model = copy.deepcopy(model)

            #required for GAM Property
            copy_gradients(model, best_model)

            
            best_iter = j
            if not early_stop_Flag:
                tolerance = 0

        else:
            if not early_stop_Flag:
                tolerance += 1
                
                if(tolerance <= patience):
                    continue
                
                #True if tolerance has gone beyond patience

                if args.calc_hessian:
                    eigen_vals, eigen_vecs = compute_hessian_spectrum(args.num_ev_hess, best_model, train_data_loader)
                    eigen_values[best_iter] = eigen_vals
                    print(eigen_vals)

                if args.hyper_eps:
                    early_stop_Flag = True

                    max_loss_nghb_dict = {}
                    max_grad_norm_nghb_dict = {}

                    for e in _eps_list(args):
                        print(f"Computing max loss and max gradient in {e}-nghb of the converged point")
                        pgd_loss_profile = max_loss_PGD(best_model, e, train_data_loader, device)
                        # Disabled to avoid CUDA OOM during evaluation
                        pgd_grad_norm_profile = None
                        #pgd_grad_norm_profile = max_grad_norm_PGD(best_model, e, train_data_loader, device)

                        max_loss_nghb_dict[e] = pgd_loss_profile
                        max_grad_norm_nghb_dict[e] = pgd_grad_norm_profile

                    max_loss_dict[best_iter] = max_loss_nghb_dict
                    max_grad_norm_dict[best_iter] = max_grad_norm_nghb_dict
                    stage_properties['at_convergence'] = {
                        'iter': best_iter,
                        'max_loss': max_loss_nghb_dict,
                        'max_grad_norm': max_grad_norm_nghb_dict}

                else:
                    early_stop_Flag = True
                    max_loss_nghb_list = max_loss_PGD(best_model, args.eps, train_data_loader, device)
                    # Disabled to avoid CUDA OOM during evaluation
                    max_grad_norm_nghb_list = None
                    #max_grad_norm_nghb_list = (max_grad_norm_PGD(best_model, args.eps, train_data_loader, device) if getattr(args, 'calc_grad_norm', True) else None)
                    
                    max_loss_dict[best_iter] = max_loss_nghb_list
                    max_grad_norm_dict[best_iter] = max_grad_norm_nghb_list
                    stage_properties['at_convergence'] = {
                        'iter': best_iter,
                        'max_loss': max_loss_nghb_list,
                        'max_grad_norm': max_grad_norm_nghb_list}

            # do not break the loop--- donot want the training to stop(even though converged) and keep training untill the last step
            # but want to keep track of when the model first converged to
            # will have two outputs if the model converged in between and at the very end
            # only have one output if it never converged in between
            
            

    # 'at completion' stage (Alg. 2 fixed-iteration): FINAL model by default,
    # best_model for legacy behaviour (completion_at='best').
    completion_model = model if str(getattr(args, 'completion_at', 'final')) == 'final' else best_model
    completion_iter = epochs - 1 if str(getattr(args, 'completion_at', 'final')) == 'final' else best_iter
    if args.calc_hessian:
        eigen_vals, eigen_vecs = compute_hessian_spectrum(args.num_ev_hess, best_model, train_data_loader)
        eigen_values[best_iter] = eigen_vals
        print(eigen_vals)
        
    if args.hyper_eps:
        max_loss_nghb_dict = {}
        max_grad_norm_nghb_dict = {}

        for e in _eps_list(args):
            print(f"Computing max loss and max gradient in {e}-nghb (completion stage)")
            pgd_loss_profile = max_loss_PGD(completion_model, e, train_data_loader, device)
            # Disabled to avoid CUDA OOM during evaluation
            pgd_grad_norm_profile = None
            #pgd_grad_norm_profile = max_grad_norm_PGD(completion_model, e, train_data_loader, device)

            max_loss_nghb_dict[e] = pgd_loss_profile
            max_grad_norm_nghb_dict[e] = pgd_grad_norm_profile

        max_loss_dict[completion_iter] = max_loss_nghb_dict
        max_grad_norm_dict[completion_iter] = max_grad_norm_nghb_dict
        stage_properties['at_completion'] = {
            'iter': completion_iter,
            'max_loss': max_loss_nghb_dict,
            'max_grad_norm': max_grad_norm_nghb_dict}

    else:
        max_loss_nghb_list = max_loss_PGD(completion_model, args.eps, train_data_loader, device)
        #max_grad_norm_nghb_list = (max_grad_norm_PGD(completion_model, args.eps, train_data_loader, device) if getattr(args, 'calc_grad_norm', True) else None)
        # Disabled to avoid CUDA OOM during evaluation
        max_grad_norm_nghb_list = None
        max_loss_dict[completion_iter] = max_loss_nghb_list
        max_grad_norm_dict[completion_iter] = max_grad_norm_nghb_list
        stage_properties['at_completion'] = {
            'iter': completion_iter,
            'max_loss': max_loss_nghb_list,
            'max_grad_norm': max_grad_norm_nghb_list}


    try:
        _st = _sharp_track
    except NameError:
        _st = None
    extras = {
        "acc_train": acc_tr_arr,
        "loss_val": loss_val_track,
        "acc_val": acc_val_track,
        "step_times": step_times,
        "stage_properties": stage_properties,   # two-stage keys (items 37/50)
        "eigen_values": eigen_values,   # was computed but never saved before
        "sharpness_track": _st,
    }
    return loss_arr,loss_val_arr,acc_val_arr, max_loss_dict, max_grad_norm_dict, extras




########################### Training ###############################
#epochs = args.n_training

train_data_loader, val_data_loader, test_data_loader = data_loader_optimizee(
    dataset, batch_size=args.batch_size, random_seed=args.seed,
    augment=getattr(args, 'augment', False),
    split_seed=int(getattr(args, 'split_seed', 42)),
    val_fraction=float(getattr(args, 'val_fraction', 0.1)),
    pacs_holdout=getattr(args, 'pacs_holdout', None),
    pacs_root_dir=getattr(args, 'pacs_root', None),
    pacs_img_size=int(getattr(args, 'pacs_img_size', 224) or 224),
    pacs_num_workers=int(getattr(args, 'pacs_num_workers', 2)))
train_iter = iter(train_data_loader)
instances_loss_tr = []
instances_loss_ts = []
instances_acc_ts = []
instances_acc_tr = []          # NEW: per-step train accuracy
instances_max_loss_nghb = []
instances_max_grad_norm = []
instances_eigen_val = []       # NEW: Hessian spectra (were discarded before)
instances_step_times = []      # NEW: efficiency (per-step wall clock)
instances_gen_gap = []         # NEW: final full-train vs test gap
instances_stage_props = []     # NEW: {'at_convergence','at_completion'} per instance (items 37/50)
instances_loss_val = []        # NEW: per-step validation loss (uniform metric set)
instances_acc_val = []         # NEW: per-step validation accuracy
instances_c10c = []            # NEW: CIFAR-10-C robustness (Experiment 6)
instances_sharpness_track = []  # NEW: per-epoch sharpness at each radius (fig 19)
instances_wallclock = []       # NEW: total wall clock per instance
instances_peak_mem_mb = []     # NEW: peak GPU memory per instance

instance_seeds = []

def _short_run_name(kind, ident, dataset, model_name, tag):
    """Compact, Windows-MAX_PATH-safe result filename base (user request:
    old names overflowed the 260-char limit). ALL consumers (plots v2,
    selector, report) select by METADATA, never by filename. Dedupes
    ident-vs-tag (sweep checkpoints embed the tag) and hash-guards
    anything still long."""
    import hashlib
    tag = tag or ""
    ident = ident or ""
    if tag and tag in ident:              # SWEEP_<tag>_<ds>_<arch> case
        ident = ""
    suffix = f"_{dataset}_{model_name}"
    if ident.endswith(suffix):            # SAMLO_mnist_mlp -> SAMLO
        ident = ident[: -len(suffix)]
    base = "_".join(b for b in ("ev", kind, ident, tag, dataset, model_name) if b)
    if len(base) > 80:
        base = base[:72] + "_" + hashlib.md5(base.encode()).hexdigest()[:6]
    return base


# ----------------------------------------------------------------------
# 24h-walltime survival: after each completed instance the accumulated
# per-instance lists are dumped to a partial file; re-running the SAME
# command resumes at the first unfinished instance (per-instance seeds
# are deterministic: seed + i). Set --set max_hours=23.0 on HPC so the
# job exits cleanly before the scheduler kills it.
# ----------------------------------------------------------------------
import time as _wt
_wall_t0 = _wt.time()
_max_hours = float(getattr(args, 'max_hours', 0) or 0)
_PERSIST = {
    "instances_loss_tr": instances_loss_tr,
    "instances_loss_ts": instances_loss_ts,
    "instances_acc_ts": instances_acc_ts,
    "instances_acc_tr": instances_acc_tr,
    "instances_max_loss_nghb": instances_max_loss_nghb,
    "instances_max_grad_norm": instances_max_grad_norm,
    "instances_eigen_val": instances_eigen_val,
    "instances_step_times": instances_step_times,
    "instances_gen_gap": instances_gen_gap,
    "instances_stage_props": instances_stage_props,
    "instances_loss_val": instances_loss_val,
    "instances_acc_val": instances_acc_val,
    "instances_c10c": instances_c10c,
    "instances_sharpness_track": instances_sharpness_track,
    "instances_wallclock": instances_wallclock,
    "instances_peak_mem_mb": instances_peak_mem_mb,
    "instance_seeds": instance_seeds,
}
_run_id = (getattr(args, 'eval_tag', '') or
           (f"bench_{args.regularizer}" if args.if_normal_optim
            else args.saved_model))
_partial_path = save_root / (
    _short_run_name("pt", "", dataset, model_name, _run_id)
    + f"_s{args.seed}.json")
_start_i = 0
# ---- multi-day chain guard (2026-07-31): if this exact run already wrote
# its FINAL timestamped JSON (partial gone = natural completion), a chained
# resubmission must NOT silently redo the whole evaluation and write a
# duplicate. resume=True (default) skips; --set overwrite=True forces.
_chain_ok = bool(getattr(args, 'resume', True)) and \
    not bool(getattr(args, 'overwrite', False)) and not _partial_path.exists()
if _chain_ok:
    import re as _re
    _kind0 = "bench" if args.if_normal_optim else "lo"
    _ident0 = getattr(args, 'model_name', '') if args.if_normal_optim else args.saved_model
    _base0 = _short_run_name(_kind0, _ident0, dataset, model_name,
                             getattr(args, 'eval_tag', '') or '')
    _pat0 = _re.compile(_re.escape(_base0) + "_[0-9]{6}-[0-9]{6}[.]json$")
    # NOTE save_root == results/ (pathlib collapses the "." in
    # get_result_file_path); FINAL jsons live in results/json/ — glob THERE
    # (this was why the guard never fired; found 2026-07-31).
    _final_dir0 = get_results_dir(experiment="dmo", result_type="json")
    _done = sorted(q.name for q in _final_dir0.glob(_base0 + "_*.json")
                   if _pat0.match(q.name))
    if _done:
        print(f"[resume] final result already present for this run "
              f"({_done[-1]}) - nothing to do. Re-run with --set "
              f"overwrite=True to force a fresh evaluation.")
        sys.exit(0)
if bool(getattr(args, 'resume', True)) and _partial_path.exists():
    with open(_partial_path) as _fh:
        _saved = json.load(_fh)
    for _k, _lst in _PERSIST.items():
        _lst[:] = _saved.get(_k, [])
    _start_i = int(_saved.get("completed", 0))
    print(f"[resume] {_partial_path.name}: {_start_i}/{args.n_training} "
          f"instances already done — continuing at instance {_start_i}")

for i in range (_start_i, args.n_training):
    print(f"instance {i} ---------------------------------- \n")
    if device.type == 'cuda':
        torch.cuda.reset_peak_memory_stats()
    inst_t0 = time.time()

    # per-instance seed (matched across Model 1 / Model 2 for PAIRED tests)
    inst_seed = int(args.seed) + i
    torch.manual_seed(inst_seed)
    np.random.seed(inst_seed)
    instance_seeds.append(inst_seed)

    model = initialize_model(dataset, model_name)
    model = model.to(device)
    if not args.if_normal_optim:
        # fresh hidden state per instance: without this, instance i>0
        # started from instance i-1's final LSTM state -> runs were not
        # i.i.d. and std-devs were contaminated (Change.MD item 33)
        meta_optimizer.reset_lstm(keep_states=False, model=model)
    if args.if_normal_optim:
        print(f"optimizing using {args.base_optimizer} with {args.regularizer}")
        loss_train_arr, loss_test_arr, acc_test_arr, max_loss_nghb, max_grad_norm_nghb, extras = fit_w_normal_optimizer(args.optimizee_epochs, train_data_loader, test_data_loader,model, val_loader=val_data_loader)
        
    else:   
        print(f"optimizing using {args.saved_model}") 
        loss_train_arr, loss_test_arr, acc_test_arr, max_loss_nghb, max_grad_norm_nghb, extras  = train_optimizee(args.optimizee_epochs, train_data_loader, test_data_loader,model,meta_optimizer, val_loader=val_data_loader)

    inst_wallclock = time.time() - inst_t0

    # ---- Generalization gap at the final model (full train split vs test split) ----
    final_train_loss, final_train_acc = evaluate_model(model, train_data_loader)
    final_test_loss, final_test_acc = evaluate_model(model, test_data_loader)
    gen_gap = {
        "final_train_loss": final_train_loss,
        "final_train_acc": final_train_acc,
        "final_test_loss": final_test_loss,
        "final_test_acc": final_test_acc,
        "acc_gap": final_train_acc - final_test_acc,
        "loss_gap": final_test_loss - final_train_loss,
    }
    print(f"instance {i}: acc gap = {gen_gap['acc_gap']:.4f}, loss gap = {gen_gap['loss_gap']:.4f}")

    instances_loss_tr.append(loss_train_arr)
    instances_loss_ts.append(loss_test_arr)
    instances_acc_ts.append(acc_test_arr)
    instances_acc_tr.append(extras["acc_train"])
    instances_max_loss_nghb.append(max_loss_nghb)
    instances_max_grad_norm.append(max_grad_norm_nghb)
    instances_eigen_val.append(extras["eigen_values"])
    instances_step_times.append(extras["step_times"])
    instances_gen_gap.append(gen_gap)
    instances_stage_props.append(extras.get("stage_properties", {}))
    instances_sharpness_track.append(extras.get('sharpness_track'))
    instances_loss_val.append(extras.get("loss_val", []))
    instances_acc_val.append(extras.get("acc_val", []))

    # ---- CIFAR-10-C robustness of the FINAL model (Experiment 6) ----
    if bool(getattr(args, 'eval_cifar10c', False)) and dataset == 'cifar10':
        from dataloaders import (cifar10c_available, cifar10c_loader,
                                 cifar10c_corruptions_present, cifar10c_root)
        _c_root = getattr(args, 'cifar10c_root', None) or None
        if cifar10c_available(_c_root):
            sev = [int(x) for x in getattr(args, 'cifar10c_severities', [1, 2, 3, 4, 5])]
            per_corr = {}
            for corr in cifar10c_corruptions_present(_c_root):
                accs = []
                for sv in sev:
                    _, a_c = evaluate_model(model, cifar10c_loader(
                        corr, sv, batch_size=args.batch_size, root=_c_root))
                    accs.append(float(a_c))
                per_corr[corr] = {"per_severity_acc": accs,
                                  "mean_acc": float(np.mean(accs))}
            mca = float(np.mean([v["mean_acc"] for v in per_corr.values()])) if per_corr else None
            instances_c10c.append({"mean_corruption_accuracy": mca,
                                   "severities": sev,
                                   "per_corruption": per_corr})
            print(f"instance {i}: CIFAR-10-C mean corruption accuracy = {mca}")
        else:
            print(f"[warn] eval_cifar10c=True but CIFAR-10-C not found at "
                  f"{cifar10c_root(_c_root)} — skipping (download from zenodo 2535967)")
            instances_c10c.append(None)
    instances_wallclock.append(inst_wallclock)
    if device.type == 'cuda':
        instances_peak_mem_mb.append(torch.cuda.max_memory_allocated() / (1024 ** 2))

    # ---- rolling partial dump + clean walltime exit ----
    with open(_partial_path, "w") as _fh:
        json.dump({"completed": i + 1, **_PERSIST}, _fh)
    if _max_hours > 0 and (_wt.time() - _wall_t0) / 3600.0 >= _max_hours \
            and i + 1 < args.n_training:
        print(f"WALLTIME_EXIT: {_max_hours:.2f}h budget reached after "
              f"instance {i}. Partial -> {_partial_path.name}. Resubmit the "
              f"SAME command/job to continue.")
        sys.exit(0)

# ---- Efficiency summary across instances ----
import numpy as _np
_all_steps = [t for inst in instances_step_times for t in inst]
efficiency = {
    "mean_step_time_ms": float(_np.mean(_all_steps) * 1e3) if _all_steps else None,
    "std_step_time_ms": float(_np.std(_all_steps) * 1e3) if _all_steps else None,
    "throughput_steps_per_sec": float(1.0 / _np.mean(_all_steps)) if _all_steps else None,
    "throughput_samples_per_sec": float(args.batch_size / _np.mean(_all_steps)) if _all_steps else None,
    # ---- META-TRAINING cost (paper metric: meta-train FLOPs). Read from
    # the LO checkpoint's traininfo; AdamW baselines do no meta-training
    # and honestly report 0. ----
    "meta_train_flops": (lambda: (
        0 if args.if_normal_optim else
        ((lambda _p: (json.load(open(_p)).get("meta_train_flops") or {})
                     .get("meta_train_flops") if _p.exists() else None)(
            Path(str(get_checkpoint_path(args.saved_model, experiment="dmo"))
                 .replace('.pt', '_traininfo.json'))))))(),
    # ---- FLOPs IN THE EXPERIMENT (user request): every eval JSON now
    # carries the per-step FLOPs accounting of ITS OWN method, from the
    # shared flops_utils module (same math as benchmark_compute.py). ----
    **(lambda: (lambda fu, tdl: fu.step_flops(
            initialize_model(dataset, model_name).to(device),
            next(iter(tdl))[0].to(device),
            bool(args.if_normal_optim), args.regularizer))(
        __import__('flops_utils'), train_data_loader))(),
    "total_wallclock_sec_per_instance": instances_wallclock,
    "peak_gpu_mem_mb": instances_peak_mem_mb,
    "n_params_optimizee": int(sum(p.numel() for p in model.parameters())),
}

# ---- Metadata: lets plotting scripts group results reliably (no filename globbing) ----
def _read_traininfo():
    if args.if_normal_optim:
        return None
    _p = Path(str(get_checkpoint_path(args.saved_model, experiment="dmo"))
              .replace('.pt', '_traininfo.json'))
    try:
        return json.load(open(_p)) if _p.exists() else None
    except Exception:
        return None

def _git_commit():
    try:
        import subprocess as _sp
        return _sp.run(["git", "rev-parse", "--short", "HEAD"],
                       capture_output=True, text=True, cwd=str(Path(__file__).resolve().parents[3]),
                       timeout=5).stdout.strip() or None
    except Exception:
        return None

_ti = _read_traininfo()

metadata = {
    "dataset": dataset,
    "optimizee_model": model_name,
    "regularizer": args.regularizer,
    # true training config of the LO (items 39 + reviewer item 3): read
    # once from the checkpoint's traininfo — regularizer AND the lambda/rho
    # values the LO was actually meta-trained with.
    "lo_regularizer": (_ti or {}).get('regularizer'),
    "lo_hparams": ({k: (_ti.get('args') or {}).get(k) for k in
                    ('reg_mode', 'lambda_reg', 'smooth_lamb', 'if_smooth',
                     'smooth_eps', 'rho', 'lr', 'base_optimizer')}
                   if _ti else None),
    "if_normal_optim": bool(args.if_normal_optim),
    "base_optimizer": args.base_optimizer if args.if_normal_optim else None,
    "saved_model": None if args.if_normal_optim else args.saved_model,
    "lo_train_dataset": getattr(args, 'lo_train_dataset', '') or None,
    "lo_train_arch": getattr(args, 'lo_train_arch', '') or None,
    # ---- explicit cross-generalization schema (plotting-friendly) ----
    "train_dataset": (getattr(args, 'lo_train_dataset', '') or dataset),
    "test_dataset": dataset,
    "train_architecture": (getattr(args, 'lo_train_arch', '') or model_name)
                          if not args.if_normal_optim else model_name,
    "test_architecture": model_name,
    "benchmark": (f"{args.base_optimizer}+{args.regularizer}"
                  if args.if_normal_optim else None),
    "rho": float(getattr(args, 'rho', 0.0)) if getattr(args, 'rho', None) is not None else None,
    "experiment_name": getattr(args, 'experiment_name', '') or None,
    "experiment_group": getattr(args, 'experiment_group', '') or None,
    "eval_tag": getattr(args, 'eval_tag', '') or None,
    "pacs_holdout": getattr(args, 'pacs_holdout', None),
    "pacs_img_size": getattr(args, 'pacs_img_size', None),
    "seed": args.seed,
    "instance_seeds": instance_seeds,   # per-instance seeds for paired tests
    "n_training": args.n_training,
    "optimizee_epochs": args.optimizee_epochs,
    "batch_size": args.batch_size,
    "augment": bool(getattr(args, 'augment', False)),
    "timestamp": datetime.now().isoformat(),
    "git_commit": _git_commit(),
    "resumed_from_instance": _start_i,
    "args": {k: v for k, v in vars(args).items()
             if isinstance(v, (int, float, str, bool, type(None)))},
}

json_dict = {
    "metadata": metadata,
    "instances_loss_tr": instances_loss_tr,
    "instances_loss_ts": instances_loss_ts,
    "instances_acc_ts": instances_acc_ts,
    "instances_acc_tr": instances_acc_tr,
    "max_loss_nghb_list": instances_max_loss_nghb,
    "max_grad_norm_nghb_list": instances_max_grad_norm,
    "eigen_values": instances_eigen_val,
    "generalization_gap": instances_gen_gap,
    "stage_properties_list": instances_stage_props,
    "instances_sharpness_track": instances_sharpness_track,   # two-stage records (items 37/50)
    "instances_loss_val": instances_loss_val,
    "instances_acc_val": instances_acc_val,
    "cifar10c_list": instances_c10c,
    "efficiency": efficiency,
}

current_date = datetime.now().strftime("%d-%m-%Y")




if args.hyper_eps:
    e = '_hyper_eps'
else:
    e = ''.join(str(args.eps).split('.'))
    
if args.if_normal_optim:
    optim_ = args.base_optimizer
    if args.regularizer == 'sam':
        optim_ += '_SAM_'
    elif args.regularizer=='gam':
        optim_ += '_GAM_'
    elif args.regularizer=='gsam':
        optim_ += '_GSAM_'
    else:
        optim_ += '_noREG_'

    # Use artifact manager for consistent result storage
    tag = f"_{args.eval_tag}" if getattr(args, 'eval_tag', '') else ''
    base_filename = _short_run_name("bench", args.model_name, dataset,
                                    model_name, getattr(args, 'eval_tag', '') or '')
    file_name = get_timestamped_result_file(base_filename, experiment="dmo", result_type="json")

else:
    optim_ = '_meta_optim_'
    meta_opt_name = args.saved_model
    
    # Use artifact manager for consistent result storage
    tag = f"_{args.eval_tag}" if getattr(args, 'eval_tag', '') else ''
    xds = f"_from-{args.lo_train_dataset}" if getattr(args, 'lo_train_dataset', '') else ''
    base_filename = _short_run_name("lo", args.saved_model, dataset,
                                    model_name, getattr(args, 'eval_tag', '') or '')
    file_name = get_timestamped_result_file(base_filename, experiment="dmo", result_type="json")

with open(file_name, "w") as json_file:
    json.dump(json_dict, json_file)
print(f"Results saved to: {file_name}")
try:
    _partial_path.unlink()
    print("[resume] run complete — partial file removed")
except FileNotFoundError:
    pass



































################################################################################################################
