from functools import reduce
from operator import mul
import copy

import torch
import torch.nn as nn
import torch.nn.functional as F


try:
    # classic API (pre-2023 releases of pytorch-hessian-eigenthings)
    from hessian_eigenthings import compute_hessian_eigenthings
except ImportError:
    compute_hessian_eigenthings = None

if compute_hessian_eigenthings is None:
    try:
        # current GitHub master exposes a new operator-based API; adapt it
        # to the classic signature so this code runs with EITHER version
        # (HPC robustness — whatever `pip install git+...` resolves to).
        from hessian_eigenthings import (HessianOperator,
                                         deflated_power_iteration,
                                         supervised_loss)

        def compute_hessian_eigenthings(model, dataloader, loss, k,
                                        **_ignored):
            import os
            # Bounded cost (paper default: 8 batches, 40 power iterations —
            # a stochastic Hessian estimate, standard practice). Override
            # via env: HESS_NUM_BATCHES / HESS_MAX_ITER.
            nb = int(os.environ.get("HESS_NUM_BATCHES", 8))
            mi = int(os.environ.get("HESS_MAX_ITER", 40))
            op = HessianOperator(model, dataloader, supervised_loss(loss),
                                 full_dataset=False, num_batches=nb)
            res = deflated_power_iteration(op, k=int(k), max_iter=mi)
            evals = getattr(res, "eigenvalues", None)
            evecs = getattr(res, "eigenvectors", None)
            if evals is None:          # tuple-style return
                evals, evecs = res
            import numpy as _np
            return _np.asarray(evals), _np.asarray(evecs)
    except ImportError:
        compute_hessian_eigenthings = None


def compute_hessian_spectrum(num_eigenvals, model, train_data_loader):
    """
    Compute Hessian eigenvalues/eigenvectors.

    If hessian_eigenthings is unavailable, skip gracefully instead
    of crashing the entire evaluation.
    """

    if compute_hessian_eigenthings is None:
        print("=" * 60)
        print("WARNING: hessian_eigenthings not installed.")
        print("Skipping Hessian spectrum computation.")
        print("=" * 60)
        return None, None

    loss_e = nn.NLLLoss()

    eigenvals, eigenvecs = compute_hessian_eigenthings(
        model,
        train_data_loader,
        loss_e,
        num_eigenvals,
    )

    return eigenvals.tolist(), eigenvecs.tolist()


def accuracy(outputs, labels):
    _, predicted = torch.max(outputs, 1)
    correct = (predicted == labels).sum().item()
    total = labels.size(0)
    return correct / total


def _resolve_device(model, device=None):
    if device is not None:
        return device
    return next(model.parameters()).device


def compute_val_loss(test_data_loader, model, device=None):
    device = _resolve_device(model, device)
    v_loss = 0
    with torch.no_grad():
        for x, y in test_data_loader:
            x, y = x.to(device), y.to(device)
            f_x = model(x)
            v_loss += F.nll_loss(f_x, y)

    return v_loss / len(test_data_loader)


def get_flat_params(model):
    params = []
    for module in model.modules():
        if len(module._parameters) != 0:
            params.append(module._parameters["weight"].view(-1))
            try:
                params.append(module._parameters["bias"].view(-1))
            except Exception:
                pass
    return torch.cat(params)


def set_flat_params(model, flat_params):
    offset = 0
    for module in model.modules():
        if len(module._parameters) != 0:
            weight_shape = module._parameters["weight"].size()
            weight_flat_size = reduce(mul, weight_shape, 1)
            module._parameters["weight"] = flat_params[
                offset : offset + weight_flat_size
            ].view(*weight_shape)
            try:
                bias_shape = module._parameters["bias"].size()
                bias_flat_size = reduce(mul, bias_shape, 1)
                module._parameters["bias"] = flat_params[
                    offset + weight_flat_size : offset + weight_flat_size + bias_flat_size
                ].view(*bias_shape)
            except Exception:
                bias_flat_size = 0
            offset += weight_flat_size + bias_flat_size

    return model


def copy_gradients(from_model, to_model):
    for from_param, to_param in zip(from_model.parameters(), to_model.parameters()):
        if from_param.grad is not None:
            if to_param.grad is None:
                to_param.grad = torch.zeros_like(to_param)
            to_param.grad.data.copy_(from_param.grad.detach().data)


def get_flat_grads(model, device=None):
    device = _resolve_device(model, device)
    flat = torch.tensor([]).to(device)
    for param in model.parameters():
        flat = torch.cat((flat, param.grad.view(-1).to(device)))

    return flat.requires_grad_()


def set_flat_grads(model, flat_grads):
    offset = 0
    for module in model.modules():
        if len(module._parameters) != 0:
            weight_shape = module._parameters["weight"].size()
            weight_flat_size = reduce(mul, weight_shape, 1)
            module._parameters["weight"].grad = flat_grads[
                offset : offset + weight_flat_size
            ].view(*weight_shape)
            try:
                bias_shape = module._parameters["bias"].size()
                bias_flat_size = reduce(mul, bias_shape, 1)
                module._parameters["bias"].grad = flat_grads[
                    offset + weight_flat_size : offset + weight_flat_size + bias_flat_size
                ].view(*bias_shape)
            except Exception:
                bias_flat_size = 0
            offset += weight_flat_size + bias_flat_size

    return model


def project_l_inf(flat_params, center_params, eps):
    return torch.min(torch.max(flat_params, center_params - eps), center_params + eps)



def _pgd_loader(train_data_loader):
    """Optional stochastic-estimate cap: PGD_NUM_BATCHES env limits how many
    batches the PGD maximisation sees (default: all). Same convention as
    HESS_NUM_BATCHES — set small for CPU smoke, leave unset for paper runs."""
    import os, itertools
    nb = int(os.environ.get("PGD_NUM_BATCHES", 0) or 0)
    if nb <= 0:
        return train_data_loader
    return list(itertools.islice(iter(train_data_loader), nb))

def max_loss_PGD(model, eps, data_loader, device, r_norm="L_inf"):
    data_loader = _pgd_loader(data_loader)
    model_p = copy.deepcopy(model)
    center_params = get_flat_params(model).detach()
    step_size = eps / 10

    actual_loss = compute_val_loss(data_loader, model_p, device)

    with torch.no_grad():
        for params in model_p.parameters():
            params.add_(torch.randn(params.size()).to(device))

    print(f"The actual test loss here is : {actual_loss}")

    flat_params = project_l_inf(get_flat_params(model_p).detach(), center_params, eps)
    perturbed_loss = [actual_loss.item()]

    for step in range(10):
        flat_params.requires_grad_()
        with torch.enable_grad():
            # ---- MEMORY FIX (exp4b/exp5 CUDA OOM) -----------------------
            # The old code summed the LOSS over the whole loader and then
            # took ONE autograd.grad, which forced every batch's graph to
            # stay alive at once: 79 CIFAR-10 test batches x VGG-16
            # activations = ~23 GB ("23.20 GiB reserved in total by
            # PyTorch" in logs/eval_exp5_*.err).
            # We instead accumulate the GRADIENT per batch. Since
            #     d/dw [ (1/N) sum_i L_i(w) ]  ==  (1/N) sum_i d/dw L_i(w)
            # this is EXACT — identical numbers, but only ONE batch's graph
            # is ever resident. actual_loss does not depend on flat_params
            # (it is computed before flat_params exists), so grad(p_loss -
            # actual_loss) == grad(p_loss).
            grad_accum = torch.zeros_like(flat_params)
            loss_sum, nb = 0.0, 0
            for x, y in data_loader:
                x, y = x.to(device), y.to(device)
                model_p = set_flat_params(model_p, flat_params)
                f_x = model_p(x)
                b_loss = F.nll_loss(f_x, y)
                g = torch.autograd.grad(b_loss, [flat_params],
                                        allow_unused=True)[0]
                if g is not None:
                    grad_accum += g.detach()
                loss_sum += float(b_loss.detach())
                nb += 1
                del f_x, b_loss, g

            nb = max(nb, 1)
            p_loss_val = loss_sum / nb
            perturbed_loss.append(p_loss_val)

            print(f"perturbed loss {step} : {perturbed_loss}")

            grads = grad_accum / nb

            if r_norm == "L_inf":
                flat_params = flat_params.detach() + step_size * torch.sign(grads.detach())
                flat_params = project_l_inf(flat_params, center_params, eps)
            elif r_norm == "euc":
                direction = grads.detach().clone()
                direction /= torch.norm(direction)
                flat_params = flat_params.detach() + step_size * direction
                norm = torch.norm(flat_params - center_params)
                if norm > eps:
                    flat_params = center_params + (flat_params - center_params) * eps / norm

    model_p = set_flat_params(model_p, flat_params.detach())
    perturbed_loss.append(compute_val_loss(data_loader, model_p, device).detach().cpu().item())

    return perturbed_loss, actual_loss.item()


def max_grad_norm_PGD(model, eps, data_loader, device, r_norm="L_inf"):
    data_loader = _pgd_loader(data_loader)
    model_p = copy.deepcopy(model)
    copy_gradients(model, model_p)
    center_params = get_flat_params(model).detach()
    step_size = eps / 10   # Alg. 2: alpha = rho/10 (Change.MD item 34)

    actual_grad_norm = torch.norm(get_flat_grads(model_p, device), p=2)

    with torch.no_grad():
        for params in model_p.parameters():
            params.add_(torch.randn(params.size()).to(device))

    print(f"The actual parameter gradient norm is : {actual_grad_norm}")

    flat_params = project_l_inf(get_flat_params(model_p).detach(), center_params, eps)
    perturb_grad_norm_list = [actual_grad_norm.item()]

    for _ in range(10):
        flat_params.requires_grad_()
        model_p = set_flat_params(model_p, flat_params)

        with torch.enable_grad():
            loss = 0
            for x, y in data_loader:
                x, y = x.to(device), y.to(device)
                loss += F.nll_loss(model_p(x), y)

            loss = loss / len(data_loader)
            model_p.zero_grad()
            for param in model_p.parameters():
                param.retain_grad()
            grad = torch.autograd.grad(loss, [flat_params], create_graph=True)[0]
            set_flat_grads(model_p, grad)

            perturb_grad_norm = torch.norm(get_flat_grads(model_p, device), p=2)
            perturb_grad_norm_list.append(perturb_grad_norm.item())

            diff = perturb_grad_norm - actual_grad_norm
            print(perturb_grad_norm, actual_grad_norm)
            grads = torch.autograd.grad(diff, [flat_params], allow_unused=True)[0]

            if r_norm == "L_inf":
                flat_params = flat_params.detach() + step_size * torch.sign(grads.detach())
                flat_params = project_l_inf(flat_params, center_params, eps)
            elif r_norm == "euc":
                direction = grads.detach().clone()
                direction /= torch.norm(direction)
                flat_params = flat_params.detach() + step_size * direction
                norm = torch.norm(flat_params - center_params)
                if norm > eps:
                    flat_params = center_params + (flat_params - center_params) * eps / norm

    print("grad_list: ", perturb_grad_norm_list)

    flat_params.requires_grad_()
    model_p = set_flat_params(model_p, flat_params)

    with torch.enable_grad():
        loss = 0
        for x, y in data_loader:
            x, y = x.to(device), y.to(device)
            loss += F.nll_loss(model_p(x), y)
        loss = loss / len(data_loader)
        grad = torch.autograd.grad(loss, [flat_params], create_graph=True)[0]
        set_flat_grads(model_p, grad)
        perturb_grad_norm = torch.norm(get_flat_grads(model_p, device), p=2)
        perturb_grad_norm_list.append(perturb_grad_norm.item())

    return perturb_grad_norm_list, actual_grad_norm.item()