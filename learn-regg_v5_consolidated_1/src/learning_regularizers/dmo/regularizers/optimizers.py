import torch
import torch.optim as optim
from functools import reduce
import torch.nn.functional as F
from operator import mul
import copy
import torch
import sys

from regularizers.schedulers import *
# from torch.distributed import ReduceOp
import contextlib



def get_flat_params(model):
    params = []
    for name, module in model.named_modules():
        if len(module._parameters) != 0:
            params.append(module._parameters['weight'].view(-1))
            try:
                params.append(module._parameters['bias'].view(-1))
            except:
                pass
    return torch.cat(params)



def set_flat_params(model, flat_params):
    # Restore original shapes
    offset = 0
    for module in model.modules():
        if len(module._parameters) != 0:
            weight_shape = module._parameters['weight'].size()
            weight_flat_size = reduce(mul, weight_shape, 1)
            module._parameters['weight'] = flat_params[
                                            offset:offset + weight_flat_size].view(*weight_shape)
            try:
                bias_shape = module._parameters['bias'].size()
                bias_flat_size = reduce(mul, bias_shape, 1)
                module._parameters['bias'] = flat_params[
                                                offset + weight_flat_size:offset + weight_flat_size + bias_flat_size].view(
                    *bias_shape)
            except:
                bias_flat_size = 0
            offset += weight_flat_size + bias_flat_size
    return model




class SAM(torch.optim.Optimizer):
    def __init__(self, params, args, **kwargs):
        assert args.rho >= 0.0, f"Invalid rho, should be non-negative: {args.rho}"

        defaults = dict(rho=args.rho, adaptive=args.adaptive, **kwargs)
        super(SAM, self).__init__(params, defaults)

        if args.base_optimizer == 'sgd':
            self.base_optimizer = optim.SGD(
                self.param_groups,
                lr=args.lr,
                momentum=args.momentum,
                weight_decay=args.weight_decay,
                **kwargs
            )

        elif args.base_optimizer == 'adam':
            self.base_optimizer = optim.Adam(
                self.param_groups,
                lr=args.lr,
                weight_decay=args.weight_decay,
                **kwargs
            )

        elif args.base_optimizer == 'AdamW':
            self.base_optimizer = optim.AdamW(
                self.param_groups,
                lr=args.lr,
                weight_decay=args.weight_decay,
                **kwargs
            )

        self.param_groups = self.base_optimizer.param_groups
        self.defaults.update(self.base_optimizer.defaults)
        self.args = args

    @torch.no_grad()
    def first_step(self, zero_grad=False):
        grad_norm = self._grad_norm()

        for group in self.param_groups:
            scale = group["rho"] / (grad_norm + 1e-12)

            for p in group["params"]:
                if p.grad is None:
                    continue

                self.state[p]["old_p"] = p.data.clone()

                e_w = (
                    (torch.pow(p, 2) if group["adaptive"] else 1.0)
                    * p.grad
                    * scale.to(p)
                )

                p.add_(e_w)

        if zero_grad:
            self.zero_grad()

    @torch.no_grad()
    def second_step(self, zero_grad=False):
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue

                p.data = self.state[p]["old_p"]

        if self.args.SAMonOptimizer:
            self.base_optimizer.step()

        if zero_grad:
            self.zero_grad()

    @torch.no_grad()
    def step(self, closure=None):
        assert closure is not None, (
            "Sharpness Aware Minimization requires closure"
        )

        closure = torch.enable_grad()(closure)

        self.first_step(zero_grad=True)
        closure()
        self.second_step()

    @torch.no_grad()
    def _grad_norm(self):
        # official ASAM norm: || (|p| if adaptive else 1) * grad ||_2
        # (the |p| factor was missing -> mis-normalised perturbation when
        #  adaptive=True; Change.MD item 31)
        shared_device = self.param_groups[0]["params"][0].device
        adaptive = bool(getattr(self.args, "adaptive", False))
        norms = []
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is not None:
                    g = (torch.abs(p) if adaptive else 1.0) * p.grad
                    norms.append(g.norm(p=2).to(shared_device))
        if len(norms) == 0:
            return torch.tensor(0.).to(shared_device)
        return torch.norm(torch.stack(norms), p=2)

    def load_state_dict(self, state_dict):
        super().load_state_dict(state_dict)
        self.base_optimizer.param_groups = self.param_groups



class smooth_regularize:

    '''
    lamb: regularization coefficient
    eps: perturbation length
    num_pga_steps: number of projected gradient ascent steps
    optimizer_model : the learned optimizer

    '''
    def __init__(self, lamb, eps, smooth_num_pga_steps,optimizer_model):
        self.lamb = lamb
        self.eps = eps
        self.smooth_num_pga_steps = smooth_num_pga_steps
        self.optimizer = optimizer_model

    def run(self, flat_grads, normalize_state=False):
        """L_smooth per Eq. 6 / Alg. 1 step (b), computed at the CALLER's
        gradient state.

        flat_grads : 1-D tensor of the optimizee's leaf gradients at the
                     final unroll step (pass utils.get_flat_grads(model)).
                     The old signature read .grad off the functional
                     meta-model, whose params are non-leaf (grad is always
                     None) -> guaranteed AttributeError (Change.MD item 22).
        normalize_state : optional RMS pre-normalisation (old behaviour);
                     OFF by default so the perturbed state s_t matches the
                     raw-gradient input that meta_update actually feeds the
                     LSTM.
        PGA step size alpha = eps / N_PGA (Alg. 1 documented default; the
        old code hardcoded eps/5). Distance = squared L2 norm (sum), per
        Eq. 6; any scale change folds into lambda_smooth via the sweep.
        """
        flat_grads = flat_grads.detach().view(-1, 1)
        if normalize_state:
            second_moment = (flat_grads ** 2).mean() + 1e-5
            flat_grads = flat_grads / torch.sqrt(second_moment)
        inputs = flat_grads.view(-1, 1).unsqueeze(1)

        state_adv = inputs + 0.01 * torch.randn(inputs.shape, device=inputs.device)
        step_size = self.eps / max(int(self.smooth_num_pga_steps), 1)
        state_snapshot = self.optimizer.snapshot_state()

        for _ in range(self.smooth_num_pga_steps):
            state_adv.requires_grad_()
            with torch.enable_grad():
                perturbed_update = self.optimizer(state_adv)
                self.optimizer.restore_state(state_snapshot)
                update = self.optimizer(inputs)
                self.optimizer.restore_state(state_snapshot)
                distance = ((perturbed_update - update) ** 2).sum()
                grad = torch.autograd.grad(distance, [state_adv], allow_unused=True)[0]
            state_adv = state_adv.detach() + step_size * torch.sign(grad.detach())
            state_adv = torch.min(torch.max(state_adv, inputs - self.eps), inputs + self.eps)

        perturbed_update = self.optimizer(state_adv.detach())
        self.optimizer.restore_state(state_snapshot)
        update = self.optimizer(inputs)
        self.optimizer.restore_state(state_snapshot)
        distance = ((update - perturbed_update) ** 2).sum()
        return distance




class GAM(torch.optim.Optimizer):
    def __init__(self, params, base_optimizer, model,  grad_rho_scheduler=None,
                 grad_norm_rho_scheduler=None, adaptive = False, perturb_eps=1e-12, args=None,
                 grad_reduce='mean', **kwargs):
        defaults = dict(adaptive=args.adaptive, **kwargs)
        super(GAM, self).__init__(params, defaults)
        self.grad_rho_scheduler = grad_rho_scheduler
        self.grad_norm_rho_scheduler = grad_norm_rho_scheduler
        self.perturb_eps = perturb_eps
        self.model = model
        self.base_optimizer = base_optimizer
        self.param_groups = self.base_optimizer.param_groups
        self.args = args

        # self.get_grad_reduce(grad_reduce)
        self.update_rho_t()
        



    @torch.no_grad()
    def update_rho_t(self):
        if self.grad_rho_scheduler is not None:
            self.grad_rho = self.grad_rho_scheduler.step()
        if self.grad_norm_rho_scheduler is not None:
            self.grad_norm_rho = self.grad_norm_rho_scheduler.step()

    @torch.no_grad()
    def perturb_weights(self, perturb_idx: int):
        grad_norm = self._grad_norm(weight_adaptive=self.args.adaptive)
        scale = self.grad_rho / (grad_norm + self.perturb_eps)

        if perturb_idx == 0:
            for group in self.param_groups:
                for p in group["params"]:
                    if p.grad is None:
                        continue
                    #step 1 
                    self.state[p]["g_0"] = p.grad.data.clone()
                    e_w = p.grad * scale.to(p)
                    if self.args.adaptive:
                        e_w *= torch.pow(p, 2)
                    #step 2
                    p.add_(e_w)
                    #theta^_2
                    self.state[p]['e_w_0'] = e_w

        elif perturb_idx == 1:
            for group in self.param_groups:
                for p in group["params"]:
                    if p.grad is None:
                        continue
                    #step 6
                    self.state[p]["g_2"] = p.grad.data.clone()
                    e_w = p.grad * scale.to(p)
                    if self.args.adaptive:
                        e_w *= torch.pow(p, 2)
                    #step 7
                    p.add_(e_w)
                    #theta^_3 = theta^2 + e_w
                    self.state[p]['e_w_1_2'] += e_w

        else:
            raise ValueError('"perturb_idx" should be one of [0, 1].')

    @torch.no_grad()
    def grad_norm_ascent(self):
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                #step 3
                self.state[p]["g_1"] = p.grad.data.clone()
                #step 4 (h_0 = g_1 - g_0)
                p.grad.data -= self.state[p]["g_0"]

        grad_norm = self._grad_norm(weight_adaptive=self.args.adaptive)
        scale = self.grad_norm_rho / (grad_norm + self.perturb_eps)

        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                e_w = p.grad * scale.to(p)  #approximate position in space with the maximum value of gradient norm
                if self.args.adaptive:
                    e_w *= torch.pow(p, 2)
                #step 5
                p.add_(e_w)
                self.state[p]['e_w_1_2'] = e_w


    @torch.no_grad()
    def unperturb(self, perturb_key: str):

        for group in self.param_groups:
            for p in group['params']:
                if perturb_key in self.state[p].keys():
                    p.data.sub_(self.state[p][perturb_key])



    ################################
    #decomposition function 
    # step 9 10 11
    ################################

    @torch.no_grad()
    def gradient_decompose(self, args=None):
        inner_prod = 0.0
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None: continue
                # update the weighted sum of grads
                
                #h_-
                self.state[p]['pro_m'] = self.state[p]['g_0'] + abs(args.grad_beta_2) * self.state[p]['g_2']
                
                #h_+
                p.grad.data = args.grad_beta_1 * self.state[p][
                    "g_1"] + args.grad_beta_3 * p.grad.data.detach().clone()
                
                #to find cosine value to find the projection
                inner_prod += torch.sum(
                    self.state[p]['pro_m'] * p.grad.data
                )

        # get norm
        new_grad_norm = self._grad_norm()
        old_grad_norm = self._grad_norm(by='pro_m')

        # get cosine
        cosine = inner_prod / (new_grad_norm * old_grad_norm + self.perturb_eps)

        # gradient decomposition
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None: continue

                #h_- ^(perp)
                vertical = self.state[p]['pro_m'] - cosine * old_grad_norm * p.grad.data / (
                        new_grad_norm + self.perturb_eps)
                p.grad.data.add_(vertical, alpha=-args.grad_gamma)


    ################################
    #Finding the gradient norm
    ################################

    @torch.no_grad()
    def _grad_norm(self, weight_adaptive: bool = False, by: str = 'grad'):
        norm = 0.0
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None: continue
                if by == 'grad':
                    g = p.grad.data
                elif by == 'pro_m':
                    g = self.state[p]['pro_m']
                # elif by == 'e_w':
                #     g = self.state[p]['e_w_0'] + self.state[p]['e_w_1_2'] + self.state[p]['e_w_2']
                elif by == 'p':
                    g = p.data
                else:
                    raise ValueError("Invalid 'by' argument in _grad_norm")

                if weight_adaptive:
                    norm += torch.sum((g * torch.abs(p.data)) ** 2)
                else:
                    norm += torch.sum(g ** 2)

        return torch.sqrt(norm)



    ################################################################
    #combining all the gradients from different GPUs of compute nodes
    ################################################################

    @torch.no_grad()
    def _sync_grad(self):
        if torch.distributed.is_initialized():  # synchronize final gardients
            for group in self.param_groups:
                for p in group['params']:
                    if p.grad is None: continue
                    if self.manual_average:
                        torch.distributed.all_reduce(p.grad, op=self.grad_reduce)
                        world_size = torch.distributed.get_world_size()
                        p.grad.div_(float(world_size))
                    else:
                        torch.distributed.all_reduce(p.grad, op=self.grad_reduce)
        return


    def maybe_no_sync(self):
        if torch.distributed.is_initialized():
            return self.model.no_sync()
        else:
            return contextlib.ExitStack()


    @torch.no_grad()
    def set_closure(self, loss_fn, inputs, targets, **kwargs):
        # create self.forward_backward_func, which is a function such that
        # self.forward_backward_func() automatically performs forward and backward passes.

        
        ###########################################
        ##### functions that updates the gradients
        ###########################################
        

        def get_grad():
            self.zero_grad()
            with torch.enable_grad():
                outputs = self.model(inputs)
                loss = loss_fn(outputs, targets, **kwargs)
            # loss_value = loss.data.clone().detach()
            loss.backward()
            # return outputs, loss_value

        self.forward_backward_func = get_grad

    @torch.no_grad()
    def LO_set_closure(self, opzee_model, dummy_model, iter_num, loss_fn, train_data_loader, batches=None, start_state=None, **kwargs):
        # create self.forward_backward_func, which is a function such that
        # self.forward_backward_func() automatically performs forward and backward passes.

        #model == meta_optimizer
        
        ###################################################
        ##### functions that updates the gradients of LO
        ###################################################

        def get_grad():
            self.zero_grad()

            for modelA, modelB in zip(dummy_model.parameters(), opzee_model.parameters()):
                modelA.data.copy_(modelB.data)

            # window-start hidden state: every geometry pass must begin from
            # the SAME state, not from the previous pass's end state
            # (Change.MD item 32d).
            if start_state is not None:
                self.model.reset_lstm(keep_states=False, model=dummy_model)
                self.model.restore_state(start_state)
            else:
                self.model.reset_lstm(keep_states=iter_num > 0, model=dummy_model)

            dev = next(dummy_model.parameters()).device
            disable_running_stats(dummy_model)   # protect BN stats (item 32)

            with torch.enable_grad():
                # cached window batches -> all passes see identical data;
                # fresh iterator only as legacy fallback (items 24, 32d).
                if batches is not None:
                    batch_iter = iter(batches)
                else:
                    batch_iter = iter(train_data_loader)
                for j in range(self.args.truncated_bptt_step):
                    try:
                        x, y = next(batch_iter)
                    except StopIteration:
                        batch_iter = iter(batches if batches is not None else train_data_loader)
                        x, y = next(batch_iter)
                    x, y = x.to(dev), y.to(dev)
                    f_x = dummy_model(x)
                    loss = loss_fn(f_x, y)
                    dummy_model.zero_grad()
                    loss.backward()
                    opzee_model_temp = self.model.meta_update(dummy_model)
                
                loss_sum = torch.zeros(1, device=dev)

                opzee_model_temp.zero_grad()
                # finding the optimizer's loss using the optimizee's state
                # for j in range(len(train_data_loader)):
                #     try:
                #         x, y = next(train_iter)
                #     except StopIteration:
                #         train_iter = iter(train_data_loader)
                #         x, y = next(train_iter)
                #     x, y = x.cuda(), y.cuda()
                f_x = opzee_model_temp(x)
                loss = loss_fn(f_x, y)
                loss_sum = loss_sum + loss

                # NOTE: the old /len(train_data_loader) scaling is removed
                # (Change.MD item 26); the meta-LR now means what it says.
                enable_running_stats(dummy_model)


                loss_sum.backward()

        self.forward_backward_func = get_grad



    def step(self, closure=None):

        if closure:
            get_grad = closure
        else:
            get_grad = self.forward_backward_func

        with self.maybe_no_sync():
            # get gradient
            # print("step 1")
            get_grad()

            # perturb weights
            # print("step 2")
            self.perturb_weights(perturb_idx=0)

            # disable running stats for second pass,
            disable_running_stats(self.model)
            # model 1
            # print("step 3")
            get_grad()
            # grad 1

            self.unperturb(perturb_key="e_w_0")
            # model 0
            # print("step 4")
            self.grad_norm_ascent()
            # model 2
            # print("step 5")
            get_grad()
            # grad 2

            self.perturb_weights(perturb_idx=1)
            # model 3
            # print("step 6")
            get_grad()
            # grad 3

            # decompose and get new update direction
            self.gradient_decompose(args=self.args)

            # unperturb
            self.unperturb(perturb_key="e_w_1_2")

        # synchronize gradients across workers
        # print("step 7")
        self._sync_grad()

        # update with new directions
        # print("step 8")
        self.base_optimizer.step()

        # enable running stats
        enable_running_stats(self.model)


    def zero_grad(self, set_to_none: bool = False):
        self.base_optimizer.zero_grad(set_to_none)

    def state_dict(self):
        return self.base_optimizer.state_dict()

    def load_state_dict(self, state_dict):
        self.base_optimizer.load_state_dict(state_dict)

    # def add_param_group(self, param_group):
    #     self.base_optimizer.add_param_group(param_group)

    def __repr__(self):
        return f'GAM({self.base_optimizer.__class__.__name__})'





############################################################
# SURROGATE GAP MINIMIZATION IMPROVES SHARPNESS-AWARE TRAINING
############################################################




class GSAM(torch.optim.Optimizer):
    def __init__(self, params, base_optimizer, model, gsam_alpha, rho_scheduler, adaptive = False, 
                 perturb_eps=1e-12, grad_reduce='mean', args=None, **kwargs):
        defaults = dict(adaptive=args.adaptive, **kwargs)
        super(GSAM, self).__init__(params, defaults)
        self.model = model
        self.base_optimizer = base_optimizer
        self.param_groups = self.base_optimizer.param_groups
        self.rho_scheduler = rho_scheduler
        self.perturb_eps = perturb_eps
        self.alpha = gsam_alpha
        self.args = args
        
        # initialize self.rho_t
        self.update_rho_t()
        

    
    @torch.no_grad()
    def update_rho_t(self):
        self.rho_t = self.rho_scheduler.step()
        return self.rho_t

    @torch.no_grad()
    def perturb_weights(self, rho=0.0):
        grad_norm = self._grad_norm( weight_adaptive = self.args.adaptive )
        for group in self.param_groups:
            scale = rho / (grad_norm + self.perturb_eps)

            for p in group["params"]:
                if p.grad is None: continue
                self.state[p]["old_g"] = p.grad.data.clone()
                e_w = p.grad * scale.to(p)
                if self.args.adaptive:
                    e_w *= torch.pow(p, 2)
                p.add_(e_w)  # climb to the local maximum "w + e(w)"
                self.state[p]['e_w'] = e_w
                
    @torch.no_grad()
    def unperturb(self):
        for group in self.param_groups:
            for p in group['params']:
                if 'e_w' in self.state[p].keys():
                    p.data.sub_(self.state[p]['e_w'])

    @torch.no_grad()
    def gradient_decompose(self, alpha=0.0):
        # calculate inner product
        inner_prod = 0.0
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None: continue
                inner_prod += torch.sum(
                    self.state[p]['old_g'] * p.grad.data
                )

        # get norm
        new_grad_norm = self._grad_norm()
        old_grad_norm = self._grad_norm(by='old_g')

        # get cosine
        cosine = inner_prod / (new_grad_norm * old_grad_norm + self.perturb_eps)

        # gradient decomposition
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None: continue
                vertical = self.state[p]['old_g'] - cosine * old_grad_norm * p.grad.data / (new_grad_norm + self.perturb_eps)
                p.grad.data.add_( vertical, alpha=-alpha)

    @torch.no_grad()
    def _sync_grad(self):
        if torch.distributed.is_initialized(): # synchronize final gardients
            for group in self.param_groups:
                for p in group['params']:
                    if p.grad is None: continue
                    if self.manual_average:
                        torch.distributed.all_reduce(p.grad, op=self.grad_reduce)
                        world_size = torch.distributed.get_world_size()
                        p.grad.div_(float(world_size))
                    else:
                        torch.distributed.all_reduce(p.grad, op=self.grad_reduce)
        return

    @torch.no_grad()
    def _grad_norm(self, by=None, weight_adaptive=False):
        #shared_device = self.param_groups[0]["params"][0].device  # put everything on the same device, in case of model parallelism
        if not by:
            norm = torch.norm(
                    torch.stack([
                        ( (torch.abs(p.data) if weight_adaptive else 1.0) *  p.grad).norm(p=2)
                        for group in self.param_groups for p in group["params"]
                        if p.grad is not None
                    ]),
                    p=2
               )
        else:
            norm = torch.norm(
                torch.stack([
                    ( (torch.abs(p.data) if weight_adaptive else 1.0) * self.state[p][by]).norm(p=2)
                    for group in self.param_groups for p in group["params"]
                    if p.grad is not None
                ]),
                p=2
            )
        return norm

    def load_state_dict(self, state_dict):
        super().load_state_dict(state_dict)
        self.base_optimizer.param_groups = self.param_groups
        
    def maybe_no_sync(self):
        if torch.distributed.is_initialized():
            return self.model.no_sync()
        else:
            return contextlib.ExitStack()

    @torch.no_grad()
    def set_closure(self, loss_fn, inputs, targets, **kwargs):
        # create self.forward_backward_func, which is a function such that
        # self.forward_backward_func() automatically performs forward and backward passes.
        # This function does not take any arguments, and the inputs and targets data
        # should be pre-set in the definition of partial-function

        def get_grad():
            self.base_optimizer.zero_grad()
            with torch.enable_grad():
                outputs = self.model(inputs)
                loss = loss_fn(outputs, targets, **kwargs)
            loss_value = loss.data.clone().detach()
            loss.backward()
            return outputs, loss_value

        self.forward_backward_func = get_grad


    @torch.no_grad()
    def LO_set_closure(self, opzee_model, dummy_model, iter_num, loss_fn, train_data_loader, batches=None, start_state=None, **kwargs):
        # create self.forward_backward_func, which is a function such that
        # self.forward_backward_func() automatically performs forward and backward passes.

        #model here means meta_optimizer
        
        ###########################################
        ##### functions that updates the gradients of LO
        ###########################################
        

        def get_grad():
            self.zero_grad()

            for modelA, modelB in zip(dummy_model.parameters(), opzee_model.parameters()):
                modelA.data.copy_(modelB.data)

            # window-start hidden state: every geometry pass must begin from
            # the SAME state, not from the previous pass's end state
            # (Change.MD item 32d).
            if start_state is not None:
                self.model.reset_lstm(keep_states=False, model=dummy_model)
                self.model.restore_state(start_state)
            else:
                self.model.reset_lstm(keep_states=iter_num > 0, model=dummy_model)

            dev = next(dummy_model.parameters()).device
            disable_running_stats(dummy_model)   # protect BN stats (item 32)

            with torch.enable_grad():
                # cached window batches -> all passes see identical data;
                # fresh iterator only as legacy fallback (items 24, 32d).
                if batches is not None:
                    batch_iter = iter(batches)
                else:
                    batch_iter = iter(train_data_loader)
                for j in range(self.args.truncated_bptt_step):
                    try:
                        x, y = next(batch_iter)
                    except StopIteration:
                        batch_iter = iter(batches if batches is not None else train_data_loader)
                        x, y = next(batch_iter)
                    x, y = x.to(dev), y.to(dev)
                    f_x = dummy_model(x)
                    loss = loss_fn(f_x, y)
                    dummy_model.zero_grad()
                    loss.backward()
                    opzee_model_temp = self.model.meta_update(dummy_model)
                
                loss_sum = torch.zeros(1, device=dev)

                opzee_model_temp.zero_grad()
                #finding the optimizer's loss using the optimizee's state
                # for j in range(len(train_data_loader)):
                #     try:
                #         x, y = next(train_iter)
                #     except StopIteration:
                #         train_iter = iter(train_data_loader)
                #         x, y = next(train_iter)
                #     x, y = x.cuda(), y.cuda()
                f_x = opzee_model_temp(x)
                loss = loss_fn(f_x, y)
                loss_sum = loss_sum + loss
                
                # NOTE: the old /len(train_data_loader) scaling is removed
                # (Change.MD item 26); the meta-LR now means what it says.
                enable_running_stats(dummy_model)

                loss_sum.backward()

        self.forward_backward_func = get_grad






    @torch.no_grad()
    def step(self, closure=None):

        if closure:
            get_grad = closure
        else:
            get_grad = self.forward_backward_func

        with self.maybe_no_sync():
            # get gradient
            get_grad()

            # perturb weights
            self.perturb_weights(rho=self.rho_t)

            # disable running stats for second pass
            disable_running_stats(self.model)

            # get gradient at perturbed weights
            get_grad()

            # decompose and get new update direction
            self.gradient_decompose(self.alpha)

            # unperturb
            self.unperturb()
            
        # synchronize gradients across workers
        self._sync_grad()    

        # update with new directions
        self.base_optimizer.step()

        # enable running stats
        enable_running_stats(self.model)







###########################################################################
###     SCHEDULER FUNCTIONS       ############
###########################################################################



def get_optim_and_schedulers(model, args): # optzee_model, data_laoder, 
    if args.base_optimizer == 'sgd':
        base_optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum,
                                  weight_decay=args.weight_decay)
    elif args.base_optimizer == 'adam':
        base_optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    elif args.base_optimizer == 'AdamW':
        base_optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    else:
        raise NotImplementedError

    

    # optional grad_rho_scheduler and grad_norm_rho_scheduler
    if args.regularizer == 'gam':
        lr_scheduler = optim.lr_scheduler.CosineAnnealingLR(base_optimizer, T_max=args.train_period)
        grad_rho_scheduler = ProportionScheduler(pytorch_lr_scheduler=lr_scheduler, max_lr=args.lr, min_lr=0.0,
                                                max_value=args.grad_rho_max, min_value=args.grad_rho_min)

        grad_norm_rho_scheduler = ProportionScheduler(pytorch_lr_scheduler=lr_scheduler, max_lr=args.lr, min_lr=0.0,
                                                  max_value=args.grad_norm_rho_max, min_value=args.grad_norm_rho_min)

    
        optimizer = GAM(params=model.parameters(), base_optimizer=base_optimizer, model=model,
                        grad_rho_scheduler=grad_rho_scheduler, grad_norm_rho_scheduler=grad_norm_rho_scheduler,
                        adaptive=args.adaptive, args=args)
    
    elif args.regularizer == 'gsam':
        scheduler = CosineScheduler(T_max=args.train_period, max_value=args.lr, min_value=0.0, optimizer=base_optimizer)
        rho_scheduler = ProportionScheduler(pytorch_lr_scheduler=scheduler, max_lr=args.lr, min_lr=0.0, 
                                            max_value=args.rho_max, min_value=args.rho_min)

        optimizer = GSAM(params=model.parameters(), base_optimizer=base_optimizer, model=model, gsam_alpha=args.alpha, 
                         rho_scheduler=rho_scheduler, adaptive=args.adaptive, args=args)
    return optimizer #, base_optimizer, lr_scheduler, grad_rho_scheduler, grad_norm_rho_scheduler




