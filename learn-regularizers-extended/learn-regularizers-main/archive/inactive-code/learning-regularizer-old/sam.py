import torch
from functools import reduce
import torch.nn.functional as F
from operator import mul
import copy


class SAM(torch.optim.Optimizer):
    def __init__(self, params, base_optimizer, rho=0.05, adaptive=False, **kwargs):
        assert rho >= 0.0, f"Invalid rho, should be non-negative: {rho}"

        defaults = dict(rho=rho, adaptive=adaptive, **kwargs)
        super(SAM, self).__init__(params, defaults)

        self.base_optimizer = base_optimizer(self.param_groups, **kwargs)
        self.param_groups = self.base_optimizer.param_groups
        self.defaults.update(self.base_optimizer.defaults)

    @torch.no_grad()
    def first_step(self, zero_grad=False):
        grad_norm = self._grad_norm()
        for group in self.param_groups:
            scale = group["rho"] / (grad_norm + 1e-12)

            for p in group["params"]:
                if p.grad is None: continue
                self.state[p]["old_p"] = p.data.clone()
                e_w = (torch.pow(p, 2) if group["adaptive"] else 1.0) * p.grad * scale.to(p)
                p.add_(e_w)  # climb to the local maximum "w + e(w)"

        if zero_grad: self.zero_grad()

    @torch.no_grad()
    def second_step(self, zero_grad=False):
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None: continue
                p.data = self.state[p]["old_p"]  # get back to "w" from "w + e(w)"
                # print(p.grad)
        self.base_optimizer.step()  # do the actual "sharpness-aware" update using the LO
        if zero_grad: self.zero_grad()

    @torch.no_grad()
    def step(self, closure=None):
        assert closure is not None, "Sharpness Aware Minimization requires closure, but it was not provided"
        closure = torch.enable_grad()(closure)  # the closure should do a full forward-backward pass

        self.first_step(zero_grad=True)
        closure()
        self.second_step()

    def _grad_norm(self):
        shared_device = self.param_groups[0]["params"][0].device  # put everything on the same device, in case of model parallelism
    
        norm = torch.norm(
                    torch.stack([
                        ((torch.abs(p) if group["adaptive"] else 1.0) * p.grad).norm(p=2).to(shared_device)
                        for group in self.param_groups for p in group["params"]
                        if p.grad is not None
                    ]),
                    p=2
               )
        return norm

    def load_state_dict(self, state_dict):
        super().load_state_dict(state_dict)
        self.base_optimizer.param_groups = self.param_groups


########################################################################################################################################
#                   #
#                   #
#                   #
#                   #
#                   #
############################################################################################################################################################



class smooth_regularize:

    '''
    lamb: regularization coefficient
    eps: perturbation length
    num_pga_steps: number of projected gradient ascent steps
    optimizer_model : the learned optimizer

    '''
    def __init__(self, lamb, eps, num_pga_steps,optimizer_model):
        self.lamb = lamb
        self.eps = eps
        self.num_pga_steps = num_pga_steps
        self.optimizer = optimizer_model

    def run(self, model_with_grads):
        grads = []
        for name, module in model_with_grads.named_modules():
            # print(name)
            if len(module._parameters) != 0:
                grads.append(module._parameters['weight'].grad.data.view(-1).unsqueeze(-1))
                try:
                    grads.append(module._parameters['bias'].grad.data.view(-1).unsqueeze(-1))
                except:
                    pass
        
        flat_grads = torch.cat(grads)
        second_moment = (flat_grads**2).mean() + 1e-5
        flat_grads = flat_grads / torch.sqrt(second_moment)
        inputs = flat_grads.view(-1, 1).unsqueeze(1)

        state_adv = inputs.detach() + 0.01 * torch.randn(inputs.shape).cuda().detach()
        step_size = self.eps / 5
        # ori_h0 = optimizer_model.h0
        ori_h0 = self.optimizer.h0.data
        ori_c0 = self.optimizer.c0.data    # saving a copy of the hidden states since they get changed after one forward pass
                                            # need to update it again later while finding the distance between two updates

        for _ in range(self.num_pga_steps):
            state_adv.requires_grad_()
            with torch.enable_grad():
                perturbed_update = self.optimizer(state_adv)    #optimizer_model(state_adv)#
                self.optimizer.h0 = ori_h0                      #optimizer_model.h0 = ori_h0 #
                self.optimizer.c0 = ori_c0
                # print(self.optimizer.h0[0])
                update = self.optimizer(inputs)                 #optimizer_model(inputs)#
                self.optimizer.h0 = ori_h0                      #optimizer_model.h0 = ori_h0 #
                self.optimizer.c0 = ori_c0
                # print(self.optimizer.h0[0][1][1])
                distance = F.mse_loss(perturbed_update, update)
                # print(state_adv)
                grad = torch.autograd.grad(distance, [state_adv], allow_unused=True)[0]

                state_adv = state_adv.detach() + step_size * torch.sign(grad.detach())
                state_adv = torch.min(torch.max(state_adv, inputs - self.eps), inputs + self.eps)

        self.optimizer.h0 = ori_h0                              #optimizer_model.h0 = ori_h0 #
        self.optimizer.c0 = ori_c0
        perturbed_inputs = state_adv.detach()
        perturbed_update = self.optimizer(perturbed_inputs)     #optimizer_model(perturbed_inputs)#
        self.optimizer.h0 = ori_h0                              #optimizer_model.h0 = ori_h0 #
        self.optimizer.c0 = ori_c0
        update = self.optimizer(inputs)                         #optimizer_model(inputs) #
        distance = F.mse_loss(update, perturbed_update)

        
        return  distance



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

