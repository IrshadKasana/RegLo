from functools import reduce
from operator import mul

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

print(torch.cuda.is_available(), device)


def initialize_rnn_hidden_state(dim_sum,n_layers,n_params):
    h0 = Variable(torch.zeros(n_layers,n_params,dim_sum),requires_grad=False)
    c0 = Variable(torch.zeros(n_layers,n_params,dim_sum),requires_grad=False)
    h0.data = h0.data.to(device)
    c0.data = h0.data.to(device)
    return h0, c0




class MetaOptimizerSDMO(nn.Module):

    def __init__(self, model, num_layers, input_dim, hidden_size):
        super(MetaOptimizerSDMO, self).__init__()
        self.meta_model = model
        self.first_order = nn.LSTM(input_dim,hidden_size,num_layers,batch_first=True,bias=False)
        self.outputer = nn.Linear(2*hidden_size,1,bias=False)
        self.outputer.weight.data.mul_(0.1)
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.alpha = Variable(torch.tensor(1))
        self.beta = Variable(torch.tensor(1))

    def reset_lstm(self, keep_states=False, model=None):
        self.meta_model.reset()
        self.meta_model.copy_params_from(model)
        if keep_states > 0:
            self.h0 = Variable(self.h0.data)
            self.c0 = Variable(self.c0.data)
        else:
            self.h0, self.c0 = initialize_rnn_hidden_state(self.hidden_size, self.num_layers, self.meta_model.get_flat_params().size(0))

    def forward(self, x):
        # print(self.h0.size())
        output1, (hn1,cn1) = self.first_order(x, (self.h0, self.c0))
        # print(output1.size(), hn1.size(), self.h0.size())

        self.h0 = hn1
        self.c0 = cn1

        sum_hidden_states = torch.zeros_like(output1[0])
        n_h = len(output1)

        # print(f"\nn_h = {n_h}\n")
        # print(f"hn1 = {hn1.size()}, cn1 = {cn1.size()}, output1 = {output1.size()}\n, output1[1]: {output1[1].size()}")

        for i in range(n_h):
            sum_hidden_states = sum_hidden_states + output1[i]
        

        # print(f"avg_hid = {sum_hidden_states.size()}")
        average_hidden = (1/n_h)*sum_hidden_states

        # print(f"tile : {torch.tile(average_hidden, (n_h, 1)).size()}, {type(torch.tile(average_hidden, (n_h, 1)))}\n")

        o1 = self.outputer(torch.cat((self.alpha*output1 , self.beta*torch.tile(average_hidden, (n_h, 1)).unsqueeze_(1)), dim=2))

        # print('o1 size:', o1.size())
        return o1.squeeze()

    def meta_update(self, model_with_grads):
        # First we need to create a flat version of parameters and gradients
        grads = []
        for name, module in model_with_grads.named_modules():
            # print(name)
            if len(module._parameters) != 0:
                grads.append(module._parameters['weight'].grad.data.view(-1).unsqueeze(-1))
                try:
                    grads.append(module._parameters['bias'].grad.data.view(-1).unsqueeze(-1))
                except:
                    pass


        flat_params = self.meta_model.get_flat_params()
        flat_grads = torch.cat(grads)
        # print(f"flat grads: {flat_grads.size()}")
        second_moment = (flat_grads**2).mean() + 1e-5
        flat_grads = flat_grads / torch.sqrt(second_moment)
        inputs = Variable(flat_grads.view(-1, 1).unsqueeze(1))
        
        # print(f"input size: {inputs.size()}")
        update = self(inputs)

        # print(f"flatparams:{flat_params.size()}, update: {update.size()}")
            
        flat_params = flat_params + update

        self.meta_model.set_flat_params(flat_params)

        # Finally, copy values from the meta model to the normal one.
        self.meta_model.copy_params_to(model_with_grads)
        return self.meta_model.model        



# A helper class that keeps track of meta updates
# It's done by replacing parameters with variables and applying updates to
# them.
class MetaModel:

    def __init__(self, model):
        self.model = model


    def reset(self):
        for module in self.model.modules():
            if len(module._parameters) != 0:
                module._parameters['weight'] = Variable(module._parameters['weight'].data)
                try:
                    module._parameters['bias'] = Variable(module._parameters['bias'].data)
                except:
                    pass


    def get_flat_params(self):
        params = []
        for name, module in self.model.named_modules():
            if len(module._parameters) != 0:
                params.append(module._parameters['weight'].view(-1))
                try:
                    params.append(module._parameters['bias'].view(-1))
                except:
                    pass
        return torch.cat(params)


    def set_flat_params(self, flat_params):
        # Restore original shapes
        offset = 0
        for module in self.model.modules():
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


    def copy_params_from(self, model):
        for modelA, modelB in zip(self.model.parameters(), model.parameters()):
            modelA.data.copy_(modelB.data)

    def copy_params_to(self, model):
        for modelA, modelB in zip(self.model.parameters(), model.parameters()):
            modelB.data.copy_(modelA.data)
