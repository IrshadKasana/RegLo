from functools import reduce
from operator import mul

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

print(torch.cuda.is_available(), device)


def initialize_rnn_hidden_state(dim_hid,n_layers,n_params):
    '''
    dim_his : hidden state dimension (for each param)
    n_layers : number of layers of LSTM
    n_params : number of parameters in the optimizee model
    '''
    h0 = torch.zeros(n_layers,n_params,dim_hid).requires_grad_(requires_grad=False).to(device)
    c0 = torch.zeros(n_layers,n_params,dim_hid).requires_grad_(requires_grad=False).to(device)

    return h0, c0




class MetaOptimizerLSTM(nn.Module):

    def __init__(self, model, num_layers, hidden_size, preproc_factor = 10.0, preproc =True):
        super(MetaOptimizerLSTM, self).__init__()
        self.meta_model = model
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.preproc = preproc

        if self.preproc:
            self.first_order = nn.LSTM(2,self.hidden_size,self.num_layers,batch_first=True,bias=False)
        else:
            self.first_order = nn.LSTM(1,self.hidden_size,self.num_layers,batch_first=True,bias=False)

        self.outputer = nn.Linear(hidden_size,1,bias=False)
        self.outputer.weight.data.mul_(0.1)
        self.preproc_factor = preproc_factor
        self.preproc_threshold = np.exp(-preproc_factor)

    def reset_lstm(self, keep_states=False, model=None):
        self.meta_model.reset()
        self.meta_model.copy_params_from(model)
        if keep_states :
            
            self.h0 = self.h0.data ####detaches and renews the computational graph (is doesnt require grad but helps in finding the grad for others)
            self.c0 = self.c0.data 
        else:
            self.h0, self.c0 = initialize_rnn_hidden_state(self.hidden_size, self.num_layers, self.meta_model.get_flat_params().size(0))

    def forward(self, x):
        # print('\n', self.preproc_factor, self.preproc_threshold, '\n')
        # print(x.size())
        if self.preproc:
            inp = torch.zeros(x.size()[0], 2).to(device) 
            
            #boolean to see which gradients pass the threshold (gradient cap)
            keep_grads = (torch.abs(x) >= self.preproc_threshold).squeeze()
            # print(inp.size(), keep_grads.size())

            inp[:, 0][keep_grads] = (torch.log(torch.abs(x[keep_grads]) + 1e-8) / self.preproc_factor).squeeze()

            #keeping all the signs of the gradients that have "true" value in keep_grads
            inp[:, 1][keep_grads] = torch.sign(x[keep_grads]).squeeze()
            inp[:, 0][~keep_grads] = -1
            inp[:, 1][~keep_grads] = (float(np.exp(self.preproc_factor)) * x[~keep_grads]).squeeze()
            x = inp.unsqueeze(1)
            x = x.to(device)
            # print(inp.size())


        # print(self.h0.size(), self.c0.size())
        output1, (hn1,cn1) = self.first_order(x, (self.h0, self.c0))
        # print(output1.size(), hn1.size(), self.h0.size())

        self.h0 = hn1
        self.c0 = cn1
        o1 = self.outputer(output1)
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

        self.meta_model.copy_params_from(model_with_grads)
        flat_params = self.meta_model.get_flat_params()     #we dont use the actual model model_with_grads to find updated params, 
                                                            #else that goes inside the computational graph and will result in updating 
                                                            #gradient of meta_optimizer in the TBTT
        flat_grads = torch.cat(grads)

        # second_moment = (flat_grads**2).mean() + 1e-5
        # flat_grads = flat_grads / torch.sqrt(second_moment)
        inputs = flat_grads.view(-1, 1).unsqueeze(1) #Variable(flat_grads.view(-1, 1).unsqueeze(1))
        
        update = self(inputs)
 
            
        flat_params = flat_params + update

        self.meta_model.set_flat_params(flat_params)

        # Finally, copy values from the meta model to the normal one.
        self.meta_model.copy_params_to(model_with_grads)
        return self.meta_model.model   #since self.meta_model is MetaModel(meta_model)





# A helper class that keeps track of meta updates
# It's done by replacing parameters with variables and applying updates to
# them.
class MetaModel:

    def __init__(self, model):
        self.model = model


    def reset(self):
        for module in self.model.modules():
            if len(module._parameters) != 0:
                module._parameters['weight'] = module._parameters['weight'].data 
                try:
                    module._parameters['bias'] = module._parameters['bias'].data 
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
