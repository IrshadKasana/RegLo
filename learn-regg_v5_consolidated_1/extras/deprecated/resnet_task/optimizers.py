import torch
import torch.nn as nn
import numpy as np
from torch.autograd import Variable
from meta_module import *


#with_bias
class SDMOptimizer(nn.Module):
    def __init__(self, preproc=False, hidden_sz=20, preproc_factor=10.0):
        super().__init__()
        self.hidden_sz = hidden_sz
        if preproc:
            self.recurs = nn.LSTMCell(2, hidden_sz) #took input as sign and pre-processes grad
        else:
            self.recurs = nn.LSTMCell(1, hidden_sz)

        self.recurs2 = nn.LSTMCell(hidden_sz, hidden_sz)
        self.output = nn.Linear(2*hidden_sz, 1)
        self.preproc = preproc
        self.preproc_factor = preproc_factor
        self.preproc_threshold = np.exp(-preproc_factor)
        #some real numbers multiplied to hidden states as parameters
        self.alpha = nn.Parameter(torch.Tensor(1))
        self.beta = nn.Parameter(torch.Tensor(1))

    def forward(self, inp, hidden, cell):
        if self.preproc:
            # Implement preproc described in Appendix A

            # Note: we do all this work on tensors, which means
            # the gradients won't propagate through inp. This
            # should be ok because the algorithm involves
            # making sure that inp is already detached.
            inp = inp.data
            inp2 = w(torch.zeros(inp.size()[0], 2))
            keep_grads = (torch.abs(inp) >= self.preproc_threshold).squeeze()
            inp2[:, 0][keep_grads] = (torch.log(torch.abs(inp[keep_grads]) + 1e-8) / self.preproc_factor).squeeze()
            inp2[:, 1][keep_grads] = torch.sign(inp[keep_grads]).squeeze()

            inp2[:, 0][~keep_grads] = -1
            inp2[:, 1][~keep_grads] = (float(np.exp(self.preproc_factor)) * inp[~keep_grads]).squeeze()
            inp = w(Variable(inp2))

        #LSTMCell takes in input (hidden, cell) and outputs the same although the initialization doesnt say it
        #eg. hx, cx = rnn(input[i], (hx, cx))
        #can take inputs in batches, hidden state in batches and cell state in batches
        hidden0, cell0 = self.recurs(inp, (hidden[0], cell[0]))
        hidden1, cell1 = self.recurs2(hidden0, (hidden[1], cell[1]))

        sum_hidden_states = torch.zeros_like(hidden1[0])
        n_h = len(hidden1)
        for i in range(n_h):
            sum_hidden_states = sum_hidden_states + hidden1[i]

        average_hidden = (1/n_h)*sum_hidden_states

        return self.output(torch.cat((self.alpha*hidden1 , self.beta*torch.tile(average_hidden, (n_h, 1))), dim=1)),\
                (hidden0, hidden1), (cell0, cell1)


