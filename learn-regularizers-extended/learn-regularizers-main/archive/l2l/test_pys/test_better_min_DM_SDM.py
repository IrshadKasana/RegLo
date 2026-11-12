import torch

import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torchvision
from torch.autograd import Variable
import itertools
import operator
from itertools import islice
from collections import OrderedDict




import functools
from tqdm import tqdm
import torch.optim as optim
from torchvision import datasets

USE_CUDA = torch.cuda.is_available()
import os


def w(v):
    if USE_CUDA:
        return v.cuda()
    return v


def to_var(x, requires_grad=True):
    if torch.cuda.is_available():
        x = x.cuda()
    return Variable(x, requires_grad=requires_grad)

class MetaModule(nn.Module):
    # adopted from: Adrien Ecoffet https://github.com/AdrienLE
    def parameters(self):
       for name, param in self.named_params(self):
            yield param

    def named_parameters(self):
       for name, param in self.named_params(self):
            yield name, param

    def named_leaves(self):
        return []

    def named_submodules(self):
        return []

    def named_params(self, curr_module=None, memo=None, prefix=''):
        if memo is None:
            memo = set()

        if hasattr(curr_module, 'named_leaves'):
            for name, p in curr_module.named_leaves():
                if p is not None and p not in memo:
                    memo.add(p)
                    yield prefix + ('.' if prefix else '') + name, p
        else:
            for name, p in curr_module._parameters.items():
                if p is not None and p not in memo:
                    memo.add(p)
                    yield prefix + ('.' if prefix else '') + name, p

        for mname, module in curr_module.named_children():
            submodule_prefix = prefix + ('.' if prefix else '') + mname
            for name, p in self.named_params(module, memo, submodule_prefix):
                yield name, p

    def update_params(self, lr_inner, first_order=False, source_params=None, detach=False):
        if source_params is not None:
            for tgt, src in zip(self.named_params(self), source_params):
                name_t, param_t = tgt
                # name_s, param_s = src
                # grad = param_s.grad
                # name_s, param_s = src
                grad = src
                if first_order:
                    grad = to_var(grad.detach().data)
                tmp = param_t - lr_inner * grad
                self.set_param(self, name_t, tmp)
        else:
            for name, param in self.named_params(self):
                if not detach:
                    grad = param.grad
                    if first_order:
                        grad = to_var(grad.detach().data)
                    tmp = param - lr_inner * grad
                    self.set_param(self, name, tmp)
                else:
                    param = param.detach_()
                    self.set_param(self, name, param)

    def set_param(self,curr_mod, name, param):
        if '.' in name:
            n = name.split('.')
            module_name = n[0]
            rest = '.'.join(n[1:])
            for name, mod in curr_mod.named_children():
                if module_name == name:
                    self.set_param(mod, rest, param)
                    break
        else:
            setattr(curr_mod, name, param)

    def detach_params(self):
        for name, param in self.named_params(self):
            self.set_param(self, name, param.detach())

    def copy(self, other, same_var=False):
        for name, param in other.named_params():
            if not same_var:
                param = to_var(param.data.clone(), requires_grad=True)
            self.set_param(name, param)


class MetaLinear(MetaModule):
    def __init__(self, *args, **kwargs):
        super().__init__()
        ignore = nn.Linear(*args, **kwargs)

        self.register_buffer('weight', to_var(ignore.weight.data, requires_grad=True))
        self.register_buffer('bias', to_var(ignore.bias.data, requires_grad=True))
        self.in_features = ignore.weight.size(1)
        self.out_features = ignore.weight.size(0)

    def forward(self, x):
        return F.linear(x, self.weight, self.bias)

    def named_leaves(self):
        return [('weight', self.weight), ('bias', self.bias)]

class MetaConv2d(MetaModule):
    def __init__(self, *args, **kwargs):
        super().__init__()
        ignore = nn.Conv2d(*args, **kwargs)

        self.stride = ignore.stride
        self.padding = ignore.padding
        self.dilation = ignore.dilation
        self.groups = ignore.groups

        self.register_buffer('weight', to_var(ignore.weight.data, requires_grad=True))

        if ignore.bias is not None:
            self.register_buffer('bias', to_var(ignore.bias.data, requires_grad=True))
        else:
            self.register_buffer('bias', None)

    def forward(self, x):
        return F.conv2d(x, self.weight, self.bias, self.stride, self.padding, self.dilation, self.groups)

    def named_leaves(self):
        return [('weight', self.weight), ('bias', self.bias)]

class MetaConvTranspose2d(MetaModule):
    def __init__(self, *args, **kwargs):
        super().__init__()
        ignore = nn.ConvTranspose2d(*args, **kwargs)

        self.stride = ignore.stride
        self.padding = ignore.padding
        self.dilation = ignore.dilation
        self.groups = ignore.groups

        self.register_buffer('weight', to_var(ignore.weight.data, requires_grad=True))

        if ignore.bias is not None:
            self.register_buffer('bias', to_var(ignore.bias.data, requires_grad=True))
        else:
            self.register_buffer('bias', None)

    def forward(self, x, output_size=None):
        output_padding = self._output_padding(x, output_size)
        return F.conv_transpose2d(x, self.weight, self.bias, self.stride, self.padding,
            output_padding, self.groups, self.dilation)

    def named_leaves(self):
        return [('weight', self.weight), ('bias', self.bias)]

class MetaBatchNorm2d(MetaModule):
    def __init__(self, *args, **kwargs):
        super().__init__()
        ignore = nn.BatchNorm2d(*args, **kwargs)

        self.num_features = ignore.num_features
        self.eps = ignore.eps
        self.momentum = ignore.momentum
        self.affine = ignore.affine
        self.track_running_stats = ignore.track_running_stats

        if self.affine:
            self.register_buffer('weight', to_var(ignore.weight.data, requires_grad=True))
            self.register_buffer('bias', to_var(ignore.bias.data, requires_grad=True))

        if self.track_running_stats:
            self.register_buffer('running_mean', torch.zeros(self.num_features))
            self.register_buffer('running_var', torch.ones(self.num_features))
        else:
            self.register_parameter('running_mean', None)
            self.register_parameter('running_var', None)


    def forward(self, x):
        return F.batch_norm(x, self.running_mean, self.running_var, self.weight, self.bias,
                        self.training or not self.track_running_stats, self.momentum, self.eps)

    def named_leaves(self):
        return [('weight', self.weight), ('bias', self.bias)]


class MetaSequential(MetaModule):
    r"""A sequential container.
    Modules will be added to it in the order they are passed in the constructor.
    Alternatively, an ordered dict of modules can also be passed in.

    To make it easier to understand, here is a small example::

        # Example of using Sequential
        model = MetaSequential(
                  MetaConv2d(1,20,5),
                  nn.ReLU(),
                  MetaConv2d(20,64,5),
                  nn.ReLU()
                )

        # Example of using Sequential with OrderedDict
        model = MetaSequential(OrderedDict([
                  ('conv1', MetaConv2d(1,20,5)),
                  ('relu1', nn.ReLU()),
                  ('conv2', MetaConv2d(20,64,5)),
                  ('relu2', nn.ReLU())
                ]))
    """

    def __init__(self, *args):
        super(MetaSequential, self).__init__()
        if len(args) == 1 and isinstance(args[0], OrderedDict):
            for key, module in args[0].items():
                self.add_module(key, module)
        else:
            for idx, module in enumerate(args):
                self.add_module(str(idx), module)

    def _get_item_by_idx(self, iterator, idx):
        """Get the idx-th item of the iterator"""
        size = len(self)
        idx = operator.index(idx)
        if not -size <= idx < size:
            raise IndexError('index {} is out of range'.format(idx))
        idx %= size
        return next(islice(iterator, idx, None))

    def __getitem__(self, idx):
        if isinstance(idx, slice):
            return self.__class__(OrderedDict(list(self._modules.items())[idx]))
        else:
            return self._get_item_by_idx(self._modules.values(), idx)

    def __setitem__(self, idx, module):
        key = self._get_item_by_idx(self._modules.keys(), idx)
        return setattr(self, key, module)

    def __delitem__(self, idx):
        if isinstance(idx, slice):
            for key in list(self._modules.keys())[idx]:
                delattr(self, key)
        else:
            key = self._get_item_by_idx(self._modules.keys(), idx)
            delattr(self, key)

    def __len__(self):
        return len(self._modules)

    def __dir__(self):
        keys = super(Sequential, self).__dir__()
        keys = [key for key in keys if not key.isdigit()]
        return keys

    def forward(self, input):
        for module in self._modules.values():
            input = module(input)
        return input


class MetaModuleList(MetaModule):
    r"""Holds submodules in a list.

    :class:`~MetaModuleList` can be indexed like a regular Python list, but
    modules it contains are properly registered, and will be visible by all
    :class:`~MetaModule` methods.

    Arguments:
        modules (iterable, optional): an iterable of modules to add

    Example::

        class MyModule(MetaModule):
            def __init__(self):
                super(MyModule, self).__init__()
                self.linears = MetaModuleList([MetaLinear(10, 10) for i in range(10)])

            def forward(self, x):
                # ModuleList can act as an iterable, or be indexed using ints
                for i, l in enumerate(self.linears):
                    x = self.linears[i // 2](x) + l(x)
                return x
    """

    def __init__(self, modules=None):
        super(MetaModuleList, self).__init__()
        if modules is not None:
            self += modules

    def _get_abs_string_index(self, idx):
        """Get the absolute index for the list of modules"""
        idx = operator.index(idx)
        if not (-len(self) <= idx < len(self)):
            raise IndexError('index {} is out of range'.format(idx))
        if idx < 0:
            idx += len(self)
        return str(idx)

    def __getitem__(self, idx):
        if isinstance(idx, slice):
            return self.__class__(list(self._modules.values())[idx])
        else:
            return self._modules[self._get_abs_string_index(idx)]

    def __setitem__(self, idx, module):
        idx = self._get_abs_string_index(idx)
        return setattr(self, str(idx), module)

    def __delitem__(self, idx):
        if isinstance(idx, slice):
            for k in range(len(self._modules))[idx]:
                delattr(self, str(k))
        else:
            delattr(self, self._get_abs_string_index(idx))
        # To preserve numbering, self._modules is being reconstructed with modules after deletion
        str_indices = [str(i) for i in range(len(self._modules))]
        self._modules = OrderedDict(list(zip(str_indices, self._modules.values())))

    def __len__(self):
        return len(self._modules)

    def __iter__(self):
        return iter(self._modules.values())

    def __iadd__(self, modules):
        return self.extend(modules)

    def __dir__(self):
        keys = super(ModuleList, self).__dir__()
        keys = [key for key in keys if not key.isdigit()]
        return keys

    def insert(self, index, module):
        r"""Insert a given module before a given index in the list.

        Arguments:
            index (int): index to insert.
            module (MetaModule): module to insert
        """
        for i in range(len(self._modules), index, -1):
            self._modules[str(i)] = self._modules[str(i - 1)]
        self._modules[str(index)] = module


    def append(self, module):
        r"""Appends a given module to the end of the list.

        Arguments:
            module (MetaModule): module to append
        """
        self.add_module(str(len(self)), module)
        return self


    def extend(self, modules):
        r"""Appends modules from a Python iterable to the end of the list.

        Arguments:
            modules (iterable): iterable of modules to append
        """
        if not isinstance(modules, container_abcs.Iterable):
            raise TypeError("ModuleList.extend should be called with an "
                            "iterable, but got " + type(modules).__name__)
        offset = len(self)
        for i, module in enumerate(modules):
            self.add_module(str(offset + i), module)
        return self



class ModuleDict(MetaModule):
    r"""Holds submodules in a dictionary.

    :class:`~MetaModuleDict` can be indexed like a regular Python dictionary,
    but modules it contains are properly registered, and will be visible by all
    :class:`~MetaModule` methods.

    :class:`~MetaModuleDict` is an **ordered** dictionary that respects

    * the order of insertion, and

    * in :meth:`~MetaModuleDict.update`, the order of the merged ``OrderedDict``
      or another :class:`~MetaModuleDict` (the argument to :meth:`~MetaModuleDict.update`).

    Note that :meth:`~MetaModuleDict.update` with other unordered mapping
    types (e.g., Python's plain ``dict``) does not preserve the order of the
    merged mapping.

    Arguments:
        modules (iterable, optional): a mapping (dictionary) of (string: module)
            or an iterable of key-value pairs of type (string, module)

    Example::

        class MyModule(MetaModule):
            def __init__(self):
                super(MyModule, self).__init__()
                self.choices = MetaModuleDict({
                        'conv': MetaConv2d(10, 10, 3),
                        'pool': nn.MaxPool2d(3)
                })
                self.activations = MetaModuleDict([
                        ['lrelu', nn.LeakyReLU()],
                        ['prelu', nn.PReLU()]
                ])

            def forward(self, x, choice, act):
                x = self.choices[choice](x)
                x = self.activations[act](x)
                return x
    """

    def __init__(self, modules=None):
        super(MetaModuleDict, self).__init__()
        if modules is not None:
            self.update(modules)

    def __getitem__(self, key):
        return self._modules[key]

    def __setitem__(self, key, module):
        self.add_module(key, module)

    def __delitem__(self, key):
        del self._modules[key]

    def __len__(self):
        return len(self._modules)

    def __iter__(self):
        return iter(self._modules)

    def __contains__(self, key):
        return key in self._modules

    def clear(self):
        """Remove all items from the ModuleDict.
        """
        self._modules.clear()


    def pop(self, key):
        r"""Remove key from the ModuleDict and return its module.

        Arguments:
            key (string): key to pop from the ModuleDict
        """
        v = self[key]
        del self[key]
        return v


    def keys(self):
        r"""Return an iterable of the ModuleDict keys.
        """
        return self._modules.keys()


    def items(self):
        r"""Return an iterable of the ModuleDict key/value pairs.
        """
        return self._modules.items()


    def values(self):
        r"""Return an iterable of the ModuleDict values.
        """
        return self._modules.values()


    def update(self, modules):
        r"""Update the :class:`~MetaModuleDict` with the key-value pairs from a
        mapping or an iterable, overwriting existing keys.

        .. note::
            If :attr:`modules` is an ``OrderedDict``, a :class:`~MetaModuleDict`, or
            an iterable of key-value pairs, the order of new elements in it is preserved.

        Arguments:
            modules (iterable): a mapping (dictionary) from string to :class:`~MetaModule`,
                or an iterable of key-value pairs of type (string, :class:`~MetaModule`)
        """
        if not isinstance(modules, container_abcs.Iterable):
            raise TypeError("ModuleDict.update should be called with an "
                            "iterable of key/value pairs, but got " +
                            type(modules).__name__)

        if isinstance(modules, container_abcs.Mapping):
            if isinstance(modules, (OrderedDict, ModuleDict)):
                for key, module in modules.items():
                    self[key] = module
            else:
                for key, module in sorted(modules.items()):
                    self[key] = module
        else:
            for j, m in enumerate(modules):
                if not isinstance(m, container_abcs.Iterable):
                    raise TypeError("ModuleDict update sequence element "
                                    "#" + str(j) + " should be Iterable; is" +
                                    type(m).__name__)
                if not len(m) == 2:
                    raise ValueError("ModuleDict update sequence element "
                                     "#" + str(j) + " has length " + str(len(m)) +
                                     "; 2 is required")
                self[m[0]] = m[1]

    def forward(self):
        raise NotImplementedError()



class LeNet(MetaModule):
    def __init__(self, n_out):
        super(LeNet, self).__init__()

        layers = []
        layers.append(MetaConv2d(1, 6, kernel_size=5))
        layers.append(nn.ReLU(inplace=True))
        layers.append(nn.MaxPool2d(kernel_size=2,stride=2))

        layers.append(MetaConv2d(6, 16, kernel_size=5))
        layers.append(nn.ReLU(inplace=True))
        layers.append(nn.MaxPool2d(kernel_size=2,stride=2))

        layers.append(MetaConv2d(16, 120, kernel_size=5))
        layers.append(nn.ReLU(inplace=True))

        self.main = nn.Sequential(*layers)

        layers = []
        layers.append(MetaLinear(120, 84))
        layers.append(nn.ReLU(inplace=True))
        layers.append(MetaLinear(84, n_out))

        self.fc_layers = nn.Sequential(*layers)

    def forward(self, x):
        x = self.main(x)
        x = x.view(-1, 120)
        return self.fc_layers(x).squeeze()


#########################################################
#           Optimizer DMO/SDMO
#########################################################


class Optimizer(nn.Module):
    def __init__(self, preproc=True, hidden_sz=20, preproc_factor=10.0):
        super().__init__()
        self.hidden_sz = hidden_sz
        if preproc:
            self.recurs = nn.LSTMCell(2, hidden_sz)
        else:
            self.recurs = nn.LSTMCell(1, hidden_sz)
        self.recurs2 = nn.LSTMCell(hidden_sz, hidden_sz)
        self.output = nn.Linear(hidden_sz, 1)
        self.preproc = preproc
        self.preproc_factor = preproc_factor
        self.preproc_threshold = np.exp(-preproc_factor)

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
        hidden0, cell0 = self.recurs(inp, (hidden[0], cell[0]))
        hidden1, cell1 = self.recurs2(hidden0, (hidden[1], cell[1]))
        return self.output(hidden1), (hidden0, hidden1), (cell0, cell1)
    


class SDMOptimizer(nn.Module):
    def __init__(self, preproc=False, hidden_sz=20, preproc_factor=10.0):
        super().__init__()
        self.hidden_sz = hidden_sz
        if preproc:
            self.recurs = nn.LSTMCell(2, hidden_sz, bias = False) #took input as sign and pre-processes grad
        else:
            self.recurs = nn.LSTMCell(1, hidden_sz, bias= False)
        
        self.recurs2 = nn.LSTMCell(hidden_sz, hidden_sz, bias=False)
        self.output = nn.Linear(2*hidden_sz, 1, bias = False)
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



#########################################################
#               Cifar Dataset
#########################################################


class Cifar10_Data:
    def __init__(self, batch_size=128):
        dataset_train = datasets.CIFAR10(
            "/nlsasfs/home/metalearning/nckris/Suraj_LSAIML/datasets/", train=True, download=True,
            transform=torchvision.transforms.Compose([ torchvision.transforms.ToTensor(), 
            torchvision.transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))]) #torchvision.transforms.Resize(32),
        )
        
        self.batch_size= batch_size
        self.loader = torch.utils.data.DataLoader(
            dataset_train, batch_size=self.batch_size, num_workers=32)

        self.batches = []
        self.cur_batch = 0
        
    def sample(self):
        if self.cur_batch >= len(self.batches):
            self.batches = []
            self.cur_batch = 0
            for b in self.loader:
                self.batches.append(b)
        batch = self.batches[self.cur_batch]
        self.cur_batch += 1
        return batch



# for validation

class Cifar10_valData:
    def __init__(self, batch_size=128):
        dataset_test = datasets.CIFAR10(
            "/nlsasfs/home/metalearning/nckris/Suraj_LSAIML/datasets/", train=False, download=True,
            transform=torchvision.transforms.Compose([ torchvision.transforms.ToTensor(), 
            torchvision.transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])) #torchvision.transforms.Resize(32),
        
    
        self.batch_size= batch_size
 
        self.loader = torch.utils.data.DataLoader(
            dataset_test, batch_size=self.batch_size, num_workers=32)

        self.batches = []
        self.cur_batch = 0
        self.flag1 = True

    def sample(self):
        if self.cur_batch >= len(self.batches):
            if (self.flag1 == False):
                self.flag1 = True
                return False

            self.batches = []
            self.cur_batch = 0
            for b in self.loader:
                self.batches.append(b)
            self.flag1 = False
      
        batch = self.batches[self.cur_batch]
        self.cur_batch += 1
        return batch

######################################################################

##########          Mnist Data          ##############################

######################################################################


class MNIST_Data:
    def __init__(self, batch_size=128):
        dataset_train = datasets.MNIST(
            "/nlsasfs/home/metalearning/nckris/Suraj_LSAIML/datasets/", train=True, download=True,
            transform=torchvision.transforms.Compose([ torchvision.transforms.ToTensor()]) #torchvision.transforms.Resize(32),
        )
        
        self.batch_size= batch_size
        indices = list(range(len(dataset_train)))
        np.random.RandomState(10).shuffle(indices)
        indices = indices[len(indices) // 2:]

        self.loader = torch.utils.data.DataLoader(
            dataset_train, batch_size=128,
            sampler=torch.utils.data.sampler.SubsetRandomSampler(indices))
        self.batches = []
        self.cur_batch = 0
        
    def sample(self):
        if self.cur_batch >= len(self.batches):
            self.batches = []
            self.cur_batch = 0
            for b in self.loader:
                self.batches.append(b)
        batch = self.batches[self.cur_batch]
        self.cur_batch += 1
        return batch



# for validation

class MNIST_valData:
    def __init__(self, batch_size=128):
        dataset_test = datasets.MNIST(
            "/nlsasfs/home/metalearning/nckris/Suraj_LSAIML/datasets/", train=False, download=True,
            transform=torchvision.transforms.Compose([ torchvision.transforms.ToTensor()])) #torchvision.transforms.Resize(32),
        
    
        self.batch_size= batch_size
 
        self.loader = torch.utils.data.DataLoader(
            dataset_test, batch_size=self.batch_size, num_workers=32)

        self.batches = []
        self.cur_batch = 0
        self.flag1 = True

    def sample(self):
        if self.cur_batch >= len(self.batches):
            if (self.flag1 == False):
                self.flag1 = True
                return False

            self.batches = []
            self.cur_batch = 0
            for b in self.loader:
                self.batches.append(b)
            self.flag1 = False
      
        batch = self.batches[self.cur_batch]
        self.cur_batch += 1
        return batch














criterion = nn.CrossEntropyLoss()
def accuracy(outputs, labels):
    
    _, predicted = torch.max(outputs, 1)
    # print(outputs.size(), predicted.size())
    correct = (predicted == labels).sum().item()
    total = labels.size(0)
    acc = correct / total
    return acc


#########################################################
#               Models for Optimizee
#########################################################


class CifarCNN(MetaModule):

    def __init__(self, num_class):
        super(CifarCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=2)
        self.conv2 = nn.Conv2d(32, 32, kernel_size=3, stride=2)
        self.conv3 = nn.Conv2d(32, 32, kernel_size=3)
        # self.linear1 = nn.Linear(128, 10)
        self.linear1 = nn.Linear(128, num_class)
        self.loss = nn.NLLLoss()
    
    def all_named_parameters(self):
        return [(k, v) for k, v in self.named_parameters()]

    def forward(self, batch):
        x, out = batch.sample()
        x = w(Variable(x))
        out = w(Variable(out))
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.avg_pool2d(x, 2)
        x = x.view(x.size(0), -1)
        x = self.linear1(x)
        x =  F.log_softmax(x, dim=-1)
        l = self.loss(x, out)
        return l, x, out






class MNISTNet(MetaModule):
    def __init__(self, layer_size=20, n_layers=1, **kwargs):
        super().__init__()

        inp_size = 28*28
        self.layers = {}
        for i in range(n_layers):
            self.layers[f'mat_{i}'] = nn.Linear(inp_size, layer_size)
            inp_size = layer_size

        self.layers['final_mat'] = nn.Linear(inp_size, 10)
        self.layers = nn.ModuleDict(self.layers)

        self.activation = nn.Sigmoid()
        self.loss = nn.NLLLoss()

    def all_named_parameters(self):
        return [(k, v) for k, v in self.named_parameters()]

    def forward(self, batch):
        # print(batch)
        inp, out = batch.sample()
        inp = w(Variable(inp.view(inp.size()[0], 28*28)))
        out = w(Variable(out))

        cur_layer = 0
        while f'mat_{cur_layer}' in self.layers:
            inp = self.activation(self.layers[f'mat_{cur_layer}'](inp))
            cur_layer += 1

        inp = F.log_softmax(self.layers['final_mat'](inp), dim=1)
        l = self.loss(inp, out)
        return l, inp, out




# training function 
def detach_var(v):
    var = w(Variable(v.data, requires_grad=True))
    var.retain_grad()
    return var

def rgetattr(obj, attr, *args):
    def _getattr(obj, attr):
        return getattr(obj, attr, *args)
    return functools.reduce(_getattr, [obj] + attr.split('.'))


def rsetattr(obj, attr, val):
    pre, _, post = attr.rpartition('.')
    return setattr(rgetattr(obj, pre) if pre else obj, post, val)




def fit_val_loss_acc(opt_net, target_data, target_optimizee, val_data, optim_it, out_mul=1):
    

    


    #use the should_train parameter to evaluate the task and not train the
    
    opt_net.eval()   #switch for some specific layers/parts of the model that behave differently during training and inference (evaluating) time. For example, Dropouts Layers, BatchNorm Layers etc.
    


    target = target_data  #it is the dataset, optimizee(target) yields loss, output
    optimizee = w(target_optimizee)
    n_params = 0
    for name, p in optimizee.named_parameters():
        n_params += int(np.prod(p.size()))

    #there are two outputs of LSTM (long term= Cell State C_t, and Hidden State = h_t)
    hidden_states = [w(Variable(torch.zeros(n_params, opt_net.hidden_sz))) for _ in range(2)]
    cell_states = [w(Variable(torch.zeros(n_params, opt_net.hidden_sz))) for _ in range(2)]

    
    
    ###############  keep track of the gradient values at the inference time  ########################

    # grad_list_test = []
    ##############################f
    
    train_loss = []
    test_loss= []
    train_acc = []
    test_acc = []
    MA = 0
    use_sgd_flag = False
    steps = 30

    optimizer = torch.optim.SGD(optimizee.parameters(), lr=1)
    count_sgd = 0

    for iteration in tqdm(range(1, optim_it + 1), "iteration: "):  #optim_it = number of iterations the optimizee is trained using LO(inner training steps)
        
        #the loss is always found using all the datapoints (not single or batch loss)
        # difference between consecutive losses will decrease quadratically


        loss, pred, real  = optimizee(target)
        train_loss_iter = loss.item()
        train_loss.append(train_loss_iter)
        tr_acc = accuracy(pred, real)
        train_acc.append(tr_acc)


        # iteration_mod_unroll = iteration % unroll
        optimizer.zero_grad()
        loss.backward(retain_graph=False)

        #gradients wrt all the parameters are updated
        

        offset = 0
        result_params = {}
        hidden_states2 = [w(Variable(torch.zeros(n_params, opt_net.hidden_sz))) for _ in range(2)]
        cell_states2 = [w(Variable(torch.zeros(n_params, opt_net.hidden_sz))) for _ in range(2)]
        
        grad_list_each_step = []

        if use_sgd_flag:
            count_sgd += 1

            if count_sgd > 100:
                break
        
        if not use_sgd_flag:
            for name, p in optimizee.named_parameters():
                cur_sz = int(np.prod(p.size()))
                # We do this so the gradients are disconnected from the graph but we still get
                # gradients from the rest
                gradients = detach_var(p.grad.view(cur_sz, 1))
                ###############  keep track of the gradient values at the inference time  ########################

                grad_list_each_step.append(gradients.tolist())

                ############################################
                
                
                updates, new_hidden, new_cell = opt_net(
                    gradients,
                    [h[offset:offset+cur_sz] for h in hidden_states],
                    [c[offset:offset+cur_sz] for c in cell_states]
                )
                #len(new_hidden) = 2

                for i in range(len(new_hidden)):
                    hidden_states2[i][offset:offset+cur_sz] = new_hidden[i]
                    cell_states2[i][offset:offset+cur_sz] = new_cell[i]
                result_params[name] = p + updates.view(*p.size()) * out_mul
                result_params[name].retain_grad()
                
                    
                
                offset += cur_sz
#######   unroll when necessary to update the params of the LO   ############
        
        #need to update the grads for optimizee parameters too since they also are connected to the RNN in the computational graph
        #reducing them to zero is necessary

            optimizee = w(target_optimizee)
            optimizee.load_state_dict(result_params)
            optimizee.zero_grad()
            hidden_states = [detach_var(v) for v in hidden_states2]
            cell_states = [detach_var(v) for v in cell_states2]


        else:
            
            optimizer.step()


#######  steps when no unrolling happens ########################## 

        # for name, p in optimizee.all_named_parameters():
        #     rsetattr(optimizee, name, result_params[name])
        # assert len(list(optimizee.all_named_parameters()))
        # hidden_states = hidden_states2
        # cell_states = cell_states2


################################################################################

########       VALIDATION: ##########
        loss_val_iter = 0
        Flag = True
        count = 0
        while(Flag):
            try:
                if count == 0:
                    loss_1, valout_pred, val_y = optimizee(val_data)
                    loss_val_iter += loss_1.item()
                    pred_y= valout_pred
                    act_y= val_y
                    count += 1
                else:
                    loss_1, valout_pred, val_y = optimizee(val_data)
                    loss_val_iter += loss_1.item()
                    torch.cat([pred_y, valout_pred], dim = 0)
                    torch.cat([act_y, val_y], dim = 0)
                    count += 1
                    
            except:
                Flag = False
        # print(pred_y, act_y)    
        acc_val_iter = accuracy(pred_y,act_y)
        test_loss.append(loss_val_iter/count)
        test_acc.append(acc_val_iter)
        
        

        ############### Moving average of loss ################

        if not use_sgd_flag:
            
            if (iteration>=steps):
                MA = np.mean(np.array(test_loss[iteration-steps:]))
                print(f"\n Moving Average : {MA} \n")
                if(iteration==steps): #initialization
                    prev_MA = MA
                else:
                    if MA > prev_MA:
                        use_sgd_flag = True
                        print("iteration at which the LO training stopped:", iteration)
                    prev_MA = MA
        
        
        print(f"Training loss:{train_loss_iter} Training Accuracy: {tr_acc} Validation Loss: {loss_val_iter/count} Validation Accuracy: {acc_val_iter}")


################################################################################
        
    


    #all_losses_ever is the loss that optimizee gives after each iteration
    return {"optimizee": optimizee, "train_loss" :train_loss, "train_acc": train_acc, "test_loss": test_loss, "test_acc": test_acc }

SDMO_dict = torch.load("/nlsasfs/home/metalearning/nckris/Suraj_LSAIML/l2l/checkpoints_LO/simple_DMO/nobias_loss_diff_CPpoint_sDMO_lr_0.001.pth")

SDMO_instance = w(SDMOptimizer(preproc= False))
SDMO_instance.load_state_dict(SDMO_dict['optimizer_state_dict'])


hist_dict = {}
for i in range(10):

    target = MNIST_Data(batch_size=128)
    val_target = MNIST_valData(batch_size = 128)
    optimizee = w(MNISTNet())
    history = fit_val_loss_acc(SDMO_instance, target, optimizee, val_target, 1000, out_mul=1)
    loss_acc = {"train_loss" : history["train_loss"], "train_acc": history["train_acc"], "test_loss": history["test_loss"], "test_acc": history["test_acc"]}
    hist_dict[i] = loss_acc

import json
file_name  = "/nlsasfs/home/metalearning/nckris/Suraj_LSAIML/l2l/checkpoints_LO/jsons/history_sdmo_mnist_sgd_list.json"
with open(file_name, "w") as json_file:
    json.dump(hist_dict, json_file)