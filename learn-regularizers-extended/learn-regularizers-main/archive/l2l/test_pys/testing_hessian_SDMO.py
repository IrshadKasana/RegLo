import torch
USE_CUDA = torch.cuda.is_available()

import os


from hessian_eigenthings import compute_hessian_eigenthings


def w(v):
    if USE_CUDA:
        return v.cuda()
    return v
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
#               Cifar Dataset
#########################################################


class Cifar10_Data:
    def __init__(self, batch_size=128):
        dataset_train = datasets.CIFAR10(
            "/nlsasfs/home/metalearning/nckris/Suraj_LSAIML/datasets", train=True, download=True,
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
            "/nlsasfs/home/metalearning/nckris/Suraj_LSAIML/datasets", train=False, download=True,
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
            "/nlsasfs/home/metalearning/nckris/Suraj_LSAIML/datasets", train=True, download=True,
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
            "/nlsasfs/home/metalearning/nckris/Suraj_LSAIML/datasets", train=False, download=True,
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






#without_bias
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
    def __init__(self, layer_size=100, n_layers=1, **kwargs):
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



class dummyMNISTNet(MetaModule):
    def __init__(self, layer_size=100, n_layers=1, **kwargs):
        super().__init__()

        inp_size = 28*28
        self.layers = {}
        for i in range(n_layers):
            self.layers[f'mat_{i}'] = nn.Linear(inp_size, layer_size)
            inp_size = layer_size
        self.layers['final_mat'] = nn.Linear(inp_size, 10)
        self.layers = nn.ModuleDict(self.layers)
        self.activation = nn.Sigmoid()
        
    def forward(self, batch):
        # print(batch)
        inp = batch
        inp = w(Variable(inp.view(inp.size()[0], 28*28)))
        cur_layer = 0
        while f'mat_{cur_layer}' in self.layers:
            inp = self.activation(self.layers[f'mat_{cur_layer}'](inp))
            cur_layer += 1
        
        inp = F.log_softmax(self.layers['final_mat'](inp), dim=1)

        return inp



dummy_dataset_train = datasets.MNIST(
            "/nlsasfs/home/metalearning/nckris/Suraj_LSAIML/datasets", train=True, download=True,
            transform=torchvision.transforms.Compose([ torchvision.transforms.ToTensor()])) #torchvision.transforms.Resize(32),
indices = list(range(len(dummy_dataset_train)))
np.random.RandomState(10).shuffle(indices)
indices = indices[len(indices) // 2:]



# training function for Resnet

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





def eig_values(best_params):
    num_eigenthings = 100  # compute top 100 eigenvalues/eigenvectors
    loss_e = nn.NLLLoss()
    dummyOptimizee = w(dummyMNISTNet())
    dummyOptimizee.load_state_dict(best_params)

    dummy_dataset_train_loader = torch.utils.data.DataLoader(dummy_dataset_train, 
                                                                batch_size=128,
                                                                sampler=torch.utils.data.sampler.SubsetRandomSampler(indices))

    eigenvals, eigenvecs = compute_hessian_eigenthings(dummyOptimizee, dummy_dataset_train_loader,loss_e, num_eigenthings)
    return eigenvals.tolist(),eigenvecs.tolist()







############## zero_order_flatness ##############


def zero_order_flat(gradient, rho = 0.05):
    e_w = (rho/(torch.norm(gradient, 2) + 1e-12))*(gradient)
    return e_w 


#################  #################  #################  


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
    zero_order_flatness_loss = []
    
    eigen_values = {}
    min_loss = 99999.99
    for iteration in tqdm(range(1, optim_it + 1), "iteration: "):  #optim_it = number of iterations the optimizee is trained using LO(inner training steps)
        
        #the loss is always found using all the datapoints (not single or batch loss)
        # difference between consecutive losses will decrease quadratically


        loss, pred, real  = optimizee(target)
        train_loss_iter = loss.item()
        train_loss.append(train_loss_iter)
        tr_acc = accuracy(pred, real)
        train_acc.append(tr_acc)


        # iteration_mod_unroll = iteration % unroll

        loss.backward(retain_graph=False)

        #gradients wrt all the parameters are updated
        

        offset = 0
        result_params = {}
        hidden_states2 = [w(Variable(torch.zeros(n_params, opt_net.hidden_sz))) for _ in range(2)]
        cell_states2 = [w(Variable(torch.zeros(n_params, opt_net.hidden_sz))) for _ in range(2)]
        gradients_each_step=None
        
        # grad_list_each_step = []

        

        for name, p in optimizee.named_parameters():
            cur_sz = int(np.prod(p.size()))
            # We do this so the gradients are disconnected from the graph but we still get
            # gradients from the rest
            gradients = detach_var(p.grad.view(cur_sz, 1))

            
            ###############  keep track of the gradient values at the inference time  ########################
            if gradients_each_step==None:
                gradients_each_step = gradients
                        
            else:
                gradients_each_step = torch.cat((gradients_each_step, gradients),0)

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
        
        if (train_loss_iter < min_loss):
            best_params = result_params
            best_iter = iteration
            min_loss = train_loss_iter

        #####finding the average of hidden states after each iteration of optim_it
        
        
        # previous_loss = loss

#######   unroll when necessary to update the params of the LO   ############
        # if iteration_mod_unroll == 0:
            

            #need to update the grads for optimizee parameters too since they also are connected to the RNN in the computational graph
            #reducing them to zero is necessary
        optimizee = w(target_optimizee)
        optimizee.load_state_dict(result_params)
        optimizee.zero_grad()
        hidden_states = [detach_var(v) for v in hidden_states2]
        cell_states = [detach_var(v) for v in cell_states2]

###################################################################################################
        #zero order flatness
###################################################################################################
        offset = 0
        result_params_ZO = {}
        z_ord_flat_update = zero_order_flat(gradient=gradients_each_step, rho = 0.05)
        for name, p in optimizee.named_parameters():
            cur_sz = int(np.prod(p.size()))
            update_w = z_ord_flat_update[offset:offset+cur_sz]
            result_params_ZO[name] = p + update_w.view(*p.size())

            offset += cur_sz

        optimizee_ZO = w(target_optimizee)
        optimizee_ZO.load_state_dict(result_params_ZO)
        nghb_loss, _ , __ = optimizee_ZO(target)

        zero_order_flatness_loss.append(nghb_loss.item() - train_loss_iter)


##################################################################################################



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


        if iteration in [1, 100, 500, 999]:
            eigenvals = eig_values(best_params)
            print(f'best iter : {best_iter}, training loss: {train_loss_iter}, valid loss: {loss_val_iter/count}')
            print(eigenvals)
            eigen_values[best_iter] = eigenvals.tolist()

            best_params = result_params
            best_iter = iteration+1


        if (iteration%10 == 0):
            # print(f"Training loss:{train_loss_iter}")
        
            print(f"Flatness_loss:{nghb_loss.item() - train_loss_iter} Training loss:{train_loss_iter} Training Accuracy: {tr_acc} Validation Loss: {loss_val_iter/count} Validation Accuracy: {acc_val_iter}")
################################################################################
        
    
    #all_losses_ever is the loss that optimizee gives after each iteration
    return {"eigen_vals": eigen_values,"flatness_loss":zero_order_flatness_loss, "train_loss" :train_loss, "train_acc": train_acc, "test_loss": test_loss, "test_acc": test_acc }
    # return eigen_values.tolist()




def get_perb_params_loss_grad(model_orig_dict, dataloader, model_perb, direction, alpha):
    perturb = {}
    # print(len(direction))
    # print(model_orig_dict.keys())
    i = 0
    for name in model_orig_dict.keys():
        assert int(np.prod(model_orig_dict[name].size())) == len(direction[i])
        # print(model_orig_dict[name].size())
        # print(torch.tensor(direction[i]).size())
        dir_param  = w(torch.tensor(direction[i]).view(*(model_orig_dict[name].size())))
        perturb[name] = model_orig_dict[name] + alpha*dir_param
        i += 1    
    model_perb.load_state_dict(perturb)
    model_perb.zero_grad()
    tot_loss = 0
    count = 0
    for x,y in dataloader:
        x , y = w(x), w(y)
        loss = F.nll_loss(model_perb(x), y)
        tot_loss += loss
        count += 1
    return tot_loss/count




def get_perb_params_loss_hess(model_orig_dict, dataloader, model_perb, direction, alpha):
    perturb = {}
    # print(len(direction))
    # print(model_orig_dict.keys())
    off_set = 0
    for name in model_orig_dict.keys():
        # print(model_orig_dict[name].size())
        # print(torch.tensor(direction[i]).size())
        cur_sz = int(np.prod(model_orig_dict[name].size()))

        dir_param  = w(torch.tensor(direction[off_set:off_set+cur_sz]).view(*(model_orig_dict[name].size())))
        perturb[name] = model_orig_dict[name] + alpha*dir_param
    model_perb.load_state_dict(perturb)
    model_perb.zero_grad()
    tot_loss = 0
    count = 0
    for x,y in dataloader:
        x , y = w(x), w(y)
        loss = F.nll_loss(model_perb(x), y)
        tot_loss += loss
        count += 1
    return tot_loss/count





def fit_val_hessian(opt_net, target_data, target_optimizee, val_data, optim_it, out_mul=1):
    
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
    
    eigen_values = {}

    diff = 1e-3
    min_eval_loss = 99999.99
    patience = 50


    for iteration in tqdm(range(1, optim_it + 1), "iteration: "):  #optim_it = number of iterations the optimizee is trained using LO(inner training steps)
        
        #the loss is always found using all the datapoints (not single or batch loss)
        # difference between consecutive losses will decrease quadratically


        loss, pred, real  = optimizee(target)
        train_loss_iter = loss.item()
        train_loss.append(train_loss_iter)
        tr_acc = accuracy(pred, real)
        train_acc.append(tr_acc)


        # iteration_mod_unroll = iteration % unroll

        loss.backward(retain_graph=False)

        #gradients wrt all the parameters are updated
        

        offset = 0
        result_params = {}
        hidden_states2 = [w(Variable(torch.zeros(n_params, opt_net.hidden_sz))) for _ in range(2)]
        cell_states2 = [w(Variable(torch.zeros(n_params, opt_net.hidden_sz))) for _ in range(2)]
        
        grad_list_each_step = []

        

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

  
   

            #need to update the grads for optimizee parameters too since they also are connected to the RNN in the computational graph
            #reducing them to zero is necessary
        optimizee = w(target_optimizee)
        optimizee.load_state_dict(result_params)
        optimizee.zero_grad()
        hidden_states = [detach_var(v) for v in hidden_states2]
        cell_states = [detach_var(v) for v in cell_states2]



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
        

        #early stopping
        check = min_eval_loss - (train_loss_iter)

        if (check > diff):
            min_eval_loss = train_loss_iter
            best_params = result_params
            best_iter = iteration
            tolerance = 0

        else:
            tolerance += 1

            if(tolerance > patience):
                eigen_values[best_iter] = eig_values(best_params)
                eigen_vals, eigen_vecs = eig_values(best_params)
                eigen_values[best_iter] = eigen_vals

                als = np.linspace(-0.5, 0.5, 21).astype(np.float32)
                hess_plot = []
                grad_plot = []
                for al in als:
                    print(f"testing for alpha: {al}")
                    dummy_dataset_train_loader = torch.utils.data.DataLoader(dummy_dataset_train, 
                                                                batch_size=128,
                                                                sampler=torch.utils.data.sampler.SubsetRandomSampler(indices))
    
                    model_perb = w(dummyMNISTNet())
                    loss_grad = get_perb_params_loss_grad(best_params, dummy_dataset_train_loader, model_perb, grad_list_each_step, al)
                    grad_plot.append(loss_grad)
                    dummy_dataset_train_loader = torch.utils.data.DataLoader(dummy_dataset_train, 
                                                                batch_size=128,
                                                                sampler=torch.utils.data.sampler.SubsetRandomSampler(indices))
                    model_perb = w(dummyMNISTNet())
                    loss_hess = get_perb_params_loss_hess(best_params, dummy_dataset_train_loader, model_perb, eigen_vecs[0], al)
                    print(f"loss in the gradient direction: {loss_grad}, loss in the hessian principal component direction: {loss_hess}")
                    hess_plot.append(loss_hess)
                print(f"grad_plot: {grad_plot}")
                print(f"hess_plot: {hess_plot}")

                break


        # if(iteration == 1):
        #     eigen_values[1] = eig_values(best_params)
        if(len(eigen_values.keys()) == 0 and iteration==optim_it):
            eigen_values[best_iter] = eig_values(best_params)
            eigen_values[best_iter] = eig_values(best_params)
            als = np.linspace(-0.5, 0.5, 21).astype(np.float32)
            hess_plot = []
            grad_plot = []
            for al in als:
                print(f"testing for alpha: {al}")
                dummy_dataset_train_loader = torch.utils.data.DataLoader(dummy_dataset_train, 
                                                            batch_size=128,
                                                            sampler=torch.utils.data.sampler.SubsetRandomSampler(indices))

                model_perb = w(dummyMNISTNet())
                loss_grad = get_perb_params_loss_grad(best_params, dummy_dataset_train_loader, model_perb, grad_list_each_step, al)
                grad_plot.append(loss_grad.item())
                dummy_dataset_train_loader = torch.utils.data.DataLoader(dummy_dataset_train, 
                                                            batch_size=128,
                                                            sampler=torch.utils.data.sampler.SubsetRandomSampler(indices))
                model_perb = w(dummyMNISTNet())
                loss_hess = get_perb_params_loss_hess(best_params, dummy_dataset_train_loader, model_perb, eigen_vecs[0], al)
                print(f"loss in the gradient direction: {loss_grad}, loss in the hessian principal component direction: {loss_hess}")
                hess_plot.append(loss_hess.item())
            print(f"grad_plot: {grad_plot}")
            print(f"hess_plot: {hess_plot}")
            
        if (iteration%10 == 0):
            print(f"Training loss:{train_loss_iter} Training Accuracy: {tr_acc} Validation Loss: {loss_val_iter/count} Validation Accuracy: {acc_val_iter}")
        
    #all_losses_ever is the loss that optimizee gives after each iteration
    return {"eigen_vals": eigen_values, "train_loss" :train_loss, "train_acc": train_acc, "test_loss": test_loss, "test_acc": test_acc, "grad_plot": grad_plot, "hess_plot": hess_plot }




if __name__ == "__main__":


    #loading pretrained LO model (DMO)
    SDMO_dict = torch.load("/nlsasfs/home/metalearning/nckris/Suraj_LSAIML/l2l/checkpoints_LO/simple_DMO/nobias_loss_diff_CPpoint_sDMO_lr_0.001.pth")

    SDMO_instance = w(SDMOptimizer(preproc= False))
    SDMO_instance.load_state_dict(SDMO_dict['optimizer_state_dict'])


    all_history = {}
    for i in range(10):
        target = MNIST_Data(batch_size=128)
        val_target = MNIST_valData(batch_size = 128)
        optimizee = w(MNISTNet())
        history = fit_val_hessian(SDMO_instance, target, optimizee, val_target, 500, out_mul=1)
        all_history[i] = history


    import json

    file_name  = "/nlsasfs/home/metalearning/nckris/Suraj_LSAIML/l2l/checkpoints_LO/jsons/AV_p50_sdmo_mnist_list.json"
    with open(file_name, "w") as json_file:
        json.dump(all_history, json_file)