import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import reduce
from operator import mul
import torch.optim as optim
from torch.autograd import Variable


DATASET_NUM_CLASSES = {
    'pacs': 7,
    'mnist': 10,
    'fmnist': 10,
    'cifar10': 10,
    'cifar100': 100,
}

DATASET_IN_CHANNELS = {
    'mnist': 1,
    'fmnist': 1,
    'cifar10': 3,
    'cifar100': 3,
}


class PACSResNet18(nn.Module):
    """torchvision ResNet-18 for PACS (7 classes, 224px, ImageNet-pretrained
    like the MetaReg baseline). log_softmax output to match F.nll_loss used
    across this codebase. Pretrained weights are fetched on the LOGIN node
    (cached in ~/.cache/torch); offline nodes fall back to random init with
    a loud warning (set PACS_PRETRAINED=0 to silence)."""
    def __init__(self, num_classes=7):
        super().__init__()
        import os
        from torchvision import models as tvm
        want = os.environ.get("PACS_PRETRAINED", "1") != "0"

        def _make(pretrained):
            """torchvision 0.10 (Agastya) only knows `pretrained=`;
            torchvision >= 0.13 deprecates and >= 0.15 REMOVES it in favour
            of `weights=`. Support both so the same code runs on the cluster
            and on modern local envs (2026-07-31)."""
            try:
                return tvm.resnet18(pretrained=pretrained)      # tv <= 0.14
            except TypeError:
                w = tvm.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
                return tvm.resnet18(weights=w)                  # tv >= 0.13
        try:
            net = _make(want)
            if want:
                print("[pacs] ResNet-18 initialised with ImageNet weights")
        except Exception as e:
            print(f"[pacs] WARNING: pretrained weights unavailable ({e}); "
                  f"using RANDOM init — paper comparison expects pretrained. "
                  f"Run prefetch_datasets.py ONCE on the login node so the "
                  f"weights land in ~/.cache/torch and offline compute "
                  f"nodes can load them.")
            net = _make(False)
        net.fc = nn.Linear(net.fc.in_features, num_classes)
        self.net = net

    def forward(self, x):
        return F.log_softmax(self.net(x), dim=1)


def initialize_model(dataset='mnist', model_name = 'mlp'):

    assert dataset in DATASET_NUM_CLASSES, \
        f"choose one of {sorted(DATASET_NUM_CLASSES)}, got '{dataset}'"

    num_classes = DATASET_NUM_CLASSES[dataset]

    if dataset == 'pacs':
        assert model_name == 'resnet18', (
            'PACS (exp6) is defined on resnet18')
        return PACSResNet18(num_classes)

    # ------------------------------------------------------------------
    # ACTIVE registry (design v2): mlp (MNIST/FMNIST optimizee),
    # resnet8 (CIFAR meta-training PROXY), resnet18 (CIFAR meta-test
    # target — canonical CIFAR ResNet-18, 11,173,962 params).
    # conv/big_conv REMOVED (superseded by the resnet8->ResNet-18 path);
    # mlp_Relu/deep_mlp/vgg* unregistered legacy (classes kept below for
    # reference only, not reachable from configs/CLI).
    # ------------------------------------------------------------------
    if (model_name == 'mlp'):
        return Model(num_classes, dataset)
    elif (model_name == 'resnet18'):
        return ResNet18(num_classes, dataset)
    elif (model_name == 'resnet8'):
        return ResNet8(num_classes, dataset)
    elif (model_name == 'vgg16'):
        # meta-TEST target for the VGG family (cross-architecture Exps
        # 10/11); meta-training THROUGH it hits the same BPTT wall as
        # ResNet-18 -> use the vgg8 proxy for training.
        return VGG('VGG16', num_classes, dataset)
    else:
        raise ValueError(f"Unknown/retired model_name '{model_name}'. Active models: mlp, resnet8, resnet18, vgg8, vgg16 (design v3 — conv/big_conv removed; see Change.MD item 91)")
    



class Model(nn.Module):

    def __init__(self, num_classes, dataset):
        super(Model, self).__init__()
        self.dataset = dataset

        if self.dataset == 'mnist' or self.dataset == 'fmnist':
            self.linear1 = nn.Linear(28 * 28, 20)
        elif self.dataset in ('cifar10', 'cifar100'):
            self.linear1 = nn.Linear(32*32*3, 20)

        self.linear2 = nn.Linear(20, num_classes)

        self.initialize_weights()

    def forward(self, inputs):
        if self.dataset == 'mnist' or self.dataset == 'fmnist':
            x = inputs.view(-1, 28 * 28)
        elif self.dataset in ('cifar10', 'cifar100'):
            x = inputs.view(-1, 32 * 32 * 3)

        x = F.sigmoid(self.linear1(x))
        x = self.linear2(x)
        return F.log_softmax(x, dim=-1)

    def initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean=0, std=0.01)
                nn.init.constant_(m.bias, 0)

    def get_flat_params(self):
        params = []

        for module in self.children():
            params.append(module._parameters['weight'].view(-1))
            params.append(module._parameters['bias'].view(-1))

        return torch.cat(params)

    def set_flat_params(self, flat_params):
        # Restore original shapes
        offset = 0
        for i, module in enumerate(self.children()):
            weight_shape = module._parameters['weight'].size()
            bias_shape = module._parameters['bias'].size()

            weight_flat_size = reduce(mul, weight_shape, 1)
            bias_flat_size = reduce(mul, bias_shape, 1)

            module._parameters['weight'].data = flat_params[
                offset:offset + weight_flat_size].view(*weight_shape).data
            module._parameters['bias'].data = flat_params[
                offset + weight_flat_size:offset + weight_flat_size + bias_flat_size].view(*bias_shape).data

            offset += weight_flat_size + bias_flat_size

class Model_Relu(nn.Module):

    def __init__(self, num_classes, dataset):
        super(Model_Relu, self).__init__()
        self.dataset = dataset

        if self.dataset == 'mnist' or self.dataset == 'fmnist':
            self.linear1 = nn.Linear(28 * 28, 20)
        elif self.dataset in ('cifar10', 'cifar100'):
            self.linear1 = nn.Linear(32*32*3, 20)
        self.linear2 = nn.Linear(20, int(num_classes))

        self.initialize_weights()

    def forward(self, inputs):
        if self.dataset == 'mnist' or self.dataset == 'fmnist':
            x = inputs.view(-1, 28 * 28)
        elif self.dataset in ('cifar10', 'cifar100'):
            x = inputs.view(-1, 32 * 32 * 3)
        x = F.relu(self.linear1(x))
        x = self.linear2(x)
        return F.log_softmax(x, dim=-1)

    def initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean=0, std=0.01)
                nn.init.constant_(m.bias, 0)


    def get_flat_params(self):
        params = []

        for module in self.children():
            params.append(module._parameters['weight'].view(-1))
            params.append(module._parameters['bias'].view(-1))

        return torch.cat(params)

    def set_flat_params(self, flat_params):
        # Restore original shapes
        offset = 0
        for i, module in enumerate(self.children()):
            weight_shape = module._parameters['weight'].size()
            bias_shape = module._parameters['bias'].size()

            weight_flat_size = reduce(mul, weight_shape, 1)
            bias_flat_size = reduce(mul, bias_shape, 1)

            module._parameters['weight'].data = flat_params[
                offset:offset + weight_flat_size].view(*weight_shape).data
            module._parameters['bias'].data = flat_params[
                offset + weight_flat_size:offset + weight_flat_size + bias_flat_size].view(*bias_shape).data

            offset += weight_flat_size + bias_flat_size



class ModelDeep(nn.Module):

    def __init__(self, num_classes, dataset):
        super(ModelDeep, self).__init__()
        self.dataset = dataset

        if self.dataset == 'mnist' or self.dataset == 'fmnist':
            self.linear1 = nn.Linear(28 * 28, 32)
        elif self.dataset in ('cifar10', 'cifar100'):
            self.linear1 = nn.Linear(32*32*3, 32)

        self.linear2 = nn.Linear(32, 32)
        self.linear3 = nn.Linear(32, num_classes)

    def forward(self, inputs):
        if self.dataset == 'mnist' or self.dataset == 'fmnist':
            x = inputs.view(-1, 28 * 28)
        elif self.dataset in ('cifar10', 'cifar100'):
            x = inputs.view(-1, 32 * 32 * 3)

        x = F.relu(self.linear1(x))
        x = F.relu(self.linear2(x))
        x = self.linear3(x)
        return F.log_softmax(x, dim=-1)

    def get_flat_params(self):
        params = []

        for module in self.children():
            params.append(module._parameters['weight'].view(-1))
            params.append(module._parameters['bias'].view(-1))

        return torch.cat(params)

    def set_flat_params(self, flat_params):
        # Restore original shapes
        offset = 0
        for i, module in enumerate(self.children()):
            weight_shape = module._parameters['weight'].size()
            bias_shape = module._parameters['bias'].size()

            weight_flat_size = reduce(mul, weight_shape, 1)
            bias_flat_size = reduce(mul, bias_shape, 1)

            module._parameters['weight'].data = flat_params[
                offset:offset + weight_flat_size].view(*weight_shape).data
            module._parameters['bias'].data = flat_params[
                offset + weight_flat_size:offset + weight_flat_size + bias_flat_size].view(*bias_shape).data

            offset += weight_flat_size + bias_flat_size








# (class Conv_NN removed — design v2: only mlp/resnet8/resnet18 remain; see Change.MD item 91)

# (class BigConvNN removed — design v2: only mlp/resnet8/resnet18 remain; see Change.MD item 91)

class FlatParamsMixin:

    def get_flat_params(self):
        params = []
        for module in self.modules():
            if len(module._parameters) == 0:
                continue
            if module._parameters.get('weight') is not None:
                params.append(module._parameters['weight'].view(-1))
            if module._parameters.get('bias') is not None:
                params.append(module._parameters['bias'].view(-1))
        return torch.cat(params)

    def set_flat_params(self, flat_params):
        offset = 0
        for module in self.modules():
            if len(module._parameters) == 0:
                continue
            if module._parameters.get('weight') is not None:
                w_shape = module._parameters['weight'].size()
                w_size = reduce(mul, w_shape, 1)
                module._parameters['weight'].data = flat_params[
                    offset:offset + w_size].view(*w_shape).data
                offset += w_size
            if module._parameters.get('bias') is not None:
                b_shape = module._parameters['bias'].size()
                b_size = reduce(mul, b_shape, 1)
                module._parameters['bias'].data = flat_params[
                    offset:offset + b_size].view(*b_shape).data
                offset += b_size


# ============================================================================
# CIFAR-style ResNet (He et al. 2016, CIFAR variant: 3x3 stem, no maxpool).
# ResNet18 ~11.2M params for meta-TEST scaling experiments (reviewer point:
# "evaluate on ResNet-18 / WideResNet on CIFAR-10/100").
# ResNet8 is a small proxy (~78K params) usable for meta-TRAINING.
#
# NOTE on usage with the LO:
#   - meta-test through the LO works (coordinate-wise LSTM); use
#     param_chunk_size in MetaOptimizerLSTM to bound memory (see
#     meta_optimizer.py) and batch_size <= 128 on a 16GB V100.
#   - full BPTT meta-training through ResNet18 is NOT feasible; meta-train
#     on 'resnet8'/'big_conv' proxies and transfer (matches the paper plan).
# ============================================================================
class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3,
                               stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1,
                          stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = out + self.shortcut(x)
        return F.relu(out)


class _ResNet(FlatParamsMixin, nn.Module):

    def __init__(self, block, num_blocks, num_classes, data, base_width=64):
        super(_ResNet, self).__init__()
        in_ch = 3 if data in ('cifar10', 'cifar100') else 1
        self.in_planes = base_width

        self.conv1 = nn.Conv2d(in_ch, base_width, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(base_width)
        self.layer1 = self._make_layer(block, base_width, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, base_width * 2, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, base_width * 4, num_blocks[2], stride=2)
        if len(num_blocks) == 4:
            self.layer4 = self._make_layer(block, base_width * 8, num_blocks[3], stride=2)
            final_planes = base_width * 8
        else:
            self.layer4 = None
            final_planes = base_width * 4
        self.linear = nn.Linear(final_planes * block.expansion, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for s in strides:
            layers.append(block(self.in_planes, planes, s))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        if self.layer4 is not None:
            out = self.layer4(out)
        out = F.adaptive_avg_pool2d(out, 1)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return F.log_softmax(out, dim=-1)


def ResNet18(num_classes, data):
    """Standard CIFAR ResNet-18 (~11.2M params). Meta-test target."""
    return _ResNet(BasicBlock, [2, 2, 2, 2], num_classes, data, base_width=64)


def ResNet8(num_classes, data):
    """Tiny ResNet proxy (~78K params, base_width=16). Meta-train friendly."""
    return _ResNet(BasicBlock, [1, 1, 1], num_classes, data, base_width=16)


# ============================================================================
# VGG with BatchNorm (CIFAR variant). VGG11 ~9.2M / VGG16 ~14.7M params.
# Meta-test scaling target requested for the revision.
# ============================================================================
VGG_CFG = {
    'VGG11': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'VGG16': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M',
              512, 512, 512, 'M', 512, 512, 512, 'M'],
}




# (VGG8Proxy REMOVED 2026-07-15: it was the VGG-family meta-training
# proxy for the retired exp11 (vgg8 -> resnet18). The final 5-experiment
# plan only ever meta-trains on the ResNet family (resnet8) and uses
# VGG-16 as a meta-TEST target (exp5), so nothing referenced it.)

class VGG(FlatParamsMixin, nn.Module):

    def __init__(self, vgg_name, num_classes, data):
        super(VGG, self).__init__()
        self.in_ch = 3 if data in ('cifar10', 'cifar100') else 1
        self.features = self._make_layers(VGG_CFG[vgg_name])
        self.classifier = nn.Linear(512, num_classes)

    def forward(self, x):
        # grayscale datasets are 28x28; pad to 32x32 so the 5 pools work
        if x.size(-1) == 28:
            x = F.pad(x, (2, 2, 2, 2))
        out = self.features(x)
        out = out.view(out.size(0), -1)
        out = self.classifier(out)
        return F.log_softmax(out, dim=-1)

    def _make_layers(self, cfg):
        layers = []
        in_channels = self.in_ch
        for v in cfg:
            if v == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                layers += [nn.Conv2d(in_channels, v, kernel_size=3, padding=1, bias=False),
                           nn.BatchNorm2d(v),
                           nn.ReLU(inplace=True)]
                in_channels = v
        return nn.Sequential(*layers)
