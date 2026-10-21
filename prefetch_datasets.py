#!/usr/bin/env python3
"""Run ONCE on the Agastya LOGIN node (compute nodes have no internet):
downloads MNIST/FMNIST/CIFAR-10 into datasets/, warms the torchvision
ResNet-18 ImageNet weights cache (~/.cache/torch) needed by Experiment 6
(PACS), and checks that CIFAR-10-C and PACS are in place."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent / "src/learning_regularizers/dmo"))
from torchvision import datasets                                    # noqa: E402
root = Path(__file__).parent / "datasets"; root.mkdir(exist_ok=True)
for cls in (datasets.MNIST, datasets.FashionMNIST, datasets.CIFAR10):
    for train in (True, False):
        cls(str(root), train=train, download=True)
    print(f"{cls.__name__}: ok")

# ---- exp6: ImageNet-pretrained ResNet-18 weights (offline cache) -------
try:
    from torchvision import models as tvm
    try:
        tvm.resnet18(pretrained=True)            # torchvision <= 0.14 (Agastya 0.10)
    except TypeError:
        tvm.resnet18(weights=tvm.ResNet18_Weights.IMAGENET1K_V1)
    print("ResNet-18 ImageNet weights: cached (compute nodes will load offline)")
except Exception as e:
    print(f"ResNet-18 ImageNet weights: FAILED ({e}) — exp6 would fall back "
          f"to RANDOM init; fix connectivity on the login node and rerun.")

c10c = root / "CIFAR-10-C" / "labels.npy"
print("CIFAR-10-C:", "present" if c10c.exists() else
      f"MISSING - download zenodo.org/record/2535967 into {c10c.parent}")
pacs = root / "PACS"
doms = ["art_painting", "cartoon", "photo", "sketch"]
have = [d for d in doms if (pacs / d).is_dir()]
print("PACS:", "present (%s)" % ", ".join(have) if len(have) == 4 else
      f"MISSING domains {sorted(set(doms)-set(have))} - place as "
      f"datasets/PACS/<domain>/<class>/*.jpg (DomainBed script or Kaggle "
      f"pacs-dataset), then rerun this check.")
