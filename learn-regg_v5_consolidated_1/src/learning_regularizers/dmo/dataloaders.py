"""
dataloaders.py
==============
Unified data loading for meta-training and meta-testing.

Supported datasets: mnist, fmnist, cifar10, cifar100

Conventions (unchanged from the original code):
  - data_loader(...)          : meta-TRAIN side. The train split is halved:
                                first half -> LO meta-training stream,
                                second half -> held-out stream.
  - data_loader_optimizee(...): meta-TEST side. Optimizee trains on the
                                (other) half / full train split and is
                                evaluated on the official test split.

New:
  - cifar100 support everywhere (100 classes handled in model.py).
  - `augment=True` adds RandomCrop(32, pad=4) + HorizontalFlip for CIFAR,
    the standard recipe for ResNet/VGG-scale meta-test runs (reviewer
    request: "with standard data augmentation").
"""
from pathlib import Path
import sys

from torchvision import datasets, transforms
import torch
import numpy as np

sys.path.append(str(Path(__file__).resolve().parents[3]))

from project_paths import datasets_dir


DATASETS_ROOT = datasets_dir(__file__)


_NORM = {
    'mnist':    ((0.1307,), (0.3081,)),
    'fmnist':   ((0.5,), (0.5,)),
    'cifar10':  ((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261)),
    'cifar100': ((0.5071, 0.4865, 0.4409), (0.2673, 0.2564, 0.2762)),
}

_DATASET_CLS = {
    'mnist':    datasets.MNIST,
    'fmnist':   datasets.FashionMNIST,
    'cifar10':  datasets.CIFAR10,
    'cifar100': datasets.CIFAR100,
}


def _check(data):
    if str(data).lower() in ('cifar10c', 'cifar10-c', 'cifar-10-c'):
        raise ValueError(
            "CIFAR-10-C is EVALUATION-ONLY corruption data — it is not a "
            "training dataset. Run a cifar10 evaluation with "
            "--set eval_cifar10c=True (Experiment 6); the final model is "
            "then scored on every corruption x severity via cifar10c_loader().")
    if data not in _DATASET_CLS:
        raise ValueError(f"Unknown dataset '{data}'. Choose from {sorted(_DATASET_CLS)}")


def _transforms(data, augment=False):
    mean, std = _NORM[data]
    eval_tf = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])
    if augment and data in ('cifar10', 'cifar100'):
        train_tf = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])
    else:
        train_tf = eval_tf
    return train_tf, eval_tf


def data_loader(data='mnist', batch_size=128, random_seed=10, augment=False,
                num_workers=0, split_seed=42, pacs_holdout=None, pacs_root_dir=None, pacs_img_size=224):
    if data == 'pacs':
        if not pacs_holdout:
            raise ValueError('dataset=pacs needs pacs_holdout=<domain> '
                             f'(one of {PACS_DOMAINS})')
        return pacs_loaders(pacs_holdout, batch_size=batch_size,
                            root=pacs_root_dir, img_size=pacs_img_size,
                            random_seed=random_seed)
    """Meta-train side: split the official train set 50/50 into a
    meta-train stream and a held-out stream (SubsetRandomSampler)."""
    _check(data)
    train_tf, _ = _transforms(data, augment)
    ds_cls = _DATASET_CLS[data]

    trainset = ds_cls(str(DATASETS_ROOT), train=True, download=True,
                      transform=train_tf)

    # split_seed (NOT the run seed) fixes the half-split so the meta-train
    # and meta-test halves are disjoint across every run/seed (item 27).
    indices = list(range(len(trainset)))
    np.random.RandomState(split_seed).shuffle(indices)
    indices_train = indices[:len(indices) // 2]
    indices_heldout = indices[len(indices) // 2:]

    train_loader = torch.utils.data.DataLoader(
        trainset, batch_size=batch_size, num_workers=num_workers,
        sampler=torch.utils.data.sampler.SubsetRandomSampler(indices_train))

    heldout_loader = torch.utils.data.DataLoader(
        trainset, batch_size=batch_size, num_workers=num_workers,
        sampler=torch.utils.data.sampler.SubsetRandomSampler(indices_heldout))

    return train_loader, heldout_loader


def data_loader_optimizee(data='mnist', batch_size=128, random_seed=10,
                          augment=False, num_workers=0, split_seed=42,
                          val_fraction=0.1,
                          pacs_holdout=None, pacs_root_dir=None,
                          pacs_img_size=224, pacs_num_workers=2):
    if data == 'pacs':
        if not pacs_holdout:
            raise ValueError(f"dataset=pacs needs pacs_holdout=<domain> "
                             f"(one of {PACS_DOMAINS})")
        return pacs_optimizee_loaders(
            pacs_holdout, batch_size=batch_size, root=pacs_root_dir,
            img_size=pacs_img_size, random_seed=random_seed,
            split_seed=split_seed, val_fraction=val_fraction,
            num_workers=int(pacs_num_workers))

    """Meta-test side. Returns (train_loader, val_loader, test_loader).

    A deterministic validation split (val_fraction of the task-train pool,
    carved under split_seed, NO augmentation) supports the per-experiment
    Validation Loss/Accuracy metrics; val_fraction=0 -> val_loader is None.
    mnist keeps the original disjoint-half behaviour vs the meta-train half.
    """
    _check(data)
    train_tf, eval_tf = _transforms(data, augment)
    ds_cls = _DATASET_CLS[data]

    trainset = ds_cls(str(DATASETS_ROOT), train=True, download=True,
                      transform=train_tf)

    indices = list(range(len(trainset)))
    np.random.RandomState(split_seed).shuffle(indices)
    pool = indices[len(indices) // 2:] if data == 'mnist' else indices

    n_val = int(len(pool) * float(val_fraction))
    val_idx, train_idx = pool[:n_val], pool[n_val:]

    train_loader = torch.utils.data.DataLoader(
        trainset, batch_size=batch_size, num_workers=num_workers,
        sampler=torch.utils.data.sampler.SubsetRandomSampler(train_idx))

    val_loader = None
    if n_val > 0:
        valset = ds_cls(str(DATASETS_ROOT), train=True, download=True,
                        transform=eval_tf)   # evaluation transform, no augment
        val_loader = torch.utils.data.DataLoader(
            valset, batch_size=batch_size, num_workers=num_workers,
            sampler=torch.utils.data.sampler.SubsetRandomSampler(val_idx))

    testset = ds_cls(str(DATASETS_ROOT), train=False, download=True,
                     transform=eval_tf)
    test_loader = torch.utils.data.DataLoader(
        testset, batch_size=batch_size, shuffle=True, num_workers=num_workers)

    return train_loader, val_loader, test_loader


# ===========================================================================
# CIFAR-10-C (Hendrycks & Dietterich corruption benchmark) — Experiment 6.
# Expects the standard release layout under <root>/CIFAR-10-C/:
#   labels.npy  +  <corruption>.npy  each (50000, 32, 32, 3) uint8,
#   stacked as 5 severities x 10000 test images.
# Download once from https://zenodo.org/record/2535967 on the cluster.
# ===========================================================================

CIFAR10C_CORRUPTIONS = [
    'gaussian_noise', 'shot_noise', 'impulse_noise',
    'defocus_blur', 'glass_blur', 'motion_blur', 'zoom_blur',
    'snow', 'frost', 'fog', 'brightness',
    'contrast', 'elastic_transform', 'pixelate', 'jpeg_compression',
]


class _CIFAR10CDataset(torch.utils.data.Dataset):
    """One (corruption, severity) slice, normalised with CIFAR-10 stats."""

    def __init__(self, images_u8, labels):
        self.images = images_u8            # (10000, 32, 32, 3) uint8
        self.labels = labels.astype(np.int64)
        mean, std = _NORM['cifar10']
        self._mean = torch.tensor(mean).view(3, 1, 1)
        self._std = torch.tensor(std).view(3, 1, 1)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, i):
        x = torch.from_numpy(self.images[i]).float().div_(255.0)
        x = x.permute(2, 0, 1)             # HWC -> CHW
        x = (x - self._mean) / self._std
        return x, int(self.labels[i])


def cifar10c_root(root=None):
    return Path(root) if root else DATASETS_ROOT / 'CIFAR-10-C'


def cifar10c_available(root=None):
    return (cifar10c_root(root) / 'labels.npy').exists()


def cifar10c_loader(corruption, severity, batch_size=512, root=None,
                    num_workers=0):
    """DataLoader for one corruption at one severity (1..5)."""
    r = cifar10c_root(root)
    labels = np.load(r / 'labels.npy')
    images = np.load(r / f'{corruption}.npy', mmap_mode='r')
    sl = slice((severity - 1) * 10000, severity * 10000)
    ds = _CIFAR10CDataset(np.ascontiguousarray(images[sl]), labels[sl])
    return torch.utils.data.DataLoader(ds, batch_size=batch_size,
                                       shuffle=False, num_workers=num_workers)


def cifar10c_corruptions_present(root=None):
    r = cifar10c_root(root)
    return [c for c in CIFAR10C_CORRUPTIONS if (r / f'{c}.npy').exists()]


# ======================================================================
# PACS (Li et al. 2017) — Experiment 6: MetaReg-style leave-one-domain-out
# domain generalization (Balaji et al., NeurIPS 2018, Sec. 4.1).
# Expected layout (not bundled; download on the LOGIN node, e.g. the
# DomainBed script or Kaggle 'pacs-dataset'):
#     datasets/PACS/{art_painting,cartoon,photo,sketch}/<class>/*.jpg
# Protocol: train the optimizee on the 3 SOURCE domains jointly, evaluate
# on the HELD-OUT domain. ImageNet normalisation, 224px (configurable so
# smoke tests can run at 64px).
# ======================================================================
PACS_DOMAINS = ["art_painting", "cartoon", "photo", "sketch"]


def _pacs_check_classes(root):
    """ImageFolder assigns labels by SORTED subdirectory name per domain.
    If any domain is missing a class folder (or has a stray extra one) the
    label indices silently misalign across domains and every cross-domain
    accuracy is garbage. Fail loudly instead (added 2026-07-31)."""
    per = {}
    for d in PACS_DOMAINS:
        per[d] = sorted(p.name for p in (root / d).iterdir() if p.is_dir())
    ref = per[PACS_DOMAINS[0]]
    bad = {d: c for d, c in per.items() if c != ref}
    if bad:
        raise RuntimeError(
            f"PACS class folders differ across domains under {root}: "
            f"{PACS_DOMAINS[0]}={ref} but {bad}. Fix the dataset layout — "
            f"labels would silently misalign.")
    return ref


def pacs_root(root=None):
    if root:
        return Path(root)
    try:                                   # repo layout: <root>/datasets/PACS
        from project_paths import PROJECT_ROOT
        return Path(PROJECT_ROOT) / "datasets" / "PACS"
    except Exception:
        return Path(__file__).resolve().parents[3] / "datasets" / "PACS"


def pacs_loaders(holdout, batch_size=64, root=None, img_size=224,
                 random_seed=10, num_workers=2):
    from torchvision import datasets as tvd
    r = pacs_root(root)
    assert holdout in PACS_DOMAINS, f"pacs_holdout must be one of {PACS_DOMAINS}"
    missing = [d for d in PACS_DOMAINS if not (r / d).is_dir()]
    if missing:
        raise FileNotFoundError(
            f"PACS domains missing under {r}: {missing}. Download PACS on "
            f"the login node into datasets/PACS/<domain>/<class>/*.jpg")
    _pacs_check_classes(r)
    mean, std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]   # ImageNet
    tf_train = transforms.Compose([
        transforms.RandomResizedCrop(img_size, scale=(0.7, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(), transforms.Normalize(mean, std)])
    tf_test = transforms.Compose([
        transforms.Resize(int(img_size * 1.15)),
        transforms.CenterCrop(img_size),
        transforms.ToTensor(), transforms.Normalize(mean, std)])
    sources = [d for d in PACS_DOMAINS if d != holdout]
    train_ds = torch.utils.data.ConcatDataset(
        [tvd.ImageFolder(str(r / d), transform=tf_train) for d in sources])
    test_ds = tvd.ImageFolder(str(r / holdout), transform=tf_test)
    g = torch.Generator(); g.manual_seed(random_seed)
    train = torch.utils.data.DataLoader(train_ds, batch_size=batch_size,
                                        shuffle=True, generator=g,
                                        num_workers=num_workers, drop_last=True)
    test = torch.utils.data.DataLoader(test_ds, batch_size=batch_size,
                                       shuffle=False, num_workers=num_workers)
    print(f"[pacs] holdout={holdout} | train on {sources} "
          f"({len(train_ds)} imgs) -> eval on {holdout} ({len(test_ds)} imgs)")
    return train, test


def pacs_optimizee_loaders(holdout, batch_size=64, root=None, img_size=224,
                           random_seed=10, split_seed=42, val_fraction=0.1,
                           num_workers=2):
    """Meta-test triple for PACS: (train, val, test). Validation is carved
    from the SOURCE-domain pool (deterministic under split_seed, eval
    transforms — no augmentation), never from the held-out domain: model
    selection must not peek at the target, per the MetaReg protocol."""
    from torchvision import datasets as tvd
    r = pacs_root(root)
    assert holdout in PACS_DOMAINS, f"pacs_holdout must be one of {PACS_DOMAINS}"
    missing = [d for d in PACS_DOMAINS if not (r / d).is_dir()]
    if missing:
        raise FileNotFoundError(
            f"PACS domains missing under {r}: {missing}. Download PACS on "
            f"the login node into datasets/PACS/<domain>/<class>/*.jpg")
    _pacs_check_classes(r)
    mean, std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
    tf_train = transforms.Compose([
        transforms.RandomResizedCrop(img_size, scale=(0.7, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(), transforms.Normalize(mean, std)])
    tf_eval = transforms.Compose([
        transforms.Resize(int(img_size * 1.15)),
        transforms.CenterCrop(img_size),
        transforms.ToTensor(), transforms.Normalize(mean, std)])
    sources = [d for d in PACS_DOMAINS if d != holdout]
    aug = torch.utils.data.ConcatDataset(
        [tvd.ImageFolder(str(r / d), transform=tf_train) for d in sources])
    plain = torch.utils.data.ConcatDataset(
        [tvd.ImageFolder(str(r / d), transform=tf_eval) for d in sources])
    n = len(aug)
    g = torch.Generator(); g.manual_seed(int(split_seed))
    perm = torch.randperm(n, generator=g).tolist()
    n_val = int(n * float(val_fraction))
    val_idx, tr_idx = perm[:n_val], perm[n_val:]
    train_ds = torch.utils.data.Subset(aug, tr_idx)
    val_ds = torch.utils.data.Subset(plain, val_idx) if n_val else None
    test_ds = tvd.ImageFolder(str(r / holdout), transform=tf_eval)
    gt = torch.Generator(); gt.manual_seed(int(random_seed))
    mk = lambda ds, sh, gen=None: torch.utils.data.DataLoader(
        ds, batch_size=batch_size, shuffle=sh, generator=gen,
        num_workers=num_workers, drop_last=sh)
    print(f"[pacs] holdout={holdout} | train {len(train_ds)} (3 src domains, "
          f"aug) | val {0 if val_ds is None else len(val_ds)} (src, plain) | "
          f"test {len(test_ds)} ({holdout})")
    return (mk(train_ds, True, gt),
            None if val_ds is None else mk(val_ds, False),
            mk(test_ds, False))
