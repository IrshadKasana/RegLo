"""Unified datasets for the pipeline.

Every dataset is presented as 3 x 32 x 32 tensors so that one optimizee architecture can be
trained on the seen dataset (SVHN) and evaluated frozen on every unseen dataset:

    svhn, cifar10, cifar100   native 3 x 32 x 32
    mnist, fmnist             28 x 28 grey  -> resized to 32 x 32, replicated to 3 channels
    imagenet                  imagenet32/{x,y}.npy (32x32, preferred) or ImageFolder (train/, val/) -> Resize + CenterCrop

Splits ("roles") mirror the base project, which keeps meta-training data disjoint from the
data used to train optimizees with a frozen optimizer:

    meta_train       first half of the train split of the meta-training dataset
    optimizee_train  second half of it (full train split for any other dataset)
    test             official test split (ImageNet: val)
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import datasets, transforms

from .config import REPO_ROOT


@dataclass(frozen=True)
class DatasetInfo:
    name: str
    num_classes: int
    mean: Tuple[float, float, float]
    std: Tuple[float, float, float]
    grayscale: bool = False


DATASETS = {
    "svhn": DatasetInfo("svhn", 10, (0.4377, 0.4438, 0.4728), (0.1980, 0.2010, 0.1970)),
    "mnist": DatasetInfo("mnist", 10, (0.1307,) * 3, (0.3081,) * 3, grayscale=True),
    "fmnist": DatasetInfo("fmnist", 10, (0.2860,) * 3, (0.3530,) * 3, grayscale=True),
    "cifar10": DatasetInfo("cifar10", 10, (0.4914, 0.4822, 0.4465), (0.2470, 0.2430, 0.2610)),
    "cifar100": DatasetInfo("cifar100", 100, (0.5071, 0.4865, 0.4409), (0.2673, 0.2564, 0.2762)),
    "imagenet": DatasetInfo("imagenet", 1000, (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
}

INPUT_SHAPE = (3, 32, 32)


class DatasetUnavailable(RuntimeError):
    """Raised when a dataset cannot be found/downloaded (e.g. ImageNet not on disk)."""


def data_root(cfg) -> Path:
    if cfg.experiment.get("data_root"):
        return Path(cfg.experiment.data_root).expanduser().resolve()
    env = os.environ.get("LSAIML_DATASETS_DIR")
    if env:
        return Path(env).expanduser().resolve()
    return REPO_ROOT / "datasets"


def num_classes(name: str) -> int:
    return DATASETS[name].num_classes


# --------------------------------------------------------------------------- synthetic data

class SyntheticImages(Dataset):
    """Learnable stand-in for a real dataset (class-specific prototypes + noise).

    Used for smoke tests and CI where real datasets cannot be downloaded. Each dataset name
    gets its own prototypes so "unseen" datasets are genuinely different tasks.
    """

    def __init__(self, name: str, split: str, size: int, n_classes: int, noise: float = 1.0):
        base_seed = sum(ord(c) for c in name) * 7919
        proto_gen = torch.Generator().manual_seed(base_seed)
        low_res = torch.randn(n_classes, 3, 8, 8, generator=proto_gen)
        self.prototypes = torch.nn.functional.interpolate(low_res, size=(32, 32), mode="bilinear",
                                                          align_corners=False)
        gen = torch.Generator().manual_seed(base_seed + (1 if split == "train" else 2))
        self.labels = torch.randint(0, n_classes, (size,), generator=gen)
        self.noise = noise * torch.randn(size, *INPUT_SHAPE, generator=gen)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        y = int(self.labels[idx])
        return self.prototypes[y] + self.noise[idx], y


# --------------------------------------------------------------------------- real data

class ImageNet32Npy(Dataset):
    """Downsampled ImageNet (32x32) stored as ``x.npy`` uint8 (N, 3, 32, 32) + ``y.npy`` int64 (N,).

    Layout written by ``prepare_imagenet32.py``: the 1,281,167 official train images first, then
    the 50,000 validation images. ``train=False`` returns the validation images (the test split).
    The arrays are memory-mapped, so nothing is loaded up front.
    """

    N_VAL = 50_000

    def __init__(self, root: Path, train: bool, info: DatasetInfo):
        self.x = np.load(root / "x.npy", mmap_mode="r")
        self.y = np.load(root / "y.npy", mmap_mode="r")
        n_train = len(self.y) - self.N_VAL
        if n_train <= 0:
            raise DatasetUnavailable(f"{root}/y.npy has {len(self.y)} rows; expected train + {self.N_VAL} val")
        self.offset, self.length = (0, n_train) if train else (n_train, self.N_VAL)
        self.mean = torch.tensor(info.mean).view(3, 1, 1)
        self.std = torch.tensor(info.std).view(3, 1, 1)

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        i = self.offset + int(idx)
        img = torch.from_numpy(np.array(self.x[i], dtype=np.float32)) / 255.0
        return (img - self.mean) / self.std, int(self.y[i])


def _transform(info: DatasetInfo, resolution: int = 32):
    ops = []
    if info.name == "imagenet":
        ops += [transforms.Resize(resolution), transforms.CenterCrop(resolution)]
    elif info.grayscale:
        ops += [transforms.Resize(32), transforms.Grayscale(num_output_channels=3)]
    ops += [transforms.ToTensor(), transforms.Normalize(info.mean, info.std)]
    return transforms.Compose(ops)


def _load_split(name: str, split: str, cfg) -> Dataset:
    info = DATASETS[name]
    if cfg.experiment.get("synthetic_data", False):
        syn = cfg.get("datasets", {}).get("synthetic", {})
        size = int(syn.get("train_size", 2048) if split == "train" else syn.get("test_size", 512))
        return SyntheticImages(name, split, size, info.num_classes)

    root = data_root(cfg)
    download = bool(cfg.experiment.get("download", True))
    train = split == "train"
    try:
        if name == "svhn":
            return datasets.SVHN(str(root / "SVHN"), split="train" if train else "test",
                                 download=download, transform=_transform(info))
        if name == "mnist":
            return datasets.MNIST(str(root), train=train, download=download, transform=_transform(info))
        if name == "fmnist":
            return datasets.FashionMNIST(str(root), train=train, download=download, transform=_transform(info))
        if name == "cifar10":
            return datasets.CIFAR10(str(root), train=train, download=download, transform=_transform(info))
        if name == "cifar100":
            return datasets.CIFAR100(str(root), train=train, download=download, transform=_transform(info))
        if name == "imagenet":
            icfg = cfg.get("datasets", {}).get("imagenet", {})
            npy_dir = Path(icfg.get("npy_root") or (root / "imagenet32")).expanduser()
            if (npy_dir / "x.npy").is_file() and (npy_dir / "y.npy").is_file():
                return ImageNet32Npy(npy_dir, train, info)
            iroot = Path(icfg.get("root") or (root / "imagenet")).expanduser()
            sub = iroot / ("train" if train else "val")
            if not sub.is_dir():
                raise DatasetUnavailable(
                    f"ImageNet split not found at {sub}. ImageNet cannot be auto-downloaded; place an "
                    f"ImageFolder layout (train/<wnid>/*.JPEG, val/<wnid>/*.JPEG) there or set "
                    f"datasets.imagenet.root.")
            return datasets.ImageFolder(str(sub), transform=_transform(info, int(icfg.get("resolution", 32))))
    except DatasetUnavailable:
        raise
    except Exception as exc:  # download / IO failure
        raise DatasetUnavailable(f"Could not load dataset '{name}' ({split}) from {root}: {exc}") from exc
    raise ValueError(f"Unknown dataset '{name}'. Known: {sorted(DATASETS)}")


_CACHE: dict = {}


def get_dataset(name: str, role: str, cfg, meta_dataset: Optional[str] = None) -> Dataset:
    """Return the dataset for a role: meta_train | optimizee_train | test."""
    meta_dataset = meta_dataset or cfg.meta_training.dataset
    key = (name, role, meta_dataset, bool(cfg.experiment.get("synthetic_data", False)), int(cfg.experiment.seed))
    if key in _CACHE:
        return _CACHE[key]

    if role == "test":
        ds = _load_split(name, "test", cfg)
    elif role in ("meta_train", "optimizee_train"):
        full = _load_split(name, "train", cfg)
        if name == meta_dataset:
            # disjoint halves, exactly like the base dataloaders (seeded shuffle, first/second half)
            indices = list(range(len(full)))
            np.random.RandomState(int(cfg.experiment.seed)).shuffle(indices)
            half = len(indices) // 2
            ds = Subset(full, indices[:half] if role == "meta_train" else indices[half:])
        else:
            ds = full
    else:
        raise ValueError(f"Unknown role '{role}'")
    _CACHE[key] = ds
    return ds


def subsample(ds: Dataset, max_samples: Optional[int], seed: int = 0) -> Dataset:
    if not max_samples or len(ds) <= max_samples:
        return ds
    idx = np.random.RandomState(seed).permutation(len(ds))[:int(max_samples)]
    return Subset(ds, idx.tolist())


def make_loader(ds: Dataset, batch_size: int, shuffle: bool, seed: int, cfg, drop_last: bool = False) -> DataLoader:
    gen = torch.Generator().manual_seed(int(seed))
    workers = 0 if cfg.experiment.get("synthetic_data", False) else int(cfg.experiment.get("num_workers", 2))
    return DataLoader(ds, batch_size=int(batch_size), shuffle=shuffle, generator=gen,
                      num_workers=workers, drop_last=drop_last,
                      pin_memory=torch.cuda.is_available(), persistent_workers=False)


class InfiniteLoader:
    """next() forever over a DataLoader, re-shuffling at every pass."""

    def __init__(self, loader: DataLoader):
        self.loader = loader
        self._it = iter(loader)

    def __len__(self):
        return len(self.loader)

    def next(self):
        try:
            return next(self._it)
        except StopIteration:
            self._it = iter(self.loader)
            return next(self._it)
