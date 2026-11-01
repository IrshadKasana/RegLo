"""
GPU-resident MNIST.

The model here has 15,910 parameters. At that size the bottleneck is not
compute but DataLoader and host-to-device overhead, so we load the dataset
once, keep it on the device as a single tensor, and index it with a
permutation. This is roughly 50x faster per epoch than a torch DataLoader and
is the main reason the whole sweep fits comfortably on a small GPU.

Memory: 60000 x 784 x 4 bytes = 188 MB for train, ~31 MB for test.
"""

from __future__ import annotations

import torch

MNIST_MEAN, MNIST_STD = 0.1307, 0.3081
DATASET_MB = 220.0          # rough resident footprint, used by require_memory


def _load_raw(root: str):
    from torchvision import datasets
    tr = datasets.MNIST(root, train=True, download=True)
    te = datasets.MNIST(root, train=False, download=True)
    return (tr.data.float().div_(255.0).view(-1, 784), tr.targets.long(),
            te.data.float().div_(255.0).view(-1, 784), te.targets.long())


def _synthetic(n_train=6000, n_test=1000, seed=0):
    """Offline stand-in so the harness can be smoke-tested without a download."""
    g = torch.Generator().manual_seed(seed)
    w = torch.randn(784, 10, generator=g)
    xtr = torch.rand(n_train, 784, generator=g)
    xte = torch.rand(n_test, 784, generator=g)
    ytr = (xtr @ w + 0.3 * torch.randn(n_train, 10, generator=g)).argmax(1)
    yte = (xte @ w + 0.3 * torch.randn(n_test, 10, generator=g)).argmax(1)
    return xtr, ytr, xte, yte


class Mnist:
    """
    Split protocol (FIXED; never depends on the run seed):

        train : 55,000   from a seed-0 permutation of the 60k train set
        val   :  5,000   -> all model and hyperparameter selection
        test  : 10,000   -> touched once, for the final table only

    Keeping the split independent of the run seed means every seed sees the
    same validation set, which is what makes the seed-paired significance
    test in `analyze` valid.

    `label_noise` flips that fraction of TRAINING labels uniformly at random
    (validation and test stay clean). Sharpness-aware methods show much larger
    effects under label noise, so this is the recommended stress condition if
    the clean-MNIST effect turns out too small to resolve.
    """

    def __init__(self, root: str = "./data", device="cuda", n_val: int = 5000,
                 label_noise: float = 0.0, allow_synthetic: bool = False,
                 eval_test: bool = False):
        """
        `eval_test=False` (the default for hyperparameter sweeps) never moves
        the test split to the device and never evaluates it. A tuning study
        should touch the test set exactly zero times: selection happens on
        validation, and the output of the study is the HYPERPARAMETERS, not an
        accuracy number. Set eval_test=True only for a final, one-off
        measurement of already-chosen settings.
        """
        try:
            xtr, ytr, xte, yte = _load_raw(root)
            self.synthetic = False
        except Exception as exc:
            if not allow_synthetic:
                raise RuntimeError(
                    f"Could not load MNIST from {root!r}: {exc}\n"
                    "Run `python -m sharpbench fetch` first (needs internet)."
                ) from exc
            xtr, ytr, xte, yte = _synthetic()
            self.synthetic = True

        xtr = (xtr - MNIST_MEAN) / MNIST_STD
        xte = (xte - MNIST_MEAN) / MNIST_STD

        perm = torch.randperm(xtr.shape[0],
                              generator=torch.Generator().manual_seed(0))
        n_val = min(n_val, xtr.shape[0] // 4)
        tr_idx, va_idx = perm[:-n_val], perm[-n_val:]

        xtr_s, ytr_s = xtr[tr_idx], ytr[tr_idx]

        self.label_noise = float(label_noise)
        if self.label_noise > 0:
            # Fixed noise pattern (seed 1234) so every arm and every seed sees
            # the SAME corrupted labels. Re-randomising per seed would add
            # variance that has nothing to do with the optimizer.
            g = torch.Generator().manual_seed(1234)
            n = ytr_s.numel()
            k = int(round(self.label_noise * n))
            idx = torch.randperm(n, generator=g)[:k]
            ytr_s = ytr_s.clone()
            ytr_s[idx] = torch.randint(0, 10, (k,), generator=g)

        self.device = torch.device(device)
        self.xtr, self.ytr = xtr_s.to(self.device), ytr_s.to(self.device)
        self.xva, self.yva = xtr[va_idx].to(self.device), ytr[va_idx].to(self.device)
        self.eval_test = bool(eval_test)
        if self.eval_test:
            self.xte, self.yte = xte.to(self.device), yte.to(self.device)
        else:
            self.xte = self.yte = None      # test split never leaves the CPU

    @property
    def n_train(self) -> int:
        return self.xtr.shape[0]

    def steps_per_epoch(self, batch_size: int) -> int:
        return self.n_train // batch_size

    def epoch_batches(self, batch_size: int, generator: torch.Generator):
        """
        Yield (x, y) minibatches for one shuffled epoch.

        The permutation is drawn on CPU then moved. This is deliberate:
        `torch.randperm(n, generator=g, device='cuda')` raises a device
        mismatch on several PyTorch builds because the small-n CUDA path falls
        back to a CPU kernel that rejects a CUDA generator. Drawing on CPU also
        makes the data order bit-identical between a CPU debug run and a GPU
        run at the same seed. Cost is ~0.4 MB per epoch.
        """
        idx = torch.randperm(self.n_train, generator=generator).to(self.device)
        for i in range(0, self.n_train - batch_size + 1, batch_size):
            sel = idx[i:i + batch_size]
            yield self.xtr[sel], self.ytr[sel]
