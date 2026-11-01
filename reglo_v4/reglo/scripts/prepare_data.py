#!/usr/bin/env python
"""Download the datasets once, convert them to fast caches, and draw the meta splits.

Run it on your laptop before the smoke test, and on a cluster's login node before the jobs
(compute nodes usually have no internet access). About 0.6 GB in all.

    python scripts/prepare_data.py                        # fmnist, svhn, cifar10, cifar100
    python scripts/prepare_data.py svhn cifar10 fmnist    # the core study (and its smoke test)
    python scripts/prepare_data.py --data-root /scratch/$USER/datasets

Afterwards every run — smoke or production — reads ``<data>/cache/*.pt`` (seconds to load) and
the saved split indices in ``<data>/splits/``. Point runs at the same folder with ``REGLO_DATA``.
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))  # works without `pip install -e .`

from reglo.config import available, load_group_file  # noqa: E402
from reglo.data import DatasetNotPrepared, load_image_data, meta_split  # noqa: E402
from reglo.utils.logging import safe_console  # noqa: E402
from reglo.utils.paths import default_data_root  # noqa: E402

REAL_DATASETS = ["fmnist", "svhn", "cifar10", "cifar100"]


def prepare(name: str, root: Path) -> str:
    cfg = load_group_file("dataset", name)
    cfg.download = True
    train = load_image_data(cfg, root, train=True)
    test = load_image_data(cfg, root, train=False)
    line = (
        f"{name:<10} train {len(train):>6}   test {len(test):>6}   "
        f"image {'×'.join(map(str, train.image_shape))}   classes {train.num_classes}"
    )
    if cfg.meta_split.enabled:
        meta, rest = meta_split(name, len(train), cfg.meta_split.fraction, cfg.meta_split.seed, root / "splits")
        line += f"   meta split: {len(meta)} meta-training / {len(rest)} optimizee"
    return line


def main(argv=None) -> int:
    safe_console()
    parser = argparse.ArgumentParser(description="Download datasets and build caches and meta splits.")
    parser.add_argument("datasets", nargs="*", help=f"default: {' '.join(REAL_DATASETS)}")
    parser.add_argument("--data-root", type=Path, default=None, help="default: $REGLO_DATA or <repo>/datasets")
    args = parser.parse_args(argv)

    names = args.datasets or REAL_DATASETS
    unknown = sorted(set(names) - set(available("dataset")))
    if unknown:
        parser.error(f"unknown dataset(s) {unknown}; choose from {available('dataset')}")
    root = (args.data_root or default_data_root()).expanduser().resolve()
    print(f"data folder: {root}")
    for name in names:
        try:
            print(prepare(name, root))
        except DatasetNotPrepared as err:
            print(f"{name:<10} FAILED: {err}", file=sys.stderr)
            return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
