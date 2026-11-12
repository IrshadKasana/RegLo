#!/usr/bin/env python
"""
prepare_imagenet32.py -- turn the downsampled-ImageNet parquet files into the two arrays the sweep reads.

    SRC_DIR/train-0000{0,1,2}-of-00003.parquet + validation-00000-of-00001.parquet      (HF: benjamin-paine/imagenet-1k-32x32)
        -> <datasets>/imagenet32/x.npy   uint8 (N, 3, 32, 32)   official train images, then the 50 000 validation images
           <datasets>/imagenet32/y.npy   int64 (N,)             class ids 0..999
           <datasets>/imagenet32/MD5.txt

N = 1,331,167 (1,281,167 train + 50,000 validation).  The sweep pools both and re-splits 70/20/10 (sweep_config.SPLIT).

    python prepare_imagenet32.py SRC_DIR [--out DATASETS_DIR] [--limit N]    # --limit: first N images of every file (tests)

Needs pyarrow + Pillow + numpy only (no torch); run it on any machine, then upload datasets/imagenet32/ (3.9 GB) -- or upload the
parquet files (0.3 GB) and run this on the cluster login node.
"""
from __future__ import annotations

import argparse
import hashlib
import io
import sys
from pathlib import Path

import numpy as np

FILES = ["train-00000-of-00003", "train-00001-of-00003", "train-00002-of-00003", "validation-00000-of-00001"]
EXPECT = {"train": 1_281_167, "validation": 50_000}


def decode(blob: bytes) -> np.ndarray:
    from PIL import Image
    im = Image.open(io.BytesIO(blob)).convert("RGB")
    if im.size != (32, 32):
        raise ValueError(f"image is {im.size}, expected 32x32")
    return np.asarray(im)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("src", type=Path)
    ap.add_argument("--out", type=Path, default=Path(__file__).resolve().parents[2] / "datasets")
    ap.add_argument("--limit", type=int, default=0, help="only the first N rows of every parquet file (smoke tests)")
    a = ap.parse_args()
    import pyarrow.parquet as pq

    out = a.out / "imagenet32"
    out.mkdir(parents=True, exist_ok=True)
    tables = []
    for name in FILES:
        p = a.src / f"{name}.parquet"
        if not p.is_file():
            sys.exit(f"missing {p}")
        pf = pq.ParquetFile(p)
        tables.append((name, pf))
    total = sum(min(pf.metadata.num_rows, a.limit) if a.limit else pf.metadata.num_rows for _, pf in tables)
    if not a.limit:
        got = {"train": sum(pf.metadata.num_rows for n, pf in tables if n.startswith("train")),
               "validation": sum(pf.metadata.num_rows for n, pf in tables if n.startswith("validation"))}
        if got != EXPECT:
            sys.exit(f"unexpected row counts {got}, expected {EXPECT}")
    x = np.lib.format.open_memmap(out / "x.npy", mode="w+", dtype=np.uint8, shape=(total, 3, 32, 32))
    y = np.empty(total, dtype=np.int64)
    pos = 0
    for name, pf in tables:
        left = a.limit or pf.metadata.num_rows
        for batch in pf.iter_batches(batch_size=20_000, columns=["image", "label"]):
            if left <= 0:
                break
            imgs = batch.column("image").to_pylist()[:left]
            labs = batch.column("label").to_pylist()[:left]
            x[pos:pos + len(imgs)] = np.stack([decode(c["bytes"]) for c in imgs]).transpose(0, 3, 1, 2)
            y[pos:pos + len(imgs)] = labs
            pos += len(imgs)
            left -= len(imgs)
            print(f"\r{name}: {pos}/{total}", end="", flush=True)
        print()
    assert pos == total, (pos, total)
    x.flush()
    np.save(out / "y.npy", y)
    if y.min() < 0 or y.max() > 999:
        sys.exit(f"labels outside 0..999: {y.min()}..{y.max()}")
    with open(out / "MD5.txt", "w") as fh:
        for f in ("x.npy", "y.npy"):
            h = hashlib.md5()
            with open(out / f, "rb") as g:
                while chunk := g.read(1 << 24):
                    h.update(chunk)
            fh.write(f"{h.hexdigest()}  {f}\n")
    print(f"wrote {out}/x.npy {x.shape}, y.npy ({len(np.unique(y))} classes), MD5.txt")


if __name__ == "__main__":
    main()
