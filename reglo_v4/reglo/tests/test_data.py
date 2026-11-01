"""Data layer: in-memory datasets, the meta split, augmentation and batch streams."""

import importlib.util
from pathlib import Path

import numpy as np
import pytest
import torch
from omegaconf import OmegaConf

from reglo.config import ConfigError, compose_meta, compose_run, load_group_file
from reglo.data import (
    Augmenter,
    BatchStream,
    DatasetNotPrepared,
    ImageData,
    build_meta_data,
    build_optimizee_data,
    holdout_split,
    load_image_data,
    meta_split,
)
from reglo.data.datasets import _cached, from_chw, from_hw, from_hwc
from reglo.data.transforms import horizontal_flip, normalize, random_crop


def synthetic(**overrides):
    return OmegaConf.merge(load_group_file("dataset", "synthetic"), overrides)


def small_data(n=10, size=4, classes=3):
    images = torch.arange(n * 3 * size * size, dtype=torch.int64).remainder(256).to(torch.uint8).view(n, 3, size, size)
    return ImageData(images, torch.arange(n) % classes, classes)


# --------------------------------------------------------------------------- datasets


def test_synthetic_is_deterministic(tmp_path):
    cfg = synthetic(train_size=64, test_size=32, image_size=16)
    a, b = load_image_data(cfg, tmp_path, True), load_image_data(cfg, tmp_path, True)
    assert torch.equal(a.images, b.images) and torch.equal(a.labels, b.labels)
    assert a.images.shape == (64, 3, 16, 16) and a.images.dtype == torch.uint8
    test = load_image_data(cfg, tmp_path, False)
    assert len(test) == 32 and not torch.equal(test.images[:32], a.images[:32])


def test_image_data_validates_its_inputs():
    with pytest.raises(ValueError, match="uint8"):
        ImageData(torch.zeros(2, 3, 4, 4), torch.zeros(2, dtype=torch.long), 2)
    with pytest.raises(ValueError, match="labels"):
        ImageData(torch.zeros(2, 3, 4, 4, dtype=torch.uint8), torch.zeros(3, dtype=torch.long), 2)


def test_subset_keeps_pairs_together():
    data = small_data()
    sub = data.subset(np.array([7, 2]))
    assert torch.equal(sub.images[0], data.images[7]) and sub.labels.tolist() == [7 % 3, 2 % 3]


def test_converters_produce_nchw():
    hwc = np.random.randint(0, 255, (5, 8, 8, 3), dtype=np.uint8)
    d = from_hwc(hwc, [0, 1, 2, 3, 4], 5)
    assert d.images.shape == (5, 3, 8, 8) and d.images[2, 1, 3, 4] == hwc[2, 3, 4, 1]
    chw = np.random.randint(0, 255, (5, 3, 8, 8), dtype=np.uint8)
    assert from_chw(chw, np.arange(5), 5).images.shape == (5, 3, 8, 8)
    hw = torch.randint(0, 255, (5, 28, 28), dtype=torch.uint8)
    assert from_hw(hw, torch.arange(5), 10).images.shape == (5, 1, 28, 28)


def test_cache_is_written_once_and_reused(tmp_path):
    calls = []

    def fetch():
        calls.append(1)
        return small_data()

    first = _cached("toy", tmp_path, True, fetch)
    second = _cached("toy", tmp_path, True, fetch)
    assert len(calls) == 1 and torch.equal(first.images, second.images)
    assert (tmp_path / "cache" / "toy_train.pt").is_file()


@pytest.mark.skipif(importlib.util.find_spec("torchvision") is None, reason="torchvision not installed")
def test_missing_real_dataset_explains_how_to_get_it(tmp_path):
    cfg = load_group_file("dataset", "cifar10")
    with pytest.raises(DatasetNotPrepared, match="prepare_data.py cifar10"):
        load_image_data(cfg, tmp_path, train=True)


# --------------------------------------------------------------------------- splits


def test_meta_split_is_disjoint_complete_and_reproducible():
    meta, rest = meta_split("x", 101, 0.5, seed=3)
    assert len(meta) + len(rest) == 101 and abs(len(meta) - len(rest)) <= 1
    assert set(meta).isdisjoint(rest) and set(meta) | set(rest) == set(range(101))
    again, _ = meta_split("x", 101, 0.5, seed=3)
    assert np.array_equal(meta, again) and np.all(np.diff(meta) > 0)
    other, _ = meta_split("x", 101, 0.5, seed=4)
    assert not np.array_equal(meta, other)


def test_meta_split_files_are_reused_and_refreshed(tmp_path):
    meta, _ = meta_split("svhn", 50, 0.5, 0, tmp_path)
    assert (tmp_path / "svhn_meta.npy").is_file() and (tmp_path / "svhn_split.json").is_file()
    np.save(tmp_path / "svhn_meta.npy", meta[:3])  # a stored split is trusted when its spec matches …
    assert len(meta_split("svhn", 50, 0.5, 0, tmp_path)[0]) == 3
    refreshed, _ = meta_split("svhn", 60, 0.5, 0, tmp_path)  # … and redrawn when it does not
    assert len(refreshed) == 30


def test_holdout_split():
    train, held = holdout_split(100, 0.1, seed=0)
    assert len(held) == 10 and len(train) == 90 and set(train).isdisjoint(held)


# --------------------------------------------------------------------------- augmentation


def test_random_crop_is_a_shifted_window_of_the_zero_padded_image():
    images = torch.randint(1, 255, (6, 3, 5, 5), dtype=torch.uint8)
    out = random_crop(images, 2, torch.Generator().manual_seed(0))
    padded = torch.nn.functional.pad(images, (2, 2, 2, 2))
    for k in range(6):
        windows = [padded[k, :, i : i + 5, j : j + 5] for i in range(5) for j in range(5)]
        assert any(torch.equal(out[k], w) for w in windows)
    assert torch.equal(random_crop(images, 0), images)


def test_horizontal_flip_flips_about_half():
    images = torch.arange(4, dtype=torch.uint8).view(1, 1, 1, 4).repeat(2000, 1, 1, 1)
    out = horizontal_flip(images, torch.Generator().manual_seed(0))
    flipped = (out[:, 0, 0, 0] == 3).float().mean().item()
    assert 0.45 < flipped < 0.55
    assert all(torch.equal(o, images[0]) or torch.equal(o, images[0].flip(-1)) for o in out[:20])


def test_normalize():
    images = torch.full((2, 3, 2, 2), 255, dtype=torch.uint8)
    out = normalize(images, torch.tensor([0.5, 0.5, 0.5]), torch.tensor([0.25, 0.5, 1.0]))
    assert torch.allclose(out[0, :, 0, 0], torch.tensor([2.0, 1.0, 0.5]))


def test_augmenter_rejects_unknown_ops():
    with pytest.raises(ValueError, match="cutout"):
        Augmenter([0.5], [0.5], augment=["cutout"])


# --------------------------------------------------------------------------- batch streams


def test_batch_stream_sizes_and_shuffling():
    data = small_data(n=10)
    stream = BatchStream(data, 4, mean=[0, 0, 0], std=[1, 1, 1], shuffle=True, seed=0)
    batches = list(stream)
    assert len(stream) == 3 and [len(y) for _, y in batches] == [4, 4, 2]
    assert batches[0][0].dtype == torch.float32
    seen = torch.cat([y for _, y in batches]).sort().values
    assert torch.equal(seen, data.labels.sort().values)  # every example exactly once per epoch
    second = torch.cat([y for _, y in stream])
    assert stream.epochs_done == 2 and len(second) == 10
    assert len(BatchStream(data, 4, mean=[0] * 3, std=[1] * 3, drop_last=True)) == 2


def test_batch_stream_resumes_with_the_same_batches():
    data = small_data(n=12)
    kwargs = dict(mean=[0] * 3, std=[1] * 3, augment=["random_crop", "horizontal_flip"], crop_padding=1, shuffle=True)
    a = BatchStream(data, 5, seed=7, **kwargs)
    list(a)
    state = a.state_dict()
    expected = [x for x, _ in a]
    b = BatchStream(data, 5, seed=999, **kwargs)
    b.load_state_dict(state)
    for x, y in zip(expected, (x for x, _ in b)):
        assert torch.equal(x, y)


def test_forever_crosses_epochs():
    stream = BatchStream(small_data(n=5), 2, mean=[0] * 3, std=[1] * 3, shuffle=True)
    it = stream.forever()
    assert len([next(it) for _ in range(7)]) == 7 and stream.epochs_done >= 2


# --------------------------------------------------------------------------- what each stage trains on


@pytest.fixture
def data_root(tmp_path, monkeypatch):
    monkeypatch.setenv("REGLO_DATA", str(tmp_path / "data"))
    return tmp_path / "data"


def test_optimizee_runs_train_on_the_half_the_lstm_never_saw(data_root):
    cfg = compose_run(["method=m1_adamw", "optimizee=convnet4", "dataset=synthetic", "dataset.train_size=200"])
    data = build_optimizee_data(cfg, "cpu")
    full = load_image_data(cfg.dataset, data_root, train=True)
    meta_idx, rest_idx = meta_split("synthetic", 200, 0.5, int(cfg.dataset.meta_split.seed), data_root / "splits")
    assert data.train.num_examples == len(rest_idx) == 100
    assert torch.equal(data.train.data.images, full.subset(rest_idx).images)
    assert data.test.num_examples == min(cfg.dataset.test_size, cfg.dataset.test_subset)  # the profile caps it
    assert not data.train_eval.shuffle and not data.train_eval.augmenter.augment

    meta = build_meta_data(compose_meta(["meta_training=m3_lstm", "dataset=synthetic", "dataset.train_size=200",
                                         "meta_tag=unit"]), "cpu")
    assert meta.num_meta == len(meta_idx) == 100
    assert meta.val.num_examples == 10 and meta.train.num_examples == 90
    meta_images = {bytes(img.numpy()) for img in full.subset(meta_idx).images}
    assert not any(bytes(img.numpy()) in meta_images for img in data.train.data.images)


def test_datasets_without_a_meta_split_use_every_training_image(data_root):
    cfg = compose_run(["method=m1_adamw", "optimizee=convnet4", "dataset=synthetic",
                       "dataset.train_size=64", "dataset.meta_split.enabled=false"])
    assert build_optimizee_data(cfg, "cpu").train.num_examples == 64


def test_a_validation_split_replaces_the_test_set_for_sweeps(data_root):
    base = ["method=m1_adamw", "optimizee=convnet4", "dataset=synthetic", "dataset.train_size=200"]
    plain = build_optimizee_data(compose_run(base), "cpu")
    swept = build_optimizee_data(compose_run(base + ["training.validation_fraction=0.2"]), "cpu")
    assert plain.train.num_examples == 100  # the optimizee half of 200 images
    assert (swept.train.num_examples, swept.test.num_examples) == (80, 20)  # 20 % of it held out …
    held_out = {bytes(img.numpy()) for img in swept.test.data.images}
    assert not any(bytes(img.numpy()) in held_out for img in swept.train.data.images)  # … and never trained on
    assert swept.train_eval.num_examples == 80
    with pytest.raises(ConfigError, match="validation_fraction"):
        compose_run(base + ["training.validation_fraction=1.0"])


def test_subsets_cap_training_and_test_sets(data_root):
    cfg = compose_run(["method=m1_adamw", "optimizee=convnet4", "dataset=synthetic", "dataset.train_subset=30",
                       "dataset.test_subset=20"])
    data = build_optimizee_data(cfg, "cpu")
    assert data.train.num_examples == 30 and data.test.num_examples == 20
    assert data.train_eval.num_examples == 30


@pytest.mark.parametrize("name, shape, classes", [("svhn", (3, 32, 32), 10), ("fmnist", (1, 28, 28), 10),
                                                  ("cifar10", (3, 32, 32), 10), ("cifar100", (3, 32, 32), 100)])
def test_offline_stand_ins_have_the_real_shape_and_their_own_split(data_root, name, shape, classes):
    from reglo.data.datasets import STANDIN_SIZES, data_key

    cfg = compose_run(["method=m1_adamw", "optimizee=convnet4", f"dataset={name}", "profile=smoke_offline",
                       "dataset.train_subset=null", "dataset.test_subset=null"])
    assert cfg.dataset.source == "synthetic" and data_key(cfg.dataset) == f"{name}-standin"
    data = build_optimizee_data(cfg, "cpu")
    assert tuple(data.train.data.images.shape[1:]) == shape and data.train.data.num_classes == classes
    expected = STANDIN_SIZES[0] // 2 if cfg.dataset.meta_split.enabled else STANDIN_SIZES[0]
    assert data.train.num_examples == expected and data.test.num_examples == STANDIN_SIZES[1]
    if cfg.dataset.meta_split.enabled:  # never the real dataset's split files
        assert (data_root / "splits" / f"{name}-standin_meta.npy").is_file()
        assert not (data_root / "splits" / f"{name}_meta.npy").exists()
    assert not (data_root / "cache").exists()  # generated in memory, never cached


def test_stand_ins_differ_between_datasets():
    from reglo.data.datasets import standin

    a = standin(compose_run(["method=m1_adamw", "optimizee=convnet4", "dataset=svhn"]).dataset, True)
    b = standin(compose_run(["method=m1_adamw", "optimizee=convnet4", "dataset=cifar10"]).dataset, True)
    assert a.images.shape == b.images.shape and not torch.equal(a.images, b.images)


def test_profiles_that_need_real_data_refuse_stand_ins():
    from reglo.config import ConfigError

    for profile in ("production", "smoke"):
        with pytest.raises(ConfigError, match="needs the real datasets"):
            compose_run(["method=m1_adamw", "optimizee=convnet4", "dataset=svhn", f"profile={profile}",
                         "dataset.source=synthetic"])
        with pytest.raises(ConfigError, match="needs the real datasets"):
            compose_run(["method=m1_adamw", "optimizee=convnet4", "dataset=synthetic", f"profile={profile}"])


def test_prepare_data_script_on_synthetic(tmp_path, capsys):
    import runpy

    module = runpy.run_path(str(Path(__file__).resolve().parents[1] / "scripts" / "prepare_data.py"))
    assert module["main"](["synthetic", "--data-root", str(tmp_path)]) == 0
    out = capsys.readouterr().out
    assert "synthetic" in out and "meta split: 2048 meta-training / 2048 optimizee" in out
    assert (tmp_path / "splits" / "synthetic_meta.npy").is_file()
