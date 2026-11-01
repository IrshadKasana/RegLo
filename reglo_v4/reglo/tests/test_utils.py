"""Registry, logging and runtime helpers."""

import json

import pytest

from reglo.registry import Registry, UnknownNameError
from reglo.utils.logging import MetricsWriter, read_json, read_jsonl, write_json


def test_registry_builds_registered_names():
    reg = Registry("widget")

    @reg.register("small")
    def small(size=1):
        return ("small", size)

    assert reg.build("small", size=3) == ("small", 3)
    assert "small" in reg and reg.names() == ["small"] and len(reg) == 1


def test_registry_unknown_name_lists_choices():
    reg = Registry("widget")
    reg.register("a")(lambda: 1)
    reg.register("b")(lambda: 2)
    with pytest.raises(UnknownNameError, match="Unknown widget 'c'. Registered: a, b"):
        reg.get("c")


def test_registry_refuses_a_second_owner_of_a_name():
    reg = Registry("widget")
    reg.register("a")(lambda: 1)
    with pytest.raises(KeyError, match="already registered"):
        reg.register("a")(lambda: 2)


def test_metrics_writer_survives_a_truncated_last_line(tmp_path):
    path = tmp_path / "metrics.jsonl"
    with MetricsWriter(path) as log:
        log.write(epoch=1, loss=0.5)
        log.write({"epoch": 2}, loss=float("nan"))
    with open(path, "a", encoding="utf-8") as fh:
        fh.write('{"epoch": 3, "lo')  # job killed mid-write
    rows = read_jsonl(path)
    assert [r["epoch"] for r in rows] == [1, 2]
    assert rows[1]["loss"] == "nan" and "timestamp" in rows[0]


def test_write_json_is_atomic_and_converts_types(tmp_path):
    np = pytest.importorskip("numpy")
    path = write_json(tmp_path / "final.json", {"acc": np.float32(0.5), "dims": (1, 2), "where": tmp_path})
    assert read_json(path) == {"acc": 0.5, "dims": [1, 2], "where": tmp_path.as_posix()}
    assert not (tmp_path / "final.json.tmp").exists()
    json.loads(path.read_text())  # strict JSON


def test_seed_everything_repeats_random_draws():
    torch = pytest.importorskip("torch")
    from reglo.utils.runtime import seed_everything

    seed_everything(7)
    first = torch.randn(3)
    seed_everything(7)
    assert torch.equal(first, torch.randn(3))


def test_resolve_device_and_environment_record():
    torch = pytest.importorskip("torch")
    from reglo.utils.runtime import environment_info, resolve_device

    device = resolve_device("auto")
    assert device.type in ("cuda", "cpu")
    assert resolve_device("cpu") == torch.device("cpu")
    info = environment_info(device)
    assert info["torch"] == torch.__version__ and "hostname" in info



def test_run_lock_admits_one_process_per_run_folder(tmp_path):
    pytest.importorskip("fcntl")  # POSIX advisory locks; on Windows runs proceed unlocked
    from reglo.utils.locking import RunLocked, run_lock

    with run_lock(tmp_path / "run"):
        with pytest.raises(RunLocked, match="another process"):
            with run_lock(tmp_path / "run"):
                pass
    with run_lock(tmp_path / "run"):  # released again
        pass
