"""Console/file logging and crash-safe metric files.

Metrics go to JSON-lines files (one JSON object per line). Every line is
flushed as it is written, so a run killed by the PBS walltime keeps everything
up to its last step; :func:`write_json` replaces files atomically, so a
``final.json`` is never half-written.
"""

from __future__ import annotations

import json
import logging
import math
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, Iterator, List, Mapping, Optional, Union

__all__ = ["MetricsWriter", "read_json", "read_jsonl", "safe_console", "setup_logging", "write_json"]

PathLike = Union[str, Path]


def safe_console() -> None:
    """Never crash on a console that cannot print 'φ' or '→' (a Windows code page, a redirected
    stream): characters it cannot encode are replaced instead."""
    for stream in (sys.stdout, sys.stderr):
        encoding = (getattr(stream, "encoding", None) or "").lower().replace("-", "")
        if encoding != "utf8" and hasattr(stream, "reconfigure"):
            try:
                stream.reconfigure(errors="replace")
            except (ValueError, OSError):  # pragma: no cover - detached or unusual streams
                pass


def setup_logging(log_file: Optional[PathLike] = None, level: str = "INFO", name: str = "reglo") -> logging.Logger:
    """A logger that prints to stdout (captured in the PBS .o file) and optionally to ``log_file``."""
    safe_console()
    logger = logging.getLogger(name)
    logger.setLevel(level)
    for old in list(logger.handlers):  # several runs in one process (make_manifest --run): close the last run's log
        old.close()
        logger.removeHandler(old)
    logger.propagate = False
    fmt = logging.Formatter("%(asctime)s | %(levelname)-7s | %(message)s", "%Y-%m-%d %H:%M:%S")
    console = logging.StreamHandler(sys.stdout)
    console.setFormatter(fmt)
    logger.addHandler(console)
    if log_file is not None:
        path = Path(log_file)
        path.parent.mkdir(parents=True, exist_ok=True)
        handler = logging.FileHandler(path, mode="a", encoding="utf-8")
        handler.setFormatter(fmt)
        logger.addHandler(handler)
    return logger


def _jsonable(value: Any) -> Any:
    """Convert tensors, NumPy scalars, paths and tuples into plain JSON types."""
    if isinstance(value, Mapping):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, Path):
        return value.as_posix()
    if hasattr(value, "item") and getattr(value, "ndim", 0) == 0:  # 0-d tensor / NumPy scalar
        value = value.item()
    if isinstance(value, float) and not math.isfinite(value):
        return str(value)  # 'nan' / 'inf' kept readable without breaking strict JSON parsers
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


class MetricsWriter:
    """Append-only JSON-lines metrics file.

    >>> with MetricsWriter(run / "metrics.jsonl") as log:
    ...     log.write(epoch=1, train_loss=0.42, test_acc=0.91)
    """

    def __init__(self, path: PathLike, append: bool = True) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._fh = open(self.path, "a" if append else "w", encoding="utf-8")

    def write(self, record: Optional[Mapping[str, Any]] = None, **fields: Any) -> Dict[str, Any]:
        row: Dict[str, Any] = {**(record or {}), **fields}
        row.setdefault("timestamp", round(time.time(), 3))
        row = _jsonable(row)
        self._fh.write(json.dumps(row) + "\n")
        self._fh.flush()
        return row

    def close(self) -> None:
        if not self._fh.closed:
            self._fh.close()

    def __enter__(self) -> "MetricsWriter":
        return self

    def __exit__(self, *exc: Any) -> None:
        self.close()


def read_jsonl(path: PathLike) -> List[Dict[str, Any]]:
    """All records of a JSON-lines file; a truncated last line (killed job) is skipped."""
    rows: List[Dict[str, Any]] = []
    with open(path, encoding="utf-8") as fh:
        for line in _nonblank(fh):
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return rows


def _nonblank(lines: Iterator[str]) -> Iterator[str]:
    for line in lines:
        line = line.strip()
        if line:
            yield line


def write_json(path: PathLike, obj: Any, indent: int = 2) -> Path:
    """Write JSON atomically: to a temporary file first, then rename over the target."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(_jsonable(obj), fh, indent=indent)
        fh.write("\n")
    os.replace(tmp, path)
    return path


def read_json(path: PathLike) -> Any:
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)
