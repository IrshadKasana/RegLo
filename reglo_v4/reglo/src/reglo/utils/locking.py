"""One process per run folder.

A grid resubmitted while its first submission is still running would otherwise let two jobs
train the same run at once and overwrite each other's checkpoints. :func:`run_lock` takes an
exclusive advisory lock on ``<run folder>/.lock`` for the duration of the run; a second process
gets :class:`RunLocked` at once. Where locks are unavailable (Windows, or a cluster file system
mounted without lock support) the run simply proceeds unlocked.
"""

from __future__ import annotations

import contextlib
import errno
import os
import socket
from pathlib import Path
from typing import Iterator, Union

__all__ = ["RunLocked", "run_lock"]


class RunLocked(RuntimeError):
    """Another process holds this run folder."""


@contextlib.contextmanager
def run_lock(folder: Union[str, Path]) -> Iterator[None]:
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / ".lock"
    try:
        import fcntl
    except ImportError:  # Windows: local runs are not concurrent anyway
        fcntl = None
    with open(path, "a+", encoding="utf-8") as fh:
        locked = False
        if fcntl is not None:
            try:
                fcntl.flock(fh.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                locked = True
            except OSError as err:
                if err.errno in (errno.EAGAIN, errno.EACCES):
                    fh.seek(0)
                    holder = fh.read().strip() or "unknown"
                    raise RunLocked(f"{folder} is being run by another process ({holder})") from err
                # ENOLCK, ENOSYS, EOPNOTSUPP …: no lock support on this file system — run unlocked
        if locked:
            fh.seek(0)
            fh.truncate()
            fh.write(f"host {socket.gethostname()}, pid {os.getpid()}, job {os.environ.get('PBS_JOBID', '-')}\n")
            fh.flush()
        try:
            yield
        finally:
            if locked:
                fcntl.flock(fh.fileno(), fcntl.LOCK_UN)
