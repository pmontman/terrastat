"""Short-lived, nonblocking process locks for local writers.

Lock files remain on disk. Ownership belongs to the open OS lock, so a killed
process releases it without stale-file removal or PID guessing.
"""
from __future__ import annotations

from contextlib import contextmanager
import os
from pathlib import Path


@contextmanager
def exclusive_lock(path: Path):
    """Exclude cooperating writers on a local filesystem; fail instead of waiting."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.is_symlink() or (path.exists() and getattr(path.lstat(), "st_file_attributes", 0) & 0x400):
        raise ValueError(f"Writer lock must not be a link: {path}")
    with open(path, "a+b") as stream:
        if os.fstat(stream.fileno()).st_size == 0:
            stream.write(b"\0")
            stream.flush()
        stream.seek(0)
        if os.name == "nt":
            import msvcrt

            acquire = lambda: msvcrt.locking(stream.fileno(), msvcrt.LK_NBLCK, 1)
            release = lambda: msvcrt.locking(stream.fileno(), msvcrt.LK_UNLCK, 1)
        else:
            import fcntl

            acquire = lambda: fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            release = lambda: fcntl.flock(stream.fileno(), fcntl.LOCK_UN)
        try:
            acquire()
        except OSError as exc:
            raise RuntimeError(f"Cannot acquire writer lock {path}; another operation may be running.") from exc
        try:
            yield
        finally:
            stream.seek(0)
            release()
