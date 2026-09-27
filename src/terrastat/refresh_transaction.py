"""Recoverable publication of one dataset's refreshed files.

The caller holds the refresh writer lock across begin/build/publish/recover. This
module does not coordinate with unrelated writers or concurrent readers. Multiple
renames are not one atomic filesystem operation: recovery restores the old release
before a commit, or completes state updates after a commit. Files and journals are
fsynced; directory entries are also fsynced where POSIX supports it. Windows does
not expose portable directory fsync, so power-loss durability additionally depends
on the filesystem. Process-interruption recovery works on both platforms.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import stat
import uuid

from terrastat.config import data_workspace
from terrastat.storage import State

_FORMAT = "terrastat-refresh-transaction"
_ID = re.compile(r"[0-9a-f]{32}\Z")
_HASH = re.compile(r"[0-9a-f]{64}\Z")
_JOURNAL_LIMIT = 64 << 20
_RESERVED = {"CON", "PRN", "AUX", "NUL", "CONIN$", "CONOUT$"} | {
    f"{prefix}{n}" for prefix in ("COM", "LPT") for n in "123456789¹²³"
}


class CommittedRefreshError(RuntimeError):
    """The new files are committed; recovery must finish state updates/cleanup."""


class RecoveryRequiredError(RuntimeError):
    """Publication could not be rolled back; stop work until journal recovery."""


def _parts(value: str) -> list[str]:
    if not isinstance(value, str) or not value or len(value.encode("utf-8")) > 4096:
        raise ValueError("transaction paths must be nonempty relative paths")
    if any(ord(c) < 32 or c in '\\:*?"<>|' for c in value):
        raise ValueError(f"unsafe transaction path: {value!r}")
    parts = value.split("/")
    for part in parts:
        if not part or part in (".", "..") or part.endswith((" ", ".")) or part.split(".")[0].upper() in _RESERVED:
            raise ValueError(f"unsafe transaction path: {value!r}")
    return parts


def _component(value: str) -> str:
    if len(_parts(value)) != 1:
        raise ValueError("source and dataset id must be single path components")
    return value


def _linked(info: os.stat_result) -> bool:
    return stat.S_ISLNK(info.st_mode) or bool(getattr(info, "st_file_attributes", 0) & 0x400)


def _exists(path: Path) -> bool:
    return os.path.lexists(path)


def _plain(path: Path, *, directory: bool | None = None) -> None:
    info = path.lstat()
    if _linked(info) or not (stat.S_ISDIR(info.st_mode) or stat.S_ISREG(info.st_mode)):
        raise ValueError(f"transaction paths must not be links or special files: {path.name!r}")
    if directory is not None and stat.S_ISDIR(info.st_mode) != directory:
        raise ValueError(f"unexpected file/directory type: {path.name!r}")


def _root(root: Path) -> Path:
    root = Path(root).absolute()
    for ancestor in reversed([root, *root.parents]):
        if _exists(ancestor):
            _plain(ancestor, directory=True)
    root.mkdir(parents=True, exist_ok=True)
    return root.resolve()


def _at(root: Path, relative: str) -> Path:
    path = root.joinpath(*_parts(relative))
    current = root
    _plain(current, directory=True)
    for part in _parts(relative):
        current /= part
        if _exists(current):
            _plain(current)
            if current != path:
                _plain(current, directory=True)
    if not path.resolve().is_relative_to(root.resolve()):
        raise ValueError("transaction path escapes its root")
    return path


def _fsync_dir(path: Path) -> None:
    if os.name != "nt":
        fd = os.open(path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
        try:
            os.fsync(fd)
        finally:
            os.close(fd)


def _mkdir(path: Path) -> None:
    missing = []
    parent = path
    while not _exists(parent):
        missing.append(parent)
        parent = parent.parent
    _plain(parent, directory=True)
    for item in reversed(missing):
        item.mkdir()
        _fsync_dir(item.parent)


def _durable_json(path: Path, value: dict) -> None:
    raw = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")
    if len(raw) > _JOURNAL_LIMIT:
        raise ValueError("refresh journal exceeds 64 MiB; publish fewer paths")
    temporary = path.with_name(path.name + ".part")
    if _exists(temporary):
        _plain(temporary, directory=False)
    if _exists(path):
        _plain(path, directory=False)
    with temporary.open("wb") as stream:
        stream.write(raw)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)
    _fsync_dir(path.parent)


def _file_record(path: Path, sync: bool = False) -> dict:
    _plain(path, directory=False)
    before = path.stat()
    digest = hashlib.sha256()
    with path.open("r+b" if sync else "rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
        if sync:
            os.fsync(stream.fileno())
    after = path.stat()
    if (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns) != (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns):
        raise OSError("file changed during refresh preparation")
    return {"bytes": after.st_size, "sha256": digest.hexdigest()}


def _snapshot(path: Path, sync: bool = False) -> dict | None:
    if not _exists(path):
        return None
    _plain(path)
    if path.is_file():
        return {"kind": "file", **_file_record(path, sync)}
    files, directories = {}, []
    pending = [path]
    names: dict[str, str] = {}
    while pending:
        current = pending.pop()
        for child in sorted(current.iterdir()):
            relative = child.relative_to(path).as_posix()
            _parts(relative)
            key = relative.casefold()
            if key in names and names[key] != relative:
                raise ValueError("case-insensitive path collision in transaction directory")
            names[key] = relative
            _plain(child)
            if child.is_dir():
                directories.append(relative)
                pending.append(child)
            else:
                files[relative] = _file_record(child, sync)
        if sync:
            _fsync_dir(current)
    return {"kind": "directory", "files": files, "directories": sorted(directories)}


def _allowed(path: str, source: str, did: str) -> str:
    parts = _parts(path)
    if path in (f"datasets/{source}/{did}", f"raw/{source}/{did}"):
        return "directory"
    if len(parts) == 4 and parts[:2] == ["series", source] and parts[2].startswith("freq=") and len(parts[2]) > 5 and parts[3] == f"{did}.parquet":
        return "file"
    if len(parts) >= 3 and parts[:2] == ["cache", source]:
        return "file"
    if path == f"refresh/checks/{source}/{did}.json":
        return "file"
    raise ValueError(f"path is outside this dataset's refresh scope: {path!r}")


def _validate_paths(paths: list[str], source: str, did: str) -> None:
    if not isinstance(paths, list):
        raise ValueError("transaction paths must be a list")
    names = []
    for path in paths:
        _allowed(path, source, did)
        parts = tuple(p.casefold() for p in _parts(path))
        for other in names:
            if parts[:len(other)] == other or other[:len(parts)] == parts:
                raise ValueError("duplicate or overlapping refresh paths")
        names.append(parts)


def _validate_record(record) -> None:
    if not isinstance(record, dict) or set(record) != {"bytes", "sha256"} or type(record["bytes"]) is not int or record["bytes"] < 0 or not isinstance(record["sha256"], str) or not _HASH.fullmatch(record["sha256"]):
        raise ValueError("invalid file checksum inventory in refresh journal")


def _validate_snapshot(value) -> None:
    if value is None:
        return
    if not isinstance(value, dict):
        raise ValueError("invalid refresh journal inventory")
    if value.get("kind") == "file" and set(value) == {"kind", "bytes", "sha256"}:
        _validate_record({k: value[k] for k in ("bytes", "sha256")})
        return
    if value.get("kind") != "directory" or set(value) != {"kind", "files", "directories"} or not isinstance(value["files"], dict) or not isinstance(value["directories"], list):
        raise ValueError("invalid refresh journal directory inventory")
    names = set()
    for path in [*value["directories"], *value["files"]]:
        _parts(path)
        if path.casefold() in names:
            raise ValueError("duplicate path in refresh journal inventory")
        names.add(path.casefold())
    for record in value["files"].values():
        _validate_record(record)


def _validate_updates(updates, source: str, did: str) -> None:
    if not isinstance(updates, list):
        raise ValueError("state updates must be a list")
    for update in updates:
        if not isinstance(update, dict) or set(update) != {"source", "dataset_id", "status", "extra"}:
            raise ValueError("invalid refresh state update")
        if update["source"] not in (source, source + "-series", source + "-refresh") or update["dataset_id"] != did or not isinstance(update["status"], str) or not isinstance(update["extra"], dict):
            raise ValueError("state update is outside this dataset's refresh scope")
        if set(update["extra"]) & {"dataset_id", "status", "at", "refresh_transaction"}:
            raise ValueError("state update extra must not replace reserved event fields")
    json.dumps(updates, allow_nan=False)


def _unique_object(pairs):
    out = {}
    for key, value in pairs:
        if key in out:
            raise ValueError("duplicate key in refresh journal")
        out[key] = value
    return out


def _read(path: Path) -> dict:
    _plain(path, directory=False)
    with path.open("rb") as stream:
        raw = stream.read(_JOURNAL_LIMIT + 1)
    if len(raw) > _JOURNAL_LIMIT:
        raise ValueError("refresh journal is too large")
    try:
        return json.loads(raw, object_pairs_hook=_unique_object)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError("invalid refresh journal JSON") from exc


def _validate_journal(journal, root: Path, tid: str) -> None:
    fields = {"format", "version", "id", "root", "source", "dataset_id", "phase", "entries", "state_updates"}
    if not isinstance(journal, dict) or set(journal) != fields or journal["format"] != _FORMAT or type(journal["version"]) is not int or journal["version"] != 1:
        raise ValueError("unrecognized refresh journal")
    if journal["id"] != tid or not _ID.fullmatch(tid) or journal["root"] != str(root):
        raise ValueError("refresh journal identity/root does not match its location")
    source, did = _component(journal["source"]), _component(journal["dataset_id"])
    if journal["phase"] not in ("building", "prepared", "committed", "cleanup") or not isinstance(journal["entries"], list):
        raise ValueError("invalid refresh journal phase or entries")
    paths = []
    for entry in journal["entries"]:
        if not isinstance(entry, dict) or set(entry) != {"path", "old", "new"}:
            raise ValueError("invalid refresh journal path record")
        paths.append(entry["path"])
        kind = _allowed(entry["path"], source, did)
        for record in (entry["old"], entry["new"]):
            _validate_snapshot(record)
            if record is not None and record["kind"] != kind:
                raise ValueError("refresh journal path has an unexpected file/directory type")
    _validate_paths(paths, source, did)
    _validate_updates(journal["state_updates"], source, did)
    if journal["phase"] == "building" and (paths or journal["state_updates"]):
        raise ValueError("building transaction cannot have prepared changes")


def _load(txn: Path) -> tuple[Path, dict]:
    txn = Path(txn).absolute()
    if not _ID.fullmatch(txn.name) or txn.parent.name != "transactions" or txn.parent.parent.name != "refresh":
        raise ValueError("unowned refresh transaction directory")
    root = _root(txn.parents[2])
    _at(root, txn.relative_to(root).as_posix())
    _plain(txn, directory=True)
    journal = _read(txn / "journal.json")
    _validate_journal(journal, root, txn.name)
    return root, journal


def begin(root: Path, source: str, did: str) -> Path:
    """Create an owned building transaction; adapters write below its ``stage/``."""
    source, did, root = _component(source), _component(did), _root(root)
    parent = _at(root, "refresh/transactions")
    _mkdir(parent)
    txn = parent / uuid.uuid4().hex
    txn.mkdir()
    journal = {"format": _FORMAT, "version": 1, "id": txn.name, "root": str(root),
               "source": source, "dataset_id": did, "phase": "building", "entries": [], "state_updates": []}
    _durable_json(txn / "journal.json", journal)
    (txn / "stage").mkdir()
    (txn / "backup").mkdir()
    _fsync_dir(txn)
    _fsync_dir(parent)
    return txn


def _rename(source: Path, target: Path) -> None:
    _plain(source)
    if _exists(target):
        raise FileExistsError(target)
    _mkdir(target.parent)
    source.rename(target)
    _fsync_dir(source.parent)
    if source.parent != target.parent:
        _fsync_dir(target.parent)


def _verify(path: Path, expected) -> None:
    if _snapshot(path) != expected:
        raise OSError(f"refresh checksum inventory mismatch: {path.name!r}; transaction retained for investigation")


def _verify_removable(path: Path, expected) -> None:
    """An interrupted rollback may have removed only part of a new directory."""
    actual = _snapshot(path)
    if actual == expected:
        return
    if actual and expected and actual["kind"] == expected["kind"] == "directory":
        if (set(actual["directories"]) <= set(expected["directories"])
                and all(expected["files"].get(name) == record for name, record in actual["files"].items())):
            return
    raise OSError("unexpected files at rollback target; transaction retained for investigation")


def _remove(path: Path, container: Path) -> None:
    if not _exists(path):
        return
    if path == container or not path.resolve().is_relative_to(container.resolve()):
        raise ValueError("refusing removal outside transaction scope")
    _snapshot(path)  # Also reject links/reparse points anywhere in a directory.
    if path.is_dir():
        shutil.rmtree(path)
    else:
        path.unlink()
    _fsync_dir(path.parent)


def _cleanup(txn: Path, root: Path, journal: dict) -> None:
    # A sibling receipt outlives recursive cleanup, including a crash after the
    # in-directory journal was removed. It proves ownership on the next recovery.
    receipt = txn.parent / (txn.name + ".cleanup.json")
    cleanup_journal = {**journal, "phase": "cleanup"}
    _durable_json(receipt, cleanup_journal)
    _remove(txn, txn.parent)
    receipt.unlink()
    _fsync_dir(receipt.parent)


def abort(txn: Path) -> None:
    """Discard an owned, unpublished building transaction. Never alter live files."""
    if not _exists(txn):
        return  # A successful publish or rollback already cleaned it up.
    root, journal = _load(txn)
    if journal["phase"] != "building":
        return  # Leave the journal and backups for recovery.
    _cleanup(Path(txn).absolute(), root, journal)


def _rollback(txn: Path, root: Path, journal: dict) -> None:
    for entry in reversed(journal["entries"]):
        live = _at(root, entry["path"])
        backup = _at(txn / "backup", entry["path"])
        if _exists(backup):
            if entry["old"] is None:
                raise OSError("unexpected backup for a previously absent refresh path")
            _verify(backup, entry["old"])
            if _exists(live):
                _verify_removable(live, entry["new"])
                _remove(live, root)
            _rename(backup, live)
        elif entry["old"] is not None:
            # Either this path has not moved yet, or a preceding recovery already
            # restored it. Both cases have the same verifiable old contents.
            _verify(live, entry["old"])
        elif _exists(live):
            _verify_removable(live, entry["new"])
            _remove(live, root)
    for entry in journal["entries"]:
        _verify(_at(root, entry["path"]), entry["old"])
    _cleanup(txn, root, journal)


def _finish(txn: Path, root: Path, journal: dict) -> None:
    for entry in journal["entries"]:
        _verify(_at(root, entry["path"]), entry["new"])
    with data_workspace(root):
        for update in journal["state_updates"]:
            state = State(update["source"])
            previous = state.events.get(update["dataset_id"], {})
            if previous.get("refresh_transaction") == txn.name and previous.get("status") == update["status"]:
                continue
            state.mark(update["dataset_id"], update["status"], refresh_transaction=txn.name, **update["extra"])
    _cleanup(txn, root, journal)


def publish(txn: Path, paths: list[str], state_updates: list[dict]) -> None:
    """Publish replacements/removals together, recovering from incomplete renames.

    A selected path present below ``stage/`` is a replacement; an absent staged
    path means removal. On pre-commit failure old contents are restored. After a
    durable commit, failures raise ``CommittedRefreshError`` and retain recovery
    records: callers must report updated files with pending bookkeeping.
    """
    txn = Path(txn).absolute()
    root, journal = _load(txn)
    if journal["phase"] != "building":
        raise ValueError("publish requires a building transaction")
    _validate_paths(paths, journal["source"], journal["dataset_id"])
    _validate_updates(state_updates, journal["source"], journal["dataset_id"])
    _plain(txn / "stage", directory=True)
    _plain(txn / "backup", directory=True)
    if any((txn / "backup").iterdir()):
        raise ValueError("building transaction has unexpected backup contents")
    entries = []
    for path in paths:
        kind = _allowed(path, journal["source"], journal["dataset_id"])
        old, new = _snapshot(_at(root, path)), _snapshot(_at(txn / "stage", path), sync=True)
        if any(record is not None and record["kind"] != kind for record in (old, new)):
            raise ValueError(f"expected a {kind} at refresh path {path!r}")
        entries.append({"path": path, "old": old, "new": new})
    journal.update(phase="prepared", entries=entries, state_updates=state_updates)
    _durable_json(txn / "journal.json", journal)
    committed = False
    try:
        for entry in entries:
            live = _at(root, entry["path"])
            staged = _at(txn / "stage", entry["path"])
            _verify(live, entry["old"])
            _verify(staged, entry["new"])
            if entry["old"] is not None:
                _rename(live, _at(txn / "backup", entry["path"]))
            if entry["new"] is not None:
                _rename(staged, live)
        for entry in entries:
            _verify(_at(root, entry["path"]), entry["new"])
        journal["phase"] = "committed"
        _durable_json(txn / "journal.json", journal)
        committed = True
        _finish(txn, root, journal)
    except BaseException as exc:
        # A journal rename can succeed before directory fsync reports a failure;
        # never roll back after the on-disk record has advanced to committed.
        disk_committed = committed or (_exists(txn / "journal.json") and _read(txn / "journal.json").get("phase") == "committed")
        if disk_committed:
            raise CommittedRefreshError("new dataset files are committed; state updates or cleanup need recovery") from exc
        try:
            _rollback(txn, root, journal)
        except BaseException as rollback_error:
            raise RecoveryRequiredError("refresh publication failed and rollback is pending; retain transaction and run recovery") from rollback_error
        raise


def recover(root: Path) -> None:
    """Recover every owned transaction under an externally held refresh lock."""
    root = _root(root)
    parent = _at(root, "refresh/transactions")
    if not _exists(parent):
        return
    _plain(parent, directory=True)
    # Finish interrupted cleanup before looking for live transaction journals.
    for receipt in sorted(parent.glob("*.cleanup.json")):
        tid = receipt.name[:-len(".cleanup.json")]
        journal = _read(receipt)
        _validate_journal(journal, root, tid)
        if journal["phase"] != "cleanup":
            raise ValueError("unrecognized refresh cleanup receipt")
        _remove(parent / tid, parent)
        receipt.unlink()
        _fsync_dir(parent)
    for txn in sorted(parent.iterdir()):
        if not _exists(txn):
            continue  # A prior cleanup may have removed its temporary receipt.
        if txn.name.endswith(".cleanup.json.part"):
            tid = txn.name[:-len(".cleanup.json.part")]
            if not _ID.fullmatch(tid) or not _exists(parent / tid / "journal.json"):
                raise ValueError("unowned refresh cleanup temporary file")
            continue  # Its owned transaction rewrites this receipt during cleanup.
        actual_root, journal = _load(txn)
        if actual_root != root:
            raise ValueError("refresh transaction root changed")
        if journal["phase"] == "building":
            abort(txn)
        elif journal["phase"] == "prepared":
            _rollback(txn, root, journal)
        elif journal["phase"] == "committed":
            try:
                _finish(txn, root, journal)
            except BaseException as exc:
                raise CommittedRefreshError("committed refresh still needs state updates or cleanup") from exc
        else:
            raise ValueError("cleanup phase belongs in an external ownership receipt")
