"""Lossless cold storage for an explicit data directory, using ordinary tar + Zstandard.

``pack`` keeps the original file bytes, including Parquet metadata, missing values,
snapshot manifests and attribution files. ``deploy`` restores those bytes into a new
directory that the existing readers can use immediately. No data is re-encoded and
no source files are deleted. Empty directories, permissions and timestamps are not
part of the format. The archive is not encrypted.

Stop writers before packing: consistency checks detect ordinary concurrent changes,
but this is not a filesystem snapshot or a lock on an active crawl. Work is streamed
with bounded buffers. An interrupted operation can be restarted; it cannot resume
halfway through a compressed archive. A process killed without running cleanup may
leave a uniquely named sibling staging file/directory, never a completed destination.

The codec comes from the already-required PyArrow, including on Python 3.11. The
archive's checksums detect damaged files; use a trusted external archive SHA-256 to
verify provenance as well. A self-supplied manifest is not a signature.
"""
from __future__ import annotations

import hashlib
import io
import json
import os
import re
import shutil
import stat
import tarfile
import tempfile
from pathlib import Path
from typing import BinaryIO

import pyarrow as pa

CHUNK = 1 << 20
MAX_MANIFEST_BYTES = 16 << 20
DEFAULT_MAX_BYTES = 1 << 40
_FORMAT = "terrastat-archive"
_SHA256 = re.compile(r"[0-9a-fA-F]{64}\Z")
_RESERVED = {"CON", "PRN", "AUX", "NUL", "CONIN$", "CONOUT$"} | {
    f"{prefix}{suffix}" for prefix in ("COM", "LPT") for suffix in "123456789¹²³"
}


def _validate_path(value: str) -> str:
    """Accept only portable, relative file names, before joining them to a root."""
    if not isinstance(value, str) or not value or len(value.encode("utf-8")) > 4096:
        raise ValueError("archive file paths must be nonempty relative strings of at most 4096 bytes")
    if any(ord(c) < 32 or c in '\\:*?"<>|' for c in value):
        raise ValueError(f"unsafe archive path: {value!r}")
    for part in value.split("/"):
        if not part or part in (".", "..") or part.endswith((" ", ".")):
            raise ValueError(f"unsafe archive path: {value!r}")
        if part.split(".", 1)[0].upper() in _RESERVED:
            raise ValueError(f"reserved Windows path: {value!r}")
        lower = part.casefold()
        if lower == ".git" or lower == ".env" or lower.startswith(".env."):
            raise ValueError(f"refusing repository or environment files in archive: {value!r}")
        if lower.endswith((".part", ".tmp")):
            raise ValueError(f"unfinished temporary file in archive: {value!r}; stop writers first")
    return value


def _register_path(path: str, names: dict[str, tuple[str, bool]]) -> None:
    """Reject duplicate files, file/directory conflicts and case-insensitive aliases."""
    parts = _validate_path(path).split("/")
    for i in range(1, len(parts) + 1):
        prefix = "/".join(parts[:i])
        key = prefix.casefold()
        is_file = i == len(parts)
        if key in names:
            previous, previous_file = names[key]
            if prefix != previous or previous_file or is_file:
                raise ValueError(f"duplicate or conflicting archive path: {path!r}")
        else:
            names[key] = (prefix, is_file)


def _is_link(info: os.stat_result) -> bool:
    return stat.S_ISLNK(info.st_mode) or bool(getattr(info, "st_file_attributes", 0) & 0x400)


def _signature(info: os.stat_result) -> tuple[int, ...]:
    # Python 3.12 on Windows can report creation time from lstat but change
    # time from fstat. Identity, size, mtime and the streamed hash still detect
    # mutations without treating that API difference as a changed file.
    change_time = info.st_ctime_ns if os.name != "nt" else 0
    return (info.st_dev, info.st_ino, info.st_mode, info.st_size, info.st_mtime_ns, change_time)


def _inventory(root: Path) -> dict[str, tuple[int, ...]]:
    inventory: dict[str, tuple[int, ...]] = {}
    names: dict[str, tuple[str, bool]] = {}
    pending = [root]
    while pending:
        directory = pending.pop()
        with os.scandir(directory) as entries:
            for entry in sorted(entries, key=lambda e: e.name):
                relative = Path(entry.path).relative_to(root).as_posix()
                _validate_path(relative)
                # DirEntry.stat() reports zero inode/device on some Windows
                # versions; Path.lstat() agrees with fstat() during verification.
                info = Path(entry.path).lstat()
                if _is_link(info):
                    raise ValueError(f"symbolic links and reparse points cannot be archived: {relative!r}")
                if stat.S_ISDIR(info.st_mode):
                    pending.append(Path(entry.path))
                elif stat.S_ISREG(info.st_mode):
                    _register_path(relative, names)
                    inventory[relative] = _signature(info)
                else:
                    raise ValueError(f"only regular files can be archived: {relative!r}")
    return dict(sorted(inventory.items()))


def _unchanged(path: Path, expected: tuple[int, ...], stream: BinaryIO | None = None) -> None:
    actual = path.lstat()
    if _is_link(actual) or _signature(actual) != expected:
        raise OSError(f"source changed while packing: {path.name!r}; stop writers and retry")
    if stream is not None and _signature(os.fstat(stream.fileno())) != expected:
        raise OSError(f"source changed while opening: {path.name!r}; stop writers and retry")


def _hash_file(path: Path, expected: tuple[int, ...] | None = None) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        if expected is not None:
            _unchanged(path, expected, stream)
        for block in iter(lambda: stream.read(CHUNK), b""):
            digest.update(block)
        if expected is not None:
            _unchanged(path, expected, stream)
    return digest.hexdigest()


def _tar_info(name: str, size: int) -> tarfile.TarInfo:
    info = tarfile.TarInfo(name)
    info.size = size
    info.mode = 0o644
    info.mtime = 0
    return info


class _HashReader:
    def __init__(self, stream: BinaryIO):
        self.stream = stream
        self.digest = hashlib.sha256()

    def read(self, size: int = -1) -> bytes:
        data = self.stream.read(size)
        self.digest.update(data)
        return data


class _ArchiveReader:
    """Bound decoded tar bytes and retain the small amount tar can read ahead."""
    def __init__(self, stream):
        self.stream = stream
        self.limit = MAX_MANIFEST_BYTES + (128 << 10)
        self.position = 0
        self.recent = b""

    def read(self, size: int = -1) -> bytes:
        if size < 0:
            size = CHUNK
        data = self.stream.read(min(size, max(1, self.limit - self.position + 1)))
        self.position += len(data)
        if self.position > self.limit:
            raise ValueError("decoded archive exceeds its permitted payload and header sizes")
        self.recent = (self.recent + data)[-(64 << 10):]
        return data

    def finish(self, payload_end: int) -> None:
        # TarFile normally accepts a missing end marker and ignores data after
        # its first zero block. Verify all already-buffered and remaining bytes
        # after the final payload, rather than silently accepting another archive.
        recent_start = self.position - len(self.recent)
        if not recent_start <= payload_end <= self.position:
            raise ValueError("invalid tar termination after the final payload")
        if any(self.recent[payload_end - recent_start:]):
            raise ValueError("unexpected data after the final archive payload")
        while block := self.read(CHUNK):
            if any(block):
                raise ValueError("unexpected data after the tar end marker")
        padding = self.position - payload_end
        if padding < 1024 or padding % 512:
            raise ValueError("missing or truncated tar end marker")


def _lexists(path: Path) -> bool:
    return os.path.lexists(path)


def _inside(path: Path, root: Path) -> bool:
    return path == root or root in path.parents


def pack(directory: str | Path, output: str | Path) -> dict:
    """Archive every regular file in ``directory`` without modifying its contents.

    ``output`` must be outside the source directory and must not already exist.
    Hidden files are included, except repository/environment files and unfinished
    ``.part``/``.tmp`` files, which cause an error rather than silently disappearing.
    The summary includes the compressed SHA-256 for a release/download receipt.
    """
    source_input = Path(directory).absolute()
    info = source_input.lstat()
    if _is_link(info) or not stat.S_ISDIR(info.st_mode):
        raise ValueError("archive source must be a directory, not a link or reparse point")
    source = source_input.resolve()
    target = Path(output).absolute()
    if _inside(target.resolve(), source):
        raise ValueError("archive output must be outside the source directory")
    if _lexists(target):
        raise FileExistsError(target)

    inventory = _inventory(source)
    files = []
    for relative, signature in inventory.items():
        files.append({"path": relative, "bytes": signature[3],
                      "sha256": _hash_file(source / relative, signature)})
    manifest = {"format": _FORMAT, "version": 1, "files": files}
    metadata = json.dumps(manifest, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    if len(metadata) > MAX_MANIFEST_BYTES:
        raise ValueError("archive manifest exceeds 16 MiB; pack smaller subdirectories separately")
    target.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary_name = tempfile.mkstemp(prefix=f".{target.name}.", suffix=".part", dir=target.parent)
    os.close(fd)
    temporary = Path(temporary_name)
    try:
        with pa.CompressedOutputStream(str(temporary), "zstd") as compressed:
            with tarfile.open(fileobj=compressed, mode="w|", format=tarfile.PAX_FORMAT) as tar:
                tar.addfile(_tar_info("archive.json", len(metadata)), io.BytesIO(metadata))
                for entry in files:
                    path = source / entry["path"]
                    with path.open("rb") as stream:
                        _unchanged(path, inventory[entry["path"]], stream)
                        reader = _HashReader(stream)
                        tar.addfile(_tar_info("payload/" + entry["path"], entry["bytes"]), reader)
                        if stream.read(1) or reader.digest.hexdigest() != entry["sha256"]:
                            raise OSError(f"source changed while packing: {entry['path']!r}")
                        _unchanged(path, inventory[entry["path"]], stream)
                if _inventory(source) != inventory:
                    raise OSError("source directory changed while packing; stop writers and retry")
        result = {"files": len(files), "original_bytes": sum(e["bytes"] for e in files),
                  "bytes": temporary.stat().st_size, "sha256": _hash_file(temporary)}
        # Windows rename refuses an existing destination and works on exFAT.
        # POSIX rename can overwrite a winner, so publish with a hard link there.
        if os.name == "nt":
            temporary.rename(target)
        else:
            os.link(temporary, target)
            temporary.unlink()
        return result
    finally:
        temporary.unlink(missing_ok=True)


def _read_manifest(tar: tarfile.TarFile, max_bytes: int) -> tuple[dict[str, dict], int]:
    first = tar.next()
    if first is None or first.name != "archive.json" or first.type not in (tarfile.REGTYPE, tarfile.AREGTYPE):
        raise ValueError("archive must start with the regular file archive.json")
    if first.sparse is not None or not 0 <= first.size <= MAX_MANIFEST_BYTES:
        raise ValueError("invalid archive manifest size (maximum 16 MiB)")
    member = tar.extractfile(first)
    if member is None:
        raise ValueError("missing archive manifest")
    with member:
        raw = member.read(MAX_MANIFEST_BYTES + 1)
    if len(raw) != first.size:
        raise ValueError("truncated archive manifest")
    try:
        manifest = json.loads(raw)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError("invalid archive manifest JSON") from exc
    if not isinstance(manifest, dict) or manifest.get("format") != _FORMAT or type(manifest.get("version")) is not int or manifest["version"] != 1:
        raise ValueError("unsupported archive format or version")
    entries = manifest.get("files")
    if not isinstance(entries, list):
        raise ValueError("archive manifest must list its files")
    names: dict[str, tuple[str, bool]] = {}
    files = {}
    total = 0
    for entry in entries:
        if not isinstance(entry, dict):
            raise ValueError("invalid file record in archive manifest")
        path = entry.get("path")
        _register_path(path, names)
        size, digest = entry.get("bytes"), entry.get("sha256")
        if type(size) is not int or size < 0 or not isinstance(digest, str) or not _SHA256.fullmatch(digest):
            raise ValueError(f"invalid size or checksum for archive file: {path!r}")
        total += size
        if total > max_bytes:
            raise ValueError(f"archive expands beyond max_bytes={max_bytes}; increase the limit explicitly")
        files[path] = {"bytes": size, "sha256": digest.lower()}
    return files, first.offset_data + ((first.size + 511) // 512) * 512


def deploy(archive: str | Path, destination: str | Path, *, sha256: str | None = None,
           max_bytes: int = DEFAULT_MAX_BYTES) -> Path:
    """Verify and restore a local archive into a new directory, returning its path.

    All payload paths, file sizes and SHA-256 hashes must match the first member's
    manifest. Links and special files are rejected; tar's extraction routines are
    never used to create files. ``max_bytes`` limits declared expanded payload size
    (default 1 TiB). Pass a trusted ``sha256`` for the compressed archive when it
    came from a remote store. Downloads are intentionally a separate operation.

    Publication uses a sibling staging directory and rename. Do not independently
    create the destination during deployment; existing destinations are refused.
    """
    if type(max_bytes) is not int or max_bytes < 0:
        raise ValueError("max_bytes must be a nonnegative integer")
    source = Path(archive).absolute()
    source_info = source.lstat()
    if _is_link(source_info) or not stat.S_ISREG(source_info.st_mode):
        raise ValueError("archive input must be a regular file, not a link or reparse point")
    target = Path(destination).absolute()
    if _lexists(target):
        raise FileExistsError(target)
    if sha256 is not None:
        if not isinstance(sha256, str) or not _SHA256.fullmatch(sha256):
            raise ValueError("sha256 must contain 64 hexadecimal characters")
        if _hash_file(source) != sha256.lower():
            raise ValueError("compressed archive SHA-256 does not match")
    target.parent.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix=f".{target.name}.deploy-", dir=target.parent))
    try:
        with pa.CompressedInputStream(str(source), "zstd") as compressed:
            reader = _ArchiveReader(compressed)
            with tarfile.open(fileobj=reader, mode="r|") as tar:
                files, payload_end = _read_manifest(tar, max_bytes)
                reader.limit = (sum(e["bytes"] for e in files.values()) +
                                2 * MAX_MANIFEST_BYTES + 2048 * len(files) + 2 * tarfile.RECORDSIZE)
                seen: set[str] = set()
                while (member := tar.next()) is not None:
                    if member.type not in (tarfile.REGTYPE, tarfile.AREGTYPE) or member.sparse is not None:
                        raise ValueError("archive payload may contain only regular, nonsparse files")
                    if not member.name.startswith("payload/"):
                        raise ValueError(f"unexpected archive member: {member.name!r}")
                    relative = _validate_path(member.name[len("payload/"):])
                    if relative not in files or relative in seen:
                        raise ValueError(f"unexpected or duplicate archive member: {relative!r}")
                    expected = files[relative]
                    if member.size != expected["bytes"]:
                        raise ValueError(f"archive file size does not match manifest: {relative!r}")
                    path = stage.joinpath(*relative.split("/"))
                    if not _inside(path.resolve(), stage.resolve()):
                        raise ValueError(f"archive path escapes staging directory: {relative!r}")
                    path.parent.mkdir(parents=True, exist_ok=True)
                    content = tar.extractfile(member)
                    if content is None:
                        raise ValueError(f"missing archive payload: {relative!r}")
                    digest, written = hashlib.sha256(), 0
                    with content, path.open("xb") as output:
                        for block in iter(lambda: content.read(CHUNK), b""):
                            written += len(block)
                            if written > expected["bytes"]:
                                raise ValueError(f"archive payload exceeds declared size: {relative!r}")
                            digest.update(block)
                            output.write(block)
                    if written != expected["bytes"] or digest.hexdigest() != expected["sha256"]:
                        raise ValueError(f"archive payload checksum or size does not match: {relative!r}")
                    seen.add(relative)
                    payload_end = member.offset_data + ((member.size + 511) // 512) * 512
                if seen != set(files):
                    raise ValueError("archive is missing files listed in its manifest")
            # tar stops at its end marker, which can precede the Zstandard footer.
            # Consume the rest so a truncated compressed stream is not published.
            reader.finish(payload_end)
        if _signature(source.lstat()) != _signature(source_info):
            raise OSError("archive changed during deployment")
        if _lexists(target):
            raise FileExistsError(target)
        stage.rename(target)
        return target
    finally:
        if stage.exists():
            # Only this operation's fresh sibling directory is ever removed.
            if stage.resolve().parent != target.parent.resolve():
                raise OSError("refusing to clean staging directory outside its original parent")
            shutil.rmtree(stage)


unpack = deploy
