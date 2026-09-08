"""Getting a published snapshot, the way scientific Python does it: ask for it, and it appears.

``fetch()`` and ``load()`` follow the pattern of ``sklearn.datasets.fetch_*``, ``torchvision``'s
``download=True`` and ``pooch``: look in a local cache, download only what is missing, verify it,
and hand back a path or a frame. Interrupted downloads resume rather than restart.

**The payload does not need this module, and that is deliberate.** A published snapshot is a
folder of plain Parquet files plus a ``manifest.json``, a dataset card and Croissant metadata.
Anyone can read it with polars, pandas, DuckDB or Hugging Face ``datasets`` without installing
terrastat, or ever having heard of it::

    pl.read_parquet("shard-*.parquet")
    duckdb.sql("select * from 'shard-*.parquet'")
    load_dataset("parquet", data_files="shard-*.parquet", streaming=True)

The manifest is what makes that self-sufficient. It lists every file with its size and sha256, so
*any* client can check that a download completed intact — the verification is a property of the
data, not a feature of this package. What this module adds is convenience, not access.
"""
from __future__ import annotations

import hashlib
import json
import logging
import os
import shutil
from pathlib import Path
from urllib.parse import urljoin

import httpx
import polars as pl
from tqdm import tqdm

log = logging.getLogger(__name__)

# Published snapshots, once they exist. A name resolves to a base URL holding manifest.json and
# the shards beside it; any other base URL works just as well, so a fork or a private mirror needs
# no change here.
SNAPSHOTS: dict[str, str] = {
    # "starter": "https://zenodo.org/records/<id>/files/",
}

CHUNK = 1 << 20


def cache_dir() -> Path:
    """Where downloaded snapshots live.

    ``TERRASTAT_CACHE`` wins; otherwise the platform's usual cache location, so a snapshot
    downloaded for one project is not downloaded again for the next.
    """
    if env := os.environ.get("TERRASTAT_CACHE"):
        return Path(env)
    if os.name == "nt":
        base = os.environ.get("LOCALAPPDATA") or (Path.home() / "AppData" / "Local")
        return Path(base) / "terrastat" / "Cache"
    return Path(os.environ.get("XDG_CACHE_HOME") or (Path.home() / ".cache")) / "terrastat"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(CHUNK), b""):
            h.update(block)
    return h.hexdigest()


def _resolve(name_or_url: str) -> str:
    if name_or_url in SNAPSHOTS:
        return SNAPSHOTS[name_or_url]
    if "://" in name_or_url:
        return name_or_url if name_or_url.endswith("/") else name_or_url + "/"
    known = ", ".join(SNAPSHOTS) or "none published yet"
    raise KeyError(
        f"{name_or_url!r} is not a known snapshot (known: {known}) and is not a URL. "
        f"Pass the base URL of a published snapshot, or build one locally with "
        f"`terrastat export {name_or_url} --public-only`."
    )


def _download(client: httpx.Client, url: str, dest: Path, expect_bytes: int | None = None,
              expect_sha: str | None = None, desc: str = "") -> Path:
    """Fetch one file, resuming a partial download rather than starting it again.

    A snapshot shard is hundreds of megabytes and a corpus is many of them, so restarting on a
    dropped connection is the difference between a nuisance and an impossibility. The partial file
    is kept beside the target; a ranged request continues from its length. Servers that ignore
    ``Range`` answer 200 instead of 206, and then the file is written from the start — handled
    rather than assumed.
    """
    dest.parent.mkdir(parents=True, exist_ok=True)
    part = dest.with_name(dest.name + ".part")
    have = part.stat().st_size if part.exists() else 0
    if expect_bytes and have > expect_bytes:      # a stale or corrupt part: start over
        part.unlink()
        have = 0

    headers = {"Range": f"bytes={have}-"} if have else {}
    with client.stream("GET", url, headers=headers) as resp:
        if resp.status_code == 416:               # already complete
            resp.close()
        else:
            resp.raise_for_status()
            resumed = resp.status_code == 206
            if have and not resumed:
                have = 0                          # server ignored the range; rewrite from scratch
            total = int(resp.headers.get("content-length", 0)) + have or expect_bytes
            mode = "ab" if have else "wb"
            with open(part, mode) as fh, tqdm(
                total=total, initial=have, unit="B", unit_scale=True, unit_divisor=1024,
                desc=desc or dest.name, dynamic_ncols=True, leave=False,
            ) as bar:
                for block in resp.iter_bytes(CHUNK):
                    fh.write(block)
                    bar.update(len(block))

    if expect_sha:
        got = sha256(part)
        if got != expect_sha:
            part.unlink(missing_ok=True)
            raise OSError(
                f"{dest.name} failed its checksum: expected {expect_sha[:16]}..., got {got[:16]}.... "
                "The partial file was removed; try again."
            )
    part.replace(dest)                            # atomic: a half-written file is never named
    return dest


def fetch(name_or_url: str = "starter", cache: Path | str | None = None,
          force: bool = False, files: list[str] | None = None) -> Path:
    """Ensure a published snapshot is on disk, and return the directory holding it.

    Already-present files are verified against the manifest and skipped, so calling this on every
    run costs a checksum rather than a download. ``force`` re-downloads regardless.
    """
    base = _resolve(name_or_url)
    root = Path(cache) if cache else cache_dir()
    name = name_or_url if name_or_url in SNAPSHOTS else base.rstrip("/").rsplit("/", 1)[-1]
    out = root / name
    out.mkdir(parents=True, exist_ok=True)

    with httpx.Client(timeout=60.0, follow_redirects=True) as client:
        mpath = out / "manifest.json"
        if force or not mpath.exists():
            _download(client, urljoin(base, "manifest.json"), mpath, desc="manifest.json")
        manifest = json.loads(mpath.read_text(encoding="utf-8"))

        wanted = list(manifest["shards"])
        if idx := manifest.get("index"):
            wanted.append(idx)
        if files:
            wanted = [w for w in wanted if w["file"] in files]

        todo = []
        for w in wanted:
            dest = out / w["file"]
            if not force and dest.exists() and dest.stat().st_size == w.get("bytes"):
                if not w.get("sha256") or sha256(dest) == w["sha256"]:
                    continue                       # present and intact
            todo.append(w)
        if not todo:
            log.info("%s: already complete in %s", name, out)
        for w in todo:
            _download(client, urljoin(base, w["file"]), out / w["file"],
                      w.get("bytes"), w.get("sha256"), desc=w["file"])

        # the human-readable companions; absent on a bare mirror, so failure is not fatal
        for extra in ("README.md", "ATTRIBUTIONS.md", "croissant.json"):
            if force or not (out / extra).exists():
                try:
                    _download(client, urljoin(base, extra), out / extra, desc=extra)
                except Exception as exc:
                    log.debug("optional file %s not fetched: %s", extra, exc)
    return out


def load(name_or_url: str = "starter", cache: Path | str | None = None,
         frequencies: list[str] | None = None, min_obs: int = 1,
         columns: list[str] | None = None, download: bool = True) -> pl.LazyFrame:
    """A published snapshot as a lazy frame, downloading it first if it is not already here.

    ``download=False`` refuses to reach the network, which is what you want in a test or on a
    machine that should stay offline.
    """
    root = Path(cache) if cache else cache_dir()
    local = Path(name_or_url)
    if local.is_dir():
        out = local
    else:
        name = name_or_url if name_or_url in SNAPSHOTS else str(name_or_url).rstrip("/").rsplit("/", 1)[-1]
        out = root / name
        if not any(out.glob("shard-*.parquet")):
            if not download:
                raise FileNotFoundError(f"{out} holds no shards and download=False")
            out = fetch(name_or_url, cache=cache)

    lf = pl.scan_parquet(str(out / "shard-*.parquet"))
    if columns:
        lf = lf.select(columns)
    if frequencies:
        lf = lf.filter(pl.col("frequency").is_in(frequencies))
    if min_obs > 1:
        lf = lf.filter(pl.col("n_obs") >= min_obs)
    return lf


def verify(directory: Path | str) -> dict:
    """Check a snapshot against its own manifest. Works on any copy, however it arrived."""
    d = Path(directory)
    manifest = json.loads((d / "manifest.json").read_text(encoding="utf-8"))
    entries = list(manifest["shards"]) + ([manifest["index"]] if manifest.get("index") else [])
    ok, bad, missing = [], [], []
    for e in entries:
        p = d / e["file"]
        if not p.exists():
            missing.append(e["file"])
        elif e.get("sha256") and sha256(p) != e["sha256"]:
            bad.append(e["file"])
        else:
            ok.append(e["file"])
    return {"ok": ok, "corrupt": bad, "missing": missing,
            "complete": not bad and not missing, "rows": manifest.get("rows")}


def clear(name: str | None = None, cache: Path | str | None = None) -> None:
    """Delete a cached snapshot, or the whole cache."""
    root = Path(cache) if cache else cache_dir()
    target = root / name if name else root
    if target.exists():
        shutil.rmtree(target)
        log.info("removed %s", target)
