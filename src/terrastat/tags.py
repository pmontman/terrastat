"""FRED tags: analyst-assigned labels ("cpi", "usa", "county", ...) attached to every series.

The bulk observations endpoint does not return tags, and asking for them series by series would
take a week. Instead we invert the relation: ``/fred/tags`` lists all ~6,000 tags with their
``series_count`` and, for about 500 of them, a ``notes`` field spelling the tag out ("cpi" ->
"Consumer Price Index"); ``/fred/tags/series`` then lists the series of one tag, 1,000 per page.

Three tag groups need no request at all because the series metadata already says the same thing
exactly: ``cc`` (copyright status), ``seas`` (sa / nsa) and ``freq``. They are derived at series
build time. The other groups are fetched, tag by tag, resumably:

    gen   concepts ("inflation", "gdp", "housing")           1,269 tags
    geo   geographies ("usa", "california", "germany")         4,497 tags
    geot  geography types ("nation", "state", "county")          17 tags
    src   data source ("bls", "census")                          114 tags
    rls   release                                                 72 tags

Files (under ``data/cache/fred/``)::

    tags.parquet             every tag: name, group_id, notes, popularity, series_count
    tag_pairs/<tag>.parquet  one file per fetched tag: its series ids
    series_tags.parquet      series_id -> list of fetched tag names (built from the pairs)
"""
from __future__ import annotations

import hashlib
import logging
import threading
import time
from pathlib import Path

import polars as pl

from terrastat.config import fred_api_key
from terrastat.http import PoliteClient, StopRequested
from terrastat.pacing import Pacer
from terrastat.storage import State, cache_dir, is_fresh, now_iso, write_parquet

log = logging.getLogger(__name__)

TAGS_URL = "https://api.stlouisfed.org/fred/tags"
TAG_SERIES_URL = "https://api.stlouisfed.org/fred/tags/series"
DERIVED_GROUPS = ("cc", "seas", "freq")
DEFAULT_GROUPS = ("gen", "geo", "geot")
ALL_FETCH_GROUPS = ("gen", "geo", "geot", "src", "rls")
PAGE = 1000


def tags_path() -> Path:
    return cache_dir("fred") / "tags.parquet"


def series_tags_path() -> Path:
    return cache_dir("fred") / "series_tags.parquet"


def pairs_dir() -> Path:
    return cache_dir("fred") / "tag_pairs"


def _slug(tag: str) -> str:
    safe = "".join(c if c.isalnum() or c in "-_." else "_" for c in tag)[:60]
    return f"{safe}_{hashlib.md5(tag.encode('utf-8')).hexdigest()[:8]}"


def make_client(min_interval: float = 2.0, max_per_minute: int | None = 30) -> PoliteClient:
    return PoliteClient(Pacer(min_interval=min_interval, max_per_minute=max_per_minute))


# -- the tag list ------------------------------------------------------------------------------


def fetch_tag_list(client: PoliteClient, refresh: bool = False) -> pl.DataFrame:
    """All FRED tags with notes and counts (6 requests); cached for a day."""
    path = tags_path()
    if path.exists() and (is_fresh(path) and not refresh):
        return pl.read_parquet(path)
    rows, offset = [], 0
    while True:
        data = client.get_json(TAGS_URL, params={"api_key": fred_api_key(), "file_type": "json", "limit": PAGE, "offset": offset})
        batch = data.get("tags", [])
        rows.extend(batch)
        if len(batch) < PAGE:
            break
        offset += PAGE
    df = pl.DataFrame(rows).select(
        pl.col("name").cast(pl.String),
        pl.col("group_id").cast(pl.String),
        pl.col("notes").cast(pl.String).fill_null(""),
        pl.col("created").cast(pl.String),
        pl.col("popularity").cast(pl.Int64),
        pl.col("series_count").cast(pl.Int64),
    )
    write_parquet(df, path)
    log.info("fred tags: %d tags, %d with notes", df.height, df.filter(pl.col("notes") != "").height)
    return df


def tag_notes() -> dict[str, str]:
    """tag name -> spelled-out form, for the tags that have one."""
    if not tags_path().exists():
        return {}
    df = pl.read_parquet(tags_path()).filter(pl.col("notes") != "")
    return dict(zip(df["name"].to_list(), df["notes"].to_list()))


# -- series of each tag --------------------------------------------------------------------------


def fetch_series_tags(
    client: PoliteClient,
    groups: list[str] | None = None,
    limit_tags: int | None = None,
    deadline: float | None = None,
    stop: threading.Event | None = None,
    bar_position: int | None = None,
) -> dict:
    """Walk the tags of ``groups`` (smallest first) and store each tag's series ids. Resumable per tag."""
    from tqdm import tqdm

    groups = list(groups or DEFAULT_GROUPS)
    if "all" in groups:
        groups = list(ALL_FETCH_GROUPS)
    tags = fetch_tag_list(client)
    state = State("fred-tags")
    todo = tags.filter(pl.col("group_id").is_in(groups)).sort("series_count")
    done = state.ids_with("done")
    pending = [r for r in todo.iter_rows(named=True) if r["name"] not in done]
    already = todo.height - len(pending)
    if limit_tags:
        pending = pending[:limit_tags]
    total_pages = sum(-(-r["series_count"] // PAGE) for r in pending)
    log.info("fred tags: %d tags in groups %s, %d done, %d to fetch (~%d requests)", todo.height, groups, todo.height - len(pending), len(pending), total_pages)
    summary = {"tags_done": 0, "pairs": 0, "requests": 0, "stopped_early": False}
    pairs_dir().mkdir(parents=True, exist_ok=True)

    def should_stop() -> bool:
        return (deadline is not None and time.monotonic() > deadline) or (stop is not None and stop.is_set())

    client.stop_check = should_stop
    bar = tqdm(pending, unit="tag", dynamic_ncols=True, position=bar_position, leave=True, total=todo.height, initial=already)
    for r in bar:
        if should_stop():
            summary["stopped_early"] = True
            break
        name = r["name"]
        bar.set_description(f"fred tags:{name[:24]}")
        ids: list[str] = []
        offset = 0
        try:
            while True:
                if should_stop():
                    raise StopRequested(f"tag {name!r} interrupted at offset {offset}; it restarts next run")
                data = client.get_json(TAG_SERIES_URL, params={"api_key": fred_api_key(), "file_type": "json", "tag_names": name, "limit": PAGE, "offset": offset, "order_by": "series_id"})
                summary["requests"] += 1
                batch = data.get("seriess", [])
                ids.extend(s["id"] for s in batch)
                count = int(data.get("count", 0) or 0)
                offset += PAGE
                if len(batch) < PAGE or offset >= count:
                    break
        except StopRequested as exc:
            log.info("fred tags: %s", exc)
            summary["stopped_early"] = True
            break
        except Exception as exc:  # noqa: BLE001
            log.error("fred tags: %r failed: %s", name, exc)
            state.mark(name, "failed", error=f"{type(exc).__name__}: {str(exc)[:200]}")
            continue
        write_parquet(pl.DataFrame({"series_id": ids}, schema={"series_id": pl.String}), pairs_dir() / f"{_slug(name)}.parquet")
        state.mark(name, "done", group=r["group_id"], n_series=len(ids))
        summary["tags_done"] += 1
        summary["pairs"] += len(ids)
    bar.close()
    log.info("fred tags transfer: %s", client.stats.summary())
    return summary


def build_series_tags_table() -> pl.DataFrame | None:
    """series_id -> sorted list of fetched tag names, from every completed tag file."""
    state = State("fred-tags")
    done = state.events
    frames = []
    for name, ev in done.items():
        if ev.get("status") != "done":
            continue
        p = pairs_dir() / f"{_slug(name)}.parquet"
        if p.exists():
            frames.append(pl.read_parquet(p).with_columns(pl.lit(name).alias("tag")))
    if not frames:
        return None
    pairs = pl.concat(frames)
    table = pairs.group_by("series_id").agg(pl.col("tag").sort().alias("tags"))
    write_parquet(table, series_tags_path())
    log.info("fred series_tags: %d series with tags from %d tags", table.height, len(frames))
    return table


def load_series_tags() -> pl.DataFrame | None:
    p = series_tags_path()
    return pl.read_parquet(p) if p.exists() else None


def run_tags(
    groups: list[str] | None = None,
    limit_tags: int | None = None,
    min_interval: float = 2.0,
    max_per_minute: int | None = 30,
    max_hours: float | None = None,
    stop: threading.Event | None = None,
    deadline: float | None = None,
    rebuild_series: bool = True,
    bar_position: int | None = None,
) -> dict:
    """The whole tags step: tag list, series per tag, the series_tags table, then the FRED series view again."""
    if max_hours is not None:
        deadline = min(deadline or float("inf"), time.monotonic() + max_hours * 3600)
    with make_client(min_interval, max_per_minute) as client:
        summary = fetch_series_tags(client, groups=groups, limit_tags=limit_tags, deadline=deadline, stop=stop, bar_position=bar_position)
    table = build_series_tags_table()
    summary["series_with_tags"] = int(table.height) if table is not None else 0
    if rebuild_series and table is not None and not summary["stopped_early"]:
        from terrastat.series import build_series

        summary["series_rebuild"] = build_series("fred", force=True)
    summary["at"] = now_iso()
    return summary


def tags_status() -> dict:
    state = State("fred-tags")
    total = pl.read_parquet(tags_path()).height if tags_path().exists() else None
    done = state.ids_with("done")
    return {
        "tags_in_fred": total,
        "tags_fetched": len(done),
        "tags_failed": len(state.ids_with("failed")),
        "series_with_tags": pl.read_parquet(series_tags_path()).height if series_tags_path().exists() else 0,
        "pairs": sum(int(state.events[t].get("n_series", 0) or 0) for t in done),
    }
